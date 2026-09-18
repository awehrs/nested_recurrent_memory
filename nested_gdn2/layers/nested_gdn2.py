"""NestedGDN-2 token-mixing layer.

Projects the input into the op's per-level arguments -- read queries, level-0
keys and values, per-level gates and mix weights -- and calls the chunkwise
kernel. Level 0's machinery mirrors fla's GatedDeltaNet2; everything above it is
the promotion apparatus.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution

from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config  # noqa: TC001
from nested_gdn2.ops.chunk import chunk_nested_gdn2
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

__all__ = ["NestedGDN2Attention"]


class NestedGDN2Attention(nn.Module):
    def __init__(self, cfg: NestedGDN2Config, layer_idx: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.L = cfg.num_levels
        H, K = cfg.num_heads, cfg.head_dim
        V = int(cfg.head_dim * cfg.expand_v)
        self.H, self.K, self.V = H, K, V
        self.key_dim, self.value_dim = H * K, H * V
        d = cfg.hidden_size

        assert len(cfg.firing_intervals) == self.L
        assert cfg.firing_intervals[0] == 1
        assert len(cfg.n_queries_per_level) == self.L - 1
        if cfg.use_triton:
            if cfg.op_chunk_size != 64:
                raise ValueError(
                    f"op_chunk_size must be 64 for the triton path, got {cfg.op_chunk_size}"
                )
            for n in cfg.n_queries_per_level:
                if n < 16 or n & (n - 1):
                    raise ValueError(
                        f"n_queries_per_level entries must be powers of two >= 16, got {n}"
                    )

        self.q_proj = nn.Linear(d, self.L * self.key_dim, bias=False)
        self.k_proj = nn.Linear(d, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d, self.value_dim, bias=False)

        if cfg.use_short_conv:
            self.q_conv1d = ShortConvolution(self.L * self.key_dim, cfg.conv_size, activation="silu")
            self.k_conv1d = ShortConvolution(self.key_dim, cfg.conv_size, activation="silu")
            self.v_conv1d = ShortConvolution(self.value_dim, cfg.conv_size, activation="silu")

        self.f_proj = nn.Sequential(nn.Linear(d, V, bias=False), nn.Linear(V, self.key_dim, bias=False))
        self.b_proj = nn.Linear(d, self.key_dim, bias=False)
        self.w_proj = nn.Linear(d, self.value_dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        if self.L > 1:
            # Upper-level gates are projected from the value written: b and w
            # per pair, g once per firing. Both arms carry them.
            self.b_projections = nn.Parameter(torch.randn(self.L - 1, H, V, K) * V**-0.5)
            self.w_projections = nn.Parameter(torch.randn(self.L - 1, H, V, V) * V**-0.5)
            self.g_projections = nn.Parameter(torch.randn(self.L - 1, H, V, K) * V**-0.5)
            if cfg.promotion == "learned":
                n_max = max(cfg.n_queries_per_level)
                self.query_banks = nn.Parameter(torch.randn(self.L - 1, H, n_max, K) * K**-0.5)
                self.key_projections = nn.Parameter(torch.randn(self.L - 1, H, K, V) * V**-0.5)
            else:
                # Merge reads neither; buffers keep them out of the parameter count.
                n_max = max(cfg.n_queries_per_level)
                self.register_buffer("query_banks", torch.zeros(self.L - 1, H, n_max, K), persistent=False)
                self.register_buffer("key_projections", torch.zeros(self.L - 1, H, K, V), persistent=False)
            self.register_buffer(
                "n_queries_per_level", torch.tensor(cfg.n_queries_per_level, dtype=torch.int), persistent=False
            )
        else:
            self.register_buffer("n_queries_per_level", torch.empty(0, dtype=torch.int), persistent=False)
        self.register_buffer(
            "firing_intervals", torch.tensor(cfg.firing_intervals, dtype=torch.int), persistent=False
        )

        self.mix_proj = nn.Linear(d, H * self.L, bias=False)

        self.g_proj = nn.Sequential(nn.Linear(d, V, bias=False), nn.Linear(V, self.value_dim, bias=True))
        self.o_norm = FusedRMSNormGated(V, activation="sigmoid", eps=cfg.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, K, V, L = self.H, self.K, self.V, self.L

        if self.cfg.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(x))
            k, _ = self.k_conv1d(self.k_proj(x))
            v, _ = self.v_conv1d(self.v_proj(x))
        else:
            q, k, v = (F.silu(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj))

        q = rearrange(q, "b t (l h d) -> b t h l d", l=L, h=H)
        k = rearrange(k, "b t (h d) -> b t h d", d=K)
        v = rearrange(v, "b t (h d) -> b t h d", d=V)
        # normalize promotes to fp32 under autocast; fla needs q/k/v to match.
        q = F.normalize(q, dim=-1).to(v.dtype)
        k = F.normalize(k, dim=-1).to(v.dtype)

        g0 = F.softplus(self.f_proj(x).float() + self.dt_bias)
        g0 = rearrange(g0, "b t (h d) -> b t h d", d=K)
        g0 = -self.A_log.float().exp().unsqueeze(-1) * g0
        b0 = rearrange(self.b_proj(x).sigmoid(), "b t (h d) -> b t h d", d=K)
        w0 = rearrange(self.w_proj(x).sigmoid(), "b t (h d) -> b t h d", d=V)

        if L > 1:
            query_banks = self.query_banks
            key_projections = self.key_projections
            b_projections, w_projections = self.b_projections, self.w_projections
            g_projections = self.g_projections
        else:
            query_banks = x.new_empty(0, H, 1, K)
            key_projections = x.new_empty(0, H, K, V)
            b_projections = x.new_empty(0, H, V, K)
            w_projections = x.new_empty(0, H, V, V)
            g_projections = x.new_empty(0, H, V, K)

        # softmax promotes to fp32 under autocast; cast back for one dtype.
        mix = rearrange(self.mix_proj(x), "b t (h l) -> b t h l", h=H).softmax(-1).to(v.dtype)

        op = chunk_nested_gdn2 if self.cfg.use_triton else naive_chunk_nested_gdn2
        o, _ = op(
            q=q,
            k=k,
            v=v,
            g=g0,
            b=b0,
            w=w0,
            mix_weights=mix,
            query_banks=query_banks,
            key_projections=key_projections,
            b_projections=b_projections,
            w_projections=w_projections,
            g_projections=g_projections,
            n_queries_per_level=self.n_queries_per_level,
            firing_intervals=self.firing_intervals,
            L=L,
            chunk_size=self.cfg.op_chunk_size,
            promotion=self.cfg.promotion,
        )

        o = self.o_norm(o, rearrange(self.g_proj(x), "b t (h d) -> b t h d", d=V))
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)")), None, None
