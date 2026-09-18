"""Phase 1 of promotion: turn the level below into pairs to write.

A promotion scheme is a function with this signature and an entry in PROBES.

    probe(state_below, ...) -> keys, values, b, w, g

        state_below  [B, NF, H, K, V]   the level below, gathered at firings
        keys         [B, NF, H, N, K]   where to write
        values       [B, NF, H, N, V]   what to write
        b            [B, NF, H, N, K]   how much to erase there
        w            [B, NF, H, N, V]   how much of the value to write
        g            [B, NF, H, K]      log-decay for the firing, fp32

NF is firings, not chunks. N varies by probe: learned returns ``n_queries``,
merge returns K.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["PROBES", "probe_learned", "probe_merge"]


# Both gates would start at 0.5. Erase compounds -- (1 - b) multiplies the state
# at every firing -- so it starts lower; write scales fresh content and does not.
B_LOGIT_INIT = -4.0
W_LOGIT_INIT = 0.0
# -softplus(0) = -0.69, exp = 0.5. -4 gives exp(g) = 0.982 per firing.
G_LOGIT_INIT = -4.0


def _gates(
    values: torch.Tensor,
    b_proj: torch.Tensor,
    w_proj: torch.Tensor,
    g_proj: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gates projected from the values being written.

    b and w are per pair; g is per firing, averaged over pairs, since decay is
    diagonal in K and multiplies the whole state once.

    The sigmoid runs in fp32: these gates saturate, and bf16 would round the
    tails to exactly 0 or 1. g stays fp32, it feeds exp.

    Args:
        values: [B, NF, H, N, V].
        b_proj: [H, V, K].
        w_proj: [H, V, V].
        g_proj: [H, V, K].
        out_dtype: cast applied to b and w.

    Returns:
        b [B, NF, H, N, K], w [B, NF, H, N, V], g [B, NF, H, K] in fp32.
    """
    vf = values.float()
    b = torch.einsum("hvk,bfhnv->bfhnk", b_proj.float(), vf)
    w = torch.einsum("hvu,bfhnv->bfhnu", w_proj.float(), vf)
    g = torch.einsum("hvk,bfhnv->bfhnk", g_proj.float(), vf)
    b = (b + B_LOGIT_INIT).sigmoid()
    w = (w + W_LOGIT_INIT).sigmoid()
    g = -F.softplus(g + G_LOGIT_INIT).mean(dim=-2)
    return b.to(out_dtype), w.to(out_dtype), g


def probe_learned(
    state_below: torch.Tensor,
    query_bank: torch.Tensor,
    key_proj: torch.Tensor,
    b_proj: torch.Tensor,
    w_proj: torch.Tensor,
    g_proj: torch.Tensor,
    n_queries: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the level below with a learned query bank; write back what comes out.

    Keys are derived from the retrieved values and L2 normalized: the delta rule
    is a contraction only for unit-norm keys.

    Args:
        state_below: [B, NF, H, K, V].
        query_bank: [H, N, K]. Rows past ``n_queries`` are ignored.
        key_proj: [H, K, V]. Value -> write key.
        b_proj: [H, V, K]. Value -> erase gate.
        w_proj: [H, V, V]. Value -> write gate.
        g_proj: [H, V, K]. Value -> log-decay.
        n_queries: probes per firing.
        out_dtype: cast applied to keys, values, b and w; g stays fp32.

    Returns:
        keys [B, NF, H, n_queries, K], unit norm along K, plus values, b, w, g.
    """
    q = query_bank[:, :n_queries].to(state_below.dtype)
    values = torch.einsum("hnk,bfhkv->bfhnv", q, state_below)
    u = torch.einsum("hkv,bfhnv->bfhnk", key_proj.float(), values.float())
    keys = F.normalize(u, dim=-1).to(out_dtype)
    b, w, g = _gates(values, b_proj, w_proj, g_proj, out_dtype)
    return keys, values.to(out_dtype), b, w, g


def probe_merge(
    state_below: torch.Tensor,
    query_bank: torch.Tensor | None = None,
    key_proj: torch.Tensor | None = None,
    b_proj: torch.Tensor | None = None,
    w_proj: torch.Tensor | None = None,
    g_proj: torch.Tensor | None = None,
    n_queries: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carry the level below up whole: each row written to its own address.

    The identity is shared across batch, firing and head, so the returned keys
    are a broadcast with stride 0 on the first three dimensions.

    Args:
        state_below: [B, NF, H, K, V].
        query_bank: ignored; accepted so every probe shares one signature.
        key_proj: ignored.
        b_proj: [H, V, K].
        w_proj: [H, V, V].
        g_proj: [H, V, K].
        n_queries: ignored; this probe always writes K pairs.
        out_dtype: cast applied to keys, values, b and w; g stays fp32.

    Returns:
        keys [B, NF, H, K, K], a broadcast identity, plus values, b, w, g.
    """
    B, NF, H, K, _ = state_below.shape
    eye = torch.eye(K, device=state_below.device, dtype=out_dtype)
    b, w, g = _gates(state_below, b_proj, w_proj, g_proj, out_dtype)
    return eye.expand(B, NF, H, K, K), state_below.to(out_dtype), b, w, g


PROBES = {
    "learned": probe_learned,
    "merge": probe_merge,
}
