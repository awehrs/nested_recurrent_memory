"""Inputs for the full op, shared by the naive and chunk tests."""

import torch
import torch.nn.functional as F
from fla.modules.l2norm import l2_norm
from fla.utils import device

__all__ = ["rand_inputs"]


def rand_inputs(B, T, H, K, V, L, N_MAX, dtype, *, firing_intervals=None, n_queries_per_level=None, b_scale=1.0, seed=42,):
    """
    Well-conditioned NestedGDN-2 inputs.

    Note: k is pre-normalized to match FLA convention — ops assume
    caller-normalized keys.
    """
    torch.manual_seed(seed)

    q = torch.randn(B, T, H, L, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = l2_norm(k)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.5

    g = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32)).to(dtype)
    # Level 0's gates only. Levels above derive theirs inside the probe.
    b = torch.rand(B, T, H, K, dtype=dtype, device=device) * b_scale
    w = torch.rand(B, T, H, V, dtype=dtype, device=device)

    mix_logits = torch.randn(B, T, H, L, dtype=dtype, device=device)
    mix_weights = torch.softmax(mix_logits, dim=-1)

    if L > 1:
        query_banks = torch.randn(L - 1, H, N_MAX, K, dtype=dtype, device=device) * (K**-0.5)
        key_projections = torch.randn(L - 1, H, K, V, dtype=dtype, device=device) * (V**-0.5)
        b_projections = torch.randn(L - 1, H, V, K, dtype=dtype, device=device) * (V**-0.5)
        w_projections = torch.randn(L - 1, H, V, V, dtype=dtype, device=device) * (V**-0.5)
        g_projections = torch.randn(L - 1, H, V, K, dtype=dtype, device=device) * (V**-0.5)
        if n_queries_per_level is None:
            n_queries_per_level = torch.full((L - 1,), N_MAX, dtype=torch.int, device=device)
        else:
            assert len(n_queries_per_level) == L - 1, \
                f"n_queries_per_level must have length L-1={L-1}, got {len(n_queries_per_level)}"

            assert all(n <= N_MAX for n in n_queries_per_level), \
                f"all n_queries_per_level entries must be <= N_MAX={N_MAX}"

            n_queries_per_level = torch.tensor(n_queries_per_level, dtype=torch.int, device=device)
    else:
        query_banks = torch.empty(0, H, N_MAX, K, dtype=dtype, device=device)
        key_projections = torch.empty(0, H, K, V, dtype=dtype, device=device)
        b_projections = torch.empty(0, H, V, K, dtype=dtype, device=device)
        w_projections = torch.empty(0, H, V, V, dtype=dtype, device=device)
        g_projections = torch.empty(0, H, V, K, dtype=dtype, device=device)
        n_queries_per_level = torch.empty(0, dtype=torch.int, device=device)

    if firing_intervals is None:
        firing_intervals = [1] + [2**i for i in range(1, L)]

    firing_intervals = torch.tensor(list(firing_intervals), dtype=torch.int, device=device)

    return (
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights,
        query_banks,
        key_projections,
        b_projections,
        w_projections,
        g_projections,
        n_queries_per_level,
        firing_intervals,
    )
