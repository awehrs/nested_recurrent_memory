"""Inputs and reference scan shared by the update forward and backward tests."""

import torch
import torch.nn.functional as F

from nested_gdn2.ops.naive import naive_update_step

__all__ = ["CONSISTENCY_TOL", "KERNEL_TOL", "inputs", "permuted", "scan"]

# tl.dot runs TF32 against a float64 reference; training is bf16, coarser still.
KERNEL_TOL = dict(rtol=2e-2, atol=1e-3)
# Kernel against itself: TF32 rounds both sides alike, only summation order moves.
CONSISTENCY_TOL = dict(rtol=1e-4, atol=1e-6)


def inputs(B, NF, H, K, V, N, *, dtype=torch.float32, seed=0, identity_keys=False):
    """Pairs and gates as a probe would hand them over."""
    torch.manual_seed(seed)
    f = dict(device="cuda", dtype=dtype)
    keys = (
        torch.eye(K, **f).expand(B, NF, H, K, K)
        if identity_keys
        else F.normalize(torch.randn(B, NF, H, N, K, **f), dim=-1)
    )
    return {
        "keys": keys,
        "values": torch.randn(B, NF, H, N, V, **f) * V**-0.5,
        "g": -F.softplus(torch.randn(B, NF, H, K, **f)),
        "b": torch.sigmoid(torch.randn(B, NF, H, N, K, **f)),
        "w": torch.sigmoid(torch.randn(B, NF, H, N, V, **f)),
    }


def scan(keys, values, g, b, w, n_groups, initial_state=None):
    """Group loop over naive_update_step, matching update_state's contract."""
    B, NF, H, _, K = keys.shape
    S = (
        initial_state
        if initial_state is not None
        else torch.zeros(
            B, H, K, values.shape[-1], dtype=values.dtype, device=values.device
        )
    )
    states = []
    for j in range(n_groups):
        states.append(S)
        if j < NF:
            S = naive_update_step(
                S, keys[:, j], values[:, j], b[:, j], w[:, j], g[:, j]
            )
    return torch.stack(states, dim=1), S


def permuted(inp, perm):
    """Reorder the pairs within every firing."""
    return {k: (t[:, :, :, perm] if t.dim() == 5 else t) for k, t in inp.items()}
