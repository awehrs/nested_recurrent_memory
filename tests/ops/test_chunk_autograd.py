"""End-to-end: chunk_nested_gdn2 must match naive_chunk_nested_gdn2, gradients included.

This is the first test that exercises the forward composition, both backward
kernels, and the dh_ext injection together against the reference.
"""

import pytest
import torch
from test_nested_gdn2 import _rand_inputs_nested

from nested_gdn2.ops.chunk import chunk_nested_gdn2
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

CASES = [
    # B,  T, H,  K,  V, L,  N, firing,     n_queries
    (1, 128, 2, 64, 64, 2, 16, (1, 1), None),
    (1, 128, 2, 64, 64, 2, 16, (1, 2), None),
    (2, 256, 2, 64, 64, 2, 16, (1, 2), None),
    (1, 256, 4, 64, 64, 2, 32, (1, 4), None),
    (1, 256, 2, 64, 64, 3, 16, (1, 2, 4), None),
    (1, 256, 2, 64, 64, 3, 32, (1, 2, 4), (32, 16)),
]
IDS = ["B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-q{}".format(*c) for c in CASES]

LEAVES = ("q", "k", "v", "g", "b", "w", "mix_weights", "query_banks", "write_projections")


TOL = dict(rtol=2e-2, atol=2e-2)


def _build(B, T, H, K, V, L, N, firing, n_queries):
    names = (*LEAVES, "n_queries_per_level", "firing_intervals")
    vals = _rand_inputs_nested(
        B, T, H, K, V, L, N, torch.float32,
        firing_intervals=firing, n_queries_per_level=n_queries,
    )
    return dict(zip(names, vals))


def _grad(t):
    """Autograd returns None for a leaf the output does not depend on; our
    Function always allocates. A degenerate config -- e.g. NT=2 with fire=2,
    where the top level is written once and never read -- hits this."""
    return torch.zeros_like(t) if t.grad is None else t.grad


def _detached(d):
    return {k: (x.detach().clone().requires_grad_(True) if k in LEAVES else x)
            for k, x in d.items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries"), CASES, ids=IDS)
def test_matches_naive_forward_and_backward(B, T, H, K, V, L, N, firing, n_queries):
    base = _build(B, T, H, K, V, L, N, firing, n_queries)
    ref, tri = _detached(base), _detached(base)

    torch.manual_seed(3)
    d_o = torch.randn(B, T, H, V, device=base["q"].device)

    o_ref, _ = naive_chunk_nested_gdn2(**ref, L=L, chunk_size=64)
    (o_ref * d_o).sum().backward()

    o_tri, _ = chunk_nested_gdn2(**tri, L=L, chunk_size=64)
    (o_tri * d_o).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, **TOL)

    for name in LEAVES:
        torch.testing.assert_close(
            _grad(tri[name]), _grad(ref[name]),
            msg=lambda m, n=name: f"{n}: {m}", **TOL,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_final_state_matches_naive():
    B, T, H, K, V, L = 2, 256, 2, 64, 64, 2
    base = _build(B, T, H, K, V, L, 16, (1, 2), None)
    ref, tri = _detached(base), _detached(base)
    _, s_ref = naive_chunk_nested_gdn2(**ref, L=L, chunk_size=64, output_final_state=True)
    _, s_tri = chunk_nested_gdn2(**tri, L=L, chunk_size=64, output_final_state=True)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_initial_state_gradient_matches_naive():
    B, T, H, K, V, L = 1, 128, 2, 64, 64, 2
    base = _build(B, T, H, K, V, L, 16, (1, 2), None)
    ref, tri = _detached(base), _detached(base)
    torch.manual_seed(5)
    init = torch.randn(B, H, L, K, V, device=base["q"].device) * 0.1
    i_ref = init.clone().requires_grad_(True)
    i_tri = init.clone().requires_grad_(True)
    d_o = torch.randn(B, T, H, V, device=base["q"].device)

    o_ref, _ = naive_chunk_nested_gdn2(**ref, L=L, chunk_size=64, initial_state=i_ref)
    (o_ref * d_o).sum().backward()
    o_tri, _ = chunk_nested_gdn2(**tri, L=L, chunk_size=64, initial_state=i_tri)
    (o_tri * d_o).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(_grad(i_tri), _grad(i_ref), **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("L", [2, 3])
def test_autocast_dtype_mix_backward(L):
    """Same dtype mix as the layer produces, through the full backward.

    Both failures wiring the kernel into the layer were autocast promoting an op
    to fp32 -- norm, then softmax -- and colliding with bf16 inside fla's
    kernels. Uniform-dtype tests cannot see that.
    """
    B, T, H, K, V, N, BT = 2, 256, 2, 64, 64, 16, 64
    firing = tuple([1] + [2**i for i in range(1, L)])
    names = (*LEAVES, "n_queries_per_level", "firing_intervals")
    vals = _rand_inputs_nested(B, T, H, K, V, L, N, torch.bfloat16, firing_intervals=firing)
    base = dict(zip(names, vals))
    base["g"] = base["g"].float()
    base["mix_weights"] = base["mix_weights"].float()

    ref, tri = _detached(base), _detached(base)
    torch.manual_seed(11)
    d_o = torch.randn(B, T, H, V, device=base["q"].device, dtype=torch.bfloat16)

    o_ref, _ = naive_chunk_nested_gdn2(**ref, L=L, chunk_size=BT)
    (o_ref.float() * d_o.float()).sum().backward()
    o_tri, _ = chunk_nested_gdn2(**tri, L=L, chunk_size=BT)
    (o_tri.float() * d_o.float()).sum().backward()

    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    for name in LEAVES:
        a, e = _grad(tri[name]).float(), _grad(ref[name]).float()
        torch.testing.assert_close(
            a, e, rtol=5e-2, atol=5e-2, msg=lambda m, n=name: f"{n}: {m}"
        )
