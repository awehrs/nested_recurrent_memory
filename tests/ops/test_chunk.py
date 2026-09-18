"""The triton composition against the naive reference, forward and backward.

The only test that exercises the level chaining, both update kernels, the probe
replay and the dh_ext injection together.
"""

import pytest
import torch
from _op_inputs import rand_inputs

from nested_gdn2.ops.chunk import chunk_nested_gdn2
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# fla's level-0 kernels bring their own precision on top of tl.dot's TF32.
TOL = dict(rtol=2e-2, atol=2e-2)

#  B,   T, H,   K,   V, L,  N, firing,    n_queries
CASES = [
    (1, 128, 2, 64, 64, 2, 16, (1, 2), None),
    (2, 256, 2, 64, 64, 2, 16, (1, 2), None),
    (1, 256, 4, 64, 64, 2, 32, (1, 4), None),
    (2, 512, 2, 64, 64, 3, 16, (1, 2, 4), None),
    (1, 512, 2, 64, 64, 3, 32, (1, 2, 4), (32, 16)),
    (1, 256, 2, 128, 128, 2, 16, (1, 2), None),
    (1, 256, 2, 64, 64, 2, 16, (1, 1), None),
]
IDS = ["B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-q{}".format(*c) for c in CASES]
ARGS = ("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries")

NAMES = (
    "q", "k", "v", "g", "b", "w", "mix_weights", "query_banks",
    "key_projections", "b_projections", "w_projections", "g_projections",
    "n_queries_per_level", "firing_intervals",
)
LEAVES = NAMES[:12]


def _build(B, T, H, K, V, L, N, firing, n_queries):
    vals = rand_inputs(
        B, T, H, K, V, L, N, torch.float32,
        firing_intervals=firing,
        n_queries_per_level=n_queries,
    )
    return dict(zip(NAMES, vals, strict=True))


def _detached(d):
    return {
        k: (x.detach().clone().requires_grad_(True) if k in LEAVES else x)
        for k, x in d.items()
    }


def _grad(t):
    """Autograd returns None for a leaf the output does not depend on; the
    Function always allocates. A top level written once and never read hits it."""
    return torch.zeros_like(t) if t.grad is None else t.grad


@pytest.mark.parametrize("promotion", ["learned", "merge"])
@pytest.mark.parametrize(ARGS, CASES, ids=IDS)
def test_matches_naive_forward_and_backward(
    B, T, H, K, V, L, N, firing, n_queries, promotion
):
    base = _build(B, T, H, K, V, L, N, firing, n_queries)
    tri, ref = _detached(base), _detached(base)
    kw = dict(L=L, chunk_size=64, promotion=promotion)

    o_tri, _ = chunk_nested_gdn2(**tri, **kw)
    o_ref, _ = naive_chunk_nested_gdn2(**ref, **kw)
    torch.testing.assert_close(o_tri, o_ref, **TOL)

    torch.manual_seed(0)
    do = torch.randn_like(o_ref)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()
    for name in LEAVES:
        torch.testing.assert_close(
            _grad(tri[name]), _grad(ref[name]), **TOL, msg=lambda m, n=name: f"{n}: {m}"
        )


@pytest.mark.parametrize("promotion", ["learned", "merge"])
def test_final_state_matches_naive(promotion):
    base = _build(2, 256, 2, 64, 64, 3, 16, (1, 2, 4), None)
    kw = dict(L=3, chunk_size=64, output_final_state=True, promotion=promotion)
    _, s_tri = chunk_nested_gdn2(**base, **kw)
    _, s_ref = naive_chunk_nested_gdn2(**base, **kw)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.parametrize("promotion", ["learned", "merge"])
def test_initial_state_gradient_matches_naive(promotion):
    base = _build(2, 256, 2, 64, 64, 2, 16, (1, 2), None)
    tri, ref = _detached(base), _detached(base)

    torch.manual_seed(1)
    init = torch.randn(2, 2, 2, 64, 64, device="cuda") * 0.1
    i_tri = init.detach().clone().requires_grad_(True)
    i_ref = init.detach().clone().requires_grad_(True)
    kw = dict(L=2, chunk_size=64, promotion=promotion)

    o_tri, _ = chunk_nested_gdn2(**tri, initial_state=i_tri, **kw)
    o_ref, _ = naive_chunk_nested_gdn2(**ref, initial_state=i_ref, **kw)
    torch.testing.assert_close(o_tri, o_ref, **TOL)

    o_tri.sum().backward()
    o_ref.sum().backward()
    torch.testing.assert_close(_grad(i_tri), _grad(i_ref), **TOL)


def test_l1_reduces_to_level_zero():
    """No promotion at L=1, so the upper-level path is never entered."""
    base = _build(2, 256, 2, 64, 64, 1, 16, (1,), None)
    kw = dict(L=1, chunk_size=64, output_final_state=True)
    o_tri, s_tri = chunk_nested_gdn2(**base, **kw)
    o_ref, s_ref = naive_chunk_nested_gdn2(**base, **kw)
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.parametrize("T", [64, 192, 320])
def test_short_and_ragged_sequences(T):
    """T not a multiple of chunk_size * firing, so the last group never fires."""
    base = _build(1, T, 2, 64, 64, 3, 16, (1, 2, 4), None)
    kw = dict(L=3, chunk_size=64, output_final_state=True)
    o_tri, s_tri = chunk_nested_gdn2(**base, **kw)
    o_ref, s_ref = naive_chunk_nested_gdn2(**base, **kw)
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_dtype_mix(dtype):
    """Under autocast the probe's casts and fla's dtype requirements have to agree."""
    base = _detached(_build(1, 256, 2, 64, 64, 3, 16, (1, 2, 4), None))
    with torch.autocast("cuda", dtype=dtype):
        o, _ = chunk_nested_gdn2(**base, L=3, chunk_size=64)
    assert torch.isfinite(o).all()
    o.sum().backward()
    for name in LEAVES:
        assert torch.isfinite(_grad(base[name])).all(), name


@pytest.mark.parametrize("bad_bt", [16, 32, 128])
def test_rejects_non_64_chunk_size(bad_bt):
    base = _build(1, 256, 2, 64, 64, 2, 16, (1, 2), None)
    with pytest.raises((ValueError, AssertionError)):
        chunk_nested_gdn2(**base, L=2, chunk_size=bad_bt)


@pytest.mark.parametrize("bad", [4, 8, 24])
def test_rejects_bad_n_queries(bad):
    base = _build(1, 256, 2, 64, 64, 2, 32, (1, 2), (bad,))
    with pytest.raises(ValueError, match="power of two"):
        chunk_nested_gdn2(**base, L=2, chunk_size=64)
