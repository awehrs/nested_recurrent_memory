"""The Triton forward must reproduce naive_chunk_nested_gdn2."""

import pytest
import torch
from fla.utils import device
from test_nested_gdn2 import _rand_inputs_nested

from nested_gdn2.ops.chunk_fwd import chunk_nested_gdn2_fwd
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

TOL = dict(rtol=2e-2, atol=2e-2)

# n_queries must be a power of two >= 16 for the kernel's tl.dot.
CASES = [
    # B,  T,  H,  K,  V,  L,   N,  firing,      n_queries,  BT
    (1, 128, 2, 64, 64, 2, 16, (1, 2), None, 64),
    (2, 256, 2, 64, 64, 2, 16, (1, 2), None, 64),
    (1, 256, 4, 64, 64, 2, 32, (1, 4), None, 64),
    (2, 512, 2, 64, 64, 3, 16, (1, 2, 4), None, 64),
    (1, 512, 2, 64, 64, 3, 32, (1, 2, 4), (32, 16), 64),
    (1, 256, 2, 128, 128, 2, 16, (1, 2), None, 64),
    (1, 256, 2, 64, 64, 2, 16, (1, 1), None, 64),
]
IDS = ["B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-q{}-BT{}".format(*c) for c in CASES]


def _run_both(B, T, H, K, V, L, N, firing, n_queries, BT, *, initial_state=None):
    inputs = _rand_inputs_nested(
        B, T, H, K, V, L, N, torch.float32,
        firing_intervals=firing,
        n_queries_per_level=n_queries,
    )
    (q, k, v, g, b, w, mix, qb, wp, nq, fi) = inputs
    kwargs = dict(
        mix_weights=mix,
        query_banks=qb,
        write_projections=wp,
        n_queries_per_level=nq,
        firing_intervals=fi,
        L=L,
        chunk_size=BT,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_ref, s_ref = naive_chunk_nested_gdn2(q, k, v, g, b, w, **kwargs)
    o_tri, s_tri = chunk_nested_gdn2_fwd(q, k, v, g, b, w, **kwargs)
    return (o_ref, s_ref), (o_tri, s_tri)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), CASES, ids=IDS)
def test_matches_naive(B, T, H, K, V, L, N, firing, n_queries, BT):
    (o_ref, s_ref), (o_tri, s_tri) = _run_both(B, T, H, K, V, L, N, firing, n_queries, BT)
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), CASES[:4], ids=IDS[:4])
def test_matches_naive_with_initial_state(B, T, H, K, V, L, N, firing, n_queries, BT):
    torch.manual_seed(7)
    init = torch.randn(B, H, L, K, V, dtype=torch.float32, device=device) * 0.1
    (o_ref, s_ref), (o_tri, s_tri) = _run_both(
        B, T, H, K, V, L, N, firing, n_queries, BT, initial_state=init
    )
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_l1_reduces_to_level_zero():
    """With L=1 the composition is stock GDN-2 scaled by its mix weight."""
    B, T, H, K, V, BT = 2, 256, 2, 64, 64, 64
    (o_ref, s_ref), (o_tri, s_tri) = _run_both(B, T, H, K, V, 1, 16, (1,), None, BT)
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    assert s_tri.shape == (B, H, 1, K, V)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bad_bt", [16, 32, 128])
def test_rejects_non_64_chunk_size(bad_bt):
    with pytest.raises(ValueError, match="chunk_size must be 64"):
        _run_both(1, 256, 2, 64, 64, 2, 16, (1, 2), None, bad_bt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("T", [16, 32, 64, 200])
def test_short_and_ragged_sequences(T):
    """Sequences shorter than a chunk, or not a multiple of one, must still match."""
    (o_ref, _), (o_tri, _) = _run_both(1, T, 2, 64, 64, 2, 16, (1, 2), None, 64)
    torch.testing.assert_close(o_tri, o_ref, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bad", [4, 8, 24, 48])
def test_rejects_bad_n_queries(bad):
    with pytest.raises(ValueError, match="power of two"):
        _run_both(1, 128, 2, 64, 64, 2, 64, (1, 2), (bad,), 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("L", [1, 2, 3])
def test_autocast_dtype_mix(L):
    """The dtypes the layer actually produces under autocast.

    q/k/v/b/w come out of projections in bf16, but ``g`` is built with an
    explicit .float() and ``mix_weights`` comes from softmax, which autocast
    runs in fp32. fla's kernels require the q/k/v triple to share a dtype, so
    this mix -- bf16 tensors alongside fp32 gates and mix weights -- is the
    combination training hits and the one uniform-dtype tests miss.
    """
    B, T, H, K, V, N, BT = 2, 256, 2, 64, 64, 16, 64
    firing = tuple([1] + [2**i for i in range(1, L)])
    q, k, v, g, b, w, mix, qb, wp, nq, fi = _rand_inputs_nested(
        B, T, H, K, V, L, N, torch.bfloat16, firing_intervals=firing
    )
    g = g.float()
    mix = mix.float()

    kwargs = dict(
        mix_weights=mix, query_banks=qb, write_projections=wp,
        n_queries_per_level=nq, firing_intervals=fi, L=L,
        chunk_size=BT, output_final_state=True,
    )
    o_ref, s_ref = naive_chunk_nested_gdn2(q, k, v, g, b, w, **kwargs)
    o_tri, s_tri = chunk_nested_gdn2_fwd(q, k, v, g, b, w, **kwargs)

    assert o_tri.dtype == o_ref.dtype, f"{o_tri.dtype} != {o_ref.dtype}"
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s_tri, s_ref, rtol=2e-2, atol=2e-2)


# =============================================================================
# permutation invariance
# =============================================================================

PERM_CASES = [
    # B,  T,  H,  K,  V,  L,   N,  firing,     n_queries,  BT
    (1, 256, 2, 64, 64, 2, 16, (1, 2), None, 64),
    (1, 256, 2, 64, 64, 2, 32, (1, 2), None, 64),
    (1, 512, 2, 64, 64, 2, 64, (1, 4), None, 64),
    (1, 512, 2, 64, 64, 3, 16, (1, 2, 4), None, 64),
]
PERM_IDS = ["B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-q{}-BT{}".format(*c) for c in PERM_CASES]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), PERM_CASES, ids=PERM_IDS)
def test_query_bank_permutation_invariance(B, T, H, K, V, L, N, firing, n_queries, BT):
    """The query bank is a set: permuting its rows must not change the output.

    A firing applies all n_queries writes simultaneously against one decayed
    state, so the row order carries no information. Sequencing the writes would
    break this.
    """
    q, k, v, g, b, w, mix, qb, wp, nq, fi = _rand_inputs_nested(
        B, T, H, K, V, L, N, torch.float32,
        firing_intervals=firing,
        n_queries_per_level=n_queries,
    )
    kwargs = dict(
        mix_weights=mix, write_projections=wp, n_queries_per_level=nq,
        firing_intervals=fi, L=L, chunk_size=BT, output_final_state=True,
    )
    o_ref, f_ref = chunk_nested_gdn2_fwd(q, k, v, g, b, w, query_banks=qb, **kwargs)

    for seed in range(3):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(int(nq.min().item()), generator=gen).to(device)
        qb_perm = qb.clone()
        qb_perm[:, :, : perm.numel()] = qb[:, :, perm]
        o, f = chunk_nested_gdn2_fwd(q, k, v, g, b, w, query_banks=qb_perm, **kwargs)

        torch.testing.assert_close(o, o_ref, **TOL)
        torch.testing.assert_close(f, f_ref, **TOL)
