import pytest
import torch
from _op_inputs import rand_inputs as _rand_inputs_nested
from fla.modules.l2norm import l2_norm
from fla.utils import device

from nested_gdn2.ops.naive import (
    naive_chunk_nested_gdn2,
    naive_recurrent_nested_gdn2,
    naive_step_nested_gdn2,
)

# =============================================================================
# naive recurrent
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "L", "N_MAX", "firing", "n_queries", "scale", "dtype", "chunk_size"),
    [
        pytest.param(
            *p,
            id="B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-qry{}-scale{}-dtype{}-{}".format(*p),
        )
        for p in [
            (1, 64, 2, 32, 32, 1, 1, (1,), None, 1.0, torch.float32, 8),
            (2, 128, 2, 64, 64, 2, 4, (1, 2), None, 0.5, torch.float32, 16),
            (2, 128, 3, 64, 64, 3, 4, (1, 2, 3), (4, 4), 1.0, torch.float32, 16),
            (2, 128, 3, 64, 64, 3, 4, (1, 2, 3), (2, 4), 1.0, torch.float32, 16),
            (1, 128, 2, 64, 128, 3, 8, (1, 2, 3), None, 1.0, torch.float16, 16),
        ]
    ],
)
def test_recurrent_naive_nested_gdn2(B, T, H, K, V, L, N_MAX, firing, n_queries, scale, dtype, chunk_size):
    inputs = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, dtype,
        firing_intervals=firing,
        n_queries_per_level=n_queries,
    )
    (
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
        n_queries,
        firing_t,
    ) = inputs

    o, final = naive_recurrent_nested_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        scale=scale,
        output_final_state=True,
        chunk_size=chunk_size,

    )

    assert o.shape == (B, T, H, V)
    assert final.shape == (B, H, L, K, V)
    assert torch.isfinite(o).all(), "Output contains inf/nan"
    assert torch.isfinite(final).all(), "Final state contains inf/nan"
    assert o.dtype == dtype

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_naive_recurrent_nested_gdn2_gradients_flow():
    B, T, H, K, V, L, N_MAX = 1, 16, 2, 16, 16, 2, 2
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32)
    (
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
        n_queries,
        firing_t,
    ) = inputs

    query_banks = query_banks.detach().requires_grad_(True)
    key_projections = key_projections.detach().requires_grad_(True)

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    mix_weights = mix_weights.detach().requires_grad_(True)

    o, _ = naive_recurrent_nested_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=1,
    )

    loss = o.sum()
    loss.backward()

    assert query_banks.grad is not None
    assert query_banks.grad.abs().sum() > 0, "query_banks got zero gradient"
    assert key_projections.grad is not None
    assert key_projections.grad.abs().sum() > 0, "key_projections got zero gradient"

    assert q.grad is not None
    assert q.grad.abs().sum() > 0, "q got zero gradient"
    assert k.grad is not None
    assert k.grad.abs().sum() > 0, "k got zero gradient"
    assert v.grad is not None
    assert v.grad.abs().sum() > 0, "v got zero gradient"
    assert mix_weights.grad is not None
    assert mix_weights.grad.abs().sum() > 0, "mix_weights got zero gradient"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-{}".format(*p))
        for p in [
            (1, 64, 2, 32, 32, torch.float32),
            (2, 128, 3, 64, 64, torch.float32),
            (1, 128, 2, 64, 128, torch.float16),
        ]
    ],
)
def test_naive_recurrent_nested_gdn_L1_matches_stock_gdn2(B, T, H, K, V, dtype):
    """L=1 has no promotion, so it must reduce to fla's stock GDN-2."""
    from fla.ops.gdn2 import naive_recurrent_gdn2

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = l2_norm(k)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.5
    g_flat = torch.empty(B, T, H, K, device=device, dtype=torch.float32).uniform_(-5.0, -0.1).to(dtype)
    b_flat = torch.rand(B, T, H, K, dtype=dtype, device=device)
    w_flat = torch.rand(B, T, H, V, dtype=dtype, device=device)

    # Stock FLA GDN-2 reference
    o_stock, _ = naive_recurrent_gdn2(q, k, v, g_flat, b_flat, w_flat)

    # L=1: only q carries a level axis.
    mix_weights = torch.ones(B, T, H, 1, dtype=dtype, device=device)
    query_banks = torch.empty(0, H, 1, K, dtype=dtype, device=device)
    key_projections = torch.empty(0, H, K, V, dtype=dtype, device=device)
    b_projections = torch.empty(0, H, V, K, dtype=dtype, device=device)
    w_projections = torch.empty(0, H, V, V, dtype=dtype, device=device)
    g_projections = torch.empty(0, H, V, K, dtype=dtype, device=device)
    n_queries = torch.empty(0, dtype=torch.int, device=device)
    firing_intervals = torch.tensor([1], dtype=torch.int, device=device)

    o_nested, _ = naive_recurrent_nested_gdn2(
        q.unsqueeze(-2), k, v, g_flat, b_flat, w_flat,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_intervals,
        L=1,
        chunk_size=1
    )

    if dtype == torch.float16:
        torch.testing.assert_close(o_stock.to(o_nested.dtype), o_nested, rtol=1e-2, atol=1e-3)
    else:
        torch.testing.assert_close(o_stock, o_nested, rtol=1e-5, atol=1e-5)

# =============================================================================
# naive chunked
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "L", "N_MAX", "firing", "n_queries", "scale", "dtype", "chunk_size"),
    [
        pytest.param(
            *p,
            id="B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-qry{}-scale{}-dtype{}-{}".format(*p),
        )
        for p in [
            (1, 64, 2, 32, 32, 1, 1, (1,), None, 1.0, torch.float32, 8),
            (2, 128, 2, 64, 64, 2, 4, (1, 2), None, 0.5, torch.float32, 16),
            (2, 128, 3, 64, 64, 3, 4, (1, 2, 3), (4, 4), 1.0, torch.float32, 16),
            (2, 128, 3, 64, 64, 3, 4, (1, 2, 3), (2, 4), 1.0, torch.float32, 16),
            (1, 128, 2, 64, 128, 3, 8, (1, 2, 3), None, 1.0, torch.float16, 16),
        ]
    ],
)
def test_naive_chunk_nested_gdn2(B, T, H, K, V, L, N_MAX, firing, n_queries, scale, dtype, chunk_size):
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, dtype, firing_intervals=firing)
    (
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
        n_queries,
        firing_t,
    ) = inputs

    o, final = naive_chunk_nested_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        scale=scale,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert o.shape == (B, T, H, V)
    assert final.shape == (B, H, L, K, V)
    assert torch.isfinite(o).all(), "Output contains inf/nan"
    assert torch.isfinite(final).all(), "Final state contains inf/nan"
    assert o.dtype == dtype

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_naive_chunk_nested_gdn2_gradients_flow():
    B, T, H, K, V, L, N_MAX = 1, 256, 2, 16, 16, 2, 2
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=(1, 2))
    (
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
        n_queries,
        firing_t,
    ) = inputs

    query_banks = query_banks.detach().requires_grad_(True)
    key_projections = key_projections.detach().requires_grad_(True)

    q = q.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    mix_weights = mix_weights.detach().requires_grad_(True)

    o, _ = naive_chunk_nested_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=64,
    )

    loss = o.sum()
    loss.backward()

    assert query_banks.grad is not None
    assert query_banks.grad.abs().sum() > 0, "query_banks got zero gradient"
    assert key_projections.grad is not None
    assert key_projections.grad.abs().sum() > 0, "key_projections got zero gradient"

    assert q.grad is not None
    assert q.grad.abs().sum() > 0, "q got zero gradient"
    assert v.grad is not None
    assert v.grad.abs().sum() > 0, "v got zero gradient"
    assert mix_weights.grad is not None
    assert mix_weights.grad.abs().sum() > 0, "mix_weights got zero gradient"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("chunk_size", "g_scale"), [(8, 16.0), (16, 8.0), (32, 4.0), (64, 2.0)])
@pytest.mark.parametrize("L", [1, 2])
def test_naive_chunk_gradients_finite_under_large_decay(chunk_size, g_scale, L):
    """Pairwise decay differences must not overflow: exp(+sum|g|) is inf for |g|*BT > 88,
    and the masked-out inf contributes 0 * inf = NaN to the backward pass.

    Level 0 only; upper-level decay comes from the probe and is bounded by softplus."""
    B, T, H, K, V, N_MAX = 1, 128, 2, 32, 32, 4
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32)
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    g = (g.float() * g_scale).detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    o, _ = naive_chunk_nested_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=chunk_size,
    )
    assert torch.isfinite(o).all(), "forward produced inf/nan"

    o.sum().backward()
    assert torch.isfinite(g.grad).all(), "g received inf/nan gradient"
    assert torch.isfinite(v.grad).all(), "v received inf/nan gradient"

# =============================================================================
# recurrent vs chunked
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "L", "N_MAX", "firing", "chunk_size", "use_initial_state"),
    [
        pytest.param(
            *p,
            id="B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-C{}-init{}".format(*p),
        )
        for p in [
            (1, 64, 2, 32, 32, 1, 1, (1,), 64, False),
            (1, 64, 2, 32, 32, 1, 1, (1,), 64, True),
            (2, 128, 2, 64, 64, 2, 4, (1, 2), 64, False),
            (2, 128, 2, 64, 64, 2, 4, (1, 2), 64, True),
            (2, 256, 3, 64, 64, 3, 4, (1, 2, 3), 64, False),
            (2, 256, 3, 64, 64, 3, 4, (1, 2, 3), 64, True),
        ]
    ],
)
def test_naive_chunk_matches_recurrent(B, T, H, K, V, L, N_MAX, firing, chunk_size, use_initial_state):
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=firing)
    (
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
        n_queries,
        firing_t,
    ) = inputs

    initial_state = None
    if use_initial_state:
        torch.manual_seed(123)
        initial_state = torch.randn(B, H, L, K, V, dtype=torch.float32, device=device) * 0.1

    o_recurrent, final_rec = naive_recurrent_nested_gdn2(
        q, k, v, g, b, w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    o_chunk, final_chunk = naive_chunk_nested_gdn2(
        q, k, v, g, b, w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(o_recurrent, o_chunk, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(final_rec, final_chunk, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promo_scale", [1.0, 8.0])
def test_promotion_normalization_matches_recurrent(promo_scale):
    """The chunked and recurrent paths must normalize write keys identically."""
    B, T, H, K, V, L, N_MAX = 1, 256, 2, 32, 32, 2, 4
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=(1, 2))
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    query_banks = query_banks * promo_scale
    key_projections = key_projections * promo_scale
    kwargs = dict(
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=64,
        output_final_state=True,
    )
    o_rec, s_rec = naive_recurrent_nested_gdn2(q, k, v, g, b, w, **kwargs)
    o_chunk, s_chunk = naive_chunk_nested_gdn2(q, k, v, g, b, w, **kwargs)

    torch.testing.assert_close(o_rec, o_chunk, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s_rec, s_chunk, rtol=1e-4, atol=1e-4)

# =============================================================================
# promotion
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("L", "firing"), [(2, (1, 2)), (3, (1, 2, 4))], ids=["L2", "L3"]
)
@pytest.mark.parametrize(
    "op", [naive_recurrent_nested_gdn2, naive_chunk_nested_gdn2],
    ids=["recurrent", "chunk"],
)
def test_query_bank_permutation_invariance(op, L, firing):
    """The query bank is a set: all n_queries pairs are written against the
    pre-firing state, so their order cannot reach the result."""
    B, T, H, K, V, N_MAX = 1, 256, 2, 32, 32, 16
    inputs = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=firing
    )
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    # Every row is live, so the permutation hits only queries that fire.
    assert int(n_queries[0]) == N_MAX
    perm = torch.randperm(N_MAX, device=device)

    kwargs = dict(
        mix_weights=mix_weights,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=16,
        output_final_state=True,
    )
    o_base, s_base = op(q, k, v, g, b, w, query_banks=query_banks, **kwargs)
    o_perm, s_perm = op(q, k, v, g, b, w, query_banks=query_banks[:, :, perm], **kwargs)

    torch.testing.assert_close(o_base, o_perm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s_base, s_perm, rtol=1e-5, atol=1e-6)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op", [naive_recurrent_nested_gdn2, naive_chunk_nested_gdn2],
    ids=["recurrent", "chunk"],
)
def test_merge_ignores_the_query_bank_and_key_projections(op):
    """Merge carries the level below up whole, so the tensors that choose what to
    promote must not reach its output. The arms must also disagree."""
    B, T, H, K, V, L, N_MAX = 1, 256, 2, 32, 32, 2, 16
    inputs = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=(1, 2)
    )
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    kwargs = dict(
        mix_weights=mix_weights,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=16,
        output_final_state=True,
    )
    chosen = dict(query_banks=query_banks, key_projections=key_projections)
    other = dict(
        query_banks=torch.randn_like(query_banks),
        key_projections=torch.randn_like(key_projections),
    )

    o_merge, s_merge = op(q, k, v, g, b, w, promotion="merge", **chosen, **kwargs)
    o_other, s_other = op(q, k, v, g, b, w, promotion="merge", **other, **kwargs)
    assert torch.equal(o_merge, o_other)
    assert torch.equal(s_merge, s_other)

    o_learned, _ = op(q, k, v, g, b, w, promotion="learned", **chosen, **kwargs)
    assert not torch.allclose(o_merge, o_learned, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promotion", ["learned", "merge"])
@pytest.mark.parametrize("which", ["b_projections", "w_projections"])
def test_upper_gates_depend_on_their_projections(promotion, which):
    """Levels >= 1 project their gates from the value being written, in both arms.
    Level 0 takes b and w directly and must not move."""
    B, T, H, K, V, L, N_MAX = 1, 256, 2, 32, 32, 2, 16
    inputs = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=(1, 2)
    )
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    kwargs = dict(
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=16,
        output_final_state=True,
        promotion=promotion,
    )
    _, s_base = naive_chunk_nested_gdn2(q, k, v, g, b, w, **kwargs)
    perturbed = {**kwargs, which: torch.randn_like(kwargs[which])}
    _, s_perturbed = naive_chunk_nested_gdn2(q, k, v, g, b, w, **perturbed)

    assert not torch.allclose(s_base[:, :, 1], s_perturbed[:, :, 1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(s_base[:, :, 0], s_perturbed[:, :, 0], rtol=0, atol=0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promo_scale", [1.0, 4.0, 16.0, 64.0, 256.0])
@pytest.mark.parametrize("T", [256, 1024])
def test_promotion_stable_under_large_key_projections(promo_scale, T):
    """Level >= 1 write keys are L2-normalized. Without it the gated delta rule stops
    being a contraction once ||k_write|| > ~2 and the state diverges over many firings."""
    B, H, K, V, L, N_MAX = 1, 2, 64, 64, 2, 4
    inputs = _rand_inputs_nested(B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=(1, 2))
    (q, k, v, g, b, w, mix_weights, query_banks, key_projections, b_projections,
     w_projections, g_projections, n_queries, firing_t) = inputs

    query_banks = (query_banks * promo_scale).detach().requires_grad_(True)
    key_projections = (key_projections * promo_scale).detach().requires_grad_(True)

    o, final_state = naive_chunk_nested_gdn2(
        q, k, v, g, b, w,
        mix_weights=mix_weights,
        query_banks=query_banks,
        key_projections=key_projections,
        b_projections=b_projections,
        w_projections=w_projections,
        g_projections=g_projections,
        n_queries_per_level=n_queries,
        firing_intervals=firing_t,
        L=L,
        chunk_size=16,
        output_final_state=True,
    )
    assert torch.isfinite(o).all(), "output diverged"
    assert torch.isfinite(final_state).all(), "state diverged"

    o.sum().backward()
    assert torch.isfinite(query_banks.grad).all()
    assert torch.isfinite(key_projections.grad).all()


# =============================================================================
# decode
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promotion", ["learned", "merge"])
@pytest.mark.parametrize(
    ("L", "firing", "prefill"),
    [(2, (1, 2), 32), (3, (1, 2, 4), 48), (3, (1, 2, 4), 0)],
    ids=["L2-prefill2chunks", "L3-prefill3chunks", "L3-no-prefill"],
)
def test_prefill_then_step_matches_one_pass(L, firing, prefill, promotion):
    """Chunked prefill over whole chunks, then one step per token, must equal a
    single chunked pass. The stepped span crosses several firing boundaries."""
    B, T, H, K, V, N_MAX, CS = 1, 128, 2, 32, 32, 16, 16
    (q, k, v, g, b, w, mix, qb, kp, bp, wp, gp, nq, fi) = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=firing
    )
    shared = dict(
        query_banks=qb, key_projections=kp, b_projections=bp, w_projections=wp,
        g_projections=gp, n_queries_per_level=nq, firing_intervals=fi,
        L=L, chunk_size=CS, promotion=promotion,
    )
    o_ref, s_ref = naive_chunk_nested_gdn2(
        q, k, v, g, b, w, mix_weights=mix, output_final_state=True, **shared
    )

    outs = []
    state = torch.zeros(B, H, L, K, V, device=device)
    if prefill:
        o_pre, state = naive_chunk_nested_gdn2(
            *(x[:, :prefill] for x in (q, k, v, g, b, w)),
            mix_weights=mix[:, :prefill], output_final_state=True, **shared,
        )
        outs.append(o_pre)
    for t in range(prefill, T):
        o_t, state = naive_step_nested_gdn2(
            q[:, t], k[:, t], v[:, t], g[:, t], b[:, t], w[:, t], mix[:, t],
            state, t, **shared,
        )
        outs.append(o_t.unsqueeze(1))

    torch.testing.assert_close(torch.cat(outs, 1), o_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, s_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promotion", ["learned", "merge"])
def test_step_handles_sequences_at_different_positions(promotion):
    """Each row fires on its own tokens, so each must match a single pass over
    that row alone."""
    B, T, H, K, V, L, N_MAX, CS = 2, 96, 2, 32, 32, 3, 16, 16
    firing, prefills = (1, 2, 4), (32, 48)
    (q, k, v, g, b, w, mix, qb, kp, bp, wp, gp, nq, fi) = _rand_inputs_nested(
        B, T, H, K, V, L, N_MAX, torch.float32, firing_intervals=firing
    )
    shared = dict(
        query_banks=qb, key_projections=kp, b_projections=bp, w_projections=wp,
        g_projections=gp, n_queries_per_level=nq, firing_intervals=fi,
        L=L, chunk_size=CS, promotion=promotion,
    )

    # Different prefill lengths, so the rows sit at different positions.
    state = torch.zeros(B, H, L, K, V, device=device)
    for row, p in enumerate(prefills):
        _, s = naive_chunk_nested_gdn2(
            *(x[row : row + 1, :p] for x in (q, k, v, g, b, w)),
            mix_weights=mix[row : row + 1, :p], output_final_state=True, **shared,
        )
        state[row] = s[0]

    pos = torch.tensor(prefills, device=device)
    steps = T - max(prefills)
    outs = []
    for i in range(steps):
        tok = [p + i for p in prefills]
        gather = lambda x, tok=tok: torch.stack([x[r, tok[r]] for r in range(B)])  # noqa: E731
        o_t, state = naive_step_nested_gdn2(
            *(gather(x) for x in (q, k, v, g, b, w, mix)), state, pos + i, **shared,
        )
        outs.append(o_t)
    got = torch.stack(outs, dim=1)

    for row, p in enumerate(prefills):
        o_ref, s_ref = naive_chunk_nested_gdn2(
            *(x[row : row + 1, : p + steps] for x in (q, k, v, g, b, w)),
            mix_weights=mix[row : row + 1, : p + steps], output_final_state=True, **shared,
        )
        torch.testing.assert_close(got[row], o_ref[0, p:], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state[row], s_ref[0], rtol=1e-4, atol=1e-4)
