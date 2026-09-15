"""Additive promotion: the control arm.

Two levels of check. The scan on its own has no tl.dot in it at all -- the
firing is `s * exp(g) + src` -- so it is held to fp32 tolerances. The full op
goes through fla's level 0 and the read einsum, which do use tensor cores, so
those comparisons use the same tolerance as test_chunk.py.
"""

import pytest
import torch
from fla.utils import device
from test_nested_gdn2 import _rand_inputs_nested

from nested_gdn2.ops.additive import additive_scan, additive_scan_bwd
from nested_gdn2.ops.chunk import chunk_nested_gdn2
from nested_gdn2.ops.chunk_fwd import chunk_nested_gdn2_fwd
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

EXACT = dict(rtol=1e-5, atol=1e-5)
TOL = dict(rtol=2e-2, atol=2e-2)


def torch_additive_scan(h, g, fire, init=None):
    """Differentiable reference with the same contract as additive_scan."""
    B, NT, H, K, V = h.shape
    S = torch.zeros(B, H, K, V, dtype=h.dtype, device=h.device) if init is None else init
    states = []
    for t in range(NT):
        states.append(S)
        if (t + 1) % fire != 0:
            continue
        S = S * g[:, t].exp().unsqueeze(-1) + h[:, t]
    return torch.stack(states, dim=1), S


def _scan_inputs(B, NT, H, K, V, seed=0, with_init=False):
    torch.manual_seed(seed)
    f = dict(device=device, dtype=torch.float32)
    return dict(
        h=(torch.randn(B, NT, H, K, V, **f) * 0.1).requires_grad_(True),
        g=torch.nn.functional.logsigmoid(torch.randn(B, NT, H, K, **f)).detach().requires_grad_(True),
        init=(torch.randn(B, H, K, V, **f) * 0.1).requires_grad_(True) if with_init else None,
    )


SCAN_CASES = [
    (2, 8, 2, 32, 32, 1),
    (2, 8, 2, 32, 32, 2),
    (1, 16, 2, 32, 32, 4),
    (2, 8, 1, 64, 64, 2),
]
SCAN_IDS = ["B{}-NT{}-H{}-K{}-V{}-f{}".format(*c) for c in SCAN_CASES]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "NT", "H", "K", "V", "FIRE"), SCAN_CASES, ids=SCAN_IDS)
def test_scan_matches_torch(B, NT, H, K, V, FIRE):
    a = _scan_inputs(B, NT, H, K, V)
    ref_s, ref_f = torch_additive_scan(a["h"], a["g"], fire=FIRE)
    # query_bank, write_proj, b, w and n_queries are ignored; passed as the
    # signature requires so the composition can swap the two scans.
    tri_s, tri_f = additive_scan(
        h=a["h"], query_bank=None, write_proj=None, g=a["g"], b=None, w=None,
        firing_interval=FIRE, n_queries=16,
    )
    torch.testing.assert_close(tri_s, ref_s, **EXACT)
    torch.testing.assert_close(tri_f, ref_f, **EXACT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize(("B", "NT", "H", "K", "V", "FIRE"), SCAN_CASES, ids=SCAN_IDS)
def test_scan_bwd_matches_autograd(B, NT, H, K, V, FIRE, with_init):
    a = _scan_inputs(B, NT, H, K, V, with_init=with_init)
    torch.manual_seed(1)
    d_states = torch.randn(B, NT, H, K, V, device=device)
    d_final = torch.randn(B, H, K, V, device=device)

    ref_s, ref_f = torch_additive_scan(a["h"], a["g"], fire=FIRE, init=a["init"])
    ((ref_s * d_states).sum() + (ref_f * d_final).sum()).backward()

    b = torch.rand(B, NT, H, K, device=device)
    w = torch.rand(B, NT, H, V, device=device)
    states, _ = additive_scan(
        h=a["h"].detach(), query_bank=None, write_proj=None, g=a["g"].detach(),
        b=b, w=w, firing_interval=FIRE, n_queries=16,
        initial_state=a["init"].detach() if with_init else None,
    )
    out = additive_scan_bwd(
        h=a["h"].detach(), states=states, query_bank=None, write_proj=None,
        g=a["g"].detach(), b=b, w=w, d_states=d_states, d_final=d_final,
        firing_interval=FIRE, n_queries=16, has_init=with_init,
    )

    torch.testing.assert_close(out["dh"], a["h"].grad, **EXACT)
    torch.testing.assert_close(out["dg"], a["g"].grad, **EXACT)
    if with_init:
        torch.testing.assert_close(out["dinit"], a["init"].grad, **EXACT)

    # Parameters additive promotion never touches must come back as zeros, not
    # as stale values: the caller scatters these into per-level buffers.
    assert out["dquery_bank"].abs().max() == 0
    assert out["dwrite_proj"].abs().max() == 0
    assert out["db"].abs().max() == 0
    assert out["dw"].abs().max() == 0


OP_CASES = [
    # B,  T,  H,  K,  V,  L,   N,  firing,      n_queries,  BT
    (1, 256, 2, 64, 64, 2, 16, (1, 2), None, 64),
    (1, 512, 2, 64, 64, 3, 16, (1, 2, 4), None, 64),
    (2, 512, 2, 64, 64, 3, 32, (1, 4, 16), None, 64),
]
OP_IDS = ["B{}-T{}-H{}-K{}-V{}-L{}-N{}-fire{}-q{}-BT{}".format(*c) for c in OP_CASES]


def _op_inputs(B, T, H, K, V, L, N, firing, n_queries):
    return _rand_inputs_nested(
        B, T, H, K, V, L, N, torch.float32,
        firing_intervals=firing, n_queries_per_level=n_queries,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), OP_CASES, ids=OP_IDS)
def test_op_matches_naive(B, T, H, K, V, L, N, firing, n_queries, BT):
    q, k, v, g, b, w, mix, qb, wp, nq, fi = _op_inputs(B, T, H, K, V, L, N, firing, n_queries)
    kwargs = dict(
        mix_weights=mix, query_banks=qb, write_projections=wp, n_queries_per_level=nq,
        firing_intervals=fi, L=L, chunk_size=BT, output_final_state=True,
        promotion="additive",
    )
    o_ref, s_ref = naive_chunk_nested_gdn2(q, k, v, g, b, w, **kwargs)
    o_tri, s_tri = chunk_nested_gdn2_fwd(q, k, v, g, b, w, **kwargs)
    torch.testing.assert_close(o_tri, o_ref, **TOL)
    torch.testing.assert_close(s_tri, s_ref, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), OP_CASES, ids=OP_IDS)
def test_op_ignores_query_bank(B, T, H, K, V, L, N, firing, n_queries, BT):
    """The dispatch test: if additive silently fell back to the learned scan,
    randomizing the query bank would move the output. It must not move at all.
    """
    q, k, v, g, b, w, mix, qb, wp, nq, fi = _op_inputs(B, T, H, K, V, L, N, firing, n_queries)
    kwargs = dict(
        mix_weights=mix, n_queries_per_level=nq, firing_intervals=fi, L=L,
        chunk_size=BT, output_final_state=True, promotion="additive",
    )
    o_ref, s_ref = chunk_nested_gdn2_fwd(
        q, k, v, g, b, w, query_banks=qb, write_projections=wp, **kwargs
    )
    o_alt, s_alt = chunk_nested_gdn2_fwd(
        q, k, v, g, b, w,
        query_banks=torch.randn_like(qb) * 100,
        write_projections=torch.randn_like(wp) * 100,
        **kwargs,
    )
    assert (o_alt - o_ref).abs().max() == 0, "additive output moved with the query bank"
    assert (s_alt - s_ref).abs().max() == 0, "additive state moved with the query bank"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "T", "H", "K", "V", "L", "N", "firing", "n_queries", "BT"), OP_CASES, ids=OP_IDS)
def test_autograd_matches_naive(B, T, H, K, V, L, N, firing, n_queries, BT):
    inputs = _op_inputs(B, T, H, K, V, L, N, firing, n_queries)
    names = ["q", "k", "v", "g", "b", "w", "mix_weights", "query_banks", "write_projections"]
    ref = {n: t.clone().requires_grad_(True) for n, t in zip(names, inputs[:9])}
    tri = {n: t.clone().requires_grad_(True) for n, t in zip(names, inputs[:9])}
    nq, fi = inputs[9], inputs[10]

    torch.manual_seed(3)
    d_o = torch.randn(B, T, H, V, device=device)

    o_ref, _ = naive_chunk_nested_gdn2(
        **ref, n_queries_per_level=nq, firing_intervals=fi, L=L,
        chunk_size=BT, promotion="additive",
    )
    (o_ref * d_o).sum().backward()

    o_tri, _ = chunk_nested_gdn2(
        **tri, n_queries_per_level=nq, firing_intervals=fi, L=L,
        chunk_size=BT, promotion="additive",
    )
    (o_tri * d_o).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, **TOL)
    for name in names:
        a, e = tri[name].grad, ref[name].grad
        if name in ("query_banks", "write_projections"):
            assert a.abs().max() == 0, f"{name} got gradient on the additive arm"
            continue
        torch.testing.assert_close(a, e, msg=lambda m, n=name: f"{n}: {m}", **TOL)
