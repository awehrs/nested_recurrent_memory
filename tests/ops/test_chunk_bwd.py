"""Backward for the promotion scan, against autograd on a torch reference.

The end-to-end check against naive_chunk_nested_gdn2 needs the autograd.Function
and level-0's backward, neither of which exists yet. Until then the oracle is
autograd through a plain-torch scan with identical semantics, kept here rather
than in the op module because it is test scaffolding.
"""

import pytest
import torch
import torch.nn.functional as F
from fla.utils import device

from nested_gdn2.ops.chunk_bwd import promotion_scan_bwd
from nested_gdn2.ops.chunk_fwd import promotion_scan


def torch_scan(h, query_bank, write_proj, g, b, w, fire, nq, init=None):
    """Differentiable reference with the same contract as promotion_scan."""
    B, NT, H, K, V = h.shape
    S = torch.zeros(B, H, K, V, dtype=h.dtype, device=h.device) if init is None else init
    states = []
    for t in range(NT):
        states.append(S)
        if (t + 1) % fire != 0:
            continue
        v_w = torch.einsum("bhkv,hnk->bhnv", h[:, t], query_bank[:, :nq])
        k_w = F.normalize(torch.einsum("hkv,bhnv->bhnk", write_proj, v_w), dim=-1)
        S = S * g[:, t].exp().unsqueeze(-1)
        # Batch write, in the literal (I - K^T K diag(b)) S + K^T (V * w) form.
        # The kernel contracts through NQ instead; keeping this side unfactored
        # means a mistake in that re-association cannot cancel on both sides.
        kb = k_w * b[:, t].unsqueeze(-2)
        gram = torch.einsum("bhnk,bhnj->bhkj", k_w, kb)
        S = (
            S
            - torch.einsum("bhkj,bhjv->bhkv", gram, S)
            + torch.einsum("bhnk,bhnv->bhkv", k_w, v_w * w[:, t].unsqueeze(-2))
        )
    return torch.stack(states, dim=1), S


def make(B, NT, H, K, V, NQ, seed=0, with_init=False):
    torch.manual_seed(seed)
    f = dict(device=device, dtype=torch.float32)
    t = dict(**f, requires_grad=True)
    return dict(
        h=(torch.randn(B, NT, H, K, V, **f) * 0.1).requires_grad_(True),
        query_bank=(torch.randn(H, NQ, K, **f) * K**-0.5).requires_grad_(True),
        write_proj=(torch.randn(H, K, V, **f) * V**-0.5).requires_grad_(True),
        g=F.logsigmoid(torch.randn(B, NT, H, K, **f)).detach().requires_grad_(True),
        b=torch.rand(B, NT, H, K, **t),
        w=torch.rand(B, NT, H, V, **t),
        init=(torch.randn(B, H, K, V, **f) * 0.1).requires_grad_(True) if with_init else None,
    )


TOL = dict(rtol=2e-2, atol=2e-2)

CASES = [
    (2, 8, 2, 32, 32, 16, 1),
    (2, 8, 2, 32, 32, 16, 2),
    (1, 16, 2, 32, 32, 16, 4),
    (2, 8, 1, 64, 64, 16, 2),
    (1, 8, 2, 32, 32, 32, 2),
]
IDS = ["B{}-NT{}-H{}-K{}-V{}-NQ{}-f{}".format(*c) for c in CASES]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "NT", "H", "K", "V", "NQ", "FIRE"), CASES, ids=IDS)
def test_forward_matches_torch_scan(B, NT, H, K, V, NQ, FIRE):
    a = make(B, NT, H, K, V, NQ)
    ref_s, ref_f = torch_scan(**a, fire=FIRE, nq=NQ)
    tri_s, tri_f = promotion_scan(
        h=a["h"], query_bank=a["query_bank"], write_proj=a["write_proj"],
        g=a["g"], b=a["b"], w=a["w"], firing_interval=FIRE, n_queries=NQ,
    )
    # tl.dot uses TF32 on tensor cores, so ~1e-3 relative against an fp32
    # reference is expected; this is the same tolerance test_chunk.py uses
    # against the naive op.
    torch.testing.assert_close(tri_s, ref_s, **TOL)
    torch.testing.assert_close(tri_f, ref_f, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize(("B", "NT", "H", "K", "V", "NQ", "FIRE"), CASES, ids=IDS)
def test_backward_matches_autograd(B, NT, H, K, V, NQ, FIRE, with_init):
    a = make(B, NT, H, K, V, NQ, with_init=with_init)
    torch.manual_seed(1)
    d_states = torch.randn(B, NT, H, K, V, device=device)
    d_final = torch.randn(B, H, K, V, device=device)

    ref_s, ref_f = torch_scan(**a, fire=FIRE, nq=NQ)
    ((ref_s * d_states).sum() + (ref_f * d_final).sum()).backward()

    states, _ = promotion_scan(
        h=a["h"].detach(), query_bank=a["query_bank"].detach(),
        write_proj=a["write_proj"].detach(), g=a["g"].detach(),
        b=a["b"].detach(), w=a["w"].detach(),
        firing_interval=FIRE, n_queries=NQ,
        initial_state=a["init"].detach() if with_init else None,
    )
    out = promotion_scan_bwd(
        h=a["h"].detach(), states=states,
        query_bank=a["query_bank"].detach(), write_proj=a["write_proj"].detach(),
        g=a["g"].detach(), b=a["b"].detach(), w=a["w"].detach(),
        d_states=d_states, d_final=d_final,
        firing_interval=FIRE, n_queries=NQ, has_init=with_init,
    )

    torch.testing.assert_close(out["dh"], a["h"].grad, **TOL)
    torch.testing.assert_close(out["dquery_bank"], a["query_bank"].grad[:, :NQ], **TOL)
    torch.testing.assert_close(out["dwrite_proj"], a["write_proj"].grad, **TOL)
    torch.testing.assert_close(out["dg"], a["g"].grad, **TOL)
    torch.testing.assert_close(out["db"], a["b"].grad, **TOL)
    torch.testing.assert_close(out["dw"], a["w"].grad, **TOL)
    if with_init:
        torch.testing.assert_close(out["dinit"], a["init"].grad, **TOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_rejects_bad_n_queries():
    a = make(1, 4, 1, 32, 32, 16)
    with pytest.raises(ValueError, match="power of two"):
        promotion_scan_bwd(
            h=a["h"], states=torch.zeros_like(a["h"]),
            query_bank=a["query_bank"], write_proj=a["write_proj"],
            g=a["g"], b=a["b"], w=a["w"],
            d_states=torch.zeros_like(a["h"]),
            d_final=torch.zeros(1, 1, 32, 32, device=device),
            firing_interval=1, n_queries=8,
        )
