"""Contract for the promotion probes.

There is one probe, shared by the naive and triton paths, so there is nothing to
compare it against. These are properties instead.
"""

import pytest
import torch
import torch.nn.functional as F

from nested_gdn2.ops.probe import (
    B_LOGIT_INIT,
    G_LOGIT_INIT,
    PROBES,
    W_LOGIT_INIT,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# N != K so merge's "always K pairs" stays distinguishable from n_queries. The
# kernel needs N >= 16; the probe does not, and nothing here reaches the kernel.
B, NF, H, K, V, N = 2, 4, 2, 16, 16, 8
ARMS = list(PROBES)


def _inputs(seed=0, **overrides):
    """Keys match the probe signatures, so the result splats."""
    torch.manual_seed(seed)
    f = dict(device="cuda", dtype=torch.float32)
    out = {
        "state_below": torch.randn(B, NF, H, K, V, **f) * K**-0.5,
        "query_bank": torch.randn(H, N, K, **f) * K**-0.5,
        "key_proj": torch.randn(H, K, V, **f) * V**-0.5,
        "b_proj": torch.randn(H, V, K, **f) * V**-0.5,
        "w_proj": torch.randn(H, V, V, **f) * V**-0.5,
        "g_proj": torch.randn(H, V, K, **f) * V**-0.5,
        "n_queries": N,
    }
    out.update(overrides)
    return out


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_shapes_and_dtypes(arm, out_dtype):
    keys, values, b, w, g = PROBES[arm](**_inputs(), out_dtype=out_dtype)
    n = K if arm == "merge" else N
    assert keys.shape == (B, NF, H, n, K)
    assert values.shape == (B, NF, H, n, V)
    assert b.shape == (B, NF, H, n, K)
    assert w.shape == (B, NF, H, n, V)
    assert g.shape == (B, NF, H, K)
    for t in (keys, values, b, w):
        assert t.dtype == out_dtype
    assert g.dtype == torch.float32, "g feeds exp; it stays fp32"


def test_learned_keys_are_unit_norm():
    """The delta rule is a contraction only for unit-norm keys."""
    keys, _, _, _, _ = PROBES["learned"](**_inputs(), out_dtype=torch.float32)
    torch.testing.assert_close(
        keys.norm(dim=-1), torch.ones_like(keys[..., 0]), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("arm", ARMS)
def test_gate_offsets_are_applied(arm):
    """Zero projections leave only the offsets, pinning the init exactly."""
    zeros = {n: torch.zeros_like(_inputs()[n]) for n in ("b_proj", "w_proj", "g_proj")}
    _, _, b, w, g = PROBES[arm](**_inputs(**zeros), out_dtype=torch.float32)

    def const(x):
        return torch.tensor(x, device="cuda")

    assert torch.allclose(b, torch.sigmoid(const(B_LOGIT_INIT)), atol=1e-7)
    assert torch.allclose(w, torch.sigmoid(const(W_LOGIT_INIT)), atol=1e-7)
    assert torch.allclose(g, -F.softplus(const(G_LOGIT_INIT)), atol=1e-7)


@pytest.mark.parametrize("arm", ARMS)
def test_retention_survives_many_firings(arm):
    """Not correctness: every implementation agrees on a state decayed to zero.
    Retention is exp(g) * (1 - b) per firing and a level sees dozens."""
    _, _, b, _, g = PROBES[arm](**_inputs(), out_dtype=torch.float32)
    per_firing = g.exp() * (1 - b.max(dim=-2).values)
    assert per_firing.min() ** 32 > 1e-2


def test_merge_keys_are_a_broadcast_identity():
    """The update addresses keys by stride, so the zeros here are load-bearing."""
    keys, _, _, _, _ = PROBES["merge"](**_inputs(), out_dtype=torch.float32)
    assert keys.stride() == (0, 0, 0, K, 1)
    assert torch.equal(keys[0, 0, 0], torch.eye(K, device="cuda"))
    assert keys.untyped_storage().size() < keys.numel() * keys.element_size()


def test_merge_is_lossless():
    """Identity addresses reproduce the state below exactly."""
    inputs = _inputs()
    keys, values, _, _, _ = PROBES["merge"](**inputs, out_dtype=torch.float32)
    assert torch.equal(keys.transpose(-2, -1) @ values, inputs["state_below"])


def test_merge_ignores_the_bank_and_key_projection():
    """The arms differ in what gets promoted and in nothing else."""
    base = PROBES["merge"](**_inputs(), out_dtype=torch.float32)
    other = _inputs()
    other["query_bank"] = torch.randn_like(other["query_bank"])
    other["key_proj"] = torch.randn_like(other["key_proj"])
    after = PROBES["merge"](**other, out_dtype=torch.float32)
    for a, c in zip(base, after, strict=True):
        assert torch.equal(a, c)


def test_permuting_the_bank_permutes_pairs_but_not_decay():
    """Pairs follow the permutation; g is a mean over pairs, so it does not."""
    inputs = _inputs()
    perm = torch.randperm(N, device="cuda")
    shuffled = _inputs(query_bank=inputs["query_bank"][:, perm])

    k0, v0, b0, w0, g0 = PROBES["learned"](**inputs, out_dtype=torch.float32)
    k1, v1, b1, w1, g1 = PROBES["learned"](**shuffled, out_dtype=torch.float32)
    # Not bitwise: a permuted operand reassociates the sums by about one ulp.
    for a, c in ((k0, k1), (v0, v1), (b0, b1), (w0, w1)):
        torch.testing.assert_close(a[:, :, :, perm], c, rtol=0, atol=1e-6)
    torch.testing.assert_close(g0, g1, rtol=0, atol=1e-6)


@pytest.mark.parametrize("arm", ARMS)
def test_decay_is_the_mean_over_pairs(arm):
    """Mean, not sum: a sum would tie a level's horizon to n_queries."""
    inputs = _inputs()
    _, values, _, _, g = PROBES[arm](**inputs, out_dtype=torch.float32)
    per_pair = -F.softplus(
        torch.einsum("hvk,bfhnv->bfhnk", inputs["g_proj"], values) + G_LOGIT_INIT
    )
    torch.testing.assert_close(g, per_pair.mean(dim=-2), rtol=1e-5, atol=1e-6)


def test_rows_past_n_queries_are_ignored():
    wide = _inputs(
        query_bank=torch.randn(H, 2 * N, K, device="cuda") * K**-0.5
    )
    base = PROBES["learned"](**wide, out_dtype=torch.float32)

    tail = wide["query_bank"].clone()
    tail[:, N:] = torch.randn_like(tail[:, N:])
    after = PROBES["learned"](**{**wide, "query_bank": tail}, out_dtype=torch.float32)
    for a, c in zip(base, after, strict=True):
        assert torch.equal(a, c)
