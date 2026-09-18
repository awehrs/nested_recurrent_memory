"""Forward write scan: ops.update_fwd.update_state, against naive_update_step."""

import pytest
import torch
from _update_ref import CONSISTENCY_TOL, KERNEL_TOL
from _update_ref import inputs as _inputs
from _update_ref import permuted as _permuted
from _update_ref import scan as _scan

from nested_gdn2.ops.update_fwd import update_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (B, NG, NF, H, K, V, N)
CASES = [(1, 4, 4, 2, 16, 32, 16), (2, 5, 4, 1, 16, 16, 16), (1, 3, 3, 2, 32, 32, 32)]
IDS = ["even", "ragged", "wide"]
ARGS = ("B", "NG", "NF", "H", "K", "V", "N")


@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize(ARGS, CASES, ids=IDS)
def test_matches_naive_scan(B, NG, NF, H, K, V, N, with_init):
    inputs = _inputs(B, NF, H, K, V, N)
    init = torch.randn(B, H, K, V, device="cuda") * K**-0.5 if with_init else None

    states, final = update_state(**inputs, n_groups=NG, initial_state=init)
    ref_states, ref_final = _scan(
        **{k: t.double() for k, t in inputs.items()},
        n_groups=NG,
        initial_state=init.double() if with_init else None,
    )
    torch.testing.assert_close(states, ref_states.float(), **KERNEL_TOL)
    torch.testing.assert_close(final, ref_final.float(), **KERNEL_TOL)


@pytest.mark.parametrize(ARGS, CASES, ids=IDS)
def test_pairs_may_be_written_in_any_order(B, NG, NF, H, K, V, N):
    """All N pairs act on the state as it stood before the firing, so their order
    cannot reach the result. Applying them one after another would make it
    matter, and every other test here would still pass."""
    inputs = _inputs(B, NF, H, K, V, N)
    perm = torch.randperm(N, device="cuda")

    base = update_state(**inputs, n_groups=NG)
    after = update_state(**_permuted(inputs, perm), n_groups=NG)
    for a, c in zip(base, after, strict=True):
        torch.testing.assert_close(a, c, **KERNEL_TOL)


def test_reference_is_order_invariant():
    """If the oracle did not have this property, the test above would be
    checking the wrong thing."""
    B, NF, H, K, V, N = 2, 3, 2, 16, 16, 16
    inputs = _inputs(B, NF, H, K, V, N, dtype=torch.float64)
    perm = torch.randperm(N, device="cuda")

    base = _scan(**inputs, n_groups=NF)
    after = _scan(**_permuted(inputs, perm), n_groups=NF)
    for a, c in zip(base, after, strict=True):
        torch.testing.assert_close(a, c, rtol=0, atol=1e-12)


def test_broadcast_identity_keys_match_a_materialized_one():
    """Merge hands over keys with stride 0 on batch, firing and head. The kernel
    reads them by stride; a contiguous copy must give the same answer."""
    inputs = _inputs(2, 4, 2, 16, 32, N=16, identity_keys=True)
    assert inputs["keys"].stride()[:3] == (0, 0, 0)

    dense = {**inputs, "keys": inputs["keys"].contiguous()}
    for a, c in zip(
        update_state(**inputs, n_groups=4),
        update_state(**dense, n_groups=4),
        strict=True,
    ):
        assert torch.equal(a, c)


@pytest.mark.parametrize("block_v", [16, 32, 64])
def test_block_v_does_not_change_the_result(block_v):
    """Value blocks partition an axis that is free in every operation."""
    inputs = _inputs(2, 4, 2, 16, 64, N=16)
    ref = update_state(**inputs, n_groups=4, block_v=16)
    got = update_state(**inputs, n_groups=4, block_v=block_v)
    for a, c in zip(ref, got, strict=True):
        torch.testing.assert_close(a, c, **CONSISTENCY_TOL)


def test_a_trailing_group_without_a_firing_holds_the_final_state():
    """NG = NF + 1 when the chunk count is not a multiple of the firing interval.
    The last group is read from but never written."""
    inputs = _inputs(2, 4, 2, 16, 32, N=16)
    states, final = update_state(**inputs, n_groups=5)
    assert torch.equal(states[:, -1], final)


def test_states_start_at_the_initial_state():
    inputs = _inputs(2, 4, 2, 16, 32, N=16)
    init = torch.randn(2, 2, 16, 32, device="cuda") * 16**-0.5
    states, _ = update_state(**inputs, n_groups=4, initial_state=init)
    torch.testing.assert_close(states[:, 0], init, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("V", "N", "n_groups", "block_v", "match"),
    [
        (32, 8, 4, 16, "power of two"),
        (32, 16, 4, 8, "block_v"),
        (24, 16, 4, 16, "divisible"),
        (32, 16, 9, 16, "inconsistent"),
    ],
    ids=["n-too-small", "block-too-small", "v-indivisible", "group-mismatch"],
)
def test_rejects_bad_shapes(V, N, n_groups, block_v, match):
    with pytest.raises(ValueError, match=match):
        update_state(**_inputs(1, 4, 1, 16, V, N), n_groups=n_groups, block_v=block_v)


def test_rejects_keys_that_are_not_unit_stride_along_k():
    """Merge's keys are a broadcast, so only the last axis is assumed dense."""
    inputs = _inputs(1, 4, 1, 16, 32, N=16)
    strided = torch.randn(1, 4, 1, 16, 32, device="cuda")[..., ::2]
    assert strided.stride(-1) == 2
    with pytest.raises(ValueError, match="unit-stride"):
        update_state(**{**inputs, "keys": strided}, n_groups=4)
