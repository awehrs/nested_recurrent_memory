"""Backward write scan: ops.update_bwd.update_state_bwd.

Every gradient is checked against autograd through the reference scan, run in
float64 so the comparison measures the kernel.
"""

import pytest
import torch
from _update_ref import CONSISTENCY_TOL
from _update_ref import inputs as _inputs
from _update_ref import scan as _scan

from nested_gdn2.ops.update_bwd import update_state_bwd
from nested_gdn2.ops.update_fwd import update_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (B, NG, NF, H, K, V, N)
CASES = [(1, 4, 4, 2, 16, 32, 16), (2, 5, 4, 1, 16, 16, 16), (1, 3, 3, 2, 32, 32, 32)]
IDS = ["even", "ragged", "wide"]
ARGS = ("B", "NG", "NF", "H", "K", "V", "N")
NAMES = ("dkeys", "dvalues", "dg", "db", "dw")

# Gradients chain more TF32 dots than the forward does. atol covers the near-zero
# tail, where dkeys differences two O(1) intermediates and inherits their error;
# rtol still governs every element of consequence.
GRAD_TOL = dict(rtol=2e-2, atol=8e-3)


def _incoming(states, final, seed=0):
    """Incoming gradients, seeded -- otherwise a borderline tolerance flakes
    between runs and the failure cannot be reproduced."""
    g = torch.Generator(device=states.device).manual_seed(seed)
    return (
        torch.randn(states.shape, generator=g, device=states.device),
        torch.randn(final.shape, generator=g, device=final.device),
    )


def _reference_grads(inputs, init, d_states, d_final, NG):
    """Autograd through the reference scan, in float64."""
    leaves = {k: t.double().detach().requires_grad_(True) for k, t in inputs.items()}
    init64 = init.double().detach().requires_grad_(True) if init is not None else None

    states, final = _scan(**leaves, n_groups=NG, initial_state=init64)
    loss = (states * d_states.double()).sum() + (final * d_final.double()).sum()

    wrt = [leaves[k] for k in ("keys", "values", "g", "b", "w")]
    if init64 is not None:
        wrt.append(init64)
    return torch.autograd.grad(loss, wrt)


@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize(ARGS, CASES, ids=IDS)
def test_matches_autograd(B, NG, NF, H, K, V, N, with_init):
    inputs = _inputs(B, NF, H, K, V, N)
    init = torch.randn(B, H, K, V, device="cuda") * K**-0.5 if with_init else None

    states, final = update_state(**inputs, n_groups=NG, initial_state=init)
    d_states, d_final = _incoming(states, final)

    out = update_state_bwd(
        states=states, **inputs, d_states=d_states, d_final=d_final, has_init=with_init
    )
    ref = _reference_grads(inputs, init, d_states, d_final, NG)

    for name, expected in zip(NAMES, ref, strict=False):
        torch.testing.assert_close(out[name], expected.float(), **GRAD_TOL)
    if with_init:
        torch.testing.assert_close(out["dinit"], ref[-1].float(), **GRAD_TOL)


@pytest.mark.parametrize("block_v", [16, 32, 64])
def test_block_v_does_not_change_the_gradients(block_v):
    """dkeys, db and dg are accumulated per value block and summed by the wrapper.
    Changing how many blocks there are is what exposes a bug in that reduction."""
    inputs = _inputs(2, 4, 2, 16, 64, N=16)
    states, final = update_state(**inputs, n_groups=4, block_v=16)
    d_states, d_final = _incoming(states, final)

    kw = dict(states=states, **inputs, d_states=d_states, d_final=d_final)
    ref = update_state_bwd(**kw, block_v=16)
    got = update_state_bwd(**kw, block_v=block_v)
    for name in NAMES:
        torch.testing.assert_close(got[name], ref[name], **CONSISTENCY_TOL)


def test_broadcast_identity_keys_match_a_materialized_one():
    """Merge's keys arrive with stride 0 on batch, firing and head. The pair
    gradients must not depend on whether they were expanded."""
    inputs = _inputs(2, 4, 2, 16, 32, N=16, identity_keys=True)
    dense = {**inputs, "keys": inputs["keys"].contiguous()}

    states, final = update_state(**inputs, n_groups=4)
    d_states, d_final = _incoming(states, final)

    a = update_state_bwd(states=states, **inputs, d_states=d_states, d_final=d_final)
    c = update_state_bwd(states=states, **dense, d_states=d_states, d_final=d_final)
    for name in NAMES:
        assert torch.equal(a[name], c[name])


def test_identity_keys_gradients_match_autograd():
    """Merge discards dkeys -- an expanded eye has no grad path -- but the other
    four still have to be right."""
    inputs = _inputs(2, 4, 2, 16, 32, N=16, identity_keys=True)
    states, final = update_state(**inputs, n_groups=4)
    d_states, d_final = _incoming(states, final)

    out = update_state_bwd(
        states=states, **inputs, d_states=d_states, d_final=d_final
    )
    ref = _reference_grads(inputs, None, d_states, d_final, 4)
    for name, expected in zip(NAMES, ref, strict=False):
        if name == "dkeys":
            continue
        torch.testing.assert_close(out[name], expected.float(), **GRAD_TOL)


@pytest.mark.parametrize("has_init", [False, True])
def test_dinit_is_returned_only_when_asked_for(has_init):
    inputs = _inputs(2, 4, 2, 16, 32, N=16)
    states, final = update_state(**inputs, n_groups=4)
    d_states, d_final = _incoming(states, final)

    out = update_state_bwd(
        states=states, **inputs, d_states=d_states, d_final=d_final, has_init=has_init
    )
    assert ("dinit" in out) is has_init


@pytest.mark.parametrize(
    ("N", "block_v", "match"),
    [(8, 16, "power of two"), (16, 8, "block_v")],
    ids=["n-too-small", "block-too-small"],
)
def test_rejects_bad_shapes(N, block_v, match):
    inputs = _inputs(1, 4, 1, 16, 32, N)
    states = torch.zeros(1, 4, 1, 16, 32, device="cuda")
    final = torch.zeros(1, 1, 16, 32, device="cuda")
    with pytest.raises(ValueError, match=match):
        update_state_bwd(
            states=states,
            **inputs,
            d_states=torch.zeros_like(states),
            d_final=torch.zeros_like(final),
            block_v=block_v,
        )
