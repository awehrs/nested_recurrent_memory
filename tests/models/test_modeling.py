"""End-to-end model checks. Requires CUDA: fla's norms are triton-only."""

import math

import pytest
import torch

from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config
from nested_gdn2.models.modeling_nested_gdn2 import NestedGDN2ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# A level's decay and erase gates only multiply the state already present, so
# they see no gradient until two of its firings have had their results read:
# one to populate the state, one to have something to decay. That needs
# NT >= 2 * f + 1 chunks. Sequence lengths below are sized accordingly.
CASES = [
    # num_levels, firing_intervals, n_queries_per_level, T
    (1, (1,), (), 256),
    (2, (1, 2), (16,), 384),
    (2, (1, 4), (32,), 640),
    (3, (1, 2, 4), (16, 16), 704),
]
IDS = ["L{}-fire{}-q{}-T{}".format(*c) for c in CASES]


def _min_tokens(firing, chunk_size=64):
    return (2 * max(firing) + 1) * chunk_size


def _config(num_levels, firing, n_queries, **kw):
    return NestedGDN2Config(
        vocab_size=512, hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64,
        num_levels=num_levels, firing_intervals=firing, n_queries_per_level=n_queries,
        fuse_cross_entropy=False, **kw,
    )


@pytest.mark.parametrize(("num_levels", "firing", "n_queries", "T"), CASES, ids=IDS)
def test_forward_backward(num_levels, firing, n_queries, T):
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(num_levels, firing, n_queries)).cuda()
    ids = torch.randint(0, 512, (2, T), device="cuda")

    out = model(input_ids=ids, labels=ids)

    # Untrained loss must sit at ln(vocab); a wildly different value means the
    # embedding init or the tying is wrong, not that training is hard.
    assert abs(out.loss.item() - math.log(512)) < 0.5, out.loss.item()
    assert out.logits.shape == (2, T, 512)
    assert torch.isfinite(out.loss)

    out.loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"no gradient reached: {no_grad}"
    for n, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


@pytest.mark.parametrize(("num_levels", "firing", "n_queries", "T"), CASES, ids=IDS)
def test_promotion_params_receive_gradient(num_levels, firing, n_queries, T):
    """Levels >= 1 must actually be driving the loss, not sitting inert."""
    if num_levels == 1:
        pytest.skip("no promotion at L=1")
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(num_levels, firing, n_queries)).cuda()
    ids = torch.randint(0, 512, (2, T), device="cuda")
    model(input_ids=ids, labels=ids).loss.backward()

    assert _min_tokens(firing) <= T, (
        f"T={T} is too short for firing={firing}; the upper gates cannot receive "
        f"gradient below {_min_tokens(firing)} tokens"
    )

    # Per level, not a global max: a single trained level would otherwise mask a
    # higher one that never fired twice.
    for i, layer in enumerate(model.model.layers):
        for name in ("query_banks", "write_projections", "g_hi", "b_hi", "w_hi"):
            g = getattr(layer.attn, name).grad
            assert g is not None, f"layer {i} {name} got no gradient"
            for lvl in range(num_levels - 1):
                assert g[lvl].abs().max() > 0, (
                    f"layer {i} {name} level {lvl + 1} gradient is all zero"
                )


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(2, (1, 2), (16,))).cuda()
    ids = torch.randint(0, 512, (2, 256), device="cuda")

    loss_a = model(input_ids=ids, labels=ids).loss
    grads_a = torch.autograd.grad(loss_a, [p for p in model.parameters() if p.requires_grad])

    model.model.gradient_checkpointing_enable()
    loss_b = model(input_ids=ids, labels=ids).loss
    grads_b = torch.autograd.grad(loss_b, [p for p in model.parameters() if p.requires_grad])

    torch.testing.assert_close(loss_a, loss_b, rtol=1e-4, atol=1e-4)
    for a, b in zip(grads_a, grads_b):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


def test_use_cache_rejected():
    model = NestedGDN2ForCausalLM(_config(2, (1, 2), (16,))).cuda()
    ids = torch.randint(0, 512, (1, 128), device="cuda")
    with pytest.raises(NotImplementedError, match="use_cache"):
        model(input_ids=ids, use_cache=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast(dtype):
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(2, (1, 2), (16,))).cuda()
    ids = torch.randint(0, 512, (2, 256), device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        out = model(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_fused_cross_entropy_matches():
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(2, (1, 2), (16,))).cuda().train()
    ids = torch.randint(0, 512, (2, 256), device="cuda")

    model.config.fuse_cross_entropy = False
    unfused = model(input_ids=ids, labels=ids).loss
    model.config.fuse_cross_entropy = True
    fused = model(input_ids=ids, labels=ids)

    assert fused.logits is None, "fused path must not materialize logits"
    torch.testing.assert_close(fused.loss, unfused, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(("T", "trained"), [(256, False), (384, True)])
def test_upper_gates_need_two_read_firings(T, trained):
    """Decay and erase gates stay dead until the level has state to act on.

    At firing_intervals=(1, 2) and chunk_size 64, level 1 fires at chunks 1 and
    3. With T=256 (NT=4) the first firing sees an empty state and the second
    lands on the last chunk, whose result only reaches the loss through the
    discarded final state -- so g_hi and b_hi get exactly zero. w_hi and the
    promotion projections are unaffected: they act on the new content, which is
    nonzero regardless.

    The practical consequence is a lower bound on context length: training at
    T < (2 * f + 1) * chunk_size leaves a level's gates untrained.
    """
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(2, (1, 2), (16,))).cuda()
    ids = torch.randint(0, 512, (2, T), device="cuda")
    model(input_ids=ids, labels=ids).loss.backward()

    attn = model.model.layers[0].attn
    for name in ("g_hi", "b_hi"):
        got = getattr(attn, name).grad.abs().max().item()
        if trained:
            assert got > 0, f"{name} should receive gradient at T={T}"
        else:
            assert got == 0, f"{name} should be dead at T={T}, got {got}"

    # Never dead, at either length.
    for name in ("w_hi", "query_banks", "write_projections"):
        assert getattr(attn, name).grad.abs().max() > 0, name


@pytest.mark.parametrize("promotion", ["learned", "additive"])
def test_promotion_arm_wiring(promotion):
    torch.manual_seed(0)
    model = NestedGDN2ForCausalLM(_config(3, (1, 2, 4), (16, 16), promotion=promotion)).cuda()
    ids = torch.randint(0, 512, (2, 640), device="cuda")
    model(input_ids=ids, labels=ids).loss.backward()

    attn = model.model.layers[0].attn
    assert attn.g_hi.grad.abs().max() > 0

    if promotion == "learned":
        assert isinstance(attn.query_banks, torch.nn.Parameter)
        for name in ("b_hi", "w_hi", "query_banks", "write_projections"):
            assert getattr(attn, name).grad.abs().max() > 0, name
    else:
        assert not isinstance(attn.query_banks, torch.nn.Parameter)
        for name in ("b_hi", "w_hi"):
            assert getattr(attn, name).grad.abs().max() == 0, name


def test_arms_differ():
    outs = []
    for promotion in ("learned", "additive"):
        torch.manual_seed(0)
        model = NestedGDN2ForCausalLM(_config(3, (1, 2, 4), (16, 16), promotion=promotion)).cuda()
        ids = torch.randint(0, 512, (2, 640), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        outs.append(model(input_ids=ids).logits)
    assert (outs[0] - outs[1]).abs().max() > 1e-3
