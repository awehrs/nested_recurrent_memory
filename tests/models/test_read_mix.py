"""Gated reads. Requires CUDA: fla's norms are triton-only.

`read_mix="sigmoid"` gates each level on its own instead of making the levels
share a budget of 1. Two things have to hold for the arm to mean anything: the
gates must be independent, and the read ablation must not renormalize them,
since rescaling the survivors would put the competition back.
"""

import pytest
import torch

from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config
from nested_gdn2.models.modeling_nested_gdn2 import NestedGDN2ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

L, FIRING, N_Q, T = 4, (1, 2, 4, 8), (16, 32, 64), 1152


def _model(read_mix):
    torch.manual_seed(0)
    cfg = NestedGDN2Config(
        vocab_size=512, hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64,
        num_levels=L, firing_intervals=FIRING, n_queries_per_level=N_Q,
        read_mix=read_mix, fuse_cross_entropy=False,
    )
    return NestedGDN2ForCausalLM(cfg).cuda().eval()


def _weights(model, ids):
    """The mix weights the layer actually computes, captured off mix_proj."""
    out = []
    layer = next(m for m in model.modules() if hasattr(m, "mix_proj"))

    def hook(mod, inp, o):
        w = o.float().view(o.shape[0], o.shape[1], layer.H, L)
        out.append(w.sigmoid() if layer.cfg.read_mix == "sigmoid" else w.softmax(-1))

    h = layer.mix_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    return out[0]


def test_gates_are_independent():
    model = _model("sigmoid")
    ids = torch.randint(0, 512, (2, T), device="cuda")
    w = _weights(model, ids)
    # Nothing forces a budget, so the weights are free to sum past 1 and a
    # token may hold two levels open at once.
    assert w.shape[-1] == L
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert w.sum(-1).max() > 1.0


def test_softmax_weights_still_sum_to_one():
    model = _model("softmax")
    ids = torch.randint(0, 512, (2, T), device="cuda")
    w = _weights(model, ids)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_keeping_all_levels_is_a_noop_when_gated():
    model = _model("sigmoid")
    ids = torch.randint(0, 512, (2, T), device="cuda")

    def logits(keep):
        for m in model.modules():
            if hasattr(m, "cfg") and hasattr(m.cfg, "num_levels"):
                m.cfg.read_levels = keep
        with torch.no_grad():
            return model(input_ids=ids).logits

    assert torch.equal(logits(None), logits(tuple(range(L))))


def _op_weights(monkeypatch, model, ids, keep):
    """The mix weights the op is handed, which is after the ablation mask."""
    from nested_gdn2.layers import nested_gdn2 as layer_mod

    seen = []
    real = layer_mod.chunk_nested_gdn2

    def spy(*a, mix_weights, **kw):
        seen.append(mix_weights.detach().float().clone())
        return real(*a, mix_weights=mix_weights, **kw)

    monkeypatch.setattr(layer_mod, "chunk_nested_gdn2", spy)
    for m in model.modules():
        if hasattr(m, "cfg") and hasattr(m.cfg, "num_levels"):
            m.cfg.read_levels = keep
    with torch.no_grad():
        model(input_ids=ids)
    return seen[0]


def test_gated_ablation_leaves_the_kept_gate_alone(monkeypatch):
    """Keeping level 0 must pass level 0's own gate through unchanged."""
    model = _model("sigmoid")
    ids = torch.randint(0, 512, (2, T), device="cuda")

    full = _op_weights(monkeypatch, model, ids, None)
    kept = _op_weights(monkeypatch, model, ids, (0,))

    assert torch.equal(kept[..., 0], full[..., 0])
    assert kept[..., 1:].abs().max() == 0.0
    assert kept[..., 0].max() < 1.0  # not rescaled to a budget of 1


def test_softmax_ablation_rescales_to_one(monkeypatch):
    """The softmax arm keeps its old behaviour: survivors sum back to 1."""
    model = _model("softmax")
    ids = torch.randint(0, 512, (2, T), device="cuda")

    kept = _op_weights(monkeypatch, model, ids, (0,))
    assert torch.allclose(kept[..., 0], torch.ones_like(kept[..., 0]), atol=1e-5)
