"""Read ablation controls. Requires CUDA: fla's norms are triton-only.

`read_levels` zeroes the unkept levels' mix weights and renormalizes over the
rest, so keeping everything has to be a bit-exact no-op.
"""

import pytest
import torch

from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config
from nested_gdn2.models.modeling_nested_gdn2 import NestedGDN2ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CASES = [(1, (1,), (), 256), (4, (1, 2, 4, 8), (16, 32, 64), 1152)]
IDS = ["L1", "L4"]


def _model(num_levels, firing, n_queries):
    torch.manual_seed(0)
    cfg = NestedGDN2Config(
        vocab_size=512, hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64,
        num_levels=num_levels, firing_intervals=firing, n_queries_per_level=n_queries,
        fuse_cross_entropy=False,
    )
    return NestedGDN2ForCausalLM(cfg).cuda().eval()


def _logits(model, ids, keep):
    for m in model.modules():
        if hasattr(m, "cfg") and hasattr(m.cfg, "num_levels"):
            m.cfg.read_levels = keep
    with torch.no_grad():
        return model(input_ids=ids).logits


@pytest.mark.parametrize(("num_levels", "firing", "n_queries", "T"), CASES, ids=IDS)
def test_keeping_all_levels_is_a_noop(num_levels, firing, n_queries, T):
    model = _model(num_levels, firing, n_queries)
    ids = torch.randint(0, 512, (2, T), device="cuda")

    base = _logits(model, ids, None)
    allk = _logits(model, ids, tuple(range(num_levels)))
    torch.testing.assert_close(base, allk, rtol=0, atol=0)


def test_ablation_changes_a_nested_model():
    """On L=4, keeping only level 0 has to change the logits."""
    model = _model(4, (1, 2, 4, 8), (16, 32, 64))
    ids = torch.randint(0, 512, (2, 1152), device="cuda")

    base = _logits(model, ids, None)
    level0 = _logits(model, ids, (0,))
    assert not torch.equal(base, level0)


def test_step_path_matches_chunked_under_ablation():
    """Ablation is applied before the split, so both read paths must agree."""
    model = _model(4, (1, 2, 4, 8), (16, 32, 64))
    ids = torch.randint(0, 512, (1, 1152), device="cuda")

    chunked = _logits(model, ids, (0,))

    for m in model.modules():
        if hasattr(m, "cfg") and hasattr(m.cfg, "num_levels"):
            m.cfg.read_levels = (0,)
    with torch.no_grad():
        cache, out = None, []
        for t in range(ids.shape[1]):
            o = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            out.append(o.logits)
    stepped = torch.cat(out, dim=1)

    torch.testing.assert_close(chunked, stepped, rtol=2e-2, atol=2e-2)
