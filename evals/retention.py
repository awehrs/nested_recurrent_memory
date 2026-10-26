"""Loss-based recall: does the model read a span it saw `distance` tokens ago?

Two forward passes per distance over sequences that differ in one place. Both
end with the same 64-token probe span at the same absolute position, so position
and content are held fixed and only the history differs:

    primed   [ filler ... S  ...(distance)...  S ]
    control  [ filler ... S' ...(distance)...  S ]

`retention` is control NLL minus primed NLL on the probe tokens: nats saved by
having seen the span before. Reported per distance, paired across trials.

Two probes, scored on the same planted spans so they are directly comparable:

    repeat        the probe text is S itself -- verbatim recall
    continuation  the probe text is what followed S in its source book

And two conditions: all levels read, or only the levels in --read-levels. The
contrast is taken within a condition, so the off-distribution cost of ablating
cancels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nested_gdn2  # noqa: F401  registers the architecture for from_pretrained

DISTANCES = (256, 1024, 2048, 4096, 8192, 15360)
BINS = 20  # top-weight histogram, 0.05 wide


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--ckpt", required=True, help="transformers-format checkpoint dir")
    p.add_argument("-d", "--data", default="fla-hub/pg19")
    p.add_argument("-s", "--split", default="test")
    p.add_argument("--field", default="text")
    p.add_argument("--arm", default=None, help="name recorded in the json; defaults to the ckpt dir")
    p.add_argument("--source", default=None, help="where the checkpoint came from, for the record")
    p.add_argument("--span", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distances", default=",".join(str(d) for d in DISTANCES))
    p.add_argument("--probes", default="repeat,continuation")
    p.add_argument("--read-levels", default="0", help="levels kept in the ablated condition")
    # fp32: the deltas are small enough that bf16 logits sit near the noise floor.
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_streams(data, split, field, tokenizer, n_docs, min_len):
    """`n_docs` token arrays, each at least `min_len` long.

    PG-19 books run far longer than seq_len, so one document is one trial's
    filler and the spans come from its neighbours.
    """
    from datasets import load_dataset

    print(f"Loading data {data}")
    ds = load_dataset(data, split=split)
    print(ds)

    # A whole book is ~1M characters and we need seq_len tokens out of it;
    # tokenizing all of it would dominate the runtime.
    cap = min_len * 8
    out = []
    for row in ds:
        ids = tokenizer(row[field][:cap], return_attention_mask=False)["input_ids"]
        if len(ids) >= min_len:
            out.append(np.asarray(ids, dtype=np.int64))
            if len(out) == n_docs:
                return out
    raise RuntimeError(
        f"{data}:{split} yielded {len(out)} documents of >= {min_len} tokens, need {n_docs}"
    )


def set_read_levels(model, keep):
    """Ablation is read off the layer's config object at forward time.

    Raises if it matches no layers, which would leave the two conditions equal.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "cfg") and hasattr(m.cfg, "num_levels"):
            m.cfg.read_levels = keep
            n += 1
    if not n:
        raise RuntimeError("found no nested layers to set read_levels on")


def attach_mix_probe(model):
    """Mean read weight per level over everything scored, overall and per layer.

    Hooked on the mix projection, so it is the mixture before the read ablation
    masks it. The per-token winner and top weight are histogrammed as well,
    since a mean cannot separate a flat mixture from a single-level read.
    """
    stats = {}
    handles = []
    for name, m in model.named_modules():
        if hasattr(m, "mix_proj") and getattr(m, "L", 1) > 1:
            stats[name] = {
                "sum": torch.zeros(m.L, dtype=torch.float64),
                "argmax": torch.zeros(m.L, dtype=torch.float64),
                "hist": torch.zeros(BINS, dtype=torch.float64),
                "entropy": 0.0,
                "n": 0,
            }

            def hook(mod, inp, out, name=name, H=m.H, L=m.L):
                w = out.float().view(-1, H, L).softmax(-1).reshape(-1, L)
                s = stats[name]
                top, who = w.max(-1)
                s["sum"] += w.sum(0).double().cpu()
                s["argmax"] += torch.bincount(who, minlength=L).double().cpu()
                s["hist"] += torch.histc(top, bins=BINS, min=0.0, max=1.0).double().cpu()
                s["entropy"] += -(w * w.clamp_min(1e-9).log()).sum().double().cpu().item()
                s["n"] += w.shape[0]

            handles.append(m.mix_proj.register_forward_hook(hook))
    return stats, handles


def mix_summary(stats):
    if not stats:
        return None
    n = sum(v["n"] for v in stats.values())
    agg = {k: sum(v[k] for v in stats.values()) for k in ("sum", "argmax", "hist", "entropy")}
    return {
        "mean": (agg["sum"] / n).tolist(),
        "argmax_frac": (agg["argmax"] / n).tolist(),
        "top_weight_hist": (agg["hist"] / n).tolist(),
        "hist_bins": BINS,
        "entropy": agg["entropy"] / n,
        "samples": n,
        "per_layer": {
            k: {
                "mean": (v["sum"] / v["n"]).tolist(),
                "argmax_frac": (v["argmax"] / v["n"]).tolist(),
                "top_weight_hist": (v["hist"] / v["n"]).tolist(),
                "entropy": v["entropy"] / v["n"],
            }
            for k, v in stats.items()
        },
    }


def summarize(primed, control):
    delta = np.array(control) - np.array(primed)
    return {
        "primed": float(np.mean(primed)),
        "control": float(np.mean(control)),
        "retention": float(delta.mean()),
        "stderr": float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0,
    }


@torch.no_grad()
def span_nll(model, tokens, start, span):
    """Mean NLL of tokens[start : start + span], teacher-forced on the whole prefix."""
    ids = torch.as_tensor(tokens, device="cuda").unsqueeze(0)
    logits = model(input_ids=ids).logits
    # logits[i] predicts token i + 1. Slice before the upcast: the full
    # seq_len x vocab tensor in fp32 is gigabytes, and 64 rows is all we score.
    pred = logits[0, start - 1 : start + span - 1].float()
    target = ids[0, start : start + span]
    return F.cross_entropy(pred, target).item()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    distances = [int(d) for d in args.distances.split(",")]
    span, seq_len = args.span, args.seq_len
    probe = seq_len - span
    if max(distances) > probe:
        raise ValueError(f"distance {max(distances)} exceeds the probe position {probe}")

    probes = args.probes.split(",")
    for pr in probes:
        if pr not in ("repeat", "continuation"):
            raise ValueError(f"probe must be 'repeat' or 'continuation', got {pr!r}")
    keep = tuple(int(i) for i in args.read_levels.split(","))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=getattr(torch, args.dtype)
    ).cuda().eval()

    # Nothing to ablate on a flat arm, so it gets the full condition only.
    num_levels = getattr(model.config, "num_levels", 1)
    conds = ["full"] + (["ablated"] if num_levels > 1 else [])
    if max(keep) >= num_levels:
        raise ValueError(f"--read-levels {keep} outside num_levels={num_levels}")

    # Each trial needs a filler document plus two others to draw spans from.
    docs = load_streams(
        args.data, args.split, args.field, tokenizer, args.trials + 2, seq_len
    )

    mix_stats, mix_handles = attach_mix_probe(model)

    rng = np.random.default_rng(args.seed)
    acc = {
        pr: {c: {d: {"p": [], "c": []} for d in distances} for c in conds} for pr in probes
    }

    for t in range(args.trials):
        filler = docs[t][:seq_len].copy()
        target = docs[(t + 1) % len(docs)]
        decoy = docs[(t + 2) % len(docs)]
        # Two spans' worth of room: S is planted, the text after it is the
        # continuation probe. Both come from the same place in the same book.
        o = rng.integers(0, len(target) - 2 * span)
        s, cont = target[o : o + span], target[o + span : o + 2 * span]
        s_prime = decoy[(o2 := rng.integers(0, len(decoy) - span)) : o2 + span]

        # One base sequence per probe: identical except for the probe text.
        base = {}
        for pr in probes:
            base[pr] = filler.copy()
            base[pr][probe : probe + span] = s if pr == "repeat" else cont

        for d in distances:
            at = probe - d
            seqs = {}
            for pr in probes:
                a = base[pr].copy()
                a[at : at + span] = s
                b = base[pr].copy()
                b[at : at + span] = s_prime
                seqs[pr] = (a, b)
            # Conditions outside the probes: all four scores come from the same
            # planted spans, so every contrast is paired within a trial.
            for c in conds:
                set_read_levels(model, None if c == "full" else keep)
                for pr in probes:
                    a, b = seqs[pr]
                    acc[pr][c][d]["p"].append(span_nll(model, a, probe, span))
                    acc[pr][c][d]["c"].append(span_nll(model, b, probe, span))

        for pr in probes:
            for c in conds:
                means = "  ".join(
                    f"{d}:{np.mean(np.array(acc[pr][c][d]['c']) - np.array(acc[pr][c][d]['p'])):+.3f}"
                    for d in distances
                )
                print(f"trial {t + 1}/{args.trials}  {pr:12s} {c:7s}  {means}", flush=True)

    set_read_levels(model, None)
    for h in mix_handles:
        h.remove()

    result = {
        "arm": args.arm or Path(args.ckpt).name,
        "span": span,
        "seq_len": seq_len,
        "trials": args.trials,
        "seed": args.seed,
        "dtype": args.dtype,
        "source": args.source or args.ckpt,
        "num_levels": num_levels,
        "read_levels": list(keep) if "ablated" in conds else None,
        "mix": mix_summary(mix_stats),
        # Per-trial deltas, so full minus ablated can be paired after the fact.
        # Means alone lose the pairing and with it the error bar on that column.
        "per_trial": {
            str(d): {
                pr: {
                    c: (
                        np.array(acc[pr][c][d]["c"]) - np.array(acc[pr][c][d]["p"])
                    ).tolist()
                    for c in conds
                }
                for pr in probes
            }
            for d in distances
        },
        "by_distance": {
            str(d): {
                pr: {c: summarize(acc[pr][c][d]["p"], acc[pr][c][d]["c"]) for c in conds}
                for pr in probes
            }
            for d in distances
        },
    }
    if result["mix"]:
        mx = result["mix"]
        print(
            f"mix mean {[round(x, 4) for x in mx['mean']]}  "
            f"argmax {[round(x, 4) for x in mx['argmax_frac']]}  "
            f"entropy {mx['entropy']:.4f} of {math.log(num_levels):.4f}",
            flush=True,
        )
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["by_distance"], indent=1))


if __name__ == "__main__":
    main()
