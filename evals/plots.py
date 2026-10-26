"""The writeup figures, from the eval JSONs and the validation curves.

Each figure is one function so a single plot can be regenerated after an edit:

    python evals/plots.py                 # all six, to assets/results/
    python evals/plots.py --only state

Nothing here touches a GPU or a checkpoint; it reads numbers that are already
on disk. Run `scripts/eval.sh` to produce new ones.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "results" / "data"
EVALS = DATA / "evals"
CURVES = DATA / "curves"

# Which eval produced what. Keyed by figure so a rerun is a one-line change.
PPL_MAIN = "20260930-1211"       # nested / flat-128 / transformer, 5722 steps
PPL_NEW = "20261008-1140"        # + flat-compute-matched, and all retention
PPL_CM = "20261019-1133"        # flat-compute-matched at 5722, run three days later
RET_PG19 = "20261008-1140"       # rebuilt probe, PG-19-trained checkpoints
RET_PG19_OLD = "20260930-1304"   # original probe, same checkpoints
RET_FINEWEB = "20260930-1331"    # FineWeb-trained checkpoints, PG-19 eval

TOKENS_PER_STEP = 524288

LABEL = {
    "nested-learned": "nested (4 levels)",
    "flat-state-matched": "flat, state-matched (128)",
    "flat-compute-matched": "flat, compute-matched (64)",
    "transformer": "transformer",
}
COLOR = {
    "nested-learned": "#c2410c",
    "flat-state-matched": "#1d4ed8",
    "flat-compute-matched": "#0d9488",
    "transformer": "#6b7280",
}
SHORT = {
    "nested-learned": "nested",
    "flat-state-matched": "flat-128",
    "flat-compute-matched": "flat-64",
    "transformer": "transformer",
}

# Buckets are 1k wide, so index 2 is the 2-3k bucket, where "past 2k" starts.
# Index 0 carries the no-context penalty and would dominate.
ZERO_AT = 2


def load(stamp, kind, arm):
    return json.loads((EVALS / stamp / f"{kind}-{arm}.json").read_text())


def val_curve(name):
    """(steps, val_loss) for one run, extracted from its training log."""
    rows = np.loadtxt(CURVES / f"{name}.csv", delimiter=",", skiprows=1)
    return rows[:, 0].astype(int), rows[:, 1]


def val_at(name, step):
    s, v = val_curve(name)
    hit = np.flatnonzero(s == step)
    if not len(hit):
        raise RuntimeError(f"{name}.csv has no val at step {step}")
    return float(v[hit[0]])


def finish(fig, out, name, dpi):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out / name}.png + .svg")


# ── 1. loss vs position ───────────────────────────────────────────────────────

def position(out, dpi):
    """Absolute loss per 1k bucket, and the same curves zeroed at the 2k bucket.

    The right panel is the claim: no arm gains much from context past ~2k, so
    the objective never asks for the range a long-range mechanism would supply.
    """
    arms = [
        ("flat-state-matched", PPL_MAIN, 5722),
        ("nested-learned", PPL_MAIN, 5722),
        ("transformer", PPL_MAIN, 5722),
        ("flat-compute-matched", PPL_CM, 5722),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))

    for arm, stamp, steps in arms:
        d = load(stamp, "ppl", arm)
        bucket = d["config"]["bucket_size"]
        nats = np.log(np.array(d["block_perplexities"]))
        x = (np.arange(len(nats)) + 0.5) * bucket
        kw = dict(color=COLOR[arm], marker="o", ms=3.5, lw=1.6)
        tag = "" if steps == 5722 else f" — {steps} steps"
        ax.plot(x, nats, label=LABEL[arm] + tag, **kw)
        bx.plot(x[ZERO_AT:], nats[ZERO_AT:] - nats[ZERO_AT], **kw)

    ax.set(xlabel="position in sequence (tokens)", ylabel="loss (nats/token)",
           title="Loss by position")
    bx.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    bx.set(xlabel="position in sequence (tokens)",
           ylabel="loss relative to the 2–3k bucket (nats)",
           title="Same curves, zeroed past 2k")
    for a in (ax, bx):
        a.grid(alpha=0.25, lw=0.5)
        a.set_xlim(0, 16384)
    ax.legend(fontsize=8, frameon=False)

    gain = {}
    for arm, stamp, _ in arms:
        nats = np.log(np.array(load(stamp, "ppl", arm)["block_perplexities"]))
        gain[arm] = nats[ZERO_AT] - nats[-1]
    note = "  ".join(f"{SHORT[a]}: {g:+.3f}" for a, g in gain.items())
    bx.text(0.5, -0.28, f"gain from 2k to 16k —  {note}", transform=bx.transAxes,
            ha="center", fontsize=8, color="#4b5563")

    finish(fig, out, "1-loss-vs-position", dpi)


# ── 2. retention vs distance ──────────────────────────────────────────────────

def retention(out, dpi, include_undertrained=False):
    """Nats saved on a 64-token span by having seen it `distance` tokens ago.

    flat-128 and nested are both at 5722 steps. flat-compute-matched is at
    2000 and 0.365 nats behind on perplexity, so its curve cannot separate
    'less state' from 'less training' and is off by default.
    """
    arms = ["flat-state-matched", "nested-learned"]
    if include_undertrained:
        arms.append("flat-compute-matched")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm in arms:
        d = load(RET_PG19, "retention", arm)["by_distance"]
        x = np.array([int(k) for k in d])
        y = np.array([v["retention"] for v in d.values()])
        e = np.array([v["stderr"] for v in d.values()])
        dashed = arm == "flat-compute-matched"
        ax.errorbar(x, y, yerr=e, label=LABEL[arm] + (" — 2000 steps" if dashed else ""),
                    color=COLOR[arm], marker="o", ms=4, lw=1.6, capsize=3,
                    ls="--" if dashed else "-", alpha=0.55 if dashed else 1.0)

    ax.set_xscale("log")
    ax.set(xlabel="distance from the span to the probe (tokens)",
           ylabel="retention (nats saved)",
           title="Loss-based recall by distance")
    ax.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=8, frameon=False)

    # Both curves decay toward zero, so the ratio is the stable statement.
    a = load(RET_PG19, "retention", "flat-state-matched")["by_distance"]
    b = load(RET_PG19, "retention", "nested-learned")["by_distance"]
    ratio = "  ".join(f"{k}: {a[k]['retention'] / b[k]['retention']:.1f}x" for k in a)
    ax.text(0.5, -0.26, f"flat-128 / nested —  {ratio}", transform=ax.transAxes,
            ha="center", fontsize=8, color="#4b5563")

    finish(fig, out, "2-retention-vs-distance", dpi)


# ── 3. validation curves ──────────────────────────────────────────────────────

def val_curves(out, dpi):
    """The four full PG-19 runs, 58 points each."""
    runs = [("flat", "flat-state-matched"), ("launch-cm-pg19", "flat-compute-matched"),
            ("nested", "nested-learned"), ("xf", "transformer")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for log, arm in runs:
        s, v = val_curve(log)
        ax.plot(s * TOKENS_PER_STEP / 1e9, v, label=f"{LABEL[arm]}  ({v[-1]:.4f})",
                color=COLOR[arm], lw=1.7)
    ax.set(xlabel="tokens seen (billions)", ylabel="validation loss (nats/token)",
           title="PG-19, 3B tokens", ylim=(2.9, 4.2))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    finish(fig, out, "3-val-curves", dpi)


# ── 4. what the training corpus sets ──────────────────────────────────────────

def corpus(out, dpi):
    """Retention of the same architecture trained on FineWeb vs on PG-19.

    Both evaluated on PG-19, so the difference is the training corpus and not
    the measurement. FineWeb's documents are ~1k tokens; PG-19's are books.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for stamp, label, color, ls in [
        (RET_FINEWEB, "trained on FineWeb (~1k-token documents)", "#9333ea", "--"),
        (RET_PG19_OLD, "trained on PG-19 (books)", COLOR["nested-learned"], "-"),
    ]:
        d = load(stamp, "retention", "nested-learned")["by_distance"]
        x = np.array([int(k) for k in d])
        y = np.array([v["retention"] for v in d.values()])
        e = np.array([v["stderr"] for v in d.values()])
        ax.errorbar(x, y, yerr=e, label=label, color=color, ls=ls,
                    marker="o", ms=4, lw=1.6, capsize=3)

    ax.set_xscale("log")
    ax.set(xlabel="distance from the span to the probe (tokens)",
           ylabel="retention (nats saved)",
           title="Training document length sets the retention horizon")
    ax.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.5, -0.26, "nested, both evaluated on PG-19; original probe",
            transform=ax.transAxes, ha="center", fontsize=8, color="#4b5563")
    finish(fig, out, "4-corpus-sets-horizon", dpi)


# ── 5. interventions ──────────────────────────────────────────────────────────

def interventions(out, dpi):
    """Each fix against the baseline at 2000 steps, with the seed band.

    The baseline is step 2000 of the full PG-19 run and the seed replicate is a
    2000-step run; both sit at a constant 3e-4 at that point, since decay_frac
    0.1 only bites after step ~5150 and the short runs used 0.
    """
    base = val_at("nested", 2000)
    seed = val_at("seed1", 2000)
    band = abs(base - seed)

    arms = [("detach", "cut promotion's gradient"),
            ("horizon", "constant n_q"),
            ("vproj", "value projection"),
            ("chained", "chained reads")]
    deltas = [(lbl, val_at(log, 2000) - base) for log, lbl in arms]
    deltas.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(deltas))
    vals = np.array([d for _, d in deltas])
    # Inside the seed band is not a result, so do not colour it as one.
    color = ["#9ca3af" if abs(v) <= band else ("#15803d" if v < 0 else "#b91c1c")
             for v in vals]
    ax.barh(y, vals, color=color, height=0.6)
    ax.axvspan(-band, band, color="#9ca3af", alpha=0.22, zorder=0,
               label=f"seed spread (±{band:.3f})")
    ax.axvline(0, color="#374151", lw=1.0)
    ax.set_yticks(y, [lbl for lbl, _ in deltas], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel=f"change in validation loss vs baseline {base:.4f}  (nats, lower is better)",
           title="Interventions at 2000 steps")
    pad = 0.006
    ax.set_xlim(vals.min() - 5 * pad, vals.max() + 5 * pad)
    for yi, v in zip(y, vals):
        # Long bars get the label inside, which means stepping back toward zero;
        # short ones get it just past the tip. Both flip with the sign.
        inside = abs(v) > 3 * pad
        ax.text(v - math.copysign(pad, v) if inside else v + math.copysign(pad, v),
                yi, f"{v:+.3f}", va="center",
                ha=("right" if v > 0 else "left") if inside else
                   ("left" if v > 0 else "right"),
                color="white" if inside else "#374151", fontsize=8,
                fontweight="bold" if inside else "normal")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, -0.18))
    finish(fig, out, "5-interventions", dpi)


# ── 6. inference state ────────────────────────────────────────────────────────

def state(out, dpi):
    """Floats of recurrent state, or KV cache, held at inference.

    Per layer and for the whole model, because the ratio is the point and
    mixing the two inflates it twelvefold.
    """
    layers, heads, seq = 12, 12, 16384
    bars = [
        ("flat, compute-matched (64)", heads * 64 * 64, COLOR["flat-compute-matched"]),
        ("nested (4 x 64)", 4 * heads * 64 * 64, COLOR["nested-learned"]),
        ("flat, state-matched (128)", heads * 128 * 128, COLOR["flat-state-matched"]),
        # K and V, per token, per layer.
        (f"transformer KV @ {seq // 1024}k", 2 * heads * 64 * seq, COLOR["transformer"]),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y = np.arange(len(bars))
    per_layer = np.array([v for _, v, _ in bars])
    ax.barh(y, per_layer, color=[c for *_, c in bars], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y, [n for n, *_ in bars], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="floats of state per layer (log scale)",
           title="Memory held at inference, independent of sequence length")
    for yi, v in zip(y, per_layer):
        ax.text(v * 1.15, yi, f"{v:,}  ({v * layers / 1e6:.2f}M total)",
                va="center", fontsize=8)
    ax.set_xlim(per_layer.min() / 2, per_layer.max() * 12)
    ax.grid(alpha=0.25, lw=0.5, axis="x", which="both")

    ratio = per_layer[-1] / per_layer[1]
    ax.text(0.5, -0.3, f"the transformer holds {ratio:.0f}x nested's state at {seq // 1024}k, "
            f"and grows with the sequence; the rest do not",
            transform=ax.transAxes, ha="center", fontsize=8, color="#4b5563")
    finish(fig, out, "6-inference-state", dpi)


FIGURES = {
    "position": position,
    "retention": retention,
    "val": val_curves,
    "corpus": corpus,
    "interventions": interventions,
    "state": state,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "assets" / "results"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--only", choices=sorted(FIGURES), action="append")
    p.add_argument("--include-undertrained", action="store_true",
                   help="add flat-compute-matched to the retention figure")
    args = p.parse_args()

    out = Path(args.out)
    for name in args.only or list(FIGURES):
        print(f"{name}:")
        fn = FIGURES[name]
        if name == "retention":
            fn(out, args.dpi, args.include_undertrained)
        else:
            fn(out, args.dpi)


if __name__ == "__main__":
    main()
