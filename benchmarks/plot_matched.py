"""Plots for bench_matched.py: reads benchmarks/results/*.csv, writes assets/bench/*.png.

    uv run python benchmarks/plot_matched.py
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).parent / "results"
ASSETS = pathlib.Path(__file__).parent.parent / "assets" / "bench"
FIGSIZE = (7.5, 4.6)
COLOR = {64: "#2a78d6", 128: "#eb6834"}
NEUTRAL = "#9a9da3"
INK, INK_2, GRID = "#111214", "#4f5257", "#e6e6e2"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def load(name):
    with (RESULTS / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("sweep", "n_q"):
                r[k] = float(v) if v else None
    return rows


def by_hd(rows, hd):
    return [r for r in rows if r["head_dim"] == hd]


def flat_line(ax):
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 3)), alpha=0.55, zorder=1)
    ax.annotate("flat", xy=(0.0, 1.0), xycoords=("axes fraction", "data"),
                xytext=(4, 3), textcoords="offset points", va="bottom", color=INK_2)


def end_labels(fig, ax, items):
    """Label line ends, pushed apart vertically where they would overlap.

    items: (x, y, text, color) in data coordinates. Call after all data is drawn.
    """
    ax.margins(y=0.12)
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72
    gap = plt.rcParams["font.size"] * 1.4 * px_per_pt
    placed = sorted(
        ((ax.transData.transform((x, y))[1], x, y, text, color) for x, y, text, color in items),
        key=lambda t: t[0],
    )
    ys = [p[0] for p in placed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (ys[-1] - placed[-1][0]) / 2 if len(ys) > 1 else 0
    for disp, (orig, x, y, text, color) in zip(ys, placed):
        ax.annotate(text, xy=(x, y), xytext=(8, (disp - shift - orig) / px_per_pt),
                    textcoords="offset points", va="center", color=color)


def line(ax, xs, ys, color, label, dashed=False):
    ax.plot(xs, ys, color=color, lw=2, ls="--" if dashed else "-", label=label,
            marker="o", ms=5, mfc="white" if dashed else color, mec=color, mew=1.5, zorder=3)


def finish(fig, ax, title, name, legend_cols=2):
    """Title above a legend row, both clear of the plot; headroom above the data."""
    ax.margins(y=0.12)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=legend_cols,
              borderaxespad=0.3, handlelength=2.4)
    path = ASSETS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def ratio_panel(rows, x, labels, setup_x):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ends = []
    for hd in (64, 128):
        pts = [(r[x], r["fb_ratio"]) for r in by_hd(rows, hd) if r["fb_ratio"] is not None]
        xs, ys = zip(*pts)
        line(ax, xs, ys, COLOR[hd], labels[hd])
        ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}×", COLOR[hd]))
    flat_line(ax)
    ax.set_ylabel("forward+backward ÷ flat")
    setup_x(ax)
    end_labels(fig, ax, ends)
    return fig, ax


def depth_x(ax):
    ax.set(xlabel="levels (L)", xticks=[2, 3, 4, 5, 6])


def batch_x(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8], labels=["2", "4", "8"])
    ax.set_xlabel("batch size")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    depth, batch, queries = load("depth"), load("batch"), load("queries")

    fig, ax = ratio_panel(depth, "L", {64: "head_dim 64", 128: "head_dim 128"}, depth_x)
    finish(fig, ax, "Training cost vs depth  (B=4)", "depth")

    labels = {hd: f"head_dim {hd}, L={int(by_hd(batch, hd)[0]['L'])}" for hd in (64, 128)}
    fig, ax = ratio_panel(batch, "B", labels, batch_x)
    finish(fig, ax, "Training cost vs batch", "batch")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ends = []
    for hd in (64, 128):
        rows = by_hd(depth, hd)
        xs = [r["L"] for r in rows]
        line(ax, xs, [r["nested_gb"] for r in rows], COLOR[hd], f"hd {hd} nested")
        line(ax, xs, [r["flat_gb"] for r in rows], COLOR[hd], f"hd {hd} flat", dashed=True)
        ends.append((xs[-1], rows[-1]["nested_gb"], f"{rows[-1]['nested_gb']:.1f} GB", COLOR[hd]))
        ends.append((xs[-1], rows[-1]["flat_gb"], f"{rows[-1]['flat_gb']:.1f} GB", COLOR[hd]))
    ax.set(xlabel="levels (L)", ylabel="peak GB", xticks=[2, 3, 4, 5, 6])
    ax.set_ylim(bottom=0)
    end_labels(fig, ax, ends)
    ax.set_ylim(bottom=0)
    finish(fig, ax, "Peak memory vs depth  (B=4)", "memory", legend_cols=4)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    groups = sorted({int(r["L"]) for r in queries})
    width = 0.34
    for i, L in enumerate(groups):
        pair = sorted((r for r in queries if r["L"] == L), key=lambda r: r["n_q"].count("16"), reverse=True)
        for j, (r, color, label) in enumerate(zip(pair, (NEUTRAL, COLOR[64]), ("uniform 16", "ladder"))):
            x = i + (j - 0.5) * (width + 0.02)
            ax.bar(x, r["fb_ratio"], width, color=color, label=label if i == 0 else None, zorder=2)
            ax.annotate(f"{r['fb_ratio']:.2f}×", xy=(x, r["fb_ratio"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=INK_2)
    flat_line(ax)
    ax.set_xticks(range(len(groups)), labels=[f"L={L}" for L in groups])
    ax.set_ylabel("forward+backward ÷ flat")
    ax.set_ylim(0, max(r["fb_ratio"] for r in queries) * 1.18)
    finish(fig, ax, "Cost of the query ladder  (B=4, head_dim 64)", "queries")


if __name__ == "__main__":
    main()
