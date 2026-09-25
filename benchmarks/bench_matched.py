"""Nested GDN-2 against flat GDN-2 at equal recurrent state.

Nested with H heads and L levels holds L*H*K*V floats; flat needs L*H heads to
match. Same state, same head_dim. Ratios are nested / flat, so below 1 means
nested is cheaper.

Three sweeps, one CSV each in benchmarks/results/$GPU/ (default h100):
    depth    L and head_dim, with the query ladder n_q = min(8 * 2^l, head_dim)
    queries  uniform 16 queries against the ladder, at fixed head_dim
    batch    batch size at the two candidate training configs

A run that runs out of memory is written as a blank row.

    GPU=a100 PULL_BACK=benchmarks/results bash scripts/run.sh benchmarks/bench_matched.py
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics

import torch
import torch.nn.functional as F
import triton
from fla.modules.l2norm import l2_norm
from fla.ops.gdn2 import chunk_gdn2

from nested_gdn2.ops.chunk import chunk_nested_gdn2

B, H, T = 4, 16, 8192
REPEATS = 3
RESULTS = pathlib.Path(__file__).parent / "results" / os.environ.get("GPU", "h100")

LEAVES = (
    "q", "k", "v", "g", "b", "w", "mix_weights", "query_banks",
    "key_projections", "b_projections", "w_projections", "g_projections",
)
COLUMNS = (
    "sweep", "B", "H", "T", "L", "head_dim", "n_q", "state",
    "nested_fwd_ms", "flat_fwd_ms", "fwd_ratio",
    "nested_fb_ms", "flat_fb_ms", "fb_ratio",
    "bwd_over_fwd", "nested_gb", "flat_gb",
)


def ladder(L, hd):
    """Queries doubling with the firing interval, so every level sees the same
    number of pairs per step. Capped at head_dim."""
    return tuple(min(8 * 2**lvl, hd) for lvl in range(1, L))


# Each row: (batch, L, head_dim, n_queries).
SWEEPS = {
    "depth": [
        (B, L, hd, ladder(L, hd))
        for hd, depths in ((64, range(2, 7)), (128, range(2, 6)))
        for L in depths
    ],
    # At L=2 the ladder is (16,), identical to uniform, so start at 3.
    "queries": [
        (B, L, 64, n_q)
        for L in (3, 4)
        for n_q in ((16,) * (L - 1), ladder(L, 64))
    ],
    "batch": [
        (batch, L, hd, ladder(L, hd))
        for L, hd in ((4, 64), (5, 128))
        for batch in (2, 4, 8)
    ],
}


def make_inputs(batch, heads, L, K, n_q, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    V = K
    d = dict(device="cuda", dtype=dtype)
    n = max(L - 1, 0)
    assert len(n_q) == n, f"n_queries needs {n} entries, got {len(n_q)}"
    assert not n_q or max(n_q) <= K, f"n_queries must be <= head_dim {K}"
    return dict(
        q=torch.randn(batch, T, heads, L, K, **d),
        k=l2_norm(torch.randn(batch, T, heads, K, **d)),
        v=torch.randn(batch, T, heads, V, **d) * 0.5,
        g=F.logsigmoid(
            torch.randn(batch, T, heads, K, device="cuda", dtype=torch.float32)
        ).to(dtype),
        b=torch.rand(batch, T, heads, K, **d),
        w=torch.rand(batch, T, heads, V, **d),
        mix_weights=torch.softmax(torch.randn(batch, T, heads, L, **d), dim=-1),
        query_banks=torch.randn(n, heads, max(n_q, default=1), K, **d) * K**-0.5,
        key_projections=torch.randn(n, heads, K, V, **d) * V**-0.5,
        b_projections=torch.randn(n, heads, V, K, **d) * V**-0.5,
        w_projections=torch.randn(n, heads, V, V, **d) * V**-0.5,
        g_projections=torch.randn(n, heads, V, K, **d) * V**-0.5,
        n_queries_per_level=torch.tensor(n_q, dtype=torch.int, device="cuda"),
        firing_intervals=torch.tensor(
            [1] + [2**i for i in range(1, L)], dtype=torch.int, device="cuda"
        ),
        L=L,
        chunk_size=64,
    )


def leaves(kw):
    out = dict(kw)
    for name in LEAVES:
        out[name] = kw[name].detach().clone().requires_grad_(True)
    return out


def nested_fn(kw):
    return lambda: chunk_nested_gdn2(**kw)


def flat_fn(kw):
    return lambda: chunk_gdn2(
        q=kw["q"][..., 0, :], k=kw["k"], v=kw["v"], g=kw["g"], b=kw["b"], w=kw["w"],
    )


def with_backward(fn):
    def step():
        o = fn()
        (o[0] if isinstance(o, tuple) else o).sum().backward()
    return step


def bench(fn):
    """Median over REPEATS separate runs; single runs varied ~12% between rows."""
    return statistics.median(
        triton.testing.do_bench(fn, return_mode="median") for _ in range(REPEATS)
    )


def time_and_memory(make_fn, kw):
    """Forward ms, forward+backward ms, and peak GB over one forward+backward."""
    fwd = bench(make_fn(kw))
    kwl = leaves(kw)
    step = with_backward(make_fn(kwl))
    fb = bench(step)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    gb = torch.cuda.max_memory_allocated() / 1e9

    del kwl
    return fwd, fb, gb


def run_one(make_fn, build):
    """time_and_memory, or None if it runs out of memory."""
    kw = None
    try:
        kw = build()
        return time_and_memory(make_fn, kw)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del kw
        torch.cuda.empty_cache()


def ratio(a, b):
    return a / b if a is not None and b is not None else None


def measure(sweep, batch, L, hd, n_q):
    n = run_one(nested_fn, lambda: make_inputs(batch, H, L, hd, n_q))
    f = run_one(flat_fn, lambda: make_inputs(batch, L * H, 1, hd, ()))
    n_fwd, n_fb, n_gb = n or (None,) * 3
    f_fwd, f_fb, f_gb = f or (None,) * 3
    return dict(
        sweep=sweep, B=batch, H=H, T=T, L=L, head_dim=hd,
        n_q="-".join(map(str, n_q)), state=L * H * hd * hd,
        nested_fwd_ms=n_fwd, flat_fwd_ms=f_fwd, fwd_ratio=ratio(n_fwd, f_fwd),
        nested_fb_ms=n_fb, flat_fb_ms=f_fb, fb_ratio=ratio(n_fb, f_fb),
        bwd_over_fwd=ratio(n_fb, n_fwd), nested_gb=n_gb, flat_gb=f_gb,
    )


def fmt(x, width, spec, suffix=""):
    return f"{'OOM':>{width}}" if x is None else f"{x:>{width - len(suffix)}{spec}}{suffix}"


def run(sweep):
    print(f"\n{sweep}  H={H} T={T}  {torch.cuda.get_device_name(0)}")
    head = (f"{'B':>3}{'L':>3}{'hd':>5}{'n_q':>16}{'state':>11}"
            f"{'fwd':>8}{'flat':>8}{'x':>7}"
            f"{'f+b':>9}{'flat':>8}{'x':>7}{'bwd/fwd':>9}{'GB':>6}{'flat GB':>9}")
    print(head)
    print("-" * len(head))

    rows = []
    for batch, L, hd, n_q in SWEEPS[sweep]:
        r = measure(sweep, batch, L, hd, n_q)
        rows.append(r)
        print(
            f"{batch:>3}{L:>3}{hd:>5}{r['n_q']:>16}{r['state']:>11,}"
            + fmt(r["nested_fwd_ms"], 8, ".2f") + fmt(r["flat_fwd_ms"], 8, ".2f")
            + fmt(r["fwd_ratio"], 7, ".2f", "x")
            + fmt(r["nested_fb_ms"], 9, ".2f") + fmt(r["flat_fb_ms"], 8, ".2f")
            + fmt(r["fb_ratio"], 7, ".2f", "x")
            + fmt(r["bwd_over_fwd"], 9, ".2f")
            + fmt(r["nested_gb"], 6, ".1f") + fmt(r["flat_gb"], 9, ".1f")
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{sweep}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(SWEEPS))
    args = parser.parse_args()
    for sweep in [args.only] if args.only else SWEEPS:
        run(sweep)


if __name__ == "__main__":
    main()
