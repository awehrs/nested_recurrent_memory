"""Forward-pass benchmark for the chunkwise NestedGDN-2 kernel.

Three comparisons:
  1. Triton vs the naive PyTorch op -- what the kernel bought.
  2. Triton at L=1 vs fla's chunk_gdn2 -- our kernel against production Triton
     on identical math, since L=1 reduces exactly.
  3. Cost per added level -- the marginal price of the hierarchy.

Requires a GPU:
    python benchmarks/bench_chunk.py
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import torch
import torch.nn.functional as F
import triton
from fla.modules.l2norm import l2_norm
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd

from nested_gdn2.ops.chunk import chunk_nested_gdn2
from nested_gdn2.ops.chunk_fwd import chunk_nested_gdn2_fwd, promotion_scan
from nested_gdn2.ops.naive import naive_chunk_nested_gdn2

# The naive op loops chunks in Python; above this it dominates the wall clock.
NAIVE_MAX_T = 2048

ROWS: list[dict] = []


def record(section: str, **fields) -> None:
    ROWS.append({"section": section, **fields})


def write_csv(path: str) -> None:
    if not ROWS:
        return
    keys: list[str] = []
    for row in ROWS:
        keys.extend(k for k in row if k not in keys)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"wrote {len(ROWS)} rows to {path}")


def make_inputs(B, T, H, K, V, L, n_q, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, T, H, L, K, dtype=dtype, device=dev)
    k = l2_norm(torch.randn(B, T, H, K, dtype=dtype, device=dev))
    v = torch.randn(B, T, H, V, dtype=dtype, device=dev) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, L, K, device=dev, dtype=torch.float32)).to(dtype)
    b = torch.rand(B, T, H, L, K, dtype=dtype, device=dev)
    w = torch.rand(B, T, H, L, V, dtype=dtype, device=dev)
    mix = torch.softmax(torch.randn(B, T, H, L, dtype=dtype, device=dev), dim=-1)
    n = max(L - 1, 0)
    return dict(
        q=q, k=k, v=v, g=g, b=b, w=w,
        mix_weights=mix,
        query_banks=torch.randn(n, H, n_q, K, dtype=dtype, device=dev) * K**-0.5,
        write_projections=torch.randn(n, H, K, V, dtype=dtype, device=dev) * V**-0.5,
        n_queries_per_level=torch.full((n,), n_q, dtype=torch.int, device=dev),
        firing_intervals=torch.tensor([1] + [2**i for i in range(1, L)], dtype=torch.int, device=dev),
        L=L,
        chunk_size=64,
    )


def peak_mem(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def bench_breakdown(B=4, T=8192, H=8, K=64, V=64, L=2, n_q=16):
    """Split one level's cost into scan / gather / read / mix."""
    kw = make_inputs(B, T, H, K, V, L, n_q)
    q, k, v, g, b, w = kw["q"], kw["k"], kw["v"], kw["g"], kw["b"], kw["w"]
    BT, NT, scale = 64, T // 64, K**-0.5

    def level0():
        return chunk_gdn2_fwd(
            q=q[..., 0, :].contiguous(), k=k, v=v,
            g=g[..., 0, :].contiguous(), b=b[..., 0, :].contiguous(),
            w_gate=w[..., 0, :].contiguous(),
            scale=scale, initial_state=None, output_final_state=True,
            chunk_size=BT, return_intermediate_states=True,
        )

    o, final = level0()[0], level0()[1]
    h = level0()[10]
    src = torch.cat([h[:, 1:], final.unsqueeze(1)], dim=1)
    last = torch.arange(NT, device=q.device) * BT + (BT - 1)

    def gather():
        return g[:, last, :, 1], b[:, last, :, 1], w[:, last, :, 1]

    gl, bl, wl = gather()

    def scan():
        return promotion_scan(
            h=src, query_bank=kw["query_banks"][0], write_proj=kw["write_projections"][0],
            g=gl, b=bl, w=wl, firing_interval=2, n_queries=n_q,
        )

    states = scan()[0]
    q_lvl = q[..., 1, :].view(B, NT, BT, H, K) * scale

    def read():
        return torch.einsum("bnthk,bnhkv->bnthv", q_lvl.float(), states)

    parts = {
        "level 0 (fla)": level0,
        "cat / shift": lambda: torch.cat([h[:, 1:], final.unsqueeze(1)], dim=1),
        "gate gather": gather,
        "promotion scan": scan,
        "read einsum": read,
        "mix + add": lambda: o + kw["mix_weights"][..., 1].unsqueeze(-1) * o,
    }
    total = triton.testing.do_bench(lambda: chunk_nested_gdn2_fwd(**kw))

    print(f"breakdown  B={B} T={T} H={H} L={L} n_q={n_q}  (B*H={B * H} programs)")
    acc = 0.0
    for name, fn in parts.items():
        ms = triton.testing.do_bench(fn)
        acc += ms
        print(f"  {name:18s}{ms:>9.3f} ms{ms / total * 100:>8.1f}%")
        record("breakdown", B=B, T=T, H=H, L=L, part=name, ms=ms, pct=ms / total * 100)
    print(f"  {'sum of parts':18s}{acc:>9.3f} ms{acc / total * 100:>8.1f}%")
    print(f"  {'full forward':18s}{total:>9.3f} ms\n")
    record("breakdown", B=B, T=T, H=H, L=L, part="full forward", ms=total, pct=100.0)


def _leaf(kw):
    """Same inputs, but as autograd leaves."""
    out = dict(kw)
    for name in ("q", "k", "v", "g", "b", "w", "mix_weights",
                 "query_banks", "write_projections"):
        out[name] = kw[name].detach().clone().requires_grad_(True)
    return out


def bench_backward(T=8192, K=64, V=64, n_q=16):
    """Forward vs forward+backward, nested against flat GDN-2 at matched state.

    The forward-only numbers understate training cost: nested's backward carries
    the cross-level injection and saves more state than flat GDN-2's does.
    """
    print(f"backward  T={T} K={K} V={V} n_q={n_q}")
    print(f"{'B':>4}{'H':>4}{'L':>3}{'nested fwd':>12}{'nested f+b':>12}{'bwd/fwd':>9}"
          f"{'flat f+b':>10}{'ratio':>8}{'MB':>8}")
    for B, H in ((4, 8), (8, 16)):
        for L in (2, 4):
            kw = _leaf(make_inputs(B, T, H, K, V, L, n_q, dtype=torch.bfloat16))
            flat = _leaf(make_inputs(B, T, L * H, K, V, 1, n_q, dtype=torch.bfloat16))
            d_o = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
            d_of = torch.randn(B, T, L * H, V, device="cuda", dtype=torch.bfloat16)

            def n_fwd():
                return chunk_nested_gdn2(**kw)[0]

            def n_both():
                o, _ = chunk_nested_gdn2(**kw)
                torch.autograd.grad(o, [kw["q"]], d_o, retain_graph=False)

            def f_both():
                o = chunk_gdn2(
                    q=flat["q"][..., 0, :], k=flat["k"], v=flat["v"],
                    g=flat["g"][..., 0, :], b=flat["b"][..., 0, :],
                    w=flat["w"][..., 0, :],
                )[0]
                torch.autograd.grad(o, [flat["k"]], d_of, retain_graph=False)

            fwd = triton.testing.do_bench(n_fwd)
            both = triton.testing.do_bench(n_both)
            flat_b = triton.testing.do_bench(f_both)
            mem = peak_mem(n_both)
            print(f"{B:>4}{H:>4}{L:>3}{fwd:>12.2f}{both:>12.2f}{both / fwd:>9.2f}x"
                  f"{flat_b:>10.2f}{both / flat_b:>8.2f}x{mem:>8.0f}")
            record("backward", B=B, H=H, L=L, T=T, nested_fwd=fwd, nested_fb=both,
                   bwd_over_fwd=both / fwd, flat_fb=flat_b, ratio=both / flat_b, mb=mem)
    print()


def bench_matched_state(T=8192, K=64, V=64, n_q=16):
    """Nested vs flat GDN-2 at equal recurrent state.

    Nested with H heads and L levels holds L*H*K*V floats; flat GDN-2 needs L*H
    heads to match. Same state, same rank -- this is the ``state_matched`` arm,
    so it is the comparison that decides whether the hierarchy costs anything
    beyond the memory it carries.
    """
    print(f"matched state  T={T} K={K} V={V} n_q={n_q}")
    print(f"{'B':>4}{'H':>4}{'L':>3}{'state':>9}{'nested ms':>11}"
          f"{'flat ms':>10}{'flat H':>8}{'ratio':>8}")
    for B, H in ((4, 8), (8, 16)):
        for L in (2, 4):
            kw = make_inputs(B, T, H, K, V, L, n_q)
            flat = make_inputs(B, T, L * H, K, V, 1, n_q)
            nested_ms = triton.testing.do_bench(lambda: chunk_nested_gdn2_fwd(**kw))
            flat_ms = triton.testing.do_bench(
                lambda: chunk_gdn2(
                    q=flat["q"][..., 0, :], k=flat["k"], v=flat["v"],
                    g=flat["g"][..., 0, :], b=flat["b"][..., 0, :], w=flat["w"][..., 0, :],
                )
            )
            state = L * H * K * V
            print(f"{B:>4}{H:>4}{L:>3}{state:>9,}{nested_ms:>11.3f}"
                  f"{flat_ms:>10.3f}{L * H:>8}{nested_ms / flat_ms:>8.2f}x")
            record("matched", B=B, H=H, L=L, T=T, state=state, nested_ms=nested_ms,
                   flat_ms=flat_ms, flat_H=L * H, ratio=nested_ms / flat_ms)
    print()


def bench_occupancy(T=4096, K=64, V=64, L=2, n_q=16):
    """Per-level cost against program count, which is B*H."""
    print(f"occupancy  T={T} L={L} n_q={n_q}   (grid is B*H programs)")
    print(f"{'B':>4}{'H':>4}{'B*H':>6}{'L=1 ms':>10}{'L=2 ms':>10}"
          f"{'level ms':>10}{'tok/ms':>10}")
    for B, H in ((2, 8), (4, 8), (8, 8), (8, 16), (16, 16)):
        # Inputs must be built outside the timed region; they are hundreds of
        # MB at the larger shapes and would otherwise dominate the measurement.
        kw1 = make_inputs(B, T, H, K, V, 1, n_q)
        kw2 = make_inputs(B, T, H, K, V, 2, n_q)
        t1 = triton.testing.do_bench(lambda: chunk_nested_gdn2_fwd(**kw1))
        t2 = triton.testing.do_bench(lambda: chunk_nested_gdn2_fwd(**kw2))
        print(f"{B:>4}{H:>4}{B * H:>6}{t1:>10.3f}{t2:>10.3f}{t2 - t1:>10.3f}{B * T / t2:>10.0f}")
        record("occupancy", B=B, H=H, programs=B * H, T=T, l1_ms=t1, l2_ms=t2,
               level_ms=t2 - t1, tok_per_ms=B * T / t2)
    print()


def bench_promotion_arms(B=4, T=8192, H=8, K=64, V=64, n_q=16):
    """Learned vs additive promotion: the price of choosing what to promote.

    Identical state and schedule on both arms -- only the scan differs. Additive
    has no probe, no normalize and no key gradients, so it is the floor for what
    a hierarchy costs at all.
    """
    print(f"promotion arms  B={B} T={T} H={H} K={K} V={V} n_q={n_q}")
    print(f"{'L':>3}{'learned fwd':>13}{'additive fwd':>14}{'fwd ratio':>11}"
          f"{'learned f+b':>13}{'additive f+b':>14}{'f+b ratio':>11}{'MB':>8}")
    for L in (2, 3, 4):
        kw = _leaf(make_inputs(B, T, H, K, V, L, n_q, dtype=torch.bfloat16))
        d_o = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)

        def fwd(p, kw=kw):
            return lambda: chunk_nested_gdn2_fwd(**kw, promotion=p)

        def both(p, kw=kw, d_o=d_o):
            def run():
                o, _ = chunk_nested_gdn2(**kw, promotion=p)
                torch.autograd.grad(o, [kw["q"]], d_o, retain_graph=False)
            return run

        lf = triton.testing.do_bench(fwd("learned"))
        af = triton.testing.do_bench(fwd("additive"))
        lb = triton.testing.do_bench(both("learned"))
        ab = triton.testing.do_bench(both("additive"))
        mem = peak_mem(both("learned"))
        print(f"{L:>3}{lf:>13.2f}{af:>14.2f}{lf / af:>10.2f}x"
              f"{lb:>13.2f}{ab:>14.2f}{lb / ab:>10.2f}x{mem:>8.0f}")
        record("arms", L=L, learned_fwd=lf, additive_fwd=af, fwd_ratio=lf / af,
               learned_fb=lb, additive_fb=ab, fb_ratio=lb / ab, mb=mem)
    print()


def bench_query_scaling(B=4, T=8192, H=8, K=64, V=64):
    """Cost against promotion width and depth.

    Per firing the kernel moves O(K*V) of state whatever NQ is; only the four
    dots scale with it. If the small shapes are latency-bound, the first
    doublings of NQ come close to free. The four dots contract through NQ, which
    is the cheap order only while NQ < K -- expect a knee at NQ = K.
    """
    print(f"query scaling  B={B} T={T} H={H} K={K} V={V}  (additive is NQ-independent)")
    print(f"{'L':>3}{'NQ':>5}{'fwd ms':>10}{'f+b ms':>10}{'fwd/add':>9}"
          f"{'ms/query':>10}{'MB':>8}")
    for L in (2, 3, 4):
        base = _leaf(make_inputs(B, T, H, K, V, L, 16, dtype=torch.bfloat16))
        add_ms = triton.testing.do_bench(
            lambda kw=base: chunk_nested_gdn2_fwd(**kw, promotion="additive")
        )
        for n_q in (16, 32, 64, 128):
            try:
                kw = _leaf(make_inputs(B, T, H, K, V, L, n_q, dtype=torch.bfloat16))
                d_o = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)

                def run_both(kw=kw, d_o=d_o):
                    o, _ = chunk_nested_gdn2(**kw)
                    torch.autograd.grad(o, [kw["q"]], d_o, retain_graph=False)

                fwd = triton.testing.do_bench(lambda kw=kw: chunk_nested_gdn2_fwd(**kw))
                both = triton.testing.do_bench(run_both)
                mem = peak_mem(run_both)
            except Exception as exc:  # noqa: BLE001
                print(f"{L:>3}{n_q:>5}   failed: {type(exc).__name__}")
                record("query_scaling", L=L, n_q=n_q, error=type(exc).__name__)
                continue
            print(f"{L:>3}{n_q:>5}{fwd:>10.2f}{both:>10.2f}{fwd / add_ms:>8.2f}x"
                  f"{fwd / n_q:>10.4f}{mem:>8.0f}")
            record("query_scaling", L=L, n_q=n_q, fwd_ms=fwd, fb_ms=both,
                   additive_ms=add_ms, vs_additive=fwd / add_ms,
                   ms_per_query=fwd / n_q, mb=mem)
        print()


def main():
    B, H, K, V, n_q = 4, 8, 64, 64, 16
    print(f"sweep  B={B} H={H} K={K} V={V} n_queries={n_q} chunk_size=64 dtype=bf16")

    print(f"{'T':>6}{'L':>3}{'naive ms':>11}{'triton ms':>11}{'speedup':>9}"
          f"{'fla ms':>9}{'vs fla':>8}{'MB':>8}")
    for T in (1024, 2048, 4096, 8192):
        for L in (1, 2, 4):
            kw = make_inputs(B, T, H, K, V, L, n_q)

            tri = triton.testing.do_bench(lambda: chunk_nested_gdn2_fwd(**kw))
            mem = peak_mem(lambda: chunk_nested_gdn2_fwd(**kw))

            if T <= NAIVE_MAX_T:
                nai = triton.testing.do_bench(lambda: naive_chunk_nested_gdn2(**kw), rep=20)
                naive_s, speed = f"{nai:.2f}", f"{nai / tri:.1f}x"
            else:
                naive_s, speed = "-", "-"

            if L == 1:
                fla = triton.testing.do_bench(
                    lambda: chunk_gdn2(
                        q=kw["q"][..., 0, :], k=kw["k"], v=kw["v"],
                        g=kw["g"][..., 0, :], b=kw["b"][..., 0, :], w=kw["w"][..., 0, :],
                    )
                )
                fla_s, vs = f"{fla:.2f}", f"{tri / fla:.2f}x"
            else:
                fla_s, vs = "-", "-"

            print(f"{T:>6}{L:>3}{naive_s:>11}{tri:>11.2f}{speed:>9}{fla_s:>9}{vs:>8}{mem:>8.0f}")
            record("sweep", B=B, H=H, T=T, L=L, triton_ms=tri, naive_ms=naive_s,
                   speedup=speed, fla_ms=fla_s, vs_fla=vs, mb=mem)
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        choices=["sweep", "breakdown", "occupancy", "matched", "backward", "arms", "queries"],
        default=None,
    )
    p.add_argument("--csv", default="benchmarks/results/bench.csv")
    args = p.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    record("device", name=torch.cuda.get_device_name(0))
    if args.only in (None, "breakdown"):
        bench_breakdown()
    if args.only in (None, "occupancy"):
        bench_occupancy()
    if args.only in (None, "matched"):
        bench_matched_state()
    if args.only in (None, "backward"):
        bench_backward()
    if args.only in (None, "arms"):
        bench_promotion_arms()
    if args.only in (None, "queries"):
        bench_query_scaling()
    if args.only in (None, "sweep"):
        main()
    write_csv(args.csv)
