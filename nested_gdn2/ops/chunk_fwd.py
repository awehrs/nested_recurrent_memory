"""Chunkwise NestedGDN-2 forward.

Level 0 is stock GDN-2, so its chunkwise states come from fla's pipeline. This
module supplies the piece with no fla analogue: the sequential scan that reads
the level below at each firing and writes it into the level above, plus the
composition of the per-level reads into the layer output.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd

from nested_gdn2.ops.additive import additive_scan
from nested_gdn2.ops.utils import from_chunks, last_token_index, to_chunks

__all__ = ["chunk_nested_gdn2_fwd", "promotion_scan"]


@triton.jit
def _promotion_scan_fwd_kernel(
    h_ptr,
    q_ptr,
    w_ptr,
    g_ptr,
    b_ptr,
    wg_ptr,
    s_ptr,
    final_ptr,
    init_ptr,
    NT,
    FIRE: tl.constexpr,
    NQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """One program per (batch, head); full K x V state resident across the chunk loop.

    K and V cannot be tiled: v_writes reduces over K, k_writes over V, and the
    erase term over K, so any tiling splits a reduction across programs.

    Args:
        h_ptr: source-level state after each chunk, [B, NT, H, K, V].
        q_ptr: learned probes, [H, NQ, K].
        w_ptr: value->key projection, [H, K, V].
        g_ptr: log-decay at each firing, [B, NT, H, K].
        b_ptr: erase gate at each firing, [B, NT, H, K].
        wg_ptr: write gate at each firing, [B, NT, H, V].
        s_ptr: out, this level's state at each chunk start, [B, NT, H, K, V].
        final_ptr: out, state after the last chunk, [B, H, K, V].
        init_ptr: initial state, unread unless HAS_INIT, [B, H, K, V].
        NT: number of chunks. Runtime, so the chunk loop is a real loop.
        FIRE: firing interval in chunks; fires when (t + 1) % FIRE == 0.
        NQ: probes, i.e. delta-rule writes applied per firing. Power of two and
            at least 16, since it is a tl.dot dimension.
        H: number of heads, used to split the program id into (batch, head).
        K: key dim. Full extent, never tiled.
        V: value dim. Full extent, never tiled.
        HAS_INIT: whether to seed the state from init_ptr rather than zeros.
    """
    # This program owns exactly one (batch, head). Nothing else is parallel:
    # K and V are held whole, and the chunk loop below is a real recurrence.
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    ok = tl.arange(0, K)
    ov = tl.arange(0, V)
    on = tl.arange(0, NQ)

    q = tl.load(
        q_ptr + i_h * NQ * K + on[:, None] * K + ok[None, :]
    ).to(tl.float32)
    wp = tl.load(w_ptr + i_h * K * V + ok[:, None] * V + ov[None, :]).to(tl.float32)

    s = tl.zeros([K, V], dtype=tl.float32)
    if HAS_INIT:
        s += tl.load(
            init_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
        ).to(tl.float32)

    for t in range(NT):
        base = ((i_b * NT + t) * H + i_h) * K * V

        tl.store(s_ptr + base + ok[:, None] * V + ov[None, :], s)

        if (t + 1) % FIRE == 0:
            src = tl.load(h_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)
            v_w = tl.dot(q, src)
            k_w = tl.dot(v_w, tl.trans(wp))
            k_w = k_w * tl.rsqrt(tl.maximum(tl.sum(k_w * k_w, axis=1), 1e-12))[:, None]

            gk = ((i_b * NT + t) * H + i_h) * K
            gv = ((i_b * NT + t) * H + i_h) * V
            g = tl.load(g_ptr + gk + ok).to(tl.float32)
            bg = tl.load(b_ptr + gk + ok).to(tl.float32)
            wg = tl.load(wg_ptr + gv + ov).to(tl.float32)

            s = s * tl.exp(g)[:, None]
            kb = k_w * bg[None, :]
            vn = v_w * wg[None, :] - tl.dot(kb, s)
            s = s + tl.dot(tl.trans(k_w), vn)

    tl.store(final_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :], s)


def promotion_scan(
    h: torch.Tensor,
    query_bank: torch.Tensor,
    write_proj: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    firing_interval: int,
    n_queries: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan one promotion level over all chunks.

    At each firing the level below is probed with ``query_bank`` to produce
    ``n_queries`` values, write keys are derived from those values and L2
    normalized, and the resulting pairs are applied to this level's state as a
    single batched gated delta-rule update.

    Args:
        h: source-level state *after* each chunk, [B, NT, H, K, V]. Caller is
            responsible for the shift: fla returns states at chunk *starts*, so
            this is ``cat(h_fla[:, 1:], final_state[:, None])``.
        query_bank: learned probes into the source level, [H, N, K]. Rows past
            ``n_queries`` are ignored, so N may exceed it.
        write_proj: value->key projection for the promoted write, [H, K, V].
        g: log-decay applied to this level's state at each firing, [B, NT, H, K].
            Gathered by the caller from the last token of each chunk.
        b: channel-wise erase gate at each firing, [B, NT, H, K]. Same gather.
        w: channel-wise write gate at each firing, [B, NT, H, V]. Same gather.
        firing_interval: this level fires when ``(chunk + 1) % interval == 0``.
            Counted in chunks, so the level activates at token
            ``interval * chunk_size``.
        n_queries: number of probes, i.e. writes applied per firing. Must be a
            power of two and at least 16, since it is a tl.dot dimension; below
            that the promotion channel is too narrow to be worth having anyway.
            Compile-time constant in the kernel, so a new value recompiles.
        initial_state: this level's state before chunk 0, [B, H, K, V], or None
            for zeros.

    Returns:
        states: this level's state at the *start* of each chunk, [B, NT, H, K, V],
            float32. Chunk-start because reads for chunk t precede its firing.
        final_state: state after the last chunk, [B, H, K, V], float32.
    """
    B, NT, H, K, V = h.shape
    if n_queries < 16 or n_queries & (n_queries - 1):
        raise ValueError(f"n_queries must be a power of two >= 16, got {n_queries}")
    q = query_bank[:, :n_queries].to(torch.float32).contiguous()

    states = h.new_empty(B, NT, H, K, V, dtype=torch.float32)
    final = h.new_empty(B, H, K, V, dtype=torch.float32)
    init = initial_state.to(torch.float32).contiguous() if initial_state is not None else h

    _promotion_scan_fwd_kernel[(B * H,)](
        h.contiguous(),
        q,
        write_proj.to(torch.float32).contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        states,
        final,
        init,
        NT,
        FIRE=firing_interval,
        NQ=n_queries,
        H=H,
        K=K,
        V=V,
        HAS_INIT=initial_state is not None,
    )
    return states, final


def chunk_nested_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    mix_weights: torch.Tensor,
    query_banks: torch.Tensor,
    write_projections: torch.Tensor,
    n_queries_per_level: torch.Tensor,
    firing_intervals: torch.Tensor,
    L: int,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    return_aux: bool = False,
    promotion: str = "learned",
):
    """Chunkwise NestedGDN-2 forward.

    Level 0 is stock GDN-2 and runs through fla's pipeline; levels >= 1 are
    promotion scans chained off it. Signature mirrors ``naive_chunk_nested_gdn2``.

    Args:
        q: per-level read queries, [B, T, H, L, K]. Level 0's slice is also its
            input-triple query.
        k: level-0 keys, [B, T, H, K]. Caller-normalized.
        v: level-0 values, [B, T, H, V].
        g: per-level log-decay, [B, T, H, L, K].
        b: per-level erase gate, [B, T, H, L, K].
        w: per-level write gate, [B, T, H, L, V].
        mix_weights: per-token per-level mix, [B, T, H, L], pre-softmaxed.
        query_banks: learned probes for levels >= 1, [L-1, H, N, K].
        write_projections: value->key projections for levels >= 1, [L-1, H, K, V].
        n_queries_per_level: probes per level >= 1, [L-1], int. Each must be a
            power of two >= 16.
        firing_intervals: per-level firing interval in chunks, [L], int, with
            ``firing_intervals[0] == 1``.
        L: number of levels.
        scale: attention scale; defaults to ``1 / sqrt(K)``.
        initial_state: [B, H, L, K, V] or None.
        output_final_state: whether to return the final state.
        return_aux: also return the intermediates the backward needs.
        promotion: ``"learned"`` probes the level below with the query bank;
            ``"additive"`` carries it up whole. The control arm -- see
            ``ops.additive``.
        chunk_size: tokens per chunk. Must be 64 -- fla's GDN-2 kernels assume
            it. Also sets where the hierarchy starts: level l first fires at
            token ``firing_intervals[l] * chunk_size``.

    Returns:
        o: [B, T, H, V].
        final_state: [B, H, L, K, V] if ``output_final_state`` else None.
        aux: dict of backward intermediates, only if ``return_aux``.
    """
    B, T, H, _, K = q.shape
    BT = chunk_size
    if BT != 64:
        # fla's chunk_intra hardcodes NC=4 sub-chunks of 16, so GDN-2 only
        # supports BT=64. chunk_gdn2_fwd skips the public wrapper's check.
        raise ValueError(f"chunk_size must be 64 for GDN-2, got {BT}")
    NT = triton.cdiv(T, BT)
    if scale is None:
        scale = K**-0.5

    o, final, g_cs, Aqk, Akk, _, _, _, _, _, h, _ = chunk_gdn2_fwd(
        q=q[..., 0, :].contiguous(),
        k=k,
        v=v,
        g=g[..., 0, :].contiguous(),
        b=b[..., 0, :].contiguous(),
        w_gate=w[..., 0, :].contiguous(),
        scale=scale,
        initial_state=initial_state[:, :, 0].contiguous() if initial_state is not None else None,
        output_final_state=True,
        chunk_size=BT,
        return_intermediate_states=True,
    )

    reads = [o]
    o = (mix_weights[..., 0].unsqueeze(-1) * o).to(o.dtype)
    finals = [final]
    all_states = [h]

    last = last_token_index(NT, BT, T, q.device)
    src = torch.cat([h[:, 1:], final.unsqueeze(1)], dim=1)

    scan = {"learned": promotion_scan, "additive": additive_scan}[promotion]

    for lvl in range(1, L):
        states, fin = scan(
            h=src,
            query_bank=query_banks[lvl - 1],
            write_proj=write_projections[lvl - 1],
            g=g[:, last, :, lvl],
            b=b[:, last, :, lvl],
            w=w[:, last, :, lvl],
            firing_interval=int(firing_intervals[lvl]),
            n_queries=int(n_queries_per_level[lvl - 1]),
            initial_state=initial_state[:, :, lvl] if initial_state is not None else None,
        )
        q_lvl = to_chunks(q[..., lvl, :], NT, BT) * scale
        read = from_chunks(torch.einsum("bnthk,bnhkv->bnthv", q_lvl.float(), states), T)
        o = o + (mix_weights[..., lvl].unsqueeze(-1) * read).to(o.dtype)

        reads.append(read)
        finals.append(fin)
        all_states.append(states)
        src = torch.cat([states[:, 1:], fin.unsqueeze(1)], dim=1)

    final_state = torch.stack(finals, dim=2) if output_final_state else None
    if not return_aux:
        return o, final_state
    aux = dict(
        g_cs=g_cs,
        Aqk=Aqk,
        Akk=Akk,
        states=all_states,
        finals=finals,
        reads=reads,
        scale=scale,
    )
    return o, final_state, aux
