"""Chunkwise NestedGDN-2 backward.

Two pieces: the reverse of a single promotion scan, and the composition that
walks the ladder top-down and hands promotion's gradient to level 0.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nested_gdn2.ops._vendor.gdn2_chunk_bwd import chunk_gdn2_bwd
from nested_gdn2.ops.additive import additive_scan_bwd
from nested_gdn2.ops.utils import from_chunks, last_token_index, to_chunks

__all__ = ["chunk_nested_gdn2_bwd", "promotion_scan_bwd"]


@triton.jit
def _promotion_scan_bwd_kernel(
    h_ptr,
    s_ptr,
    q_ptr,
    w_ptr,
    g_ptr,
    b_ptr,
    wg_ptr,
    ds_ptr,
    dfinal_ptr,
    dh_ptr,
    dq_ptr,
    dw_ptr,
    dg_ptr,
    db_ptr,
    dwg_ptr,
    dinit_ptr,
    NT,
    FIRE: tl.constexpr,
    NQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """Reverse of ``_promotion_scan_fwd_kernel``. One program per (batch, head).

    Walks chunks in reverse carrying ``ds``, the gradient of this level's state.
    At a firing it recomputes the forward chain from the saved chunk-start state
    and reverses it, emitting parameter and gate gradients and the injection into
    the level below.

    Args:
        h_ptr: source-level state after each chunk, [B, NT, H, K, V].
        s_ptr: this level's state at each chunk start, [B, NT, H, K, V].
        q_ptr: learned probes, [H, NQ, K].
        w_ptr: value->key projection, [H, K, V].
        g_ptr: log-decay at each firing, [B, NT, H, K].
        b_ptr: erase gate at each firing, [B, NT, H, K].
        wg_ptr: write gate at each firing, [B, NT, H, V].
        ds_ptr: incoming grad w.r.t. ``states``, [B, NT, H, K, V].
        dfinal_ptr: incoming grad w.r.t. the final state, [B, H, K, V].
        dh_ptr: out, grad w.r.t. the source level, [B, NT, H, K, V].
        dq_ptr: out, per-batch partial for the probes, [B, H, NQ, K].
        dw_ptr: out, per-batch partial for the projection, [B, H, K, V].
        dg_ptr: out, [B, NT, H, K].
        db_ptr: out, [B, NT, H, K].
        dwg_ptr: out, [B, NT, H, V].
        dinit_ptr: out, grad w.r.t. the initial state, [B, H, K, V].
        NT: number of chunks.
        FIRE: firing interval in chunks.
        NQ: probes per firing.
        H: number of heads.
        K: key dim.
        V: value dim.
        HAS_INIT: whether to write dinit_ptr.
    """
    # ---------------------------------------------------------------------
    # THE MATH
    #
    # Per (batch, head), one firing. The query bank is a set, so the whole
    # firing is one batched update; the sequence runs across firings.
    #   H  [K,V]   source level's state AFTER chunk t
    #   S  [K,V]   this level's state entering the firing
    #   Q  [NQ,K]  probes          W [K,V]  write projection
    #   g,b [K]    decay, erase    w [V]    write gate
    #
    # Forward:
    #   (1) V_w = Q H                                [NQ,V]
    #   (2) U   = V_w W^T                            [NQ,K]
    #   (3) n_i = ||u_i||^-1 ,  K_w = diag(n) U      [NQ,K]
    #   (4) S0  = S * exp(g)      S0[k,v] = S[k,v]*exp(g_k)
    #   (5) kb  = K_w * b         kb[i,j] = K_w[i,j] b[j]
    #   (6) VN  = V_w * w - kb S0                    [NQ,V]
    #   (7) S_out = S0 + K_w^T VN
    #
    # The erase contracts through NQ, never forming K_w^T kb: at NQ < K that is
    # both cheaper and one fewer [K,K] live in registers.
    #
    # Reverse, all from the incoming ds = dL/dS_out:
    #   dVN  = K_w ds                       (7)
    #   dK_w = VN ds^T                      (7), K_w's direct appearance
    #        + (-dVN S0^T) * b              (6) -> (5), through kb
    #   db   = sum_i (-dVN S0^T) * K_w      (5)
    #   dw   = sum_i dVN * V_w              (6)
    #   dS0  = ds - kb^T dVN                (7) identity + (6) erase
    #   dg   = sum_v dS0 * S0 ,  ds <- dS0 * exp(g)          (4)
    #   du   = n (dK_w - <dK_w,khat> khat)                   (3)
    #   dW   = du^T V_w ,  dV_w += du W                      (2)
    #   dQ   = dV_w H^T ,  dH = Q^T dV_w                     (1)
    #
    # Between firings nothing happens, so the reverse pass only accumulates.
    # ---------------------------------------------------------------------
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    ok = tl.arange(0, K)
    ov = tl.arange(0, V)
    on = tl.arange(0, NQ)

    # Q and W are shared across all chunks, so their gradients accumulate in
    # registers for the whole scan and are stored once at the end.
    q = tl.load(q_ptr + i_h * NQ * K + on[:, None] * K + ok[None, :]).to(tl.float32)
    wp = tl.load(w_ptr + i_h * K * V + ok[:, None] * V + ov[None, :]).to(tl.float32)

    ds = tl.load(
        dfinal_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
    ).to(tl.float32)

    dq_acc = tl.zeros([NQ, K], dtype=tl.float32)
    dw_acc = tl.zeros([K, V], dtype=tl.float32)

    for t in range(NT - 1, -1, -1):
        base = ((i_b * NT + t) * H + i_h) * K * V
        gk = ((i_b * NT + t) * H + i_h) * K
        gv = ((i_b * NT + t) * H + i_h) * V

        if (t + 1) % FIRE == 0:
            # recompute (1)-(6)
            src = tl.load(h_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)
            v_w = tl.dot(q, src)
            u = tl.dot(v_w, tl.trans(wp))
            nrm = tl.rsqrt(tl.maximum(tl.sum(u * u, axis=1), 1e-12))
            k_w = u * nrm[:, None]

            g = tl.load(g_ptr + gk + ok).to(tl.float32)
            bg = tl.load(b_ptr + gk + ok).to(tl.float32)
            wg = tl.load(wg_ptr + gv + ov).to(tl.float32)

            s = tl.load(s_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)
            s = s * tl.exp(g)[:, None]
            kb = k_w * bg[None, :]
            vn = v_w * wg[None, :] - tl.dot(kb, s)

            # (7) and (6) reversed. Every term reads the incoming ds, so dS0 is
            # held aside and only becomes ds once the rest is done.
            dvn = tl.dot(k_w, ds)
            dwg = tl.sum(dvn * v_w, axis=0)
            dv_w = dvn * wg[None, :]
            dk_w = tl.dot(vn, tl.trans(ds))

            # (5) reversed, through kb = K_w * b.
            dkb = -tl.dot(dvn, tl.trans(s))
            dk_w += dkb * bg[None, :]
            dbg = tl.sum(dkb * k_w, axis=0)

            d_s0 = ds - tl.dot(tl.trans(kb), dvn)

            # (4) reversed. dg contracts against the DECAYED state.
            dg = tl.sum(d_s0 * s, axis=1)
            ds = d_s0 * tl.exp(g)[:, None]

            # (3) reversed: projection onto the tangent space of the sphere.
            dot = tl.sum(dk_w * k_w, axis=1)
            du = (dk_w - dot[:, None] * k_w) * nrm[:, None]

            # (2) and (1) reversed. dv_w already holds the dVN * w term, so both
            # paths into V_w rejoin before dQ and dH.
            dw_acc += tl.dot(tl.trans(du), v_w)
            dv_w += tl.dot(du, wp)
            dq_acc += tl.dot(dv_w, tl.trans(src))

            # dH: the injection into the level below. Level 1 hands this to fla
            # as dh_ext; levels >= 2 un-shift it onto their chunk-start states.
            tl.store(
                dh_ptr + base + ok[:, None] * V + ov[None, :],
                tl.dot(tl.trans(q), dv_w),
            )
            tl.store(dg_ptr + gk + ok, dg)
            tl.store(db_ptr + gk + ok, dbg)
            tl.store(dwg_ptr + gv + ov, dwg)

        # This level's read during chunk t uses the chunk-START state, so its
        # gradient joins after the firing at t has been reversed.
        ds += tl.load(ds_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)

    tl.store(dq_ptr + (i_b * H + i_h) * NQ * K + on[:, None] * K + ok[None, :], dq_acc)
    tl.store(dw_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :], dw_acc)
    if HAS_INIT:
        tl.store(
            dinit_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :], ds
        )

def promotion_scan_bwd(
    h: torch.Tensor,
    states: torch.Tensor,
    query_bank: torch.Tensor,
    write_proj: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    d_states: torch.Tensor,
    d_final: torch.Tensor,
    firing_interval: int,
    n_queries: int,
    has_init: bool = False,
) -> dict:
    """Backward for ``promotion_scan``.

    Args mirror the forward, plus ``states`` (its output) and the incoming
    gradients ``d_states`` [B, NT, H, K, V] and ``d_final`` [B, H, K, V].

    Returns a dict with ``dh`` (the injection into the level below), ``dquery_bank``,
    ``dwrite_proj``, ``dg``, ``db``, ``dw``, and ``dinit`` when ``has_init``.
    """
    B, NT, H, K, V = h.shape
    if n_queries < 16 or n_queries & (n_queries - 1):
        raise ValueError(f"n_queries must be a power of two >= 16, got {n_queries}")

    # Only written at firings, so non-firing chunks must already be zero.
    dh = torch.zeros_like(states)
    dq = torch.empty(B, H, n_queries, K, device=h.device, dtype=torch.float32)
    dw_proj = torch.empty(B, H, K, V, device=h.device, dtype=torch.float32)
    dg = torch.zeros_like(g, dtype=torch.float32)
    db = torch.zeros_like(b, dtype=torch.float32)
    dwg = torch.zeros_like(w, dtype=torch.float32)
    dinit = torch.empty(B, H, K, V, device=h.device, dtype=torch.float32)

    _promotion_scan_bwd_kernel[(B * H,)](
        h.contiguous(),
        states.contiguous(),
        query_bank[:, :n_queries].to(torch.float32).contiguous(),
        write_proj.to(torch.float32).contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        d_states.contiguous(),
        d_final.contiguous(),
        dh,
        dq,
        dw_proj,
        dg,
        db,
        dwg,
        dinit,
        NT,
        FIRE=firing_interval,
        NQ=n_queries,
        H=H,
        K=K,
        V=V,
        HAS_INIT=has_init,
    )
    out = {
        "dh": dh,
        "dquery_bank": dq.sum(0),
        "dwrite_proj": dw_proj.sum(0),
        "dg": dg,
        "db": db,
        "dw": dwg,
    }
    if has_init:
        out["dinit"] = dinit
    return out


def chunk_nested_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    mix_weights: torch.Tensor,
    query_banks: torch.Tensor,
    write_projections: torch.Tensor,
    initial_state: torch.Tensor | None,
    g_cs: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    states: list,
    reads: list,
    finals: list,
    do: torch.Tensor,
    d_final: torch.Tensor | None,
    L: int,
    scale: float,
    chunk_size: int,
    firing_intervals: list,
    n_queries_per_level: list,
    has_init: bool,
    promotion: str,
) -> tuple:
    """Backward for ``chunk_nested_gdn2_fwd``, in the order its inputs appear.

    Levels are reversed top-down: each level's ``dh`` from ``promotion_scan_bwd``
    is the gradient of the level below's *shifted* states, so it is un-shifted
    back onto that level before its own reversal. Level 1's un-shifted gradient
    becomes ``dh_ext`` for the patched GDN-2 backward, which is how promotion's
    contribution enters level 0's state recurrence.

    Args:
        q, k, v, g, b, w, mix_weights, query_banks, write_projections,
            initial_state: the forward's inputs.
        g_cs, Aqk, Akk: fla intermediates from the level-0 forward.
        states: per-level chunk-start states, ``aux["states"]``.
        reads: per-level unmixed reads, ``aux["reads"]``.
        finals: per-level final states, ``aux["finals"]``.
        do: incoming grad w.r.t. the output, [B, T, H, V].
        d_final: incoming grad w.r.t. the stacked final state, or None.
        L: number of levels.
        scale: the forward's attention scale.
        chunk_size: tokens per chunk.
        firing_intervals: per-level firing interval, as python ints.
        n_queries_per_level: probes per level >= 1, as python ints.
        has_init: whether the forward was given an initial state.
        promotion: which scan the forward used, ``"learned"`` or ``"additive"``.

    Returns:
        (dq, dk, dv, dg, db, dw, dmix, dquery_banks, dwrite_projections, dinit).
    """
    mix = mix_weights
    qb = query_banks
    wp = write_projections
    init = initial_state
    BT = chunk_size
    firing = firing_intervals
    n_q = n_queries_per_level

    B, T, H, _, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    last = last_token_index(NT, BT, T, q.device)
    do = do.contiguous()
    do32 = do.float()

    dmix = torch.stack([(do32 * r.float()).sum(-1) for r in reads], dim=-1)
    dq = torch.zeros_like(q)
    dg = torch.zeros_like(g)
    db = torch.zeros_like(b)
    dw = torch.zeros_like(w)
    dqb = torch.zeros_like(qb)
    dwp = torch.zeros_like(wp)
    dinit = torch.zeros_like(init) if has_init else None

    # Gradient of each level's read, and of the states it read from.
    d_states = [None] * L
    dh_ext = None
    d_finals = [
        (d_final[:, :, i].float() if d_final is not None else
         torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32))
        for i in range(L)
    ]
    for lvl in range(1, L):
        d_read = to_chunks((mix[..., lvl].unsqueeze(-1) * do).float(), NT, BT)
        q_lvl = to_chunks(q[..., lvl, :].float(), NT, BT) * scale
        d_states[lvl] = torch.einsum("bnthv,bnthk->bnhkv", d_read, q_lvl)
        dq[..., lvl, :] = from_chunks(
            torch.einsum("bnthv,bnhkv->bnthk", d_read, states[lvl]), T
        ).to(q.dtype) * scale

    # Reverse the ladder top-down, un-shifting each level's dh onto the one below.
    scan_bwd = {"learned": promotion_scan_bwd, "additive": additive_scan_bwd}[promotion]

    for lvl in range(L - 1, 0, -1):
        out = scan_bwd(
            h=torch.cat([states[lvl - 1][:, 1:], finals[lvl - 1].unsqueeze(1)], dim=1),
            states=states[lvl],
            query_bank=qb[lvl - 1],
            write_proj=wp[lvl - 1],
            g=g[:, last, :, lvl].contiguous(),
            b=b[:, last, :, lvl].contiguous(),
            w=w[:, last, :, lvl].contiguous(),
            d_states=d_states[lvl],
            d_final=d_finals[lvl],
            firing_interval=firing[lvl],
            n_queries=n_q[lvl - 1],
            has_init=has_init,
        )
        dqb[lvl - 1, :, :n_q[lvl - 1]] = out["dquery_bank"].to(qb.dtype)
        dwp[lvl - 1] = out["dwrite_proj"].to(wp.dtype)
        dg[:, last, :, lvl] = out["dg"].to(g.dtype)
        db[:, last, :, lvl] = out["db"].to(b.dtype)
        dw[:, last, :, lvl] = out["dw"].to(w.dtype)
        if has_init:
            dinit[:, :, lvl] = out["dinit"].to(init.dtype)

        # promotion read src = cat(states_below[1:], final_below), so this is
        # the gradient of the level below's *after-chunk* state.
        below = out["dh"]
        if lvl == 1:
            # fla's dh[t] is already the after-chunk convention, so it goes in
            # unshifted; its last entry lands on b_dh seeded by dht, which is
            # where promotion's read of the final state belongs.
            dh_ext = below
        else:
            # Our own states[t] is the chunk-*start* state, so un-shift.
            if d_states[lvl - 1] is None:
                d_states[lvl - 1] = torch.zeros_like(below)
            d_states[lvl - 1][:, 1:] += below[:, :-1]
            d_finals[lvl - 1] += below[:, -1]

    dq0, dk, dv, db0, dw0, dg0, dh0 = chunk_gdn2_bwd(
        q=q[..., 0, :].contiguous(),
        k=k,
        v=v,
        b=b[..., 0, :].contiguous(),
        w_gate=w[..., 0, :].contiguous(),
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=init[:, :, 0].contiguous() if has_init else None,
        # fla requires do to match v's dtype; mix may arrive in fp32.
        do=(mix[..., 0].unsqueeze(-1) * do).to(v.dtype).contiguous(),
        dht=d_finals[0],
        g=g_cs,
        chunk_size=BT,
        dh_ext=dh_ext,
    )[:7]

    dq[..., 0, :] = dq0
    dg[..., 0, :] = dg0.to(g.dtype)
    db[..., 0, :] = db0.to(b.dtype)
    dw[..., 0, :] = dw0.to(w.dtype)
    if has_init:
        dinit[:, :, 0] = dh0.to(init.dtype)

    return dq, dk, dv, dg, db, dw, dmix.to(mix.dtype), dqb, dwp, dinit
