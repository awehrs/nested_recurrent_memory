"""Additive promotion: the control arm.

Level l takes the whole of level l-1 at each firing, with no say in what comes
up:

    S <- S * exp(g) + H

against learned promotion's probe-and-write. Same recurrence, same firing
schedule, same state, so the arms differ in exactly one thing: whether the
promotion is chosen or fixed. The aggregation itself has no parameters -- the
erase gate has no key to act on and the write gate is dropped, which matches
log-linear attention, where the merge is unlearned and the learning sits in the
decay and the per-level read weights. ``g`` and the output mix stay learned in
both arms.

Signatures mirror ``chunk_fwd.promotion_scan`` and ``chunk_bwd.promotion_scan_bwd``
so the composition can swap one pair for the other.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["additive_scan", "additive_scan_bwd"]


@triton.jit
def _additive_scan_fwd_kernel(
    h_ptr,
    g_ptr,
    s_ptr,
    final_ptr,
    init_ptr,
    NT,
    FIRE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """One program per (batch, head); full K x V state resident across the loop.

    Args:
        h_ptr: source-level state after each chunk, [B, NT, H, K, V].
        g_ptr: log-decay at each firing, [B, NT, H, K].
        s_ptr: out, this level's state at each chunk start, [B, NT, H, K, V].
        final_ptr: out, state after the last chunk, [B, H, K, V].
        init_ptr: initial state, unread unless HAS_INIT, [B, H, K, V].
        NT: number of chunks. Runtime, so the chunk loop is a real loop.
        FIRE: firing interval in chunks; fires when (t + 1) % FIRE == 0.
        H: number of heads, used to split the program id into (batch, head).
        K: key dim. Full extent, never tiled.
        V: value dim. Full extent, never tiled.
        HAS_INIT: whether to seed the state from init_ptr rather than zeros.
    """
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    ok = tl.arange(0, K)
    ov = tl.arange(0, V)

    s = tl.zeros([K, V], dtype=tl.float32)
    if HAS_INIT:
        s += tl.load(
            init_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
        ).to(tl.float32)

    for t in range(NT):
        base = ((i_b * NT + t) * H + i_h) * K * V

        tl.store(s_ptr + base + ok[:, None] * V + ov[None, :], s)

        if (t + 1) % FIRE == 0:
            gk = ((i_b * NT + t) * H + i_h) * K
            g = tl.load(g_ptr + gk + ok).to(tl.float32)
            src = tl.load(h_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)
            s = s * tl.exp(g)[:, None] + src

    tl.store(final_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :], s)


@triton.jit
def _additive_scan_bwd_kernel(
    s_ptr,
    g_ptr,
    ds_ptr,
    dfinal_ptr,
    dh_ptr,
    dg_ptr,
    dinit_ptr,
    NT,
    FIRE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """Reverse of ``_additive_scan_fwd_kernel``. One program per (batch, head).

    With S_out = S0 + H and S0 = S * exp(g), the whole reversal is three lines:
    H enters additively so dH is ds unchanged, dg contracts ds against the
    decayed state, and ds picks up exp(g) on its way to the previous chunk.
    Nothing is recomputed and there is no tape.

    Args:
        s_ptr: this level's state at each chunk start, [B, NT, H, K, V].
        g_ptr: log-decay at each firing, [B, NT, H, K].
        ds_ptr: incoming grad w.r.t. ``states``, [B, NT, H, K, V].
        dfinal_ptr: incoming grad w.r.t. the final state, [B, H, K, V].
        dh_ptr: out, grad w.r.t. the source level, [B, NT, H, K, V].
        dg_ptr: out, [B, NT, H, K].
        dinit_ptr: out, grad w.r.t. the initial state, [B, H, K, V].
        NT: number of chunks.
        FIRE: firing interval in chunks.
        H: number of heads.
        K: key dim.
        V: value dim.
        HAS_INIT: whether to write dinit_ptr.
    """
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    ok = tl.arange(0, K)
    ov = tl.arange(0, V)

    ds = tl.load(
        dfinal_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
    ).to(tl.float32)

    for t in range(NT - 1, -1, -1):
        base = ((i_b * NT + t) * H + i_h) * K * V

        if (t + 1) % FIRE == 0:
            gk = ((i_b * NT + t) * H + i_h) * K
            g = tl.load(g_ptr + gk + ok).to(tl.float32)
            s = tl.load(s_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)
            s = s * tl.exp(g)[:, None]

            tl.store(dh_ptr + base + ok[:, None] * V + ov[None, :], ds)
            tl.store(dg_ptr + gk + ok, tl.sum(ds * s, axis=1))
            ds = ds * tl.exp(g)[:, None]

        # This level's read during chunk t uses the chunk-START state, so its
        # gradient joins after the firing at t has been reversed.
        ds += tl.load(ds_ptr + base + ok[:, None] * V + ov[None, :]).to(tl.float32)

    if HAS_INIT:
        tl.store(
            dinit_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :], ds
        )


def additive_scan(
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
    """Scan one additive promotion level over all chunks.

    ``query_bank``, ``write_proj``, ``b`` and ``n_queries`` are accepted so the
    signature matches ``promotion_scan``, and are unused: additive promotion has
    no probe, no write keys and no erase. ``w`` is unused for the same reason --
    there is no write to gate, only a whole state carried up.

    Args:
        h: source-level state *after* each chunk, [B, NT, H, K, V]. Caller is
            responsible for the shift, as in ``promotion_scan``.
        query_bank: ignored.
        write_proj: ignored.
        g: log-decay applied to this level's state at each firing, [B, NT, H, K].
        b: ignored.
        w: ignored.
        firing_interval: fires when ``(chunk + 1) % interval == 0``.
        n_queries: ignored.
        initial_state: this level's state before chunk 0, [B, H, K, V], or None.

    Returns:
        states: this level's state at the *start* of each chunk, [B, NT, H, K, V],
            float32.
        final_state: state after the last chunk, [B, H, K, V], float32.
    """
    B, NT, H, K, V = h.shape

    states = h.new_empty(B, NT, H, K, V, dtype=torch.float32)
    final = h.new_empty(B, H, K, V, dtype=torch.float32)
    init = initial_state.to(torch.float32).contiguous() if initial_state is not None else h

    _additive_scan_fwd_kernel[(B * H,)](
        h.contiguous(),
        g.contiguous(),
        states,
        final,
        init,
        NT,
        FIRE=firing_interval,
        H=H,
        K=K,
        V=V,
        HAS_INIT=initial_state is not None,
    )
    return states, final


def additive_scan_bwd(
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
    """Backward for ``additive_scan``.

    Returns the same keys as ``promotion_scan_bwd`` so the composition needs no
    branch. ``dquery_bank``, ``dwrite_proj``, ``db`` and ``dw`` are zeros: those
    parameters do not participate in additive promotion, so they receive no
    gradient from this level. They still have to be the right shape, because the
    caller scatters them into the per-level gradient buffers.
    """
    B, NT, H, K, V = h.shape

    dh = torch.zeros_like(states)
    dg = torch.zeros_like(g, dtype=torch.float32)
    dinit = torch.empty(B, H, K, V, device=h.device, dtype=torch.float32)

    _additive_scan_bwd_kernel[(B * H,)](
        states.contiguous(),
        g.contiguous(),
        d_states.contiguous(),
        d_final.contiguous(),
        dh,
        dg,
        dinit,
        NT,
        FIRE=firing_interval,
        H=H,
        K=K,
        V=V,
        HAS_INIT=has_init,
    )
    out = {
        "dh": dh,
        "dquery_bank": torch.zeros(H, n_queries, K, device=h.device, dtype=torch.float32),
        "dwrite_proj": torch.zeros(H, K, V, device=h.device, dtype=torch.float32),
        "dg": dg,
        "db": torch.zeros_like(b, dtype=torch.float32),
        "dw": torch.zeros_like(w, dtype=torch.float32),
    }
    if has_init:
        out["dinit"] = dinit
    return out
