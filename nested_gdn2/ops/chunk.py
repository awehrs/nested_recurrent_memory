"""Chunkwise NestedGDN-2: the composition, and the autograd seam.

Level 0 is stock GDN-2 from fla. Each level above it is a gather, a probe and an
update, chained: level l reads the finished state of level l-1. Reads come off
every level and are mixed per token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd

from nested_gdn2.ops._vendor.gdn2_chunk_bwd import chunk_gdn2_bwd
from nested_gdn2.ops.probe import PROBES
from nested_gdn2.ops.update_bwd import update_state_bwd
from nested_gdn2.ops.update_fwd import update_state

__all__ = ["chunk_nested_gdn2"]

# Tensors saved before the per-level runs: thirteen forward inputs plus the
# three fla intermediates. Keep in step with save_for_backward below.
N_FIXED = 16


def n_groups(n_chunks: int, firing_interval: int) -> int:
    """How many runs of ``firing_interval`` chunks cover ``n_chunks``.

    The last run may be short; it is read but never fired.
    """
    return (n_chunks + firing_interval - 1) // firing_interval


def n_firings(n_chunks: int, firing_interval: int) -> int:
    """How many firings occur in ``n_chunks``. One per complete run."""
    return n_chunks // firing_interval


def _source_index(
    firing_below: int,
    firing_here: int,
    n_chunks: int,
) -> list[int]:
    """Which of the level below's groups each of this level's firings reads.

    This level's i-th firing ends chunk ``(i+1)*firing_here - 1``, by which
    point the level below has completed ``(i+1)*firing_here // firing_below``
    firings; its state is the one entering that group. Strictly increasing, and
    only the last entry can run past the stored groups, where it means the final
    state instead.
    """
    return [
        ((i + 1) * firing_here) // firing_below
        for i in range(n_firings(n_chunks, firing_here))
    ]


def gather_source(
    states_below: torch.Tensor,
    final_below: torch.Tensor,
    firing_below: int,
    firing_here: int,
    n_chunks: int,
) -> torch.Tensor:
    """The level below's state *after* each of this level's firing chunks.

    The level below stores one state per its own group. After chunk t it has
    completed (t+1) // firing_below firings, and its state is the one entering
    that group; nested intervals make the division exact. The final entry comes
    from ``final_below`` when the index runs past the stored groups.

    Args:
        states_below: [B, NG_below, H, K, V].
        final_below: [B, H, K, V].
        firing_below: the level below's interval, in chunks.
        firing_here: this level's interval, in chunks.
        n_chunks: NT.

    Returns:
        [B, NF_here, H, K, V].
    """
    idx = _source_index(firing_below, firing_here, n_chunks)
    ng_below = states_below.shape[1]
    inside = [j for j in idx if j < ng_below]

    # dtype is explicit: inside is empty when every firing reads final_below,
    # and torch.tensor([]) would be float and unusable as an index.
    idx_t = torch.tensor(inside, device=states_below.device, dtype=torch.long)
    src = states_below[:, idx_t]
    if len(inside) < len(idx):
        tail = final_below.unsqueeze(1).expand(-1, len(idx) - len(inside), -1, -1, -1)
        src = torch.cat([src, tail], dim=1)
    return src


def scatter_source(
    d_source: torch.Tensor,
    firing_below: int,
    firing_here: int,
    n_chunks: int,
    n_groups_below: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``gather_source``, for the backward.

    Returns:
        d_states_below: [B, NG_below, H, K, V], zero where this level did not
            read.
        d_final_below: [B, H, K, V].
    """
    idx = _source_index(firing_below, firing_here, n_chunks)
    B, _, H, K, V = d_source.shape
    opts = dict(device=d_source.device, dtype=d_source.dtype)

    d_states = torch.zeros(B, n_groups_below, H, K, V, **opts)
    d_final = torch.zeros(B, H, K, V, **opts)

    inside = [j for j in idx if j < n_groups_below]
    if inside:
        d_states.index_add_(
            1,
            torch.tensor(inside, device=d_source.device, dtype=torch.long),
            d_source[:, : len(inside)],
        )
    if len(inside) < len(idx):
        d_final += d_source[:, len(inside):].sum(1)
    return d_states, d_final


def read_level(
    q_level: torch.Tensor,
    states: torch.Tensor,
    firing_interval: int,
    chunk_size: int,
    n_tokens: int,
    scale: float,
) -> torch.Tensor:
    """Read a level at every token.

    Every token in a group reads the state that group started with, so the
    queries are viewed as [B, NG, group_tokens, H, K] and contracted against the
    per-group states in one einsum -- no gather, no repeat.

    Returns:
        [B, T, H, V].
    """
    B, _, H, K = q_level.shape
    ng = states.shape[1]
    group_tokens = firing_interval * chunk_size

    pad = ng * group_tokens - n_tokens
    if pad:
        q_level = F.pad(q_level, (0, 0, 0, 0, 0, pad))
    q_grouped = q_level.view(B, ng, group_tokens, H, K) * scale

    read = torch.einsum("bnthk,bnhkv->bnthv", q_grouped.float(), states)
    return read.reshape(B, ng * group_tokens, H, -1)[:, :n_tokens]


def chunk_nested_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    mix_weights: torch.Tensor,
    query_banks: torch.Tensor,
    key_projections: torch.Tensor,
    b_projections: torch.Tensor,
    w_projections: torch.Tensor,
    g_projections: torch.Tensor,
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

    Args:
        q: per-level read queries, [B, T, H, L, K]. Level 0's slice is also its
            input-triple query.
        k: level-0 keys, [B, T, H, K]. Caller-normalized.
        v: level-0 values, [B, T, H, V].
        g: level-0 log-decay, [B, T, H, K]. Levels above get theirs from the
            probe, one per firing.
        b: level-0 erase gate, [B, T, H, K]. Levels above get theirs from the
            probe, per pair.
        w: level-0 write gate, [B, T, H, V]. Same.
        mix_weights: per-token per-level mix, [B, T, H, L], pre-softmaxed.
        query_banks: probes for levels >= 1, [L-1, H, N, K]. Padded to the
            widest level; each level slices what it uses.
        key_projections: value->key, for levels >= 1, [L-1, H, K, V].
        b_projections: value->erase gate, [L-1, H, V, K].
        w_projections: value->write gate, [L-1, H, V, V].
        g_projections: value->log-decay, [L-1, H, V, K].
        n_queries_per_level: probes per level >= 1, [L-1], int.
        firing_intervals: per-level interval in chunks, [L], int, with
            ``firing_intervals[0] == 1`` and each a multiple of the one below.
        L: number of levels.
        scale: attention scale; defaults to ``1 / sqrt(K)``.
        initial_state: [B, H, L, K, V] or None.
        output_final_state: whether to return the final state.
        chunk_size: tokens per chunk. Must be 64 -- fla's GDN-2 kernels assume it.
        return_aux: also return the intermediates the backward needs.
        promotion: which entry of ops.probe.PROBES to use, ``"learned"`` or
            ``"merge"``. Both arms gate identically; only the pairs differ.

    Returns:
        o: [B, T, H, V].
        final_state: [B, H, L, K, V] if ``output_final_state`` else None.
        aux: dict of backward intermediates, only if ``return_aux``.

    Outline:
        level 0 through fla -> o_0, h, final_0
        for each level above:
            gather the level below at this level's firings
            probe  -> keys, values
            update -> states (one per group), final
            read, scale by mix, accumulate into o
    """
    B, T, H, _, K = q.shape
    BT = chunk_size
    if BT != 64:
        raise ValueError(f"chunk_size must be 64 for GDN-2, got {BT}")
    NT = (T + BT - 1) // BT
    if scale is None:
        scale = K**-0.5

    o, final, g_cs, Aqk, Akk, _, _, _, _, _, h, _ = chunk_gdn2_fwd(
        q=q[..., 0, :].contiguous(),
        k=k,
        v=v,
        g=g.contiguous(),
        b=b.contiguous(),
        w_gate=w.contiguous(),
        scale=scale,
        initial_state=initial_state[:, :, 0].contiguous() if initial_state is not None else None,
        output_final_state=True,
        chunk_size=BT,
        return_intermediate_states=True,
    )

    reads = [o]
    o = (mix_weights[..., 0].unsqueeze(-1) * o).to(o.dtype)
    finals = [final]
    states_per_level = [h]

    probe = PROBES[promotion]

    for lvl in range(1, L):
        f_here = int(firing_intervals[lvl])
        f_below = int(firing_intervals[lvl - 1])

        src = gather_source(
            states_per_level[lvl - 1], finals[lvl - 1], f_below, f_here, NT
        )
        keys, values, b_f, w_f, g_f = probe(
            src,
            query_banks[lvl - 1],
            key_projections[lvl - 1],
            b_projections[lvl - 1],
            w_projections[lvl - 1],
            g_projections[lvl - 1],
            int(n_queries_per_level[lvl - 1]),
            v.dtype,
        )

        states, fin = update_state(
            keys=keys,
            values=values,
            g=g_f,
            b=b_f,
            w=w_f,
            n_groups=n_groups(NT, f_here),
            initial_state=initial_state[:, :, lvl] if initial_state is not None else None,
        )

        read = read_level(q[..., lvl, :], states, f_here, BT, T, scale)
        o = o + (mix_weights[..., lvl].unsqueeze(-1) * read).to(o.dtype)

        reads.append(read)
        finals.append(fin)
        states_per_level.append(states)

    final_state = torch.stack(finals, dim=2) if output_final_state else None
    if not return_aux:
        return o, final_state
    aux = dict(
        g_cs=g_cs,
        Aqk=Aqk,
        Akk=Akk,
        states=states_per_level,
        finals=finals,
        reads=reads,
        scale=scale,
    )
    return o, final_state, aux


def read_level_bwd(
    d_read: torch.Tensor,
    q_level: torch.Tensor,
    states: torch.Tensor,
    firing_interval: int,
    chunk_size: int,
    n_tokens: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for ``read_level``.

    Returns:
        d_states: [B, NG, H, K, V]. Every token in a group contributed to the
            same state, so their gradients sum into one entry.
        d_q_level: [B, T, H, K].
    """
    B, _, H, K = q_level.shape
    ng = states.shape[1]
    group_tokens = firing_interval * chunk_size
    pad = ng * group_tokens - n_tokens

    if pad:
        d_read = F.pad(d_read, (0, 0, 0, 0, 0, pad))
        q_level = F.pad(q_level, (0, 0, 0, 0, 0, pad))
    d_read = d_read.view(B, ng, group_tokens, H, -1).float()
    q_grouped = q_level.view(B, ng, group_tokens, H, K).float() * scale

    d_states = torch.einsum("bnthv,bnthk->bnhkv", d_read, q_grouped)
    d_q = torch.einsum("bnthv,bnhkv->bnthk", d_read, states) * scale
    return d_states, d_q.reshape(B, ng * group_tokens, H, K)[:, :n_tokens]


def probe_replay(
    probe,
    source: torch.Tensor,
    query_bank: torch.Tensor,
    key_proj: torch.Tensor,
    b_proj: torch.Tensor,
    w_proj: torch.Tensor,
    g_proj: torch.Tensor,
    n_queries: int,
    out_dtype: torch.dtype,
) -> tuple[tuple, tuple]:
    """Re-run a probe with a live graph, so autograd can reverse it.

    The forward runs with gradients off, so nothing it built is differentiable
    later. Replaying costs two matmuls and means a new probe needs a forward
    only -- no hand-derived normalization Jacobian, no matrix-map gradients.

    It also serves the update's backward, which needs the keys, values and gates
    to rebuild its own intermediates. Replaying covers both, so none of them are
    saved from the forward.

    Returns:
        outputs: (keys, values, b, w, g), still attached to the replay graph.
        leaves: (source, query_bank, key_proj, b_proj, w_proj, g_proj), the
            tensors to differentiate with respect to.
    """
    with torch.enable_grad():
        leaves = tuple(
            t.detach().requires_grad_(True)
            for t in (source, query_bank, key_proj, b_proj, w_proj, g_proj)
        )
        outputs = probe(*leaves, n_queries, out_dtype)
    return outputs, leaves


def chunk_nested_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    mix_weights: torch.Tensor,
    query_banks: torch.Tensor,
    key_projections: torch.Tensor,
    b_projections: torch.Tensor,
    w_projections: torch.Tensor,
    g_projections: torch.Tensor,
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

    Walks the ladder top-down. Each level's key and value gradients go back
    through its probe, producing the gradient of the level below; that is
    scattered onto the level below's groups, or -- for level 1 -- handed to
    fla's patched backward as ``dh_ext``, which is the only way promotion's
    contribution can join level 0's state recurrence.

    Returns:
        (dq, dk, dv, dg, db, dw, dmix, dquery_banks, dkey_projections,
        db_projections, dw_projections, dinit).
    """
    B, T, H, _, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    do = do.contiguous()
    do32 = do.float()

    dmix = torch.stack([(do32 * r.float()).sum(-1) for r in reads], dim=-1)
    dq = torch.zeros_like(q)
    db = torch.zeros_like(b)
    dw = torch.zeros_like(w)
    dqb = torch.zeros_like(query_banks)
    dkp = torch.zeros_like(key_projections)
    dbp = torch.zeros_like(b_projections)
    dwp = torch.zeros_like(w_projections)
    dgp = torch.zeros_like(g_projections)
    dinit = torch.zeros_like(initial_state) if has_init else None

    d_states = [None] * L
    d_finals = [
        (d_final[:, :, i].float() if d_final is not None else
         torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32))
        for i in range(L)
    ]

    # Level 0's read is fla's own output; its gradient goes in as `do`.
    for lvl in range(1, L):
        d_read = (mix_weights[..., lvl].unsqueeze(-1) * do).float()
        ds, dq_lvl = read_level_bwd(
            d_read, q[..., lvl, :], states[lvl],
            int(firing_intervals[lvl]), BT, T, scale,
        )
        d_states[lvl] = ds
        dq[..., lvl, :] = dq_lvl.to(q.dtype)

    probe = PROBES[promotion]
    dh_ext = None

    for lvl in range(L - 1, 0, -1):
        f_here = int(firing_intervals[lvl])
        f_below = int(firing_intervals[lvl - 1])
        source = gather_source(
            states[lvl - 1], finals[lvl - 1], f_below, f_here, NT
        )
        (keys, values, b_pair, w_pair, g_pair), leaves = probe_replay(
            probe,
            source,
            query_banks[lvl - 1],
            key_projections[lvl - 1],
            b_projections[lvl - 1],
            w_projections[lvl - 1],
            g_projections[lvl - 1],
            int(n_queries_per_level[lvl - 1]),
            v.dtype,
        )

        out = update_state_bwd(
            states=states[lvl],
            keys=keys.detach(),
            values=values.detach(),
            g=g_pair.detach(),
            b=b_pair.detach(),
            w=w_pair.detach(),
            d_states=d_states[lvl],
            d_final=d_finals[lvl],
            has_init=has_init,
        )
        if has_init:
            dinit[:, :, lvl] = out["dinit"].to(initial_state.dtype)

        # Merge's keys are a constant identity with no graph; the rest carry
        # gradient back in one call.
        outs = [keys, values, b_pair, w_pair, g_pair]
        douts = [
            out["dkeys"].to(keys.dtype),
            out["dvalues"].to(values.dtype),
            out["db"].to(b_pair.dtype),
            out["dw"].to(w_pair.dtype),
            out["dg"].to(g_pair.dtype),
        ]
        live = [(o, d) for o, d in zip(outs, douts) if o.requires_grad]
        if not live:
            raise RuntimeError(
                f"level {lvl}: no probe output carries a graph, so nothing "
                f"upstream can receive gradient"
            )
        grads = torch.autograd.grad(
            outputs=[o for o, _ in live],
            inputs=list(leaves),
            grad_outputs=[d for _, d in live],
            allow_unused=True,
        )
        d_source = grads[0] if grads[0] is not None else torch.zeros_like(source)
        for buf, gr in (
            (dqb, grads[1]),
            (dkp, grads[2]),
            (dbp, grads[3]),
            (dwp, grads[4]),
            (dgp, grads[5]),
        ):
            if gr is not None:
                buf[lvl - 1] += gr.to(buf.dtype)

        below_states, below_final = scatter_source(
            d_source.float(), f_below, f_here, NT, states[lvl - 1].shape[1]
        )
        if lvl == 1:
            # fla's dh[t] is the gradient of the state *leaving* chunk t; ours
            # is the state entering chunk t, so the injection shifts down one.
            # Entry 0 is always zero: no level reads chunk 0's entering state.
            dh_ext = torch.zeros_like(below_states)
            dh_ext[:, :-1] = below_states[:, 1:]
            d_finals[0] = d_finals[0] + below_final
        else:
            d_states[lvl - 1] = (
                below_states if d_states[lvl - 1] is None
                else d_states[lvl - 1] + below_states
            )
            d_finals[lvl - 1] = d_finals[lvl - 1] + below_final

    dq0, dk, dv, db0, dw0, dg0, dh0 = chunk_gdn2_bwd(
        q=q[..., 0, :].contiguous(),
        k=k,
        v=v,
        b=b.contiguous(),
        w_gate=w.contiguous(),
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=initial_state[:, :, 0].contiguous() if has_init else None,
        # fla requires do to match v's dtype; mix may arrive in fp32.
        do=(mix_weights[..., 0].unsqueeze(-1) * do).to(v.dtype).contiguous(),
        dht=d_finals[0],
        g=g_cs,
        chunk_size=BT,
        dh_ext=dh_ext,
    )[:7]

    dq[..., 0, :] = dq0
    dg = dg0.to(g.dtype)
    db = db0.to(b.dtype)
    dw = dw0.to(w.dtype)
    if has_init:
        dinit[:, :, 0] = dh0.to(initial_state.dtype)

    return (
        dq, dk, dv, dg, db, dw, dmix.to(mix_weights.dtype),
        dqb, dkp, dbp, dwp, dgp, dinit,
    )


class ChunkNestedGDN2Function(torch.autograd.Function):
    """Autograd node for the whole op."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights,
        query_banks,
        key_projections,
        b_projections,
        w_projections,
        g_projections,
        n_queries_per_level,
        firing_intervals,
        L,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        promotion,
    ):
        o, final_state, aux = chunk_nested_gdn2_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            mix_weights=mix_weights,
            query_banks=query_banks,
            key_projections=key_projections,
            b_projections=b_projections,
            w_projections=w_projections,
            g_projections=g_projections,
            n_queries_per_level=n_queries_per_level,
            firing_intervals=firing_intervals,
            L=L,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            return_aux=True,
            promotion=promotion,
        )
        # Three runs of L. Keys, values and gates come back from the probe replay.
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            b,
            w,
            mix_weights,
            query_banks,
            key_projections,
            b_projections,
            w_projections,
            g_projections,
            initial_state,
            aux["g_cs"],
            aux["Aqk"],
            aux["Akk"],
            *aux["states"],
            *aux["reads"],
            *aux["finals"],
        )
        ctx.meta = (
            L,
            aux["scale"],
            chunk_size,
            [int(x) for x in firing_intervals],
            [int(x) for x in n_queries_per_level],
            initial_state is not None,
            promotion,
        )
        return o, (final_state if output_final_state else None)

    @staticmethod
    def backward(ctx, do, dfinal):
        (L, scale, chunk_size, firing, n_q, has_init, promotion) = ctx.meta
        saved = list(ctx.saved_tensors)
        (q, k, v, g, b, w, mix, qb, kp, bp, wp, gp, init,
         g_cs, Aqk, Akk) = saved[:N_FIXED]

        per_level = saved[N_FIXED:]
        assert len(per_level) == 3 * L, (
            f"save_for_backward layout changed: expected {3 * L} per-level "
            f"tensors after {N_FIXED} fixed ones, got {len(per_level)}"
        )
        states = per_level[0 * L:1 * L]
        reads = per_level[1 * L:2 * L]
        finals = per_level[2 * L:3 * L]

        grads = chunk_nested_gdn2_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            mix_weights=mix,
            query_banks=qb,
            key_projections=kp,
            b_projections=bp,
            w_projections=wp,
            g_projections=gp,
            initial_state=init,
            g_cs=g_cs,
            Aqk=Aqk,
            Akk=Akk,
            states=states,
            reads=reads,
            finals=finals,
            do=do,
            d_final=dfinal,
            L=L,
            scale=scale,
            chunk_size=chunk_size,
            firing_intervals=firing,
            n_queries_per_level=n_q,
            has_init=has_init,
            promotion=promotion,
        )
        dq, dk, dv, dg, db, dw, dmix, dqb, dkp, dbp, dwp, dgp, dinit = grads

        return (
            dq,
            dk,
            dv,
            dg,
            db,
            dw,
            dmix,
            dqb,
            dkp,
            dbp,
            dwp,
            dgp,
            None,   # n_queries_per_level
            None,   # firing_intervals
            None,   # L
            None,   # scale
            dinit,
            None,   # output_final_state
            None,   # chunk_size
            None,   # promotion
        )


def chunk_nested_gdn2(
    q,
    k,
    v,
    g,
    b,
    w,
    mix_weights,
    query_banks,
    key_projections,
    b_projections,
    w_projections,
    g_projections,
    n_queries_per_level,
    firing_intervals,
    L,
    scale=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    promotion="learned",
):
    """Differentiable chunkwise NestedGDN-2. Signature matches the naive op."""
    return ChunkNestedGDN2Function.apply(
        q,
        k,
        v,
        g,
        b,
        w,
        mix_weights,
        query_banks,
        key_projections,
        b_projections,
        w_projections,
        g_projections,
        n_queries_per_level,
        firing_intervals,
        L,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        promotion,
    )
