"""Public entry point for the chunkwise NestedGDN-2 op.

Ties the forward composition to both backward kernels. The work lives in
``chunk_fwd`` and ``chunk_bwd``; this module is only the autograd seam.
"""

from __future__ import annotations

import torch

from nested_gdn2.ops.chunk_bwd import chunk_nested_gdn2_bwd
from nested_gdn2.ops.chunk_fwd import chunk_nested_gdn2_fwd

__all__ = ["chunk_nested_gdn2"]

# Tensors saved before the per-level runs: the ten forward inputs plus the three
# fla intermediates. Keep in step with the save_for_backward call below.
N_FIXED = 13


class ChunkNestedGDN2Function(torch.autograd.Function):
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
        write_projections,
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
            write_projections=write_projections,
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
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            b,
            w,
            mix_weights,
            query_banks,
            write_projections,
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
        (q, k, v, g, b, w, mix, qb, wp, init, g_cs, Aqk, Akk) = saved[:N_FIXED]
        # Three per-level runs of length L, in the order they were saved.
        per_level = saved[N_FIXED:]
        assert len(per_level) == 3 * L, (
            f"save_for_backward layout changed: expected {3 * L} per-level tensors "
            f"after {N_FIXED} fixed ones, got {len(per_level)}"
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
            write_projections=wp,
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
        dq, dk, dv, dg, db, dw, dmix, dqb, dwp, dinit = grads

        return (
            dq,
            dk,
            dv,
            dg,
            db,
            dw,
            dmix,
            dqb,
            dwp,
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
    write_projections,
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
        write_projections,
        n_queries_per_level,
        firing_intervals,
        L,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        promotion,
    )
