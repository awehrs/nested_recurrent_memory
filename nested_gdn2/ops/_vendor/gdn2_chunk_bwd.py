"""Vendored from fla/ops/gdn2/chunk_bwd.py (MIT): the ``chunk_gdn2_bwd``
orchestration only, so the state scan can be routed to our patched
``chunk_gated_delta_rule_bwd_dhu``.

No kernels are copied. Everything except that one call is imported from fla, so
this stays a thin sequencing layer. Re-sync when bumping flash-linear-attention.
"""

from __future__ import annotations

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused
from fla.ops.gdn2.chunk_intra import chunk_gdn2_bwd_intra
from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

from nested_gdn2.ops._vendor.common_chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

__all__ = ["chunk_gdn2_bwd"]


def chunk_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    state_v_first: bool = False,
    w_wy: torch.Tensor | None = None,
    u_wy: torch.Tensor | None = None,
    qg: torch.Tensor | None = None,
    kg: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    disable_recompute: bool = False,
    dh_ext: torch.Tensor | None = None,
):
    """End-to-end GDN-2 backward, with promotion's gradient injected.

    Returns (dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad). ``db`` has
    shape [B, T, H, K] (channel-wise erase gate); ``dw`` has shape [B, T, H, V]
    (channel-wise write gate).

    ``dh_ext`` is the one addition over upstream: a per-chunk [B, NT, H, K, V]
    gradient added into the state recurrence at each chunk, carrying the
    contribution from levels above. ``None`` reproduces upstream exactly.
    """
    if not disable_recompute:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        w_wy, u_wy, qg, kg = recompute_w_u_fwd_gdn2(
            k=k,
            v=v,
            b=b,
            w_gate=w_gate,
            A=Akk,
            q=q,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w_wy,
            u=u_wy,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w_wy,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        dh_ext=dh_ext,
    )

    dq, dk, dv, db, dw, dg, dAkk = chunk_gdn2_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        b=b,
        w_gate=w_gate,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    dq, dk, db, dg = chunk_gdn2_bwd_intra(
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    dA_log, dt_bias_grad = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_gate_in_kernel:
        dg, dA_log, dt_bias_grad = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound,
        )

    return dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad
