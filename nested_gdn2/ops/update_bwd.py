"""Reverse of the write scan.

Splits the gradient of this level's state three ways: back to the previous
group, out to the key-value pairs, out to the gates.

dkeys, db and dg are partial -- each program covers one value block and the
caller sums them. dvalues, dw and dinit are complete as written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["update_state_bwd"]


@triton.jit
def _update_state_bwd_kernel(
    states_ptr,
    keys_ptr,
    values_ptr,
    g_ptr,
    b_ptr,
    w_ptr,
    d_states_ptr,
    d_final_ptr,
    dkeys_ptr,
    dvalues_ptr,
    dg_ptr,
    db_ptr,
    dw_ptr,
    dinit_ptr,
    sk_b,
    sk_f,
    sk_h,
    sk_n,
    NG,
    NF,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """One program per (batch, head, value block), walking groups in reverse.

    Args:
        states_ptr: state entering each group, [B, NG, H, K, V].
        keys_ptr: [B, NF, H, N, K], addressed by the strides below.
        values_ptr: [B, NF, H, N, V], contiguous.
        g_ptr: log-decay at each firing, [B, NF, H, K].
        b_ptr: erase gate per pair, [B, NF, H, N, K].
        w_ptr: write gate per pair, [B, NF, H, N, V].
        d_states_ptr: incoming, one per group, [B, NG, H, K, V].
        d_final_ptr: incoming, [B, H, K, V].
        dkeys_ptr: out, partial, [B, NF, H, NV, N, K].
        dvalues_ptr: out, complete, [B, NF, H, N, V].
        dg_ptr: out, partial, [B, NF, H, NV, K].
        db_ptr: out, partial, [B, NF, H, NV, N, K].
        dw_ptr: out, complete, [B, NF, H, N, V].
        dinit_ptr: out, complete, [B, H, K, V].
        sk_b: elements to advance keys_ptr by to move one step along B; zero
            when keys are broadcast along it. See ops.update_fwd.
        sk_f: same, along the firing axis.
        sk_h: same, along the head axis.
        sk_n: same, along the pair axis. Last axis assumed unit-stride.
        NG: number of groups.
        NF: number of firings.
        N: pairs per firing. A tl.dot dimension.
        H: number of heads.
        K: key dim, never tiled.
        V: value dim; BV of it per program.
        BV: columns this program owns.
        HAS_INIT: whether to write dinit_ptr.
    """
    NV: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_v = pid % NV
    i_h = (pid // NV) % H
    i_b = pid // (NV * H)

    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    on = tl.arange(0, N)

    ds = tl.load(
        d_final_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
    ).to(tl.float32)

    for j in range(NG - 1, -1, -1):
        if j < NF:
            gk = ((i_b * NF + j) * H + i_h) * K
            g = tl.load(g_ptr + gk + ok).to(tl.float32)

            pn = ((i_b * NF + j) * H + i_h) * N
            bg = tl.load(
                b_ptr + pn * K + on[:, None] * K + ok[None, :]
            ).to(tl.float32)
            wg = tl.load(
                w_ptr + pn * V + on[:, None] * V + ov[None, :]
            ).to(tl.float32)

            keys = tl.load(
                keys_ptr + i_b * sk_b + j * sk_f + i_h * sk_h
                + on[:, None] * sk_n + ok[None, :]
            ).to(tl.float32)
            values = tl.load(
                values_ptr + ((i_b * NF + j) * H + i_h) * N * V
                + on[:, None] * V + ov[None, :]
            ).to(tl.float32)

            # Recompute the forward's intermediates from the group's state.
            s = tl.load(
                states_ptr + ((i_b * NG + j) * H + i_h) * K * V
                + ok[:, None] * V + ov[None, :]
            ).to(tl.float32)
            s = s * tl.exp(g)[:, None]
            kb = keys * bg
            vn = values * wg - tl.dot(kb, s)

            # Every term below reads the incoming ds, so dS0 lands last.
            dvn = tl.dot(keys, ds)
            dkeys = tl.dot(vn, tl.trans(ds))
            dvalues = dvn * wg
            dw = dvn * values

            dkb = -tl.dot(dvn, tl.trans(s))
            dkeys += dkb * bg
            db = dkb * keys

            d_s0 = ds - tl.dot(tl.trans(kb), dvn)
            dg = tl.sum(d_s0 * s, axis=1)
            ds = d_s0 * tl.exp(g)[:, None]

            pk = (((i_b * NF + j) * H + i_h) * NV + i_v) * N * K
            tl.store(dkeys_ptr + pk + on[:, None] * K + ok[None, :], dkeys)
            tl.store(
                dvalues_ptr + ((i_b * NF + j) * H + i_h) * N * V
                + on[:, None] * V + ov[None, :],
                dvalues,
            )
            tl.store(db_ptr + pk + on[:, None] * K + ok[None, :], db)
            tl.store(
                dw_ptr + pn * V + on[:, None] * V + ov[None, :],
                dw,
            )
            tl.store(
                dg_ptr + (((i_b * NF + j) * H + i_h) * NV + i_v) * K + ok, dg
            )

        # This group read the state entering it, so its gradient joins here.
        ds += tl.load(
            d_states_ptr + ((i_b * NG + j) * H + i_h) * K * V
            + ok[:, None] * V + ov[None, :]
        ).to(tl.float32)

    if HAS_INIT:
        tl.store(
            dinit_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :],
            ds,
        )


def update_state_bwd(
    states: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    d_states: torch.Tensor,
    d_final: torch.Tensor,
    has_init: bool = False,
    block_v: int = 16,
) -> dict:
    """Backward for ``update_state``.

    Allocates the partial buffers, launches, and reduces them over value blocks.

    Args:
        states: one state per group, [B, NG, H, K, V], from the forward.
        keys: [B, NF, H, N, K]. May be a broadcast view.
        values: [B, NF, H, N, V].
        g: log-decay at each firing, [B, NF, H, K].
        b: erase gate per pair, [B, NF, H, N, K].
        w: write gate per pair, [B, NF, H, N, V].
        d_states: incoming gradient, one per group, [B, NG, H, K, V].
        d_final: incoming gradient of the final state, [B, H, K, V].
        has_init: whether the forward was given an initial state.
        block_v: columns per program; must match the forward.

    Returns:
        dict with dkeys, dvalues, dg, db, dw, and dinit when has_init.
    """
    B, NF, H, N, K = keys.shape
    NG = states.shape[1]
    V = values.shape[-1]

    if N < 16 or N & (N - 1):
        raise ValueError(f"pairs per firing must be a power of two >= 16, got {N}")
    if block_v < 16 or block_v & (block_v - 1):
        raise ValueError(f"block_v must be a power of two >= 16, got {block_v}")
    if V % block_v:
        raise ValueError(f"V={V} must be divisible by block_v={block_v}")
    if keys.stride(-1) != 1:
        raise ValueError("keys must be unit-stride along K")
    if not (NG - 1 <= NF <= NG):
        raise ValueError(f"NF={NF} inconsistent with NG={NG}")

    nv = V // block_v
    f32 = dict(device=keys.device, dtype=torch.float32)

    # Partial over value blocks; reduced before returning.
    dkeys_p = torch.zeros(B, NF, H, nv, N, K, **f32)
    db_p = torch.zeros(B, NF, H, nv, N, K, **f32)
    dg_p = torch.zeros(B, NF, H, nv, K, **f32)
    # Complete as written: their value index is free.
    dvalues = torch.zeros(B, NF, H, N, V, **f32)
    dw = torch.zeros(B, NF, H, N, V, **f32)
    dinit = torch.empty(B, H, K, V, **f32)

    _update_state_bwd_kernel[(B * H * nv,)](
        states.contiguous(),
        keys,
        values.contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        d_states.contiguous(),
        d_final.contiguous(),
        dkeys_p,
        dvalues,
        dg_p,
        db_p,
        dw,
        dinit,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        NG,
        NF,
        N=N,
        H=H,
        K=K,
        V=V,
        BV=block_v,
        HAS_INIT=has_init,
    )

    out = {
        "dkeys": dkeys_p.sum(3),
        "dvalues": dvalues,
        "dg": dg_p.sum(3),
        "db": db_p.sum(3),
        "dw": dw,
    }
    if has_init:
        out["dinit"] = dinit
    return out
