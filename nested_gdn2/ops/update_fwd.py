"""Phase 2 of promotion: apply key-value pairs to this level's state.

The sequential half. Knows nothing about where the pairs came from.

A level's state only changes when it fires, so the f chunks between firings all
read the same state: one state is stored per group, not per chunk.

Nothing reduces over value space, so a program owns a slice of columns and the
grid is B*H*NV.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["update_state"]


@triton.jit
def _update_state_kernel(
    keys_ptr,
    values_ptr,
    g_ptr,
    b_ptr,
    w_ptr,
    init_ptr,
    states_ptr,
    final_ptr,
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
    """One program per (batch, head, value block); state slice held in registers.

    Key space cannot be tiled: the erase reduces across all of it to produce the
    values being written.

    Args:
        keys_ptr: [B, NF, H, N, K]. Addressed by the strides below rather than
            by an assumed layout; merge returns a broadcast identity.
        values_ptr: [B, NF, H, N, V], contiguous.
        g_ptr: log-decay at each firing, [B, NF, H, K]. Per firing, not per
            pair: decay multiplies the whole state before anything lands.
        b_ptr: erase gate per pair, [B, NF, H, N, K].
        w_ptr: write gate per pair, [B, NF, H, N, V].
        init_ptr: state before group 0, unread unless HAS_INIT, [B, H, K, V].
        states_ptr: out, one state per group, [B, NG, H, K, V].
        final_ptr: out, state after the last firing, [B, H, K, V].
        sk_b: elements to advance keys_ptr by to move one step along B. Zero
            when keys are broadcast along this axis.
        sk_f: same, along the firing axis.
        sk_h: same, along the head axis.
        sk_n: same, along the pair axis. Last axis assumed unit-stride.
        NG: number of groups.
        NF: number of firings. NG - 1 < NF <= NG; they differ only when the
            chunk count is not a multiple of the firing interval.
        N: pairs written per firing. n_queries for learned, K for merge.
            A tl.dot dimension, so compile-time.
        H: number of heads, used to split the program id.
        K: key dim. Full extent, never tiled.
        V: value dim. Full extent; BV of it per program.
        BV: columns this program owns. At least 16, a tl.dot dimension.
        HAS_INIT: whether to seed from init_ptr rather than zeros.
    """
    NV: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_v = pid % NV
    i_h = (pid // NV) % H
    i_b = pid // (NV * H)

    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    on = tl.arange(0, N)

    s = tl.zeros([K, BV], dtype=tl.float32)
    if HAS_INIT:
        s += tl.load(
            init_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :]
        ).to(tl.float32)

    for j in range(NG):
        tl.store(
            states_ptr + ((i_b * NG + j) * H + i_h) * K * V
            + ok[:, None] * V + ov[None, :],
            s,
        )

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

            s = s * tl.exp(g)[:, None]
            kb = keys * bg
            vn = values * wg - tl.dot(kb, s)
            s = s + tl.dot(tl.trans(keys), vn)

    tl.store(
        final_ptr + (i_b * H + i_h) * K * V + ok[:, None] * V + ov[None, :],
        s,
    )


def update_state(
    keys: torch.Tensor,
    values: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    n_groups: int,
    initial_state: torch.Tensor | None = None,
    block_v: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one level's write scan over all groups.

    Args:
        keys: [B, NF, H, N, K]. May be a broadcast view; its strides are read
            here and passed to the kernel.
        values: [B, NF, H, N, V].
        g: log-decay at each firing, [B, NF, H, K].
        b: erase gate per pair, [B, NF, H, N, K].
        w: write gate per pair, [B, NF, H, N, V].
        n_groups: NG. A trailing group without a firing is stored but not
            updated.
        initial_state: [B, H, K, V], or None for zeros.
        block_v: columns per program. V must be divisible by it.

    Returns:
        states: one state per group, [B, NG, H, K, V], float32.
        final_state: after the last firing, [B, H, K, V], float32.
    """
    B, NF, H, N, K = keys.shape
    V = values.shape[-1]

    if N < 16 or N & (N - 1):
        raise ValueError(f"pairs per firing must be a power of two >= 16, got {N}")
    if block_v < 16 or block_v & (block_v - 1):
        raise ValueError(f"block_v must be a power of two >= 16, got {block_v}")
    if V % block_v:
        raise ValueError(f"V={V} must be divisible by block_v={block_v}")
    if keys.stride(-1) != 1:
        raise ValueError("keys must be unit-stride along K")
    if not (n_groups - 1 <= NF <= n_groups):
        raise ValueError(f"NF={NF} inconsistent with n_groups={n_groups}")

    states = values.new_empty(B, n_groups, H, K, V, dtype=torch.float32)
    final = values.new_empty(B, H, K, V, dtype=torch.float32)
    # Unread unless HAS_INIT, but still a valid pointer for the launch.
    init = (
        initial_state.to(torch.float32).contiguous()
        if initial_state is not None
        else values
    )

    _update_state_kernel[(B * H * (V // block_v),)](
        keys,
        values.contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        init,
        states,
        final,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        n_groups,
        NF,
        N=N,
        H=H,
        K=K,
        V=V,
        BV=block_v,
        HAS_INIT=initial_state is not None,
    )
    return states, final
