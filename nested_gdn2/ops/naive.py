import torch
import torch.nn.functional as F

from nested_gdn2.ops.probe import PROBES


def naive_update_step(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """One firing: decay, erase at the keys, write the values.

    All N pairs act on the state as it stood before the firing, so their order
    cannot reach the result. Unfactored on purpose: the kernel contracts through
    N instead, so a mistake in that re-association cannot cancel on both sides.

    Args:
        state: [B, H, K, V].
        keys: [B, H, N, K].
        values: [B, H, N, V].
        b: [B, H, N, K].
        w: [B, H, N, V].
        g: [B, H, K].

    Returns:
        [B, H, K, V].
    """
    S = state * g.exp().unsqueeze(-1)
    kb = keys * b
    gram = torch.einsum("bhnk,bhnj->bhkj", keys, kb)
    return (
        S
        - torch.einsum("bhkj,bhjv->bhkv", gram, S)
        + torch.einsum("bhnk,bhnv->bhkv", keys, values * w)
    )


def naive_step_nested_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    mix_weights: torch.Tensor,
    state: torch.Tensor,
    position: int | torch.Tensor,
    query_banks: torch.Tensor,
    key_projections: torch.Tensor,
    b_projections: torch.Tensor,
    w_projections: torch.Tensor,
    g_projections: torch.Tensor,
    n_queries_per_level: torch.Tensor,
    firing_intervals: torch.Tensor,
    L: int,
    scale: float | None = None,
    chunk_size: int = 64,
    promotion: str = "learned",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token: level 0 writes, every level is read and mixed, then each level
    whose firing this token completes promotes, bottom-up.

    Level 0 writes before the read, levels above after, so a firing is first
    visible at the next token.

    Args:
        q: [B, H, L, K].
        k: [B, H, K].
        v: [B, H, V].
        g: level-0 log-decay, [B, H, K].
        b: level-0 erase gate, [B, H, K].
        w: level-0 write gate, [B, H, V].
        mix_weights: [B, H, L].
        state: [B, H, L, K, V], float32.
        position: index of this token, from 0. An int, or [B] when sequences in
            the batch sit at different positions and so fire on different tokens.
        The rest are as for naive_chunk_nested_gdn2.

    Returns:
        o: [B, H, V], in q's dtype.
        state: [B, H, L, K, V], float32.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out_dtype = q.dtype
    q, k, v, g, b, w, mix_weights = (
        x.float() for x in (q, k, v, g, b, w, mix_weights)
    )
    S = list(state.float().unbind(2))

    S[0] = naive_update_step(
        S[0], k.unsqueeze(2), v.unsqueeze(2), b.unsqueeze(2), w.unsqueeze(2), g
    )
    reads = torch.einsum("bhlk,bhlkv->bhlv", q * scale, torch.stack(S, dim=2))
    o = (reads * mix_weights.unsqueeze(-1)).sum(2)

    end = 1 + (
        position
        if torch.is_tensor(position)
        else torch.full(k.shape[:1], position, device=k.device, dtype=torch.long)
    )
    at_boundary = end % chunk_size == 0
    chunk = torch.div(end, chunk_size, rounding_mode="floor")

    # Only the rows that fire are promoted; with rows at different positions,
    # some row is at a boundary on most steps.
    for lvl in range(1, L):
        idx = (at_boundary & (chunk % int(firing_intervals[lvl]) == 0)).nonzero(
            as_tuple=True
        )[0]
        if idx.numel() == 0:
            continue
        keys, values, b_pair, w_pair, g_pair = (
            x.squeeze(1)
            for x in PROBES[promotion](
                S[lvl - 1][idx].unsqueeze(1),
                query_banks[lvl - 1],
                key_projections[lvl - 1],
                b_projections[lvl - 1],
                w_projections[lvl - 1],
                g_projections[lvl - 1],
                int(n_queries_per_level[lvl - 1]),
                torch.float32,
            )
        )
        S[lvl] = S[lvl].index_copy(
            0, idx, naive_update_step(S[lvl][idx], keys, values, b_pair, w_pair, g_pair)
        )

    return o.to(out_dtype), torch.stack(S, dim=2)


def naive_recurrent_nested_gdn2(
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
    chunk_size: int = 1,
    promotion: str = "learned",
):
    """Token-by-token reference forward pass for NestedGDN-2.

    Every level uses its own read query, supplied by the caller. Level 0's
    read query (q_reads[..., 0, :]) also serves as its input-triple query
    (Option (b): tied at level 0).

    Args:
        q: per-level read queries, shape [B, T, H, L, K].
            Caller pre-applies per-level read projections. Level 0 uses
            q_reads[..., 0, :] for state reads; higher levels use their
            corresponding slice.
        k: keys of shape [B, T, H, K] (level 0's input-triple key).
            Caller-normalized (L2 along last dim) per FLA convention.
        v: values of shape [B, T, H, V] (level 0's input-triple value).
        g: level-0 log-decay of shape [B, T, H, K]. Levels above derive
            theirs in the probe.
        b: level-0 channel-wise erase gate of shape [B, T, H, K]. Levels above
            get theirs from the probe, per promoted pair.
        w: level-0 channel-wise write gate of shape [B, T, H, V]. Same.
        mix_weights: per-token per-level mix weights of shape [B, T, H, L]
            (expected pre-softmaxed across L).
        query_banks: learned extraction queries for levels ≥ 1, shape
            [L-1, H, N_MAX, K].
        key_projections: learned value→key projections for levels ≥ 1,
            shape [L-1, H, K, V].
        b_projections: value→erase-gate projections, shape [L-1, H, V, K].
        w_projections: value→write-gate projections, shape [L-1, H, V, V].
        g_projections: value→log-decay projections, shape [L-1, H, V, K].
        n_queries_per_level: number of extraction queries per level ≥ 1,
            shape [L-1], dtype int.
        firing_intervals: per-level firing interval, in chunks. Shape [L],
            dtype int. Level ℓ fires when (chunk_index + 1) % firing_intervals[ℓ] == 0.
            Convention: firing_intervals[0] = 1.
        L: number of memory levels.
        scale: attention scale; defaults to 1 / sqrt(K).
        initial_state: optional [B, H, L, K, V] initial state in float32.
        output_final_state: whether to return the final state.
        chunk_size: BT, tokens per chunk. Used to convert token index t to
            chunk index for firing checks: chunk_idx = t // chunk_size.
            Defaults to 1 (every token is its own chunk).

    Returns:
        o: outputs of shape [B, T, H, V].
        final_state: [B, H, L, K, V] if output_final_state else None.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    state = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, L, K, V, device=v.device, dtype=torch.float32)
    )

    o = []
    for t in range(T):
        o_t, state = naive_step_nested_gdn2(
            q[:, t], k[:, t], v[:, t], g[:, t], b[:, t], w[:, t], mix_weights[:, t],
            state, t,
            query_banks, key_projections, b_projections, w_projections, g_projections,
            n_queries_per_level, firing_intervals, L,
            scale=scale, chunk_size=chunk_size, promotion=promotion,
        )
        o.append(o_t)

    return torch.stack(o, dim=1), (state if output_final_state else None)


def naive_chunk_nested_gdn2(
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
    promotion: str = "learned",
):
    """

    Args:
        q: per-level read queries, shape [B, T, H, L, K].
            Caller pre-applies per-level read projections. Level 0 uses
            q[..., 0, :] for state reads; higher levels use their
            corresponding slice.
        k: keys of shape [B, T, H, K] (level 0's input-triple key).
            Caller-normalized (L2 along last dim) per FLA convention.
        v: values of shape [B, T, H, V] (level 0's input-triple value).
        g: level-0 log-decay of shape [B, T, H, K]. Levels above derive
            theirs in the probe.
        b: level-0 channel-wise erase gate of shape [B, T, H, K]. Levels above
            get theirs from the probe, per promoted pair.
        w: level-0 channel-wise write gate of shape [B, T, H, V]. Same.
        mix_weights: per-token per-level mix weights of shape [B, T, H, L]
            (expected pre-softmaxed across L).
        query_banks: learned extraction queries for levels ≥ 1, shape
            [L-1, H, N_MAX, K].
        key_projections: learned value→key projections for levels ≥ 1,
            shape [L-1, H, K, V].
        b_projections: value→erase-gate projections, shape [L-1, H, V, K].
        w_projections: value→write-gate projections, shape [L-1, H, V, V].
        g_projections: value→log-decay projections, shape [L-1, H, V, K].
        n_queries_per_level: number of extraction queries per level ≥ 1,
            shape [L-1], dtype int.
        firing_intervals: per-level firing interval, in chunks. Shape [L],
            dtype int. Level ℓ fires when (chunk_index + 1) % firing_intervals[ℓ] == 0.
            Convention: firing_intervals[0] = 1.
        L: number of memory levels.
        scale: attention scale; defaults to 1 / sqrt(K).
        initial_state: optional [B, H, L, K, V] initial state in float32.
        output_final_state: whether to return the final state.
        chunk_size: BT, tokens per chunk.

    Returns:
        o: outputs of shape [B, T, H, V].
        final_state: [B, H, L, K, V] if output_final_state else None.
    """

    if scale is None:
        scale = q.shape[-1] ** -0.5
    BT = chunk_size

    orig_dtype = q.dtype
    q, k, v, g, b, w, mix_weights = (
        x.transpose(1, 2).contiguous().float()
        for x in (q, k, v, g, b, w, mix_weights)
    )
    # q: [B, H, T, L, K]

    B, H, T, K = k.shape
    V = v.shape[-1]

    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, 0, 0, pad_len))
        b = F.pad(b, (0, 0, 0, pad_len))
        w = F.pad(w, (0, 0, 0, pad_len))
        mix_weights = F.pad(mix_weights, (0, 0, 0, pad_len))
    T_pad = k.shape[2]
    NT = T_pad // BT

    q = q * scale

    q0 = q[..., 0, :]
    g0 = g
    b0, w0 = b, w

    def chunk(x):
        return x.view(B, H, NT, BT, -1)

    q0, k, v = (chunk(x) for x in (q0, k, v))
    g0, b0, w0 = (chunk(x) for x in (g0, b0, w0))

    mix_weights = mix_weights.view(B, H, NT, BT, L)
    q_reads = q.view(B, H, NT, BT, L, K)

    g_cum = g0.cumsum(-2)
    g_last = g_cum[..., -1:, :]

    k_g = k * g_cum.exp()
    k_g_b = k_g * b0

    decay_ij = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3))
    decay_ij_exp = decay_ij.clamp(max=0).exp()
    tril_mask = torch.tril(
        torch.ones(BT, BT, device=k.device, dtype=torch.bool),
        diagonal=-1,
    )
    bk = b0 * k
    T_lower = torch.einsum('bhnik,bhnjk,bhnijk->bhnij',
                           bk, k, decay_ij_exp)
    T_lower = T_lower.masked_fill(~tril_mask, 0.0)

    # (I + T_lower)^-1 @ rhs, solved directly rather than by forming the inverse.
    A = torch.eye(BT, device=k.device, dtype=torch.float32) + T_lower
    wy = torch.linalg.solve_triangular(
        A, torch.cat([w0 * v, k_g_b], dim=-1), upper=False, unitriangular=True
    )
    u_wy, w_wy = wy.split([V, K], dim=-1)
    k_tail = k * (g_last - g_cum).exp()

    decay_qk = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)).clamp(max=0).exp()
    causal_mask = torch.tril(
        torch.ones(BT, BT, device=k.device, dtype=torch.bool),
        diagonal=0,
    )

    if initial_state is not None:
        S_list = [initial_state[:, :, lvl].to(torch.float32).clone() for lvl in range(L)]
    else:
        S_list = [torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32) for _ in range(L)]

    o = torch.zeros(B, H, NT, BT, V, device=v.device, dtype=torch.float32)

    for n in range(NT):
        q0_n = q0[:, :, n]
        k_n = k[:, :, n]
        g_n = g_cum[:, :, n]
        g_last_n = g_last[:, :, n].squeeze(-2)
        w_n = w_wy[:, :, n]
        u_n = u_wy[:, :, n]
        k_tail_n = k_tail[:, :, n]

        v_new = u_n - w_n @ S_list[0]
        A_qk = torch.einsum('bhik,bhjk,bhijk->bhij',
                            q0_n, k_n, decay_qk[:, :, n]).masked_fill(~causal_mask, 0.0)
        r_0 = A_qk @ v_new + (q0_n * g_n.exp()) @ S_list[0]

        reads = [r_0]
        for lvl in range(1, L):
            q_lvl_n = q_reads[:, :, n, :, lvl, :]
            r_lvl = torch.einsum("bhtk,bhkv->bhtv", q_lvl_n, S_list[lvl])
            reads.append(r_lvl)

        reads_stacked = torch.stack(reads, dim=3)
        mix_n = mix_weights[:, :, n]

        o[:, :, n] = (reads_stacked * mix_n.unsqueeze(-1)).sum(-2)

        S_list[0] = S_list[0] * g_last_n.unsqueeze(-1).exp() + k_tail_n.transpose(-1, -2) @ v_new

        for lvl in range(1, L):
            f_lvl = int(firing_intervals[lvl].item())
            if (n + 1) % f_lvl != 0:
                continue

            keys, vals, b_pair, w_pair, g_pair = PROBES[promotion](
                S_list[lvl - 1].unsqueeze(1),
                query_banks[lvl - 1],
                key_projections[lvl - 1],
                b_projections[lvl - 1],
                w_projections[lvl - 1],
                g_projections[lvl - 1],
                int(n_queries_per_level[lvl - 1].item()),
                torch.float32,
            )
            k_writes, v_writes = keys.squeeze(1), vals.squeeze(1)
            b_pair, w_pair = b_pair.squeeze(1), w_pair.squeeze(1)
            b_g = g_pair.squeeze(1)

            S_list[lvl] = naive_update_step(
                S_list[lvl], k_writes, v_writes, b_pair, w_pair, b_g
            )

    o = o.reshape(B, H, T_pad, V)[:, :, :T].transpose(1, 2).contiguous().to(orig_dtype)

    final_state = torch.stack(S_list, dim=2) if output_final_state else None
    return o, final_state
