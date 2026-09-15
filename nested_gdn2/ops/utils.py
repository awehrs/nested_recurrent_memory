"""Chunk-layout helpers shared by the forward and backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["from_chunks", "last_token_index", "to_chunks"]


def to_chunks(
    x: torch.Tensor,
    NT: int,
    BT: int,
) -> torch.Tensor:
    """[B, T, H, D] -> [B, NT, BT, H, D], zero-padding a ragged last chunk."""
    B, T, H, D = x.shape
    pad = NT * BT - T
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
    return x.view(B, NT, BT, H, D)


def from_chunks(
    x: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """[B, NT, BT, H, D] -> [B, T, H, D], dropping any padding."""
    B, NT, BT, H, D = x.shape
    return x.reshape(B, NT * BT, H, D)[:, :T]


def last_token_index(
    NT: int,
    BT: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """Index of each chunk's last token, [NT].

    Where levels >= 1 gather their gates. Clamped so a ragged final chunk reads
    its own last token rather than past the end. Cheap enough to rebuild in the
    backward, which keeps it out of the autograd context.
    """
    return (torch.arange(NT, device=device) * BT + (BT - 1)).clamp(max=T - 1)
