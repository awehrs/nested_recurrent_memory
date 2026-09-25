"""Memmapped reader for the shards written by ``train/prepare_data.py``.

No DataLoader and no worker processes: a batch is a handful of slices out of the
page cache. Order is a function of (seed, epoch, step), so a resumed run replays
exactly the batches it would have seen.

Labels are the inputs; the model shifts them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


class TokenShards:
    """One logical token stream over equal-sized uint16 shards.

    The last shard is validation; everything before it is training, and those
    are all full, so shard index is just ``//stride``.
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        names = meta["shards"]
        if len(names) < 2:
            raise ValueError(f"{d} has {len(names)} shards; need at least one train and one val")
        names = names[:-1] if split == "train" else names[-1:]

        self.shards = [np.memmap(d / n, dtype=np.uint16, mode="r") for n in names]
        self.sizes = [len(s) for s in self.shards]
        self.stride = self.sizes[0]
        self.tokens = sum(self.sizes)
        self.eos_token_id = meta["eos_token_id"]
        self.vocab_size = meta["vocab_size"]
        self.tokenizer = meta["tokenizer"]

    def __len__(self) -> int:
        return self.tokens

    def read(self, start: int, length: int) -> np.ndarray:
        """``length`` tokens from ``start``, crossing shard boundaries as needed."""
        out = np.empty(length, dtype=np.uint16)
        n = 0
        while n < length:
            i, off = divmod(start + n, self.stride)
            take = min(length - n, self.sizes[i] - off)
            out[n:n + take] = self.shards[i][off:off + take]
            n += take
        return out

    def batch(self, starts, seq_len: int) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([self.read(int(s) * seq_len, seq_len) for s in starts]).astype(np.int64)
        )


@lru_cache(maxsize=2)
def _perm(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def train_stream(shards: TokenShards, seq_len: int, micro_batch: int, rank: int,
                 world_size: int, seed: int = 0, start_slot: int = 0):
    """Infinite micro-batches, one rank's share.

    A slot is one global micro-batch: ranks split it, so they never overlap and
    together cover the epoch. ``start_slot`` skips ahead on resume.
    """
    per = micro_batch * world_size
    n = (len(shards) // seq_len // per) * per   # truncate so no batch straddles an epoch
    if n == 0:
        raise ValueError(f"{len(shards)} tokens is under one global batch of {per}x{seq_len}")
    slot = start_slot
    while True:
        epoch, base = divmod(slot * per, n)
        idx = _perm(n, seed, epoch)[base + rank * micro_batch: base + (rank + 1) * micro_batch]
        yield shards.batch(idx, seq_len)
        slot += 1


def val_batches(shards: TokenShards, seq_len: int, micro_batch: int, rank: int,
                world_size: int, n_batches: int):
    """The same windows every time, in order, so val curves are comparable."""
    per = micro_batch * world_size
    n = max(len(shards) // seq_len - micro_batch + 1, 1)
    for b in range(n_batches):
        base = (b * per + rank * micro_batch) % n
        yield shards.batch(range(base, base + micro_batch), seq_len)
