"""Configuration for NestedGDN-2."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

__all__ = ["NestedGDN2Config"]


class NestedGDN2Config(PretrainedConfig):
    model_type = "nested_gdn2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_heads: int = 16,
        head_dim: int = 64,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        fuse_swiglu: bool = True,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        use_cache: bool = True,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        num_levels: int = 2,
        firing_intervals: tuple[int, ...] = (1, 4),
        n_queries_per_level: tuple[int, ...] = (64,),
        promotion: str = "learned",
        op_chunk_size: int = 64,
        use_triton: bool = True,
        **kwargs,
    ):
        """
        Args:
            num_levels: depth of the memory hierarchy, L. Level 0 is stock GDN-2.
            firing_intervals: per-level firing interval in chunks, length L, with
                ``firing_intervals[0] == 1``. Level l first fires at token
                ``firing_intervals[l] * op_chunk_size``.
            n_queries_per_level: probes per level >= 1, length L-1. Each must be
                a power of two >= 16 on the triton path. Validated on both arms
                even though merge promotion ignores it, so that two configs
                differing only in ``promotion`` are always both valid.
                Cost is close to flat up to ``head_dim`` and rises sharply past
                it: measured on an H100 at head_dim 64, going 16 -> 64 costs
                ~2% of forward time for 4x the promotion width, while 64 -> 128
                costs ~20%. The promotion kernel contracts through this
                dimension, which is the cheaper order only while it is below
                head_dim. Default 64 accordingly.
            promotion: how a level is filled from the one below. ``"learned"``
                probes it with the query bank and writes the result; ``"merge"``
                carries it up whole. The two arms of the experiment: same state,
                same schedule, differing only in whether promotion is chosen.
            op_chunk_size: tokens per chunk. Must be 64 for the triton path.
            use_triton: use the chunkwise triton op rather than the naive
                reference. The reference is for testing; it is far slower.
            head_dim: the promotion kernel holds a full head_dim x head_dim
                state in registers, so large values risk spilling.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.num_levels = num_levels
        self.firing_intervals = tuple(firing_intervals)
        self.n_queries_per_level = tuple(n_queries_per_level)
        self.promotion = promotion
        self.op_chunk_size = op_chunk_size
        self.use_triton = use_triton

        if len(self.firing_intervals) != num_levels:
            raise ValueError(
                f"firing_intervals must have length num_levels={num_levels}, "
                f"got {len(self.firing_intervals)}"
            )
        if self.firing_intervals[0] != 1:
            raise ValueError(f"firing_intervals[0] must be 1, got {self.firing_intervals[0]}")
        if any(b < a for a, b in zip(self.firing_intervals, self.firing_intervals[1:])):
            raise ValueError(
                f"firing_intervals must be non-decreasing, got {self.firing_intervals}"
            )
        if promotion not in ("learned", "merge"):
            raise ValueError(f"promotion must be 'learned' or 'merge', got {promotion!r}")
        if len(self.n_queries_per_level) != num_levels - 1:
            raise ValueError(
                f"n_queries_per_level must have length num_levels-1={num_levels - 1}, "
                f"got {len(self.n_queries_per_level)}"
            )
        if use_triton:
            if op_chunk_size != 64:
                raise ValueError(f"op_chunk_size must be 64 on the triton path, got {op_chunk_size}")
            for n in self.n_queries_per_level:
                if n < 16 or n & (n - 1):
                    raise ValueError(
                        f"n_queries_per_level entries must be powers of two >= 16, got {n}"
                    )

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
