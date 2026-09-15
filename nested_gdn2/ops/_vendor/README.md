# Vendored from flash-linear-attention

Copied from `flash-linear-attention==0.5.2` (MIT) -- the version in `uv.lock`
and in the local venv. Everything here is upstream's code with one feature
added; nothing in this directory is original work beyond the `dh_ext` plumbing
described below.

**Version discrepancy, unresolved.** `scripts/test.sh` and `scripts/run.sh` pin
`0.5.0` on the pod, downgrading after `uv sync`. So the vendored files are
0.5.2-derived but are validated, and run, against 0.5.0's imports. The suite
passes that way, so the seven helpers `gdn2_chunk_bwd.py` pulls from fla have
compatible signatures across both. Worth collapsing to one version before any
long run -- set `FLA_VERSION=0.5.2` for run.sh, or bump the pin in test.sh, and
confirm the suite still passes.

| file | upstream path | copied |
|---|---|---|
| `common_chunk_delta_h.py` | `fla/ops/common/chunk_delta_h.py` | backward half only (kernel + wrapper); the forward is used unmodified from fla |
| `gdn2_chunk_bwd.py` | `fla/ops/gdn2/chunk_bwd.py` | the `chunk_gdn2_bwd` orchestration only; no kernels |

## The change: `dh_ext`

Promotion delivers gradient to level 0's state at *every firing chunk*, not just
at the final state. That gradient has to participate in the reverse state
recurrence, because gradient entering at chunk `t` must keep flowing back to
chunks `< t`.

fla carries `b_dh` in registers across the reverse scan and never materializes
it between chunks, so there is no way to add to it from outside. Adding to the
returned `dh` tensor instead would give the right value at `t` and zero
propagation below it.

Superposition does not rescue this. The `dh` recurrence is linear, so in
principle the injection's contribution could be computed by a separate decay
scan and added. But `dh` and `dv` are mutually coupled inside the kernel
(`b_dv = dot(b_k, b_dh)`, then `b_dh += ... - dot(b_w, b_dv)`), so a second pass
would have to reproduce that coupling -- i.e. reimplement the kernel.

So `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` takes a `dh_ext` pointer
to a per-chunk `[B, NT, H, K, V]` buffer, added into `b_dh` immediately before
each chunk's store -- before the decay-and-accumulate that carries `b_dh` to the
previous chunk. Four add sites, because fla unrolls up to four 64-wide K
sub-blocks into separate named variables.

Diff against upstream is confined to:
- `HAS_DH_EXT` heuristic
- `dh_ext` kernel parameter and its base-offset computation
- four `b_dhN += tl.load(dh_ext + ...)` blocks
- `dh_ext` threaded through both Python wrappers

## Re-syncing

On a flash-linear-attention bump, diff the two upstream files against these and
re-apply the four add sites. `tests/ops/test_chunk_bwd.py` and
`tests/ops/test_chunk_autograd.py` will catch a bad merge.

Better: upstream it. `dh_ext` is small, general, and additively composable --
any hierarchical or multi-timescale state model needs the same hook. A merged PR
deletes this directory.
