#!/bin/bash
# Train every arm back to back. Each waits for capacity, trains, uploads, and
# terminates its own instance before the next starts.
#
#   nohup bash scripts/sweep.sh > outputs/sweep.log 2>&1 &
#
# A failed arm doesn't stop the ones after it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ARMS="${ARMS:-nested-learned nested-merge flat-compute-matched flat-state-matched}"

for a in $ARMS; do
    echo "=== $a  $(date -u +%FT%TZ) ==="
    ARM="$a" bash scripts/pretrain.sh "$@" || echo "=== $a FAILED (continuing) ==="
done
echo "=== sweep done $(date -u +%FT%TZ) ==="
