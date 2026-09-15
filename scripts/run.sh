#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

source .env

# ── config ─────────────────────────────────────────────────────────────────────
# Runs an arbitrary python target on a GPU pod. Same lifecycle as test.sh; the
# only difference is what gets executed remotely.
#
# Usage: bash scripts/run.sh <target.py> [target args...]
#   bash scripts/run.sh benchmarks/bench_chunk.py
#   bash scripts/run.sh benchmarks/bench_chunk.py --only breakdown
#   PULL_BACK="outputs" bash scripts/run.sh train/pretrain.py --steps 100

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/run.sh <target.py> [args...]" >&2
    exit 2
fi

RUN_TARGET="$1"; shift
RUN_ARGS="${*}"

if [ ! -f "$RUN_TARGET" ]; then
    echo "no such target: ${RUN_TARGET}" >&2
    exit 2
fi

GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3}"
GPU_COUNT="${GPU_COUNT:-1}"
IMAGE="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DISK_GB="${DISK_GB:-40}"
SSH_KEY=~/.ssh/id_ed25519
SSH_OPTS="-o StrictHostKeyChecking=no"
SSH_USER="root"
PROJECT_NAME=nested_recurrent_memory
# Directories to copy back when the run finishes, space separated.
PULL_BACK="${PULL_BACK:-}"

RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# ── helpers ────────────────────────────────────────────────────────────────────
gql() {
    curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg q "$1" '{query: $q}')"
}

# ── create pod (unbounded wait for capacity) ───────────────────────────────────
RETRY_DELAY="${LAUNCH_RETRY_DELAY:-60}"
echo "Creating RunPod pod (${GPU_COUNT}x ${GPU_TYPE}) — polling every ${RETRY_DELAY}s for capacity, Ctrl-C to stop..."

POD_ID=""
retry_attempt=0
while [ -z "$POD_ID" ] || [ "$POD_ID" = "null" ]; do
    retry_attempt=$((retry_attempt + 1))
    CREATE_RESP=$(gql "mutation {
        podFindAndDeployOnDemand(input: {
            gpuCount: ${GPU_COUNT},
            containerDiskInGb: ${DISK_GB},
            minVcpuCount: $((4 * GPU_COUNT)),
            minMemoryInGb: $((32 * GPU_COUNT)),
            gpuTypeId: \"${GPU_TYPE}\",
            name: \"${PROJECT_NAME}-run\",
            imageName: \"${IMAGE}\",
            ports: \"22/tcp\",
            startSsh: true,
            startJupyter: false,
            supportPublicIp: true
        }) { id imageName }
    }")

    POD_ID=$(echo "$CREATE_RESP" | jq -r '.data.podFindAndDeployOnDemand.id // ""')

    if [ -n "$POD_ID" ] && [ "$POD_ID" != "null" ]; then
        echo "Pod created: $POD_ID"
        break
    fi

    ERROR_CODE=$(echo "$CREATE_RESP" | jq -r '.errors[0].extensions.code // ""')
    ERROR_MSG=$(echo "$CREATE_RESP" | jq -r '.errors[0].message // "no capacity"')

    if [ -n "$ERROR_CODE" ] && [ "$ERROR_CODE" != "SUPPLY_CONSTRAINT" ]; then
        echo "Failed to create pod (non-retryable, code=${ERROR_CODE}):"
        echo "$CREATE_RESP" | jq '.errors'
        exit 1
    fi

    echo "   [${retry_attempt}] no capacity (${ERROR_MSG}) — retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

# ── cleanup trap ───────────────────────────────────────────────────────────────
cleanup() {
    echo "Terminating pod $POD_ID..."
    gql "mutation { podTerminate(input: { podId: \"${POD_ID}\" }) }" > /dev/null
    echo "Pod terminated"
}
trap cleanup EXIT

# ── wait for IP and port ───────────────────────────────────────────────────────
echo "Waiting for pod to start..."
SSH_HOST=""
SSH_PORT=""

for attempt in $(seq 1 120); do
    POD_RESP=$(gql "query { pod(input: { podId: \"${POD_ID}\" }) { desiredStatus machineId runtime { ports { ip isIpPublic privatePort publicPort } } } }")
    MACHINE_ID=$(echo "$POD_RESP" | jq -r '.data.pod.machineId // "none"')
    SSH_HOST=$(echo "$POD_RESP" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort | tostring == "22") | select(.isIpPublic == true)] | first | .ip // ""' 2>/dev/null) || SSH_HOST=""
    SSH_PORT=$(echo "$POD_RESP" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort | tostring == "22") | select(.isIpPublic == true)] | first | .publicPort // ""' 2>/dev/null) || SSH_PORT=""
    echo "   Attempt $attempt/120 — Machine: ${MACHINE_ID}  IP: ${SSH_HOST}  Port: ${SSH_PORT}"

    if [ -n "$SSH_HOST" ] && [ "$SSH_HOST" != "null" ] && [ -n "$SSH_PORT" ] && [ "$SSH_PORT" != "null" ]; then
        echo "Pod running at ${SSH_HOST}:${SSH_PORT}"
        break
    fi

    if [ $attempt -eq 30 ] && [ "$MACHINE_ID" = "none" ]; then
        echo "No GPU assigned after 5 minutes"
        exit 1
    fi

    if [ $attempt -eq 120 ]; then
        echo "Pod never got a public IP"
        exit 1
    fi

    sleep 10
done

# ── wait for SSH ───────────────────────────────────────────────────────────────
echo "Waiting for SSH..."
SSH_READY=false
for attempt in $(seq 1 60); do
    SSH_ERR=$(ssh -o BatchMode=yes $SSH_OPTS -o ConnectTimeout=10 \
        -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} 'echo ok' 2>&1) || true
    if echo "$SSH_ERR" | grep -q "^ok$"; then
        echo "SSH ready"
        SSH_READY=true
        break
    fi
    echo "   Attempt $attempt/60 — $SSH_ERR"
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo "SSH never became ready"
    exit 1
fi

# ── rsync project ──────────────────────────────────────────────────────────────
ssh $SSH_OPTS -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} "apt-get update -qq && apt-get install -y rsync -qq"

echo "Syncing project..."
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='outputs' \
    --exclude='wandb' \
    -e "ssh $SSH_OPTS -i ${SSH_KEY} -p ${SSH_PORT}" \
    . ${SSH_USER}@${SSH_HOST}:~/${PROJECT_NAME}/

echo "Code synced"

# ── remote setup + run ─────────────────────────────────────────────────────────
cat > /tmp/setup_and_run.sh << EOF
#!/bin/bash
set -e

PROJECT_NAME=${PROJECT_NAME}

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.local/bin/env

export MAX_JOBS=\$(nproc)

cd ~/\${PROJECT_NAME}

retry() { for a in 1 2 3 4 5 6 7 8; do "\$@" && return 0; echo "  [retry \$a] '\$*' failed, sleep 20s..."; sleep 20; done; return 1; }

echo "Syncing dependencies..."
# Let uv match the torch CUDA wheel to the pod's driver; RunPod hands out mixed
# driver versions and a cu130 wheel on a 12.8 driver fails at first cuda call.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
retry uv sync
uv pip install ninja

echo "Installing flash-linear-attention (pinned; must match ops/_vendor/README.md)..."
MAX_JOBS=\$(nproc) uv pip install flash-linear-attention==${FLA_VERSION:-0.5.0}

echo "Verifying fla import:"
uv run python -c "import fla; print('fla OK')" || { echo "fla import FAILED"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/workspace/triton_cache
mkdir -p /workspace/triton_cache

echo "GPU info:"
nvidia-smi

echo "Verifying torch sees the GPU:"
uv run python - <<'PYEOF' || { echo "torch cannot use the GPU on this pod; aborting early"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} cuda {torch.version.cuda} available={ok} "
      f"devices={torch.cuda.device_count() if ok else 0}")
sys.exit(0 if ok else 1)
PYEOF

echo "Running: uv run python ${RUN_TARGET} ${RUN_ARGS}"
PYTHONUNBUFFERED=1 uv run python ${RUN_TARGET} ${RUN_ARGS} 2>&1 | tee run.log
STATUS=\${PIPESTATUS[0]}
echo "Run complete"
exit \$STATUS
EOF

scp $SSH_OPTS -i $SSH_KEY -P $SSH_PORT \
    /tmp/setup_and_run.sh ${SSH_USER}@${SSH_HOST}:~/

echo "Running ${RUN_TARGET}..."
ssh $SSH_OPTS -o ConnectTimeout=30 \
    -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
    "apt-get install -y -qq tmux && chmod +x setup_and_run.sh && tmux new-session -d -s run 'bash setup_and_run.sh > ~/run.log 2>&1; echo EXIT_CODE=\$? >> ~/run.log'"

# Reconnecting tail: the job runs detached in tmux, ssh drops don't tear down the pod.
echo "Tailing logs (reconnects on ssh drop; job runs detached in tmux)..."
misses=0
while ! ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
         "grep -q EXIT_CODE ~/run.log 2>/dev/null"; do
    if ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} "true" 2>/dev/null; then
        misses=0
        ssh $SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
            -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
            "tail -f ~/run.log --pid=\$(tmux list-panes -t run -F '#{pane_pid}' 2>/dev/null | head -1)" || true
    else
        misses=$((misses + 1))
        if [ "$misses" -ge 30 ]; then echo "pod unreachable ${misses}x; stopping tail"; break; fi
    fi
    sleep 5
done

# Pull the log back, plus anything the caller asked for.
scp $SSH_OPTS -i $SSH_KEY -P $SSH_PORT \
    ${SSH_USER}@${SSH_HOST}:~/run.log ./run.log 2>/dev/null || true

for d in $PULL_BACK; do
    echo "Pulling back ${d}..."
    rsync -a -e "ssh $SSH_OPTS -i ${SSH_KEY} -p ${SSH_PORT}" \
        ${SSH_USER}@${SSH_HOST}:~/${PROJECT_NAME}/${d}/ ./${d}/ 2>/dev/null || true
done

EXIT_CODE=$(grep -oE 'EXIT_CODE=[0-9]+' run.log 2>/dev/null | tail -1 | cut -d= -f2)
echo "Done! (cleanup will terminate pod)"
exit "${EXIT_CODE:-1}"
