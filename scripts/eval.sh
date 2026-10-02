#!/bin/bash
set -e

# Positional perplexity for finished checkpoints, on one GPU. Launches an
# instance, runs, uploads results to GCS, pulls them back, and terminates.
#
#   bash scripts/eval.sh
#   ARMS="nested-learned flat-compute-matched" bash scripts/eval.sh
#   RUN_FILTER='-2026092[34]-' bash scripts/eval.sh        # pin older runs
#   KEEP_ALIVE=1 bash scripts/eval.sh                      # no self-termination

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

source .env

# ── config ─────────────────────────────────────────────────────────────────────
# Only provisioning differs between providers; everything downstream is shared.
PROVIDER="${PROVIDER:-lambda}"
case "$PROVIDER" in
    lambda)
        INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_a100_sxm4}"
        SSH_USER=ubuntu
        SSH_PORT=22
        ;;
    runpod)
        INSTANCE_TYPE="${INSTANCE_TYPE:-NVIDIA A100-SXM4-80GB}"
        IMAGE="${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
        DISK_GB="${DISK_GB:-120}"
        CUDA_VERSION="${CUDA_VERSION:-12.8}"
        SSH_USER=root
        ;;
    *) echo "PROVIDER must be lambda or runpod, got ${PROVIDER}" >&2; exit 2 ;;
esac
REGION="${REGION:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/scoobdoob.pem}"
SSH_KEY_NAME="${SSH_KEY_NAME:-scoobdoob}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"
PROJECT_NAME=nested_recurrent_memory
FLA_VERSION="${FLA_VERSION:-0.5.2}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"
SERVICE_KEY=~/.config/gcloud/servicekey.json

GCS_BASE="${GCS_BASE:-gs://nested-gdn}"
ARMS="${ARMS:-nested-learned nested-merge flat-compute-matched flat-state-matched}"
# A wrapped paste can put a newline inside this, which lands verbatim in the
# generated remote script and breaks the for-loops. Unquoted word splitting
# collapses runs of whitespace and trims the ends.
ARMS=$(echo $ARMS)

PPL_DATA="${PPL_DATA:-fla-hub/pg19}"
# The train split is 28,602 books and ppl.py tokenizes the whole split before
# evaluating; at the default num_proc that OOMs the box. 100 books is plenty.
PPL_SPLIT="${PPL_SPLIT:-test}"
PPL_PROC="${PPL_PROC:-8}"
PPL_BLOCK="${PPL_BLOCK:-16384}"
PPL_BUCKET="${PPL_BUCKET:-1024}"

LAMBDA_API="https://cloud.lambdalabs.com/api/v1"
api() {
    local method=$1 path=$2 body=${3:-}
    if [ -n "$body" ]; then
        curl -s -u "${LAMBDA_LABS_API_KEY}:" -X "$method" "${LAMBDA_API}${path}" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -s -u "${LAMBDA_LABS_API_KEY}:" -X "$method" "${LAMBDA_API}${path}"
    fi
}
gql() {
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" -d "$(jq -n --arg q "$1" '{query: $q}')"
}

# Each provisioner sets INSTANCE_ID, or leaves it empty when there is no
# capacity so the caller retries.
launch_lambda() {
    local regions
    if [ -n "$REGION" ]; then regions="$REGION"; else
        regions=$(api GET /instance-types \
            | jq -r --arg t "$INSTANCE_TYPE" \
              '.data[$t].regions_with_capacity_available[]?.name' 2>/dev/null || true)
    fi
    local resp code
    for r in $regions; do
        resp=$(api POST /instance-operations/launch "$(jq -n \
            --arg r "$r" --arg t "$INSTANCE_TYPE" --arg k "$SSH_KEY_NAME" --arg n "${PROJECT_NAME}-eval" \
            '{region_name:$r, instance_type_name:$t, ssh_key_names:[$k],
              file_system_names:[], quantity:1, name:$n}')")
        INSTANCE_ID=$(echo "$resp" | jq -r '.data.instance_ids[0] // ""')
        [ -n "$INSTANCE_ID" ] && { echo "Instance launched in ${r}: ${INSTANCE_ID}"; return; }
        code=$(echo "$resp" | jq -r '.error.code // ""')
        case "$code" in
            *capacity*|"") ;;
            *) echo "Launch failed (code=${code})" >&2; echo "$resp" | jq '.error' >&2; exit 1 ;;
        esac
    done
}

launch_runpod() {
    local resp code
    resp=$(gql "mutation {
        podFindAndDeployOnDemand(input: {
            gpuCount: 1, containerDiskInGb: ${DISK_GB}, minVcpuCount: 8, minMemoryInGb: 64,
            gpuTypeId: \"${INSTANCE_TYPE}\", allowedCudaVersions: [\"${CUDA_VERSION}\"],
            name: \"${PROJECT_NAME}-eval\", imageName: \"${IMAGE}\",
            ports: \"22/tcp\", startSsh: true, startJupyter: false, supportPublicIp: true
        }) { id }
    }")
    INSTANCE_ID=$(echo "$resp" | jq -r '.data.podFindAndDeployOnDemand.id // ""')
    [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != null ] && { echo "Pod created: ${INSTANCE_ID}"; return; }
    INSTANCE_ID=""
    code=$(echo "$resp" | jq -r '.errors[0].extensions.code // ""')
    if [ -n "$code" ] && [ "$code" != SUPPLY_CONSTRAINT ]; then
        echo "Launch failed (code=${code})" >&2; echo "$resp" | jq '.errors' >&2; exit 1
    fi
}

# Sets SSH_HOST, SSH_PORT and STATUS for the current instance.
poll_lambda() {
    local inst
    inst=$(api GET "/instances/${INSTANCE_ID}")
    STATUS=$(echo "$inst" | jq -r '.data.status // "unknown"')
    SSH_HOST=$(echo "$inst" | jq -r '.data.ip // ""')
    SSH_PORT=22
}

poll_runpod() {
    local resp
    resp=$(gql "query { pod(input: { podId: \"${INSTANCE_ID}\" }) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }")
    STATUS=$(echo "$resp" | jq -r '.data.pod.desiredStatus // "unknown"' | tr 'A-Z' 'a-z')
    SSH_HOST=$(echo "$resp" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort==22) | select(.isIpPublic)] | first | .ip // ""')
    SSH_PORT=$(echo "$resp" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort==22) | select(.isIpPublic)] | first | .publicPort // ""')
    # RunPod reports RUNNING before ssh is mapped; treat a mapped port as ready.
    [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && STATUS=active || STATUS=booting
}

terminate_lambda() {
    api POST /instance-operations/terminate \
        "$(jq -n --arg i "$1" '{instance_ids:[$i]}')" >/dev/null
}

terminate_runpod() {
    gql "mutation { podTerminate(input: { podId: \"$1\" }) }" >/dev/null
}

# ── preflight ──────────────────────────────────────────────────────────────────
[ -f "$SERVICE_KEY" ] || { echo "Missing ${SERVICE_KEY}" >&2; exit 2; }
if [ "$PROVIDER" = lambda ]; then
    KEY_RESP=$(api GET /ssh-keys)
    if echo "$KEY_RESP" | jq -e '.error' >/dev/null 2>&1; then
        echo "Lambda API rejected the key: $(echo "$KEY_RESP" | jq -r '.error.message')" >&2
        exit 2
    fi
    echo "$KEY_RESP" | jq -r '.data[].name' | grep -qx "$SSH_KEY_NAME" || {
        echo "SSH_KEY_NAME '${SSH_KEY_NAME}' not registered with Lambda" >&2; exit 2; }
else
    [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY not set in .env" >&2; exit 2; }
fi

# Resolve each arm to the newest run that actually has a transformers-format
# checkpoint; a run without one can't be evaluated.
gcloud auth activate-service-account --key-file="$SERVICE_KEY" >/dev/null 2>&1
CKPTS=""
for arm in $ARMS; do
    # RUN_FILTER pins an older run; without it the newest one for the arm wins,
    # which silently mixes corpora once an arm has been retrained.
    run=$(gsutil ls "${GCS_BASE}/runs/" 2>/dev/null | grep "/${arm}-" | grep -- "${RUN_FILTER:-}" | tail -1)
    [ -n "$run" ] || { echo "No runs for ${arm} under ${GCS_BASE}/runs/" >&2; exit 2; }
    hf=$(gsutil ls -d "${run}step_*/hf" 2>/dev/null | tail -1)
    [ -n "$hf" ] || { echo "No hf/ checkpoint under ${run}" >&2; exit 2; }
    CKPTS="${CKPTS}${arm}=${hf%/} "
    echo "${arm} -> ${hf%/}"
done

STAMP="${STAMP:-$(date +%Y%m%d-%H%M)}"
GCS_DEST="${GCS_BASE}/evals/${STAMP}"
echo "Results:     ${GCS_DEST}/"

# ── launch ─────────────────────────────────────────────────────────────────────
RETRY_DELAY="${LAUNCH_RETRY_DELAY:-60}"
echo "Launching ${INSTANCE_TYPE} — polling every ${RETRY_DELAY}s, Ctrl-C to stop..."
INSTANCE_ID=""
attempt=0
while [ -z "$INSTANCE_ID" ]; do
    attempt=$((attempt + 1))
    launch_${PROVIDER}
    [ -n "$INSTANCE_ID" ] && break
    echo "   [${attempt}] no capacity — retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

STARTED=0
cleanup() {
    if [ "$STARTED" = 0 ]; then
        echo "Setup did not finish; terminating ${INSTANCE_ID}..."
        terminate_${PROVIDER} "$INSTANCE_ID"
    else
        echo "Instance ${INSTANCE_ID} terminates itself when the evals finish."
        echo "Reattach: ssh ${SSH_OPTS} -i ${SSH_KEY} -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} 'tail -f ~/eval.log'"
    fi
}
trap cleanup EXIT

echo "Waiting for the instance..."
SSH_HOST=""
for attempt in $(seq 1 120); do
    poll_${PROVIDER}
    echo "   $attempt/120 — ${STATUS}  ${SSH_HOST:-none}:${SSH_PORT:-}"
    [ "$STATUS" = active ] && [ -n "$SSH_HOST" ] && [ "$SSH_HOST" != null ] && break
    case "$STATUS" in terminated|error|exited) echo "Instance ${STATUS}" >&2; exit 1 ;; esac
    sleep 10
done

for attempt in $(seq 1 60); do
    ssh -o BatchMode=yes $SSH_OPTS -o ConnectTimeout=10 -i $SSH_KEY -p $SSH_PORT \
        ${SSH_USER}@${SSH_HOST} 'echo ok' 2>/dev/null | grep -q ok && break
    echo "   ssh $attempt/60"
    sleep 10
done

# RunPod containers run as root with no sudo installed; Lambda gives an
# unprivileged user that needs it.
ssh $SSH_OPTS -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
    'S=$(command -v sudo || true); $S apt-get update -qq && $S apt-get install -y -qq rsync tmux git'

cp "$SERVICE_KEY" ./servicekey.json
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='.env' --exclude='.venv' \
    --exclude='outputs' --exclude='wandb' --exclude='*.log' \
    -e "ssh $SSH_OPTS -i ${SSH_KEY} -p ${SSH_PORT}" . ${SSH_USER}@${SSH_HOST}:~/${PROJECT_NAME}/
rm -f ./servicekey.json
echo "Code synced"

# ── on-instance script ─────────────────────────────────────────────────────────
SETUP=$(mktemp)
cat > "$SETUP" << EOF
#!/bin/bash
set -uo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
LAMBDA_LABS_API_KEY="${LAMBDA_LABS_API_KEY:-}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
INSTANCE_ID="${INSTANCE_ID}"
KEEP_ALIVE="${KEEP_ALIVE}"
GCS_DEST="${GCS_DEST}"

cd ~/${PROJECT_NAME}

finish() {
    status=\$?
    command -v gsutil >/dev/null && gsutil -m -q cp -r ~/results/* "\$GCS_DEST/" 2>/dev/null
    command -v gsutil >/dev/null && gsutil -q cp ~/eval.log "\$GCS_DEST/eval.log" 2>/dev/null
    echo "EXIT_CODE=\$status"
    [ "\$KEEP_ALIVE" = 1 ] && { echo "KEEP_ALIVE=1: leaving the instance up."; return; }
    sleep 90
    if [ -z "\$INSTANCE_ID" ]; then
        echo "No instance id: TERMINATE THIS INSTANCE BY HAND." >&2
        return
    fi
    echo "Terminating \$INSTANCE_ID (${PROVIDER})"
    if [ "${PROVIDER}" = lambda ]; then
        curl -s -u "\${LAMBDA_LABS_API_KEY}:" -X POST \
            https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
            -H "Content-Type: application/json" \
            -d "{\"instance_ids\":[\"\$INSTANCE_ID\"]}" >/dev/null
    else
        curl -s -X POST "https://api.runpod.io/graphql?api_key=\$RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\\\"\$INSTANCE_ID\\\\\" }) }\"}" >/dev/null
    fi
}
trap finish EXIT

mkdir -p ~/results

curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.local/bin/env
uv sync --extra train || exit 1
uv pip install flash-linear-attention==${FLA_VERSION} || exit 1

# ppl.py lives in the fla repo, not the wheel; it needs one import to see our
# architecture because it does not pass trust_remote_code.
git clone --depth 1 https://github.com/fla-org/flash-linear-attention ~/fla || exit 1
sed -i 's/^import fla\$/import fla\nimport nested_gdn2/' ~/fla/evals/ppl.py

if [ ! -d \$HOME/google-cloud-sdk ]; then
    curl -s https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=\$HOME >/dev/null 2>&1
fi
source \$HOME/google-cloud-sdk/path.bash.inc
gcloud auth activate-service-account --key-file=./servicekey.json >/dev/null 2>&1 || exit 1

mkdir -p ~/ckpt
for pair in ${CKPTS}; do
    arm=\${pair%%=*}; src=\${pair#*=}
    echo "fetching \$arm from \$src"
    mkdir -p ~/ckpt/\$arm
    gsutil -m -q cp -r "\$src/*" ~/ckpt/\$arm/ || exit 1
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch,sys; n=torch.cuda.device_count(); print('devices',n); sys.exit(0 if n else 1)" || exit 1

for pair in ${CKPTS}; do
    arm=\${pair%%=*}
    echo "--- perplexity: \$arm ---"
    PYTHONPATH=~/${PROJECT_NAME} uv run python ~/fla/evals/ppl.py \
        -p ~/ckpt/\$arm -d ${PPL_DATA} -s ${PPL_SPLIT} --num_proc ${PPL_PROC} \
        --block_size ${PPL_BLOCK} --bucket_size ${PPL_BUCKET} \
        --output ~/results/ppl-\$arm.json || echo "  \$arm FAILED"
done

echo "=== done \$(date -u) ==="
EOF

scp $SSH_OPTS -i $SSH_KEY -P $SSH_PORT "$SETUP" ${SSH_USER}@${SSH_HOST}:~/setup_eval.sh
rm -f "$SETUP"

ssh $SSH_OPTS -o ConnectTimeout=30 -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
    "chmod +x setup_eval.sh && tmux new-session -d -s eval 'bash setup_eval.sh 2>&1 | tee ~/eval.log'"
STARTED=1

echo "Tailing (Ctrl-C stops watching, not the run)..."
misses=0
while ! ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
         "grep -q EXIT_CODE ~/eval.log 2>/dev/null"; do
    if ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} "true" 2>/dev/null; then
        misses=0
        ssh $SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
            -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} \
            "tail -n 50 -f ~/eval.log --pid=\$(tmux list-panes -t eval -F '#{pane_pid}' 2>/dev/null | head -1)" || true
    else
        misses=$((misses + 1))
        [ "$misses" -ge 30 ] && { echo "Instance unreachable; the run continues."; exit 0; }
    fi
    sleep 5
done

mkdir -p outputs/evals
gsutil -m -q cp -r "${GCS_DEST}/*" outputs/evals/ 2>/dev/null || true
echo "Results: outputs/evals/  and  ${GCS_DEST}/"
