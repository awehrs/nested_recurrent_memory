#!/bin/bash
set -e

# Provision a Lambda Cloud instance, sync the repo, and train one arm.
#
# The instance outlives this script. Training runs in tmux; when it ends, for
# any reason, the instance uploads its checkpoints to GCS, checks they landed,
# and terminates itself. This script only tails the log, so closing it, Ctrl-C,
# or a dropped connection leaves the run going. It terminates the instance
# itself only if something fails before training starts.
#
# Usage: ARM=<arm> bash scripts/pretrain.sh [key=value ...]
#   ARM=nested-learned bash scripts/pretrain.sh
#   ARM=nested-learned bash scripts/pretrain.sh max_steps=100 save_every=50      # smoke run
#   RESUME_FROM_GCS=gs://nested-gdn/runs/<run>/step_004000 \
#       ARM=nested-learned bash scripts/pretrain.sh
#   KEEP_ALIVE=1 ARM=... bash scripts/pretrain.sh                                # no self-termination
#   REGION=us-east-1 ARM=... bash scripts/pretrain.sh                            # pin a region

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

source .env

# ── config ─────────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_8x_a100_80gb_sxm4}"
GPU_COUNT="${GPU_COUNT:-8}"
# The cu128 torch pin lives in pyproject.toml, so uv sync resolves a coherent
# CUDA 12 stack rather than being corrected afterwards.
REGION="${REGION:-}"                 # empty: take whichever region has capacity
SSH_KEY="${SSH_KEY:-$HOME/.ssh/scoobdoob.pem}"
SSH_KEY_NAME="${SSH_KEY_NAME:-scoobdoob}"   # name registered with Lambda, not a path
SSH_OPTS="-o StrictHostKeyChecking=accept-new"
SSH_USER="ubuntu"
PROJECT_NAME=nested_recurrent_memory
FLA_VERSION="${FLA_VERSION:-0.5.2}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"
SERVICE_KEY=~/.config/gcloud/servicekey.json

GCS_BASE="${GCS_BASE:-gs://nested-gdn}"
DATA_GCS="${DATA_GCS:-${GCS_BASE}/data/fineweb-edu-gpt2}"
RESUME_FROM_GCS="${RESUME_FROM_GCS:-}"

LAMBDA_API="https://cloud.lambdalabs.com/api/v1"
api() {  # api <method> <path> [json body]
    local method=$1 path=$2 body=${3:-}
    if [ -n "$body" ]; then
        curl -s -u "${LAMBDA_LABS_API_KEY}:" -X "$method" "${LAMBDA_API}${path}" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -s -u "${LAMBDA_LABS_API_KEY}:" -X "$method" "${LAMBDA_API}${path}"
    fi
}

# ── preflight: fail before paying for anything ─────────────────────────────────
if [ -z "${ARM:-}" ] || [ ! -f "configs/${ARM}.yaml" ] || [ "$ARM" = base ]; then
    echo "Set ARM to one of: $(ls configs/*.yaml 2>/dev/null | xargs -n1 basename \
        | sed 's/\.yaml$//' | grep -v '^base$' | tr '\n' ' ')" >&2
    exit 2
fi
if [ ! -f "$SERVICE_KEY" ]; then
    echo "Missing ${SERVICE_KEY}; needed to write checkpoints to GCS." >&2
    exit 2
fi
# A missing bucket can't be fixed from the instance, and a run that can't save
# is wasted money.
if command -v gsutil >/dev/null; then
    gcloud auth activate-service-account --key-file="$SERVICE_KEY" >/dev/null 2>&1
    if ! gsutil ls "${GCS_BASE}" >/dev/null 2>&1; then
        echo "Cannot read ${GCS_BASE} as the service account. Create the bucket and" >&2
        echo "grant roles/storage.objectAdmin, or set GCS_BASE." >&2
        exit 2
    fi
    if ! gsutil ls "${DATA_GCS}" >/dev/null 2>&1; then
        echo "No data at ${DATA_GCS}; run train/prepare_data.py --gcs first." >&2
        exit 2
    fi
fi
# Lambda launches against a key name in their account, not a local file.
KEY_RESP=$(api GET /ssh-keys)
if echo "$KEY_RESP" | jq -e '.error' >/dev/null 2>&1; then
    echo "Lambda API rejected the key: $(echo "$KEY_RESP" | jq -r '.error.message')" >&2
    exit 2
fi
KEY_NAMES=$(echo "$KEY_RESP" | jq -r '.data[].name')
if [ -z "$SSH_KEY_NAME" ]; then
    SSH_KEY_NAME=$(echo "$KEY_NAMES" | head -1)
fi
if ! echo "$KEY_NAMES" | grep -qx "$SSH_KEY_NAME"; then
    echo "SSH_KEY_NAME '${SSH_KEY_NAME}' is not registered with Lambda. Known: $(echo "$KEY_NAMES" | tr '\n' ' ')" >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-${ARM}-$(date +%Y%m%d-%H%M)}"
GCS_RUN_DEST="${GCS_BASE}/runs/${RUN_NAME}"
# Defaults first, so anything passed on the command line overrides them.
OVERRIDES="config=configs/${ARM}.yaml run_name=${RUN_NAME} gcs_dest=${GCS_RUN_DEST}"
[ -n "$RESUME_FROM_GCS" ] && OVERRIDES="${OVERRIDES} resume_from=\$HOME/resume"
OVERRIDES="${OVERRIDES} ${*}"

echo "Run:         ${RUN_NAME}"
echo "Checkpoints: ${GCS_RUN_DEST}/"
echo "SSH key:     ${SSH_KEY_NAME}"
[ -n "$RESUME_FROM_GCS" ] && echo "Resuming:    ${RESUME_FROM_GCS}"

# ── launch (unbounded wait for capacity) ───────────────────────────────────────
# Capacity shortages retry forever across regions; any other error fails fast.
RETRY_DELAY="${LAUNCH_RETRY_DELAY:-60}"
echo "Launching ${INSTANCE_TYPE} — polling every ${RETRY_DELAY}s, Ctrl-C to stop..."

INSTANCE_ID=""
attempt=0
while [ -z "$INSTANCE_ID" ]; do
    attempt=$((attempt + 1))
    if [ -n "$REGION" ]; then
        REGIONS="$REGION"
    else
        REGIONS=$(api GET /instance-types \
            | jq -r --arg t "$INSTANCE_TYPE" \
              '.data[$t].regions_with_capacity_available[]?.name' 2>/dev/null || true)
    fi

    for r in $REGIONS; do
        RESP=$(api POST /instance-operations/launch "$(jq -n \
            --arg r "$r" --arg t "$INSTANCE_TYPE" --arg k "$SSH_KEY_NAME" \
            --arg n "${PROJECT_NAME}-${ARM}" \
            '{region_name:$r, instance_type_name:$t, ssh_key_names:[$k],
              file_system_names:[], quantity:1, name:$n}')")
        INSTANCE_ID=$(echo "$RESP" | jq -r '.data.instance_ids[0] // ""')
        if [ -n "$INSTANCE_ID" ]; then
            echo "Instance launched in ${r}: ${INSTANCE_ID}"
            break
        fi
        CODE=$(echo "$RESP" | jq -r '.error.code // ""')
        case "$CODE" in
            *insufficient-capacity*|*capacity*) ;;   # try the next region
            "") ;;
            *)
                echo "Launch failed (non-retryable, code=${CODE}):" >&2
                echo "$RESP" | jq '.error' >&2
                exit 1
                ;;
        esac
    done

    [ -n "$INSTANCE_ID" ] && break
    echo "   [${attempt}] no capacity for ${INSTANCE_TYPE} — retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

# ── cleanup: only before training starts ───────────────────────────────────────
STARTED=0
terminate() {
    api POST /instance-operations/terminate \
        "$(jq -n --arg i "$1" '{instance_ids:[$i]}')" > /dev/null
}
cleanup() {
    if [ "$STARTED" = 0 ]; then
        echo "Setup did not finish; terminating ${INSTANCE_ID}..."
        terminate "$INSTANCE_ID"
        echo "Instance terminated"
    else
        echo "Instance ${INSTANCE_ID} is still running and terminates itself once checkpoints are in GCS."
        echo "Reattach: ssh ${SSH_OPTS} -i ${SSH_KEY} ${SSH_USER}@${SSH_HOST} 'tail -f ~/train.log'"
    fi
}
trap cleanup EXIT

# ── wait for the instance to boot ──────────────────────────────────────────────
echo "Waiting for the instance to become active..."
SSH_HOST=""
for attempt in $(seq 1 120); do
    INST=$(api GET "/instances/${INSTANCE_ID}")
    STATUS=$(echo "$INST" | jq -r '.data.status // "unknown"')
    SSH_HOST=$(echo "$INST" | jq -r '.data.ip // ""')
    echo "   Attempt $attempt/120 — status: ${STATUS}  ip: ${SSH_HOST:-none}"
    if [ "$STATUS" = "active" ] && [ -n "$SSH_HOST" ] && [ "$SSH_HOST" != "null" ]; then
        echo "Instance active at ${SSH_HOST}"
        break
    fi
    if [ "$STATUS" = "terminated" ] || [ "$STATUS" = "error" ]; then
        echo "Instance entered ${STATUS} before booting" >&2
        exit 1
    fi
    if [ $attempt -eq 120 ]; then
        echo "Instance never became active" >&2
        exit 1
    fi
    sleep 10
done

# ── wait for SSH ───────────────────────────────────────────────────────────────
echo "Waiting for SSH..."
SSH_READY=false
for attempt in $(seq 1 60); do
    SSH_ERR=$(ssh -o BatchMode=yes $SSH_OPTS -o ConnectTimeout=10 \
        -i $SSH_KEY ${SSH_USER}@${SSH_HOST} 'echo ok' 2>&1) || true
    if echo "$SSH_ERR" | grep -q "^ok$"; then
        echo "SSH ready"
        SSH_READY=true
        break
    fi
    echo "   Attempt $attempt/60 — $SSH_ERR"
    sleep 10
done
if [ "$SSH_READY" = false ]; then
    echo "SSH never became ready" >&2
    exit 1
fi

# ── sync project ───────────────────────────────────────────────────────────────
ssh $SSH_OPTS -i $SSH_KEY ${SSH_USER}@${SSH_HOST} \
    "sudo apt-get update -qq && sudo apt-get install -y -qq rsync tmux"

cp "$SERVICE_KEY" ./servicekey.json
echo "Syncing project..."
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='outputs' \
    --exclude='wandb' \
    --exclude='*.log' \
    -e "ssh $SSH_OPTS -i ${SSH_KEY}" \
    . ${SSH_USER}@${SSH_HOST}:~/${PROJECT_NAME}/
rm -f ./servicekey.json
echo "Code synced"

# ── on-instance script ─────────────────────────────────────────────────────────
# Written with secrets expanded, so it goes to a private temp file and is
# removed locally once copied.
SETUP=$(mktemp)
cat > "$SETUP" << EOF
#!/bin/bash
# Not set -e: the shutdown in finish() must run however the script ends.
set -uo pipefail

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
LAMBDA_LABS_API_KEY="${LAMBDA_LABS_API_KEY}"
GCS_RUN_DEST="${GCS_RUN_DEST}"
KEEP_ALIVE="${KEEP_ALIVE}"
# Baked in rather than discovered on the instance: an unset id under set -u
# aborts the trap and leaves the instance billing.
INSTANCE_ID="${INSTANCE_ID}"

cd ~/${PROJECT_NAME}

# Upload checkpoints, confirm the newest one is in GCS, then terminate.
# If the upload can't be confirmed the instance stays up rather than lose the run.
finish() {
    status=\$?
    echo "Exit status \$status; syncing checkpoints to \$GCS_RUN_DEST/"
    verified=1
    if [ -d outputs/checkpoints ] && command -v gsutil >/dev/null; then
        gsutil -m rsync -r outputs/checkpoints/ "\$GCS_RUN_DEST/" || verified=0
        latest=\$(ls -1 outputs/checkpoints | sort | tail -1)
        if [ -n "\$latest" ] && ! gsutil ls "\$GCS_RUN_DEST/\$latest/" >/dev/null 2>&1; then
            verified=0
        fi
    fi
    command -v gsutil >/dev/null && gsutil -q cp ~/train.log "\$GCS_RUN_DEST/train.log" || true

    echo "EXIT_CODE=\$status"
    if [ "\$KEEP_ALIVE" = 1 ]; then
        echo "KEEP_ALIVE=1: leaving the instance up."
        return
    fi
    if [ "\$verified" = 0 ]; then
        echo "Checkpoint upload not verified: leaving the instance up so nothing is lost."
        return
    fi
    # Let the local tail see EXIT_CODE and pull the log before the instance goes.
    sleep 120
    if [ -z "\$INSTANCE_ID" ]; then
        echo "No instance id: TERMINATE THIS INSTANCE BY HAND." >&2
        return
    fi
    echo "Terminating \$INSTANCE_ID"
    curl -s -u "\${LAMBDA_LABS_API_KEY}:" -X POST \
        https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"\$INSTANCE_ID\"]}" >/dev/null
}
trap finish EXIT

retry() { for a in 1 2 3 4 5 6 7 8; do "\$@" && return 0; echo "  [retry \$a] '\$*' failed, sleep 20s..."; sleep 20; done; return 1; }

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.local/bin/env
export MAX_JOBS=\$(nproc)

echo "Syncing dependencies..."
retry uv sync --extra train || exit 1
uv pip install flash-linear-attention==${FLA_VERSION} || exit 1

echo "Installing gcloud..."
if [ ! -d \$HOME/google-cloud-sdk ]; then
    curl -s https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=\$HOME >/dev/null 2>&1
fi
source \$HOME/google-cloud-sdk/path.bash.inc
gcloud auth activate-service-account --key-file=./servicekey.json >/dev/null 2>&1 || exit 1

echo "Fetching data from ${DATA_GCS}..."
mkdir -p \$HOME/data
gsutil -m -q rsync -r "${DATA_GCS}" \$HOME/data || exit 1

if [ -n "${RESUME_FROM_GCS}" ]; then
    echo "Fetching resume checkpoint from ${RESUME_FROM_GCS}..."
    mkdir -p \$HOME/resume
    gsutil -m cp -r "${RESUME_FROM_GCS}/*" \$HOME/resume/ || exit 1
fi

nvidia-smi
uv run python - <<'PYEOF' || exit 1
import sys, torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"torch {torch.__version__} cuda {torch.version.cuda} devices={n}")
sys.exit(0 if n == ${GPU_COUNT} else 1)
PYEOF

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=\$HOME/triton_cache
mkdir -p \$HOME/triton_cache
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

echo "Training ${RUN_NAME}..."
PYTHONUNBUFFERED=1 uv run accelerate launch \
    --mixed_precision bf16 \
    --num_processes ${GPU_COUNT} \
    --num_machines 1 \
    --dynamo_backend no \
    -m train.pretrain data_dir=\$HOME/data out_dir=outputs ${OVERRIDES}
EOF

scp $SSH_OPTS -i $SSH_KEY "$SETUP" ${SSH_USER}@${SSH_HOST}:~/setup_and_train.sh
rm -f "$SETUP"

echo "Starting training in tmux..."
ssh $SSH_OPTS -o ConnectTimeout=30 -i $SSH_KEY ${SSH_USER}@${SSH_HOST} \
    "chmod +x setup_and_train.sh && tmux new-session -d -s train 'bash setup_and_train.sh 2>&1 | tee ~/train.log'"
STARTED=1

# ── tail ───────────────────────────────────────────────────────────────────────
# Reconnects on drops. Giving up only stops the viewing; the run is unaffected.
echo "Tailing logs (Ctrl-C stops watching, not training)..."
misses=0
while ! ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY ${SSH_USER}@${SSH_HOST} \
         "grep -q EXIT_CODE ~/train.log 2>/dev/null"; do
    if ssh $SSH_OPTS -o ConnectTimeout=20 -i $SSH_KEY ${SSH_USER}@${SSH_HOST} "true" 2>/dev/null; then
        misses=0
        ssh $SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
            -i $SSH_KEY ${SSH_USER}@${SSH_HOST} \
            "tail -n 50 -f ~/train.log --pid=\$(tmux list-panes -t train -F '#{pane_pid}' 2>/dev/null | head -1)" || true
    else
        misses=$((misses + 1))
        if [ "$misses" -ge 30 ]; then
            echo "Instance unreachable ${misses}x; stopping the tail. The run continues."
            exit 0
        fi
    fi
    sleep 5
done

mkdir -p outputs
scp $SSH_OPTS -i $SSH_KEY \
    ${SSH_USER}@${SSH_HOST}:~/train.log "./outputs/${RUN_NAME}.log" 2>/dev/null || true
grep -E "EXIT_CODE|not verified|KEEP_ALIVE" "./outputs/${RUN_NAME}.log" | tail -3 || true
echo "Log: ./outputs/${RUN_NAME}.log   Checkpoints: ${GCS_RUN_DEST}/"
