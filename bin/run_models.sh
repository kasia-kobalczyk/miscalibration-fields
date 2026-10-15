#!/usr/bin/env bash
# Launch grid sweeps for multiple models sequentially.
# Each model's grid runs in parallel tmux sessions; we wait for all sessions
# in a batch to finish before moving to the next model.
#
# Usage:
#   bash bin/run_models.sh [dataset] [--device <device>]
#
# If --device is not given, defaults to "cuda:0".
# Configure MODEL_NAMES, sigma/lambda grids, and DATASET below.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-}"  # optional: conda env to activate in each tmux session
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Models to sweep (edit this list) ─────────────────────────────────────
MODEL_NAMES=(
    "Qwen3-1.7B-Base"
    "Qwen3-4B-Base"
    "Qwen3-8B-Base"
    "gemma-3-1b-pt"
    "gemma-3-4b-pt"
    "gemma-3-12b-pt"
    "Llama-3.2-1B"
    "Llama-3.2-3B"
    "Llama-3.1-8B"
    "Ministral-3-3B-Base-2512"
    "Ministral-3-8B-Base-2512"
    "Ministral-3-14B-Base-2512"
)

# ── Hyperparameter grid ──────────────────────────────────────────────────
sigma_ls=(1.0 0.5 0.1 0.05)
lambda_ls=(1e-4 1e-3 1e-2 1e-1)

# ── Parse arguments ──────────────────────────────────────────────────────
DATASET="hellaswag"
DEVICE="cuda:0"
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
if [ ${#POSITIONAL[@]} -gt 0 ]; then
    DATASET="${POSITIONAL[0]}"
fi

# ── Poll interval (seconds) while waiting for tmux sessions ─────────────
POLL_INTERVAL=30

# ── Helper: wait until none of the tracked sessions exist ────────────────
wait_for_sessions() {
    local sessions=("$@")
    echo "Waiting for ${#sessions[@]} tmux sessions to finish..."
    while true; do
        local running=0
        for s in "${sessions[@]}"; do
            if tmux has-session -t "$s" 2>/dev/null; then
                running=$((running + 1))
            fi
        done
        if [ "$running" -eq 0 ]; then
            break
        fi
        echo "  … $running session(s) still running"
        sleep "$POLL_INTERVAL"
    done
    echo "All sessions finished."
}

# ── Main loop ────────────────────────────────────────────────────────────
for model in "${MODEL_NAMES[@]}"; do
    echo "============================================================"
    echo "MODEL: $model   DATASET: $DATASET   DEVICE: $DEVICE"
    echo "============================================================"

    # Read embed_dim from dataset config
    EMBED_CONFIG="$PROJECT_DIR/datasets/$DATASET/$model/embeddings/config.json"
    if [ ! -f "$EMBED_CONFIG" ]; then
        echo "ERROR: $EMBED_CONFIG not found – skipping $model"
        continue
    fi
    IN_DIM=$(python -c "import json; print(json.load(open('$EMBED_CONFIG'))['embed_dim'])")
    echo "  in_dim=$IN_DIM (from $EMBED_CONFIG)"

    session_names=()

    for sigma in "${sigma_ls[@]}"; do
    for lambda in "${lambda_ls[@]}"; do
        SAFE_MODEL="${model//./_}"
        SESSION_NAME="${DATASET}_${SAFE_MODEL}_sigma${sigma//./_}_lambda${lambda//./_}"
        # Truncate to 128 chars (tmux limit)
        SESSION_NAME="${SESSION_NAME:0:128}"

        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session $SESSION_NAME already exists – skipping"
            session_names+=("$SESSION_NAME")
            continue
        fi

        tmux new-session -d -s "$SESSION_NAME"
        if [[ -n "$CONDA_ENV" ]]; then
            tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV" C-m
        fi
        tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && export PYTHONPATH=src" C-m

        CMD="python -m calib_atlas.train --config-name=config"
        CMD+=" ++data.model_name=$model"
        CMD+=" ++data.name=$DATASET"
        CMD+=" ++loss.sigma=$sigma"
        CMD+=" ++loss.lambda_mass=$lambda"
        CMD+=" ++model.in_dim=$IN_DIM"
        CMD+=" ++device=$DEVICE"
        CMD+="; exit"

        tmux send-keys -t "$SESSION_NAME" "$CMD" C-m

        session_names+=("$SESSION_NAME")
        echo "Launched: $SESSION_NAME"
        sleep 1
    done
    done

    # Wait for every session in this batch to exit
    wait_for_sessions "${session_names[@]}"

    # Find the best run for this model/dataset
    echo "Finding best run for $model on $DATASET ..."
    python "$SCRIPT_DIR/find_best_run.py" \
        --output_dir "$PROJECT_DIR/outputs" \
        --dataset "$DATASET" \
        --model "$model"

    echo ""
done

echo "All models done."
