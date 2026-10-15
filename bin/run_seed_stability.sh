#!/usr/bin/env bash
# Launch K seed-stability training runs in parallel tmux sessions for each
# (model, dataset) pair in PAIRS. After each pair's batch finishes, call
# generate_seed_stability.py to compute per-seed metrics + pairwise
# correlation / sign-agreement summary.
#
# Each run reuses the (sigma, lambda_mass, in_dim) chosen by the real
# best run (from outputs/<dataset>/<model>/best_run.json). cfg.seed stays
# fixed (default 0) so the train/val/test split is identical across seeds;
# only +train_seed=k varies, controlling phi initialization and minibatch
# shuffle order.
#
# Usage:
#   bash bin/run_seed_stability.sh [--K 5] [--device cuda:0]

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-}"  # optional: conda env to activate in each tmux session
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── (model, dataset) pairs to run (edit this list) ───────────────────────
PAIRS=(
    #"Qwen3-8B-Base hhrlhf_judge"
    "Llama-3.2-3B mmlu"
    "Ministral-3-14B-Base-2512 hhrlhf_judge"
    "Qwen3-1.7B-Base hhrlhf_judge"
    "Llama-3.2-1B medmcqa"
    "Ministral-3-3B-Base-2512 hhrlhf_judge"
)

# ── Defaults ─────────────────────────────────────────────────────────────
K=5
DEVICE="cuda:0"
POLL_INTERVAL=30

# ── Parse arguments ──────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --K)        K="$2"; shift 2 ;;
        --K=*)      K="${1#*=}"; shift ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --device=*) DEVICE="${1#*=}"; shift ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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
        echo "  ... $running session(s) still running"
        sleep "$POLL_INTERVAL"
    done
    echo "All sessions finished."
}

# ── Per-pair loop ────────────────────────────────────────────────────────
for pair in "${PAIRS[@]}"; do
    read -r MODEL DATASET <<< "$pair"

    OUTPUT_DIR="$PROJECT_DIR/outputs/$DATASET/$MODEL"
    BEST_RUN="$OUTPUT_DIR/best_run.json"
    if [ ! -f "$BEST_RUN" ]; then
        echo "ERROR: $BEST_RUN not found - skipping $DATASET / $MODEL"
        continue
    fi

    SIGMA=$(python -c "import json; print(json.load(open('$BEST_RUN'))['overrides']['loss.sigma'])")
    LAMBDA=$(python -c "import json; print(json.load(open('$BEST_RUN'))['overrides']['loss.lambda_mass'])")
    IN_DIM=$(python -c "import json; print(json.load(open('$BEST_RUN'))['overrides']['model.in_dim'])")

    echo "============================================================"
    echo "DATASET=$DATASET  MODEL=$MODEL  DEVICE=$DEVICE"
    echo "K=$K  sigma=$SIGMA  lambda=$LAMBDA  in_dim=$IN_DIM"
    echo "============================================================"

    session_names=()
    SAFE_MODEL="${MODEL//./_}"

    for ((k=0; k<K; k++)); do
        KK=$(printf "%02d" "$k")
        SESSION_NAME="seedstab_${DATASET}_${SAFE_MODEL}_s${KK}"
        SESSION_NAME="${SESSION_NAME:0:128}"

        SEED_DIR="$OUTPUT_DIR/seed_runs/s${KK}"

        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session $SESSION_NAME already running - skipping"
            session_names+=("$SESSION_NAME")
            continue
        fi

        if [ -f "$SEED_DIR/best_phi.pt" ]; then
            echo "Already trained at $SEED_DIR/best_phi.pt - skipping"
            continue
        fi

        mkdir -p "$SEED_DIR"

        tmux new-session -d -s "$SESSION_NAME"
        if [[ -n "$CONDA_ENV" ]]; then
            tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV" C-m
        fi
        tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && export PYTHONPATH=src" C-m

        CMD="python -m calib_atlas.train --config-name=config"
        CMD+=" ++data.model_name=$MODEL"
        CMD+=" ++data.name=$DATASET"
        CMD+=" ++loss.sigma=$SIGMA"
        CMD+=" ++loss.lambda_mass=$LAMBDA"
        CMD+=" ++model.in_dim=$IN_DIM"
        CMD+=" ++device=$DEVICE"
        CMD+=" +train_seed=$k"
        CMD+=" hydra.run.dir=$SEED_DIR"
        CMD+="; exit"

        tmux send-keys -t "$SESSION_NAME" "$CMD" C-m

        session_names+=("$SESSION_NAME")
        echo "Launched: $SESSION_NAME -> $SEED_DIR"
        sleep 2
    done

    if [ ${#session_names[@]} -gt 0 ]; then
        wait_for_sessions "${session_names[@]}"
    else
        echo "All seed checkpoints already exist for this pair; nothing to train."
    fi
done

echo "All pairs done."
