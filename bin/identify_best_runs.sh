#!/usr/bin/env bash
# For each dataset in DATASETS, run find_best_run.py for every model
# subdirectory found under outputs/<dataset>/.
#
# Usage: bash bin/identify_best_runs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/outputs"

DATASETS=(
    #"medmcqa"
    #"hhrlhf_judge"
    # "mmlu"
    # "mmlu_pro"
    "hellaswag"
)

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$OUTPUT_DIR/$dataset"
    if [ ! -d "$dataset_dir" ]; then
        echo "WARN: $dataset_dir missing – skipping"
        continue
    fi

    for model_path in "$dataset_dir"/*/; do
        [ -d "$model_path" ] || continue
        model="$(basename "$model_path")"
        echo ">>> find_best_run: $dataset / $model"
        python "$SCRIPT_DIR/find_best_run.py" \
            --output_dir "$OUTPUT_DIR" \
            --dataset "$dataset" \
            --model "$model" \
          || echo "find_best_run.py failed for $dataset/$model"
    done
done

echo "Done."
