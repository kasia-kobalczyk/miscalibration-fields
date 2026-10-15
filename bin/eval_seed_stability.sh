export PYTHONPATH=src

PAIRS=(
    #"Qwen3-8B-Base hhrlhf_judge"
    "Llama-3.2-3B mmlu"
    "Ministral-3-14B-Base-2512 hhrlhf_judge"
    "Qwen3-1.7B-Base hhrlhf_judge"
    "Ministral-3-3B-Base-2512 hhrlhf_judge"
    "Llama-3.2-1B medmcqa"
)

for pair in "${PAIRS[@]}"; do
    read -r MODEL DATASET <<< "$pair"
    echo "=== $DATASET / $MODEL ==="
    python bin/generate_seed_stability.py \
        --dataset "$DATASET" \
        --model   "$MODEL" \
        --K 5
done
