#!/bin/sh

set -e 
set -x

# This script now evaluates diversity on the AlpacaEval dataset.

# --- Step 0: Configuration ---

# Allow downloading the AlpacaEval dataset if it's not cached
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0" # Use a single GPU for evaluation

# Define all paths clearly
MODEL_CHECKPOINT_PATH="/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
DATASET_NAME="tatsu-lab/alpaca_eval" # Use the standard AlpacaEval dataset
SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_diversity_alpaca" # Save results alongside the model
mkdir -p $SAVE_DIR

# Define generation parameters
SEED=42
NUM_SAMPLES=16  # Generate 16 responses per prompt
TEMPERATURE=1.0
TOP_K=50
TOP_P=0.9

# --- Step 1: Generate Responses ---

echo "Generating ${NUM_SAMPLES} responses for each prompt in AlpacaEval..."

# Note the corrected path to the Python scripts (../.../evaluation/)
python ../../evaluation/generate_response.py \
    --model_name_or_path $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATASET_NAME \
    --dataset_split "eval" \
    --prompt_key "instruction" \
    --max_size 100 \
    --seed $SEED \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --max_new_tokens 512 \
    --n $NUM_SAMPLES \
    --use_vllm True \
    --do_sample True \
    --remove_old True \
    --save_path "${SAVE_DIR}/generated_responses.json"


# --- Step 2: Evaluate Diversity ---

echo "Calculating diversity metrics for the generated responses..."

# Note the corrected path to the Python scripts (../.../evaluation/)
python ../../evaluation/evaluation_diversity.py \
    --tokenizer_path $TOKENIZER_PATH \
    --detokenizer_path $TOKENIZER_PATH \
    --response_path "${SAVE_DIR}/generated_responses.json" \
    2>&1 | tee ${SAVE_DIR}/diversity_metrics.log

echo "Diversity evaluation complete. Results saved in ${SAVE_DIR}"