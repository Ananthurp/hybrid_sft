#!/bin/sh
set -e 
set -x

# This script ONLY generates responses for AlpacaEval.

# --- Configuration ---
# Comment out OFFLINE for the first run to download alpaca_eval
# export TRANSFORMERS_OFFLINE=1 
export CUDA_VISIBLE_DEVICES="2"

# MODEL_CHECKPOINT_PATH="/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
MODEL_CHECKPOINT_PATH="/data/ananthu/gem_project/results/qwen_1.5b_ce_run/output_dir"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
DATASET_PATH="tatsu-lab/alpaca_eval"
SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_chat_alpaca"
mkdir -p $SAVE_DIR

echo "Generating 32 responses for each prompt in AlpacaEval..."

python evaluation/generate_response.py \
    --model_name_or_path $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATASET_PATH \
    --split "eval" \
    --column_name "instruction" \
    --seed 42 \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 0.9 \
    --max_new_tokens 2048 \
    --max_size 200 \
    --n 32 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.7 \
    --do_sample True \
    --remove_old True \
    --save_path "${SAVE_DIR}/generated_responses.json"

echo "Response generation complete."