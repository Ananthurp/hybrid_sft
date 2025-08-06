#!/bin/sh
set -e
set -x

# This script ONLY scores pre-generated responses with a reward model.

# --- Configuration ---
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="2"

# --- Define Paths ---
# MODEL_RESULTS_DIR="/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
MODEL_RESULTS_DIR="/data/ananthu/gem_project/results/qwen_1.5b_ce_run/output_dir"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
REWARD_MODEL_PATH="sfairXC/FsfairX-LLaMA3-RM-v0.1"
RESPONSE_FILE_PATH="${MODEL_RESULTS_DIR}/evaluation_chat_alpaca/generated_responses.json"
SAVE_DIR="${MODEL_RESULTS_DIR}/evaluation_chat_alpaca"

# --- Evaluate the generated responses with the reward model ---
echo "Scoring pre-generated responses with a reduced batch size..."

python evaluation/evaluation_reward.py \
    --model_name_or_path $REWARD_MODEL_PATH \
    --batch_size 2 \
    --detokenizer_path $TOKENIZER_PATH \
    --data_path "${RESPONSE_FILE_PATH}" \
    --save_path "${SAVE_DIR}/reward_scores.json"  \
    2>&1 | tee ${SAVE_DIR}/reward_eval.log

echo "Reward scoring complete."