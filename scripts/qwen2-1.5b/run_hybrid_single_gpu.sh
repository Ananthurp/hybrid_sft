#!/bin/bash
# This master script trains a model with the hybrid loss on a SINGLE GPU,
# and then runs the full evaluation suite automatically.

set -e # Exit immediately if any command fails.
set -x # Print each command before it is executed.

# --- Section 0: Global Configuration ---
echo "--- CONFIGURATION ---"
# Define the single GPU we want to use (e.g., 0, 1, 2, or 3)
export CUDA_VISIBLE_DEVICES="0"

# Define all paths and variables
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"

EXPERIMENT_NAME="qwen_1.5b_hybrid_single_gpu"
RESULTS_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"

# --- Section 1: Supervised Fine-Tuning on a Single GPU ---
echo "--- STARTING SINGLE-GPU TRAINING ---"
mkdir -p $RESULTS_DIR

# We now call the train.py script directly with python, NOT deepspeed.
python ${TRAIN_SCRIPT_PATH} \
    --seed 1234 \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir ${RESULTS_DIR} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 132 \
    --num_train_epochs 3 \
    --save_strategy "epoch" \
    --loss "hybrid" \
    --ns_alpha 0.5 \
    --ns_tau 0.065 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    --use_flash_attn True \
    2>&1 | tee $RESULTS_DIR/training.log

echo "--- TRAINING COMPLETE ---"

# --- Section 2: Run Full Evaluation Suite ---
echo "--- STARTING EVALUATION ---"
# The final trained model is saved directly in the RESULTS_DIR.
FINAL_MODEL_PATH=${RESULTS_DIR}

# 2.1: AlpacaEval Generation
bash scripts/qwen2-1.5b/eval/run_generation.sh ${FINAL_MODEL_PATH}

# 2.2: AlpacaEval Reward Scoring
bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh ${FINAL_MODEL_PATH}

# 2.3: Diversity Evaluation
bash scripts/qwen2-1.5b/eval/run_diversity_eval.sh ${FINAL_MODEL_PATH}

# 2.4: GSM8K Standard Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_eval.sh ${FINAL_MODEL_PATH}

# 2.5: GSM8K Voting Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_voting_eval.sh ${FINAL_MODEL_PATH}

echo "--- ALL EXPERIMENTS COMPLETE ---"