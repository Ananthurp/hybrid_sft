#!/bin/bash
# This master script trains a model with the hybrid loss, consolidates the
# checkpoint, and then runs the full evaluation suite automatically.

set -e # Exit immediately if any command fails.
set -x # Print each command before it is executed.

# --- Section 0: Global Configuration ---
echo "--- CONFIGURATION ---"
# Define the two GPUs we are allowed to use
export CUDA_VISIBLE_DEVICES="0,3"

# Define all paths and variables in one place for easy management
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"

# Define a unique name for this experiment
EXPERIMENT_NAME="qwen_1.5b_hybrid_run_full"
RESULTS_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"

TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"
CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# --- Section 1: Supervised Fine-Tuning ---
echo "--- STARTING TRAINING ---"
mkdir -p $RESULTS_DIR

# Launch on 2 GPUs (0 and 3). Note: DeepSpeed re-indexes these to local ranks 0 and 1.
deepspeed --include localhost:0,3 ${TRAIN_SCRIPT_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    --seed 1234 \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir ${RESULTS_DIR} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 33 \
    --num_train_epochs 3 \
    --save_strategy "epoch" \
    --loss "hybrid" \
    --ns_alpha 0.5 \
    --ns_tau 0.1 \
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


# --- Section 2: Consolidate Checkpoint ---
echo "--- STARTING CHECKPOINT CONSOLIDATION ---"
# Define the path to the final partitioned checkpoint and the new consolidated directory
FINAL_CHECKPOINT_DIR="${RESULTS_DIR}/checkpoint-1392" # Assuming 1392 steps for 3 epochs
CONSOLIDATED_MODEL_DIR="${RESULTS_DIR}/output_dir_consolidated"
mkdir -p $CONSOLIDATED_MODEL_DIR

python ${CONVERSION_SCRIPT_PATH} ${FINAL_CHECKPOINT_DIR} ${CONSOLIDATED_MODEL_DIR}/pytorch_model.bin

# Copy the necessary config/tokenizer files to make the directory self-contained
cp ${BASE_MODEL_PATH}/*.json ${CONSOLIDATED_MODEL_DIR}/
cp ${BASE_MODEL_PATH}/*.txt ${CONSOLIDATED_MODEL_DIR}/

echo "--- CONSOLIDATION COMPLETE ---"


# --- Section 3: Run Full Evaluation Suite ---
echo "--- STARTING EVALUATION ---"
# We will now call our evaluation scripts, passing the path to the
# new consolidated model as an argument to make them reusable.

# Note: We run evaluations on a single GPU (the first one available: GPU 0)
export CUDA_VISIBLE_DEVICES="0"

# 3.1: AlpacaEval Generation
bash scripts/qwen2-1.5b/eval/run_generation.sh ${CONSOLIDATED_MODEL_DIR}

# 3.2: AlpacaEval Reward Scoring
bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh ${CONSOLIDATED_MODEL_DIR}

# 3.3: Diversity Evaluation
bash scripts/qwen2-1.5b/eval/run_diversity_eval.sh ${CONSOLIDATED_MODEL_DIR}

# 3.4: GSM8K Standard Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_eval.sh ${CONSOLIDATED_MODEL_DIR}

# 3.5: GSM8K Voting Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_voting_eval.sh ${CONSOLIDATED_MODEL_DIR}

echo "--- ALL EXPERIMENTS COMPLETE ---"