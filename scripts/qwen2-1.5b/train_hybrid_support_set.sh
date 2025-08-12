#!/bin/bash
set -euo pipefail
set -x

# --- Comm / Memory guards (keep if needed on your fabric) ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO

# --- GPUs to use ---
GPUS_TO_USE="0,1"
echo "--- Using GPUs: ${GPUS_TO_USE} ---"

# --- Paths ---
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"

# --- Hyperparams (YOUR choices) ---
NS_ALPHA=0.25                   # keep your alpha
EXPERIMENT_NAME="hybrid_alpha${NS_ALPHA}_epochs3_support_set"
OUTPUT_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "--- LAUNCHING: ${EXPERIMENT_NAME} ---"
deepspeed --include "localhost:${GPUS_TO_USE}" "${TRAIN_SCRIPT_PATH}" \
  --deepspeed "${DEEPSPEED_CONFIG_PATH}" \
  --seed 1234 \
  --model_name_or_path "${BASE_MODEL_PATH}" \
  --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
  --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --loss "hybrid" \
  --ns_type "support_set" \
  --ns_alpha "${NS_ALPHA}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 66 \
  --save_strategy "epoch" \
  --learning_rate 2e-6 \
  --max_grad_norm 0.5 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --bf16 True \
  --use_flash_attn True \
  --report_to "wandb" \
  --run_name "${EXPERIMENT_NAME}" \
  2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "--- SUCCESS: Finished ${EXPERIMENT_NAME} ---"
