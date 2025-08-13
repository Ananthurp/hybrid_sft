#!/bin/bash
set -euo pipefail
set -x

# --- Comms / memory guards ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO

# --- GPUs to use (edit if needed) ---
GPUS_TO_USE="0,1"
echo "--- Using GPUs: ${GPUS_TO_USE} ---"

# --- Paths ---
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"
RESULTS_ROOT="/data/ananthu/gem_project/results"

# --- Fixed HPs for this retrain ---
ALPHA=0.5
NUM_EPOCHS=3
TOPK_LIST=(3 5 10)

# Helper: ensure unique output dir by adding _r1, _r2, ...
ensure_unique_dir () {
  local base_dir="$1"
  if [[ ! -d "$base_dir" ]]; then
    echo "$base_dir"; return
  fi
  local i=1
  while [[ -d "${base_dir}_r${i}" ]]; do
    ((i++))
  done
  echo "${base_dir}_r${i}"
}

for K in "${TOPK_LIST[@]}"; do
  EXP_BASE_NAME="hybrid_${ALPHA}alpha_${NUM_EPOCHS}epochs_topk${K}"
  OUTPUT_DIR_BASE="${RESULTS_ROOT}/${EXP_BASE_NAME}"
  OUTPUT_DIR="$(ensure_unique_dir "${OUTPUT_DIR_BASE}")"
  RUN_NAME="$(basename "${OUTPUT_DIR}")"

  mkdir -p "${OUTPUT_DIR}"
  echo "--- LAUNCHING: ${RUN_NAME} on GPUs ${GPUS_TO_USE} ---"

  deepspeed --include "localhost:${GPUS_TO_USE}" "${TRAIN_SCRIPT_PATH}" \
    --deepspeed "${DEEPSPEED_CONFIG_PATH}" \
    --seed 1234 \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir "${OUTPUT_DIR}" \
    --loss "hybrid" \
    --ns_type "top_k" \
    --ns_alpha "${ALPHA}" \
    --ns_top_k "${K}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --save_total_limit 2 \
    --report_to "wandb" \
    --run_name "${RUN_NAME}" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

  echo "--- SUCCESS: Finished ${RUN_NAME} ---"
done

