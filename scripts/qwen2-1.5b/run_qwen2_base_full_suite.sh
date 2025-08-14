#!/bin/bash
set -euo pipefail
set -x

# ---- Comms / memory guards ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO

# ---- GPUs to use ----
GPUS_TO_USE="${GPUS_TO_USE:-0,1}"   # edit or export before calling
echo "--- Using GPUs: ${GPUS_TO_USE} ---"

# ---- Paths ----
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B"   # <— base model
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"

# ---- Common training knobs ----
EPOCHS=3
BATCH=4
GAS=8
LR=2e-6
LOG_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=100
SAVE_LIMIT=2

run() {
  NAME="$1"; shift
  OUT="/data/ananthu/gem_project/results/${NAME}"
  mkdir -p "$OUT"
  echo "--- LAUNCHING: ${NAME} ---"

  deepspeed --include "localhost:${GPUS_TO_USE}" "${TRAIN_SCRIPT_PATH}" \
    --deepspeed "${DEEPSPEED_CONFIG_PATH}" \
    --seed 1234 \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file  "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir "${OUT}" \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH} \
    --gradient_accumulation_steps ${GAS} \
    --learning_rate ${LR} \
    --max_grad_norm 0.5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps ${LOG_STEPS} \
    --gradient_checkpointing True \
    --bf16 True \
    --use_flash_attn True \
    --evaluation_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --prediction_loss_only True \
    --per_device_eval_batch_size 2 \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_LIMIT} \
    --report_to "wandb" \
    --run_name "${NAME}" \
    "$@" 2>&1 | tee "${OUT}/training.log"

  echo "--- SUCCESS: Finished ${NAME} ---"
}

############################
# 1) Cross-entropy (baseline)
############################
run "qwen_base_ce_run" \
  --loss "ce"

############################
# 2) GEM (paper baseline)
############################
run "qwen_base_gem_run" \
  --loss "gem" \
  --gem_beta 0.7 \
  --gem_h "linear"

############################################
# 3) Sparsemax-only (FY loss, no NS term)
############################################
run "qwen_base_sparsemax_only_epochs3" \
  --loss "hybrid" \
  --ns_type "support_set" \
  --ns_alpha 0.0

######################################################
# 4) Hybrid (FY + NS on sparse support) alpha = 0.25
######################################################
run "qwen_base_hybrid_alpha0.25_epochs3_support_set" \
  --loss "hybrid" \
  --ns_type "support_set" \
  --ns_alpha 0.25

######################################################
# 5) Hybrid (FY + NS on sparse support) alpha = 0.5
######################################################
run "qwen_base_hybrid_alpha0.5_epochs3_support_set" \
  --loss "hybrid" \
  --ns_type "support_set" \
  --ns_alpha 0.5

######################################################
# 6) Hybrid (FY + NS on sparse support) alpha = 0.7
######################################################
run "qwen_base_hybrid_alpha0.7_epochs3_support_set" \
  --loss "hybrid" \
  --ns_type "support_set" \
  --ns_alpha 0.7

############################################
# 7) NS-only (Top-K=3), no sparsemax loss
############################################
run "qwen_base_ns_only_topk3_epochs3" \
  --loss "hybrid" \
  --ns_only True \
  --ns_alpha 1.0 \
  --ns_type "top_k" \
  --ns_top_k 3

#####################################################
# 8) Hybrid with Top-K=3 (FY + NS)
#####################################################
run "qwen_base_hybrid_alpha0.5_topk3_epochs3" \
  --loss "hybrid" \
  --ns_type "top_k" \
  --ns_top_k 3 \
  --ns_alpha 0.5
