#!/bin/bash
# Master script for hyperparameter tuning of the hybrid loss function.

set -e; set -x;

# --- Global Configuration ---
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"

# --- Hyperparameter Arrays ---
ALPHAS=(0.5 1.0)
EPOCHS=(3 6)
TOP_K_VALS=(3 5 10)
BOTTOM_P_VALS=(0.5 0.7 0.9 0.95)

# --- Main Experiment Loop ---

for alpha in "${ALPHAS[@]}"; do
  for num_epochs in "${EPOCHS[@]}"; do
    
    # --- Run Top-K Experiments ---
    for top_k in "${TOP_K_VALS[@]}"; do
      EXPERIMENT_NAME="hybrid_alpha${alpha}_epochs${num_epochs}_topk${top_k}"
      OUTPUT_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"
      echo "--- LAUNCHING: ${EXPERIMENT_NAME} ---"
      mkdir -p $OUTPUT_DIR

      deepspeed --include localhost:2,3 ${TRAIN_SCRIPT_PATH} \
        --deepspeed ${DEEPSPEED_CONFIG_PATH} \
        --model_name_or_path ${BASE_MODEL_PATH} \
        --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
        --output_dir ${OUTPUT_DIR} \
        --loss "hybrid" \
        --ns_type "top_k" \
        --ns_alpha ${alpha} \
        --ns_top_k ${top_k} \
        --num_train_epochs ${num_epochs} \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 66 \
        --save_strategy "epoch" \
        --learning_rate 2e-6 \
        --max_grad_norm 0.5 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --bf16 True \
        --use_flash_attn True \
        2>&1 | tee $OUTPUT_DIR/training.log
    done

    # --- Run Bottom-P Experiments ---
    for bottom_p in "${BOTTOM_P_VALS[@]}"; do
      EXPERIMENT_NAME="hybrid_alpha${alpha}_epochs${num_epochs}_bottomp${bottom_p}"
      OUTPUT_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"
      echo "--- LAUNCHING: ${EXPERIMENT_NAME} ---"
      mkdir -p $OUTPUT_DIR

      deepspeed --include localhost:2,3 ${TRAIN_SCRIPT_PATH} \
        --deepspeed ${DEEPSPEED_CONFIG_PATH} \
        --model_name_or_path ${BASE_MODEL_PATH} \
        --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
        --output_dir ${OUTPUT_DIR} \
        --loss "hybrid" \
        --ns_type "bottom_p" \
        --ns_alpha ${alpha} \
        --ns_bottom_p ${bottom_p} \
        --num_train_epochs ${num_epochs} \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 66 \
        --save_strategy "epoch" \
        --learning_rate 2e-6 \
        --max_grad_norm 0.5 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --bf16 True \
        --use_flash_attn True \
        --report_to "wandb"\
        2>&1 | tee $OUTPUT_DIR/training.log
    done
  done
done

echo "--- ALL HYPERPARAMETER TUNING EXPERIMENTS LAUNCHED ---"