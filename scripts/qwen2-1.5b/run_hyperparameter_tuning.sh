#!/bin/bash
# Master script for hyperparameter tuning of the hybrid loss function.

set -e; set -x;

# --- Global Configuration ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO

# --- Manually specify the GPUs to use ---
# As confirmed by nvidia-smi, we will use GPUs 0 and 1.
# The script will run on GPUs 0 and 1.
# You can change this to "2,3" or any other pair if needed.
GPUS_TO_USE="2,3"
echo "--- Using manually specified GPUs: ${GPUS_TO_USE} ---"

# --- Global Paths ---
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
      echo "--- LAUNCHING: ${EXPERIMENT_NAME} on GPUs ${GPUS_TO_USE} ---"
      mkdir -p $OUTPUT_DIR

      deepspeed --include localhost:${GPUS_TO_USE} ${TRAIN_SCRIPT_PATH} \
        --deepspeed ${DEEPSPEED_CONFIG_PATH} \
        --seed 1234 \
        --model_name_or_path ${BASE_MODEL_PATH} \
        --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
        --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
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
        --gradient_checkpointing True \
        ---bf16 True \
        --use_flash_attn True \
        --report_to "wandb" \
        --run_name "${EXPERIMENT_NAME}" \
        2>&1 | tee $OUTPUT_DIR/training.log
      
      echo "--- SUCCESS: Finished ${EXPERIMENT_NAME} ---"
    done

    # --- Run Bottom-P Experiments ---
    for bottom_p in "${BOTTOM_P_VALS[@]}"; do
      EXPERIMENT_NAME="hybrid_alpha${alpha}_epochs${num_epochs}_bottomp${bottom_p}"
      OUTPUT_DIR="/data/ananthu/gem_project/results/${EXPERIMENT_NAME}"
      echo "--- LAUNCHING: ${EXPERIMENT_NAME} on GPUs ${GPUS_TO_USE} ---"
      mkdir -p $OUTPUT_DIR

      deepspeed --include localhost:${GPUS_TO_USE} ${TRAIN_SCRIPT_PATH} \
        --deepspeed ${DEEPSPEED_CONFIG_PATH} \
        --seed 1234 \
        --model_name_or_path ${BASE_MODEL_PATH} \
        --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
        --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
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
        --gradient_checkpointing True \
        --bf16 True \
        --use_flash_attn True \
        --report_to "wandb" \
        --run_name "${EXPERIMENT_NAME}" \
        2>&1 | tee $OUTPUT_DIR/training.log
        
      echo "--- SUCCESS: Finished ${EXPERIMENT_NAME} ---"
    done
  done
done

echo "--- ALL HYPERPARAMETER TUNING EXPERIMENTS HAVE FINISHED ---"