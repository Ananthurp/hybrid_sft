#!/bin/bash

set -e 
set -x

# --- 1. DEFINE OUR ABSOLUTE PATHS ---
MODEL_PATH="/data/ananthu/gem_project/models/Llama-3.1-8B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_llama3.1-8b"
OUTPUT_DIR="/data/ananthu/gem_project/results/full_test_run"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_48gb.json"


# --- 2. CREATE THE OUTPUT DIRECTORY ---
mkdir -p $OUTPUT_DIR


# --- 3. SET ENVIRONMENT VARIABLES ---
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"


# --- 4. LAUNCH THE TRAINING JOB ---
deepspeed ${TRAIN_SCRIPT_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    --seed 1234 \
    --model_name_or_path ${MODEL_PATH} \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_steps 50 \
    --save_strategy "no" \
    --loss "gem" \
    --gem_beta 0.7 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    --use_flash_attn False \
    2>&1 | tee $OUTPUT_DIR/training.log