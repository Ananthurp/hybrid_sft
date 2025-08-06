#!/bin/bash
set -e 
set -x

# --- Define Our Paths ---
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
INPUT_DATA_PATH="/data/ananthu/gem_project/datasets/ultrafeedback_binarized"
OUTPUT_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
PYTHON_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/preprocess_data.py"

# --- Create the output directory if it doesn't exist ---
mkdir -p ${OUTPUT_DATA_DIR}

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Run Tokenization for the Training Split ---
python ${PYTHON_SCRIPT_PATH} \
    --tokenizer_name_or_path ${TOKENIZER_PATH} \
    --dataset_name_or_path ${INPUT_DATA_PATH} \
    --split "train_sft" \
    --output_file "${OUTPUT_DATA_DIR}/train.jsonl" 

# --- Run Tokenization for the Test Split ---
python ${PYTHON_SCRIPT_PATH} \
    --tokenizer_name_or_path ${TOKENIZER_PATH} \
    --dataset_name_or_path ${INPUT_DATA_PATH} \
    --split "test_sft" \
    --output_file "${OUTPUT_DATA_DIR}/test.jsonl"

echo "Tokenization for Qwen2-1.5B complete."