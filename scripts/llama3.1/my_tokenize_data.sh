#!/bin/bash

set -e 
set -x

# --- Define Our Project Paths ---

# Path to the local model folder, which contains the tokenizer
TOKENIZER_PATH="/data/ananthu/gem_project/models/Llama-3.1-8B-Instruct"

# Path to the local raw dataset folder
INPUT_DATA_PATH="/data/ananthu/gem_project/datasets/ultrafeedback_binarized"

# Path to the NEW directory where the tokenized output files will be saved
OUTPUT_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_llama3.1-8b"

# --- Create the output directory if it doesn't exist ---
mkdir -p ${OUTPUT_DATA_DIR}

# --- Set Offline Mode and other environment variables ---
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_VISIBLE_DEVICES="0"

# --- Run Tokenization for the Training Split ---
# We replaced the Hugging Face names with our local paths
python ../../preprocess_data.py \
    --dataset_name_or_path ${INPUT_DATA_PATH} \
    --split "train_sft" \
    --tokenizer_name_or_path ${TOKENIZER_PATH} \
    --max_seq_length 2048 \
    --output_file "${OUTPUT_DATA_DIR}/train.jsonl" 

# --- Run Tokenization for the Test Split ---
# We do the same for the test data
python ../../preprocess_data.py \
    --dataset_name_or_path ${INPUT_DATA_PATH} \
    --split "test_sft" \
    --tokenizer_name_or_path ${TOKENIZER_PATH} \
    --max_seq_length 2048 \
    --output_file "${OUTPUT_DATA_DIR}/test.jsonl"

echo "Tokenization complete. Output saved to ${OUTPUT_DATA_DIR}"