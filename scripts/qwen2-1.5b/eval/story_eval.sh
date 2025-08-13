#!/bin/sh
set -e; set -x;

MODEL_CHECKPOINT_PATH=$1

export CUDA_VISIBLE_DEVICES="2"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_story"
mkdir -p "$SAVE_DIR"

python evaluation/generate_response.py \
  --model_name_or_path "$MODEL_CHECKPOINT_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "data/story_generation/test.jsonl" \
  --dataset_split "train" \
  --prompt_key "instruction" \
  --max_size 200 \
  --n 1 \
  --temperature 0.8 \
  --use_vllm True \
  --vllm_gpu_memory_utilization 0.7 \
  --save_path "${SAVE_DIR}/generated_responses.json"

python evaluation/evaluation_diversity.py \
  --tokenizer_path "$TOKENIZER_PATH" \
  --detokenizer_path "$TOKENIZER_PATH" \
  --response_path "${SAVE_DIR}/generated_responses.json" \
  2>&1 | tee "${SAVE_DIR}/diversity_metrics.log"
