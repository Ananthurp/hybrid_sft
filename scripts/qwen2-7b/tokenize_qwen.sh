#!/usr/bin/env bash
set -euo pipefail
set -x

# Absolute project paths (no uninitialized vars)
PROJ="$HOME/LLMDiversity"
REPO="$PROJ/hybrid_sft"

# If base tokenizer lacks a chat template, switch to Qwen2-7B-Instruct below.
export TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJ/models/Qwen2-7B}"

# Dataset ID (binarized UltraFeedback with train_sft/test_sft)
export INPUT_DATA_PATH="${INPUT_DATA_PATH:-HuggingFaceH4/ultrafeedback_binarized}"

# Output dir for tokenized JSONL
export OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-$PROJ/datasets/ultrafeedback_tokenized_qwen2-7b}"
PYTHON_SCRIPT_PATH="$REPO/preprocess_data.py"

mkdir -p "$OUTPUT_DATA_DIR"

# Keep caches on scratch
export PRJ=prj0000000224
export SCRATCH="/scratch/$PRJ"
export HF_HOME="$SCRATCH/LLMDiversity_work/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Tunables
MAXLEN="${MAXLEN:-4096}"
WORKERS="${WORKERS:-16}"

# Precheck: ensure tokenizer has a chat template
python - <<'PY'
import os, sys
from transformers import AutoTokenizer
p=os.environ['TOKENIZER_PATH']
tok=AutoTokenizer.from_pretrained(p, use_fast=True)
if not getattr(tok, "chat_template", None):
    sys.stderr.write(f"ERROR: Tokenizer at {p} has no chat_template.\n"
                     "Use Qwen2-7B-Instruct tokenizer or provide a chat template.\n")
    sys.exit(2)
print("Tokenizer chat template present.")
PY

# Train split
python "$PYTHON_SCRIPT_PATH" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --dataset_name_or_path "$INPUT_DATA_PATH" \
  --split "train_sft" \
  --max_seq_length "$MAXLEN" \
  --preprocessing_num_workers "$WORKERS" \
  --output_file "$OUTPUT_DATA_DIR/train.jsonl"

# Test split
python "$PYTHON_SCRIPT_PATH" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --dataset_name_or_path "$INPUT_DATA_PATH" \
  --split "test_sft" \
  --max_seq_length "$MAXLEN" \
  --preprocessing_num_workers "$WORKERS" \
  --output_file "$OUTPUT_DATA_DIR/test.jsonl"

echo "Tokenization complete → $OUTPUT_DATA_DIR"
