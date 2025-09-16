#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J codegen_gen
#SBATCH -o codegen_gen-%j.out
#SBATCH -e codegen_gen-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# Node-local caches
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$HF_HOME/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"

# --- Required inputs (pass via --export) ---
: "${MODEL_DIR:?set via --export MODEL_DIR=/path/to/finetuned/model}"
: "${RUN_NAME:?set via --export RUN_NAME=name_for_outputs}"
: "${TOKENIZER_PATH:?set via --export TOKENIZER_PATH=/path/to/base/tokenizer}"

# --- Generation knobs ---
OUT_BASE="${OUT_BASE:-$HOME/LLMDiversity/evals_codegen}"
DATASET="${DATASET:-humaneval}"          # humaneval or mbpp
TEMPLATE="${TEMPLATE:-normal}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW="${MAX_NEW:-512}"
N_PER_PROBLEM="${N_PER_PROBLEM:-100}"    # generate 100 once → we’ll compute all pass@k from this
N_PER_BATCH="${N_PER_BATCH:-100}"
N_BATCHES="${N_BATCHES:-1}"
USE_VLLM="${USE_VLLM:-True}"

OUT_DIR="$OUT_BASE/$RUN_NAME"
mkdir -p "$OUT_DIR"

python -m hybrid_sft.evaluation.text2code \
  --model_key "deepseek-ai/deepseek-coder-7b-base" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --save_path "$OUT_DIR/evalplus-$DATASET.jsonl" \
  --dataset "$DATASET" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_new_tokens "$MAX_NEW" \
  --n_problems_per_batch "$N_PER_BATCH" \
  --n_samples_per_problem "$N_PER_PROBLEM" \
  --n_batches "$N_BATCHES" \
  --use_vllm "$USE_VLLM" \
  --template "$TEMPLATE"

echo "[done] Generated -> $OUT_DIR/evalplus-$DATASET.jsonl"