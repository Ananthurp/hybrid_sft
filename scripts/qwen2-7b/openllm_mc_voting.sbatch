#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J OLLM_MC
#SBATCH -o OLLM_MC-%j.out
#SBATCH -e OLLM_MC-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# --- Node-local caches ---
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$HF_HOME/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

# --- Inputs (REQUIRED from --export) ---
: "${MODEL_DIR:?set via --export MODEL_DIR=/path/to/model}"
: "${RUN_NAME:?set via --export RUN_NAME=name_for_outputs}"
: "${TOKENIZER_PATH:?set via --export TOKENIZER_PATH=/path/to/tokenizer}"
: "${TASK:?set via --export TASK=arc|hellaswag|winogrande|mmlu|truthfulqa}"

# --- Sane defaults ---
OUT_BASE="${OUT_BASE:-$HOME/LLMDiversity/evals_openllm/$RUN_NAME/$TASK}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"               # -1 disables top-k in vLLM 0.7.1
MAX_NEW="${MAX_NEW:-256}"
N="${N:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_VLLM="${USE_VLLM:-True}"

mkdir -p "$OUT_BASE"
DUMP="$OUT_BASE/${TASK}_voting_dump.json"
SUMMARY="$OUT_BASE/${TASK}_voting_summary.json"
LOG="$OUT_BASE/${TASK}_voting.log"

python -u $HOME/LLMDiversity/hybrid_sft/evaluation/multi_openllm/evaluation_mc_voting.py \
  --task "$TASK" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --dtype bf16 \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --n "$N" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW" \
  --use_vllm "$USE_VLLM" \
  --vllm_gpu_memory_utilization 0.9 \
  --remove_old True \
  --save_path "$DUMP" \
  --summary_path "$SUMMARY" \
  2>&1 | tee "$LOG"

echo "[done] wrote:"
ls -lh "$OUT_BASE"