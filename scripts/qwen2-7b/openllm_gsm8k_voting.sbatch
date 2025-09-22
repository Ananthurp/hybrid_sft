#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J gsm8k_vote
#SBATCH -o gsm8k_vote-%j.out
#SBATCH -e gsm8k_vote-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# Node-local caches to avoid /scratch pressure
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$HF_HOME/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

# Required inputs
: "${MODEL_DIR:?set via --export MODEL_DIR=/path/to/model}"
: "${RUN_NAME:?set via --export RUN_NAME=my_run_name}"
: "${TOKENIZER_PATH:?set via --export TOKENIZER_PATH=/path/to/tokenizer}"

# Optional knobs
N="${N:-32}"                    # samples per question
BATCH="${BATCH:-16}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"            # 0 => disabled (use top_p)
MAX_NEW="${MAX_NEW:-512}"
USE_VLLM="${USE_VLLM:-True}"   # must be True (script requires vLLM)

OUT_BASE="${OUT_BASE:-$HOME/LLMDiversity/evals_openllm}"
OUT_DIR="$OUT_BASE/$RUN_NAME/gsm8k"
mkdir -p "$OUT_DIR"

python ~/LLMDiversity/hybrid_sft/evaluation/multi_openllm/evaluation_gsm8k_voting.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --dtype bf16 \
  --n "$N" \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --max_new_tokens "$MAX_NEW" \
  --use_vllm "$USE_VLLM" \
  --vllm_gpu_memory_utilization 0.9 \
  --seed 42 \
  --batch_size "$BATCH" \
  --save_path "$OUT_DIR/gsm8k_voting_dump.json" \
  --summary_path "$OUT_DIR/gsm8k_voting_summary.json" \
  --remove_old \
  2>&1 | tee "$OUT_DIR/gsm8k_voting.log"

echo "[done] summary at: $OUT_DIR/gsm8k_voting_summary.json"