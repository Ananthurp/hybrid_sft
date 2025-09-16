#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J CODEGEN
#SBATCH -o CODEGEN-%j.out
#SBATCH -e CODEGEN-%j.err

set -euo pipefail
set -x

# --- Env (CUDA 11.8 + infer118) ---
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
export VLLM_CACHE_DIR="$LOCAL_SCRATCH/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

# --- Inputs (pass via sbatch --export) ---
MODEL_DIR="${MODEL_DIR:?}"
TOKENIZER_DIR="${TOKENIZER_DIR:?$HOME/LLMDiversity/models/Qwen2-7B}"
RUN_NAME="${RUN_NAME:?}"
OUT_BASE="${OUT_BASE:-$HOME/LLMDiversity/evals_codegen}"

# Generation knobs
MODEL_KEY="deepseek-ai/deepseek-coder-7b-base"  # just a key for the helper; safe for our Qwen tokenizer
DATASET="humaneval"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW="${MAX_NEW:-512}"
N_PER_PROBLEM="${N_PER_PROBLEM:-100}"    # <-- budgets up to 100
N_PER_BATCH="${N_PER_BATCH:-100}"
N_BATCHES="${N_BATCHES:-1}"
USE_VLLM="${USE_VLLM:-True}"
TEMPLATE="${TEMPLATE:-normal}"

OUT_DIR="$OUT_BASE/$RUN_NAME"
mkdir -p "$OUT_DIR"
echo "[info] OUT_DIR: $OUT_DIR"

SAVE_PATH="$OUT_DIR/evalplus-${DATASET}.jsonl"
LOG_PATH="$OUT_DIR/${DATASET}.log"

# --- Generate completions ---
python -m hybrid_sft.evaluation.text2code \
  --model_key "$MODEL_KEY" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_DIR" \
  --save_path "$SAVE_PATH" \
  --dataset "$DATASET" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_new_tokens "$MAX_NEW" \
  --n_problems_per_batch "$N_PER_BATCH" \
  --n_samples_per_problem "$N_PER_PROBLEM" \
  --n_batches "$N_BATCHES" \
  --use_vllm "$USE_VLLM" \
  --template "$TEMPLATE"

# --- Evaluate (pass@k incl. k up to 100) ---
# HumanEval does not need sanitize; EvalPlus writes eval_results JSON next to samples.
evalplus.evaluate --dataset "$DATASET" --samples "$SAVE_PATH" 2>&1 | tee "$LOG_PATH"

# Copy the JSON result (EvalPlus standard name) into OUT_DIR with a stable filename
RESULT_JSON="$(dirname "$SAVE_PATH")/evalplus-${DATASET}_eval_results.json"
if [[ -f "$RESULT_JSON" ]]; then
  cp "$RESULT_JSON" "$OUT_DIR/${DATASET}_eval_results.json"
fi

echo "[done] $RUN_NAME"