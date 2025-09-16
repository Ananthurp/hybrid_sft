#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J codegen_eval
#SBATCH -o codegen_eval-%j.out
#SBATCH -e codegen_eval-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Node-local caches (fast + avoids /scratch clutter) ---
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$HF_HOME/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

# --- Inputs from sbatch --export (REQUIRED) ---
: "${MODEL_DIR:?set via --export MODEL_DIR=/path/to/model}"
: "${RUN_NAME:?set via --export RUN_NAME=name_for_outputs}"
: "${TOKENIZER_PATH:?set via --export TOKENIZER_PATH=Qwen tokenizer path}"

# Sane defaults
MODEL_KEY="${MODEL_KEY:-deepseek-ai/deepseek-coder-7b-base}"   # satisfies magicoder enum; weights still loaded from MODEL_DIR
OUT_BASE="${OUT_BASE:-$HOME/LLMDiversity/evals_codegen}"
DATASET_LIST="${DATASET_LIST:-humaneval}"                       # or "humaneval,mbpp"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW="${MAX_NEW:-512}"
N_PER_PROBLEM="${N_PER_PROBLEM:-100}"                           # produce 100 for all budgets
N_PER_BATCH="${N_PER_BATCH:-100}"
N_BATCHES="${N_BATCHES:-1}"
USE_VLLM="${USE_VLLM:-True}"
TEMPLATE="${TEMPLATE:-normal}"

OUT_DIR="$OUT_BASE/$RUN_NAME"
mkdir -p "$OUT_DIR"
echo "[info] OUT_DIR: $OUT_DIR"

# -------------------------
# Run per dataset
# -------------------------
IFS=',' read -ra DATASETS <<< "$DATASET_LIST"
for DATASET in "${DATASETS[@]}"; do
  SAVE_PATH="$OUT_DIR/evalplus-$DATASET.jsonl"
  LOG_PATH="$OUT_DIR/$DATASET.log"

  # generate 100 samples/problem once
  python -m hybrid_sft.evaluation.text2code \
    --model_key "$MODEL_KEY" \
    --model_name_or_path "$MODEL_DIR" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
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

  # MBPP needs sanitized file
  if [[ "$DATASET" == "mbpp" ]]; then
    SANITIZED="$OUT_DIR/evalplus-mbpp-sanitized.jsonl"
    python -m evalplus.sanitize --dataset mbpp --samples "$SAVE_PATH"
    mv "$OUT_DIR/evalplus-mbpp-sanitized.jsonl" "$SANITIZED"
  fi

  # Evaluate pass@1, pass@10, pass@100 (evalplus default)
  if [[ "$DATASET" == "mbpp" ]]; then
    evalplus.evaluate --dataset mbpp --samples "$SANITIZED" 2>&1 | tee "$LOG_PATH"
  else
    evalplus.evaluate --dataset humaneval --samples "$SAVE_PATH" 2>&1 | tee "$LOG_PATH"
  fi

  # Extra budgets 20, 50 via truncation
  for BUDGET in 20 50; do
    SRC="$SAVE_PATH"
    [[ "$DATASET" == "mbpp" ]] && SRC="$SANITIZED"
    TGT="${SRC%.jsonl}_top${BUDGET}.jsonl"
    python ~/LLMDiversity/hybrid_sft/evaluation/truncate_evalplus_jsonl.py \
      "$SRC" "$TGT" "$BUDGET"

    # Re-evaluate the truncated file -> produces *_eval_results.json next to it
    if [[ "$DATASET" == "mbpp" ]]; then
      evalplus.evaluate --dataset mbpp --samples "$TGT" \
        2>&1 | tee "${LOG_PATH%.log}_top${BUDGET}.log"
    else
      evalplus.evaluate --dataset humaneval --samples "$TGT" \
        2>&1 | tee "${LOG_PATH%.log}_top${BUDGET}.log"
    fi
  done
done

echo "[done] results at $OUT_DIR"