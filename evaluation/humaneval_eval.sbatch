#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J humaneval
#SBATCH -o humaneval-%j.out
#SBATCH -e humaneval-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Caches on /scratch (you said it's expanded now) ---
export PRJ=prj0000000224
export SCRATCH="/scratch/$PRJ"
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$HF_HOME/vllm"
export TMPDIR="$SCRATCH/tmp/$USER/$SLURM_JOB_ID"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"

# --- Inputs (export via sbatch) ---
MODEL_PATH="${MODEL_PATH:?set model path}"
# Use Qwen Instruct tokenizer for robust chat formatting across all runs
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen2-7B-Instruct}"
OUT_DIR="${OUT_DIR:?set output dir}"
mkdir -p "$OUT_DIR"

# Sampling budgets: generate once with N=100
N_SAMPLES="${N_SAMPLES:-100}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW="${MAX_NEW:-512}"

# Dummy key just to select context in helper (harmless)
MODEL_KEY="bigcode/starcoder"

JSONL="$OUT_DIR/evalplus-humaneval.jsonl"

# 1) Generate 100 samples/problem with vLLM
python -m hybrid_sft.evaluation.text2code \
  --model_key "$MODEL_KEY" \
  --model_name_or_path "$MODEL_PATH" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --save_path "$JSONL" \
  --dataset humaneval \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --max_new_tokens "$MAX_NEW" \
  --n_problems_per_batch 200 \
  --n_samples_per_problem "$N_SAMPLES" \
  --n_batches 1 \
  --use_vllm True \
  --template normal

# 2) Evaluate. If your evalplus has --k we get all budgets at once.
if evalplus.evaluate --help | grep -q -- "--k"; then
  evalplus.evaluate --dataset humaneval --samples "$JSONL" --k 1,10,20,50,100 \
    2>&1 | tee "$OUT_DIR/humaneval_eval.log"
else
  # Older evalplus: still run once (it will report pass@1/10/100)
  evalplus.evaluate --dataset humaneval --samples "$JSONL" \
    2>&1 | tee "$OUT_DIR/humaneval_eval.log"
fi