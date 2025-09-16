#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J gsm8k
#SBATCH -o gsm8k-%j.out
#SBATCH -e gsm8k-%j.err

set -euo pipefail

# --- modules / env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118

module load cuda/11.8.0
export CUDA_HOME=/apps/cuda/11.8.0

# threads & tokenizers
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=true

# scratch caches (keeps HF stuff off your home)
export JOB_SCRATCH="/tmp/$USER/${SLURM_JOB_ID:-gsm8k}"
mkdir -p "$JOB_SCRATCH"/{hf,tmp}
export HF_HOME="$JOB_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$JOB_SCRATCH/tmp"

# --- required via --export ---
# MODEL_DIR, TOKENIZER_DIR, RUN_NAME, OUT_BASE
# optional: USE_VLLM(True/False), MAX_NEW_TOKENS, TEMP, TOP_P, TOP_K, BATCH, VLLM_MEM

OUT_DIR="${OUT_BASE}/${RUN_NAME}"
mkdir -p "$OUT_DIR"

echo "[info] MODEL_DIR: ${MODEL_DIR}"
echo "[info] TOKENIZER_DIR: ${TOKENIZER_DIR}"
echo "[info] OUT_DIR: ${OUT_DIR}"
echo "[info] USE_VLLM: ${USE_VLLM:-True}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
BATCH="${BATCH:-16}"
VLLM_MEM="${VLLM_MEM:-0.90}"

SAVE_PATH="${OUT_DIR}/gsm8k_${SLURM_JOB_ID:-manual}.json"
LOG_PATH="${OUT_DIR}/gsm8k_${SLURM_JOB_ID:-manual}.log"

python "$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_gsm8k.py" \
  --dataset_name_or_path "openai/gsm8k" \
  --model_name_or_path "${MODEL_DIR}" \
  --tokenizer_name_or_path "${TOKENIZER_DIR}" \
  --use_vllm "${USE_VLLM:-True}" \
  --vllm_gpu_memory_utilization "${VLLM_MEM}" \
  --batch_size "${BATCH}" \
  --top_k "${TOP_K}" \
  --top_p "${TOP_P}" \
  --temperature "${TEMP}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --save_path "${SAVE_PATH}" \
  2>&1 | tee "${LOG_PATH}"

echo "[done] wrote:"
echo " - ${SAVE_PATH}"
echo " - ${LOG_PATH}"