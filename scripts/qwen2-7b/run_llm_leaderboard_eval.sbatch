#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J openllm
#SBATCH -o openllm-%j.out
#SBATCH -e openllm-%j.err

set -euo pipefail

# --- modules / env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118

module load cuda/11.8.0
export CUDA_HOME=/apps/cuda/11.8.0

# threads / tokenizer / allocator
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# scratch caches
export JOB_SCRATCH="/tmp/$USER/${SLURM_JOB_ID:-eval}"
mkdir -p "$JOB_SCRATCH"/{hf,logs,tmp}
export HF_HOME="$JOB_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$JOB_SCRATCH/tmp"

# --- inputs via --export ---
# MODEL_DIR, TOKENIZER_DIR, RUN_NAME, OUT_BASE, USE_VLLM (True/False), DTYPE(=bfloat16|float16),
# TASKS_OVERRIDE (optional), GEN_KW_OVERRIDE (optional)

INCLUDE_DIR="$HOME/LLMDiversity/hybrid_sft/tasks"

# default 6 OpenLLM leaderboard tasks; override with TASKS_OVERRIDE
TASKS="${TASKS_OVERRIDE:-arc_challenge,hellaswag,winogrande,truthfulqa_mc2,mmlu,gsm8k}"

OUT_DIR="${OUT_BASE}/${RUN_NAME}"
mkdir -p "$OUT_DIR"

echo "[info] MODEL_DIR: $MODEL_DIR"
echo "[info] TOKENIZER_DIR: $TOKENIZER_DIR"
echo "[info] OUT_DIR: $OUT_DIR"
echo "[info] USE_VLLM: ${USE_VLLM:-False}"

DTYPE="${DTYPE:-bfloat16}"

HF_MODEL_ARGS="pretrained=${MODEL_DIR},tokenizer=${TOKENIZER_DIR},trust_remote_code=True,dtype=${DTYPE}"
VLLM_MODEL_ARGS="pretrained=${MODEL_DIR},tokenizer=${TOKENIZER_DIR},tensor_parallel_size=1,dtype=${DTYPE},gpu_memory_utilization=0.90,trust_remote_code=True"

# Remove temperature (it’s ignored with greedy); make sampling explicit off
GEN_KW="${GEN_KW_OVERRIDE:-do_sample=False,top_p=1.0,max_new_tokens=256}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RES_JSON="$OUT_DIR/results-${TIMESTAMP}.json"
RES_SUMMARY="$OUT_DIR/summary-${TIMESTAMP}.txt"

if [[ "${USE_VLLM:-False}" == "True" ]]; then
  BACKEND="vllm"
  MODEL_ARGS="$VLLM_MODEL_ARGS"
else
  BACKEND="hf"
  MODEL_ARGS="$HF_MODEL_ARGS"
fi

python -m lm_eval \
  --model "${BACKEND}" \
  --tasks "${TASKS}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --max_batch_size 1 \
  --gen_kwargs "${GEN_KW}" \
  --device cuda:0 \
  --include_path "${INCLUDE_DIR}" \
  --output_path "${RES_JSON}" \
  2>&1 | tee "${RES_SUMMARY}"

echo "[done] Wrote:"
echo "  - ${RES_JSON}"
echo "  - ${RES_SUMMARY}"