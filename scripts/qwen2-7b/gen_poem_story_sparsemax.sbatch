#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J POEM_STORY_SPMX
#SBATCH -o POEM_STORY_SPMX-%j.out
#SBATCH -e POEM_STORY_SPMX-%j.err
set -euo pipefail
set -x

module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120_spmx
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
export VLLM_CACHE_DIR="$LOCAL_SCRATCH/vllm"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$VLLM_CACHE_DIR" || true

: "${MODEL_DIR:?set model dir}"
: "${TOKENIZER_PATH:?set tokenizer dir}"
: "${OUT_DIR:?set output dir root}"
USE_VLLM="${USE_VLLM:-True}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GEN_TOK_SHORT="${GEN_TOK_SHORT:-256}"
POEM_MAX="${POEM_MAX:-300}"
POEM_N="${POEM_N:-8}"
STORY_MAX="${STORY_MAX:-300}"
STORY_N="${STORY_N:-8}"

EVAL_DIR_POEM="$OUT_DIR/evaluation_poem"
EVAL_DIR_STORY="$OUT_DIR/evaluation_story"
mkdir -p "$EVAL_DIR_POEM" "$EVAL_DIR_STORY"
rm -f "$EVAL_DIR_POEM/responses.json" "$EVAL_DIR_STORY/responses.json"

# Generate POEM (sparsemax via patched vLLM)
python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/generate_response.py" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/poem_generation" \
  --split test \
  --n "$POEM_N" \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens "$GEN_TOK_SHORT" \
  --save_path "$EVAL_DIR_POEM/responses.json" \
  --remove_old \
  --max_size "$POEM_MAX"

# Diversity POEM
python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py" \
  --response_path "$EVAL_DIR_POEM/responses.json" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --detokenizer_path "$TOKENIZER_PATH" \
  --out_path "$EVAL_DIR_POEM/diversity_summary.json" \
  > "$EVAL_DIR_POEM/diversity_metrics.log" 2>&1 || echo "[warn] Poem diversity failed"

# Generate STORY (sparsemax via patched vLLM)
python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/generate_response.py" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/story_generation" \
  --split test \
  --n "$STORY_N" \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens "$GEN_TOK_SHORT" \
  --save_path "$EVAL_DIR_STORY/responses.json" \
  --remove_old \
  --max_size "$STORY_MAX"

# Diversity STORY
python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py" \
  --response_path "$EVAL_DIR_STORY/responses.json" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --detokenizer_path "$TOKENIZER_PATH" \
  --out_path "$EVAL_DIR_STORY/diversity_summary.json" \
  > "$EVAL_DIR_STORY/diversity_metrics.log" 2>&1 || echo "[warn] Story diversity failed"

echo "[done] wrote:"
ls -lh "$EVAL_DIR_POEM"/responses.json "$EVAL_DIR_POEM"/diversity_summary.json \
       "$EVAL_DIR_STORY"/responses.json "$EVAL_DIR_STORY"/diversity_summary.json || true