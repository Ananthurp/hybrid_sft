#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J gen-$SHORTNAME
#SBATCH -o gen-$SHORTNAME-%j.out
#SBATCH -e gen-$SHORTNAME-%j.err

set -euo pipefail
set -x

module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

export HF_HOME="/scratch/prj0000000224/LLMDiversity_work/cache/huggingface"
mkdir -p "$HF_HOME"

# REQUIRED env vars:
#   MODEL_DIR, TOKENIZER_DIR, DATASET_KEY, SPLIT, SAVE_JSON, EVAL_DIR, SHORTNAME
python "$EVAL_DIR/generate_response.py" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$TOKENIZER_DIR" \
  --dataset_path "$DATASET_KEY" \
  --split "$SPLIT" \
  --use_vllm True \
  --vllm_gpu_memory_utilization 0.90 \
  --batch_size 10 \
  --n 4 \
  --do_sample True \
  --top_k 50 \
  --top_p 0.9 \
  --temperature 0.7 \
  --max_new_tokens 1024 \
  --remove_old True \
  --save_path "$SAVE_JSON"