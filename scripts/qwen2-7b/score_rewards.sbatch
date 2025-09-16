#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J rwd-$SHORTNAME
#SBATCH -o rwd-$SHORTNAME-%j.out
#SBATCH -e rwd-$SHORTNAME-%j.err

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
#   EVAL_DIR, TOKENIZER_DIR, RESP_JSON, OUT_JSON, SUM_JSON, SHORTNAME
python "$EVAL_DIR/evaluation_reward.py" \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --tokenizer_path     "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --detokenizer_path   "$TOKENIZER_DIR" \
  --data_path  "$RESP_JSON" \
  --batch_size 2 \
  --save_path  "$OUT_JSON" \
  --summary_path "$SUM_JSON"