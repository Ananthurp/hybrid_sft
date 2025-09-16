#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -J div-$SHORTNAME
#SBATCH -o div-$SHORTNAME-%j.out
#SBATCH -e div-$SHORTNAME-%j.err

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
#   EVAL_DIR, TOKENIZER_DIR, RESP_JSON, DIVERSITY_LOG, SHORTNAME

# Save printed metrics into a deterministic .log for later parsing
python "$EVAL_DIR/evaluation_diversity.py" \
  --response_path "$RESP_JSON" \
  --tokenizer_path "$TOKENIZER_DIR" | tee "$DIVERSITY_LOG"