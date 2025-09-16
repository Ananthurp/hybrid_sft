#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J bt-$SHORTNAME
#SBATCH -o bt-$SHORTNAME-%j.out
#SBATCH -e bt-$SHORTNAME-%j.err

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
#   EVAL_DIR, CAND_JSON, BASE_JSON, OUT_JSON, SHORTNAME
python "$EVAL_DIR/bt_winrate_v2.py" \
  --candidate_json "$CAND_JSON" \
  --baseline_json  "$BASE_JSON" \
  --reward_model "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --dtype bfloat16 \
  --batch_size 8 \
  --max_len 4096 \
  --out_file "$OUT_JSON"