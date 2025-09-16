#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p cpun
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J CE_MERGE
#SBATCH -o CE_MERGE-%j.out
#SBATCH -e CE_MERGE-%j.err
set -euo pipefail
set -x

module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss  # same env as training
# No CUDA module needed

mkdir -p "$OUT_DIR"

# Write as safetensors (preferred), with a proper index
python "$CKPT_DIR/zero_to_fp32.py" \
  "$CKPT_DIR" "$OUT_DIR" \
  --tag "$TAG" \
  --safe_serialization