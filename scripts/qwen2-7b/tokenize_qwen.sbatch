#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=tok-qwen2-7b
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpun
#SBATCH --qos=normal

set -euxo pipefail

# ===== Conda activation (robust) =====
module load miniforge/24.11.3-2
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate hybrid_loss

# Align workers with CPUs
export MAXLEN=4096
export WORKERS=16

# Run the tokenization
bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/tokenize_qwen.sh"
