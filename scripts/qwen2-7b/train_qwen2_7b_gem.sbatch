#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p h200n
#SBATCH --gres=gpu:h200:4
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH -t 14:00:00
#SBATCH -J qwen2-7b-gem
#SBATCH -o qwen2-7b-gem-%j.out
#SBATCH -e qwen2-7b-gem-%j.err

set -euo pipefail
set -x

# --- Env / modules ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Give this job its own rendezvous port ---
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
if command -v ss >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$MASTER_PORT$"; then
    export MASTER_PORT=$((MASTER_PORT + 1))
  fi
fi
export DS_MASTER_PORT="$MASTER_PORT"
echo "Rendezvous: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# --- Training knobs ---
export GPUS_PER_NODE=4
export MICRO_BS=8
export GA=4
export EPOCHS=3
export SAVE_STRATEGY=steps
export SAVE_STEPS=1000
export EVAL_STRATEGY=no
export LOGGING_STEPS=10
export LR=2e-6
export REPORT_TO=wandb
export MAX_STEPS=0
export MAX_TRAIN_SAMPLES=0

# Run only GEM
export RUN_CE=0
export RUN_GEM=1
export RUN_HYB75=0
export RUN_HYB50=0

# W&B API key (from your secret file)
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.secrets/wandb_api_key" ]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_api_key")"
fi

bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/train_qwen2_7b_pipeline.sh"