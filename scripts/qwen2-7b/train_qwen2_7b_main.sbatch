#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p h200n
#SBATCH --gres=gpu:h200:4
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH -t 3-00:00:00
#SBATCH -J qwen2-7b-main
#SBATCH -o qwen2-7b-main-%j.out
#SBATCH -e qwen2-7b-main-%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail
set -x

module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# W&B key from secret file if not already set
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.secrets/wandb_api_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_api_key")"
fi

# tunables (env overrides allowed)
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export MICRO_BS=${MICRO_BS:-8}
export GA=${GA:-4}
export EPOCHS=${EPOCHS:-3}
export SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
export SAVE_STEPS=${SAVE_STEPS:-500}
export EVAL_STRATEGY=${EVAL_STRATEGY:-no}
export LOGGING_STEPS=${LOGGING_STEPS:-10}
export LR=${LR:-2e-6}
export REPORT_TO=${REPORT_TO:-wandb}
export MAX_STEPS=${MAX_STEPS:-0}
export MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}

trap 'echo "[SLURM] Pre-timeout signal (USR1) received – finishing current step; periodic checkpoints will handle resume."' USR1

bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/train_qwen2_7b_pipeline.sh"
