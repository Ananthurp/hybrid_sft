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
# send USR1 5 minutes before timeout so we can finish a step and hit a save point
#SBATCH --signal=B:USR1@300

set -euo pipefail
set -x

# --- Conda & CUDA ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

# this CUDA module sets CUDA_HOME, PATH and LD_LIBRARY_PATH cleanly
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Optional: W&B API key from secret file (avoid printing secrets in logs) ---
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.secrets/wandb_api_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_api_key")"
fi

# --- Train knobs you can tweak at submit time (env overrides) ---
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# With 4×H200 + ZeRO-3 + gradient checkpointing, this is a good starting point.
# You can push higher later if headroom allows (see note below).
export MICRO_BS=${MICRO_BS:-8}
export GA=${GA:-4}
export EPOCHS=${EPOCHS:-3}

# frequent-enough checkpoints so a timeout never loses much progress
export SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
export SAVE_STEPS=${SAVE_STEPS:-500}

# eval = off for max throughput
export EVAL_STRATEGY=${EVAL_STRATEGY:-no}
export LOGGING_STEPS=${LOGGING_STEPS:-10}
export LR=${LR:-2e-6}
export REPORT_TO=${REPORT_TO:-wandb}   # set to "none" to disable W&B

# use epochs by default; set MAX_STEPS>0 for short range tests
export MAX_STEPS=${MAX_STEPS:-0}
export MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}

# If Slurm warns us we’re close to timeout, just message in logs.
# (We already save every SAVE_STEPS steps; reruns resume automatically.)
trap 'echo "[SLURM] Pre-timeout signal (USR1) received – will finish current step and rely on periodic checkpoints."' USR1

# --- Launch the 4-phase pipeline ---
bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/train_qwen2_7b_pipeline.sh"

