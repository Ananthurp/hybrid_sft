#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p h200n
#SBATCH --gres=gpu:h200:4
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH -t 15:00:00
#SBATCH -J qwen2-7b-sparsemax
#SBATCH -o qwen2-7b-sparsemax-%j.out
#SBATCH -e qwen2-7b-sparsemax-%j.err

set -euo pipefail
set -x

# --- Env / modules ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Use scratch for caches + tmp + extensions + wandb ---
export PRJ=prj0000000224
export SCRATCH="/scratch/$PRJ"
export HF_HOME="$SCRATCH/LLMDiversity_work/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"    # ok even if deprecated
export TMPDIR="$SCRATCH/tmp/$USER"
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"
export WANDB_DIR="$SCRATCH/LLMDiversity_work/wandb"
export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" \
         "$TORCH_EXTENSIONS_DIR" "$WANDB_DIR" "$XDG_CACHE_HOME"

# Clear possibly stale HF json schema cache (safe)
CACHE_ROOT="${HF_DATASETS_CACHE:-$SCRATCH/LLMDiversity_work/cache/huggingface/datasets}"
echo "[preflight] clearing cached JSON datasets under: $CACHE_ROOT/json"
rm -rf "$CACHE_ROOT/json" || true

# --- Rendezvous port ---
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$MASTER_PORT$"; then
  export MASTER_PORT=$((MASTER_PORT + 1))
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

# Run only Sparsemax α=0
export RUN_CE=0
export RUN_GEM=0
export RUN_HYB75=0
export RUN_HYB50=0
export RUN_SPARSEMAX=1

# Resume (set to "none" to force fresh start)
export RESUME_FROM=last

# --- OUTPUT DIR ON SCRATCH (FIXES ENOSPC) ---
export OUT="$SCRATCH/LLMDiversity_work/results/qwen2-7b/phase5_sparsemax"
mkdir -p "$OUT"

# Optional: convenience symlink back into HOME (so your old paths still “exist”)
mkdir -p "$HOME/LLMDiversity/results/qwen2-7b"
ln -sfn "$OUT" "$HOME/LLMDiversity/results/qwen2-7b/phase5_sparsemax"

# W&B resume id (read from SCRATCH now)
export WANDB_RESUME=allow
RID="$(cat "$OUT"/runs/*/wandb/latest-run/id 2>/dev/null || true)"
export WANDB_RUN_ID="${RID:-sparsemax-$(date +%s)}"

# Launch
bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/train_qwen2_7b_pipeline.sh"