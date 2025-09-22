#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p h200n
#SBATCH --gres=gpu:h200:4
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH -t 14:00:00
#SBATCH -J qwen2-7b-ns-only
#SBATCH -o qwen2-7b-ns-only-%j.out
#SBATCH -e qwen2-7b-ns-only-%j.err

set -euo pipefail
set -x

# --- Env / modules ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# (Optional) silence DS first-time build warning & speed up compile for H200 (sm_90)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# --- Rendezvous port ---
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
export DS_MASTER_PORT="$MASTER_PORT"

# --- Disable W&B for this job ---
export WANDB_DISABLED=true
unset WANDB_API_KEY

# --- Run config (same as your pipeline defaults) ---
GPUS=4
MICRO_BS=8
GA=4
EPOCHS=3
SAVE_STRATEGY=steps
SAVE_STEPS=1000
EVAL_STRATEGY=no
LOGGING_STEPS=10
LR=2e-6

PROJ="$HOME/LLMDiversity"
REPO="$PROJ/hybrid_sft"
MODEL_DIR="$PROJ/models/Qwen2-7B"
DATA_DIR="$PROJ/datasets/ultrafeedback_tokenized_qwen2-7b"
OUT_DIR="$PROJ/results/qwen2-7b/phase9_ns_only_bottomp90"
DS_CFG="$REPO/scripts/deepspeed_config_qwen.json"
mkdir -p "$OUT_DIR"

deepspeed --num_gpus "$GPUS" "$REPO/train.py" \
  --deepspeed "$DS_CFG" \
  --seed 1234 \
  --model_name_or_path "$MODEL_DIR" \
  --train_tokenized_file "$DATA_DIR/train.jsonl" \
  --test_tokenized_file  "$DATA_DIR/test.jsonl" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir True \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$MICRO_BS" \
  --gradient_accumulation_steps "$GA" \
  --save_strategy "$SAVE_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 2 \
  --learning_rate "$LR" \
  --max_grad_norm 0.5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps "$LOGGING_STEPS" \
  --gradient_checkpointing True \
  --evaluation_strategy "$EVAL_STRATEGY" \
  --per_device_eval_batch_size "$MICRO_BS" \
  --prediction_loss_only True \
  --eval_accumulation_steps 2 \
  --bf16 True \
  --use_flash_attn True \
  --report_to none \
  --run_name "phase9_ns_only_bottomp90_${SLURM_JOB_ID}" \
  --loss ns_only \
  --ns_type bottom_p --ns_bottom_p 0.9 \
  --ns_alpha 1.0 \
  --ns_temperature 1.0 \
  --include_tokens_per_second True --include_num_input_tokens_seen True