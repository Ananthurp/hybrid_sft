# #!/usr/bin/env bash
# # Qwen2-7B training launcher (independent phases)
# # Enable exactly one phase per Slurm job via env flags:
# #   RUN_CE=1  or  RUN_GEM=1  or  RUN_HYB75=1  or  RUN_HYB50=1
# set -euo pipefail
# set -x

# # -------- Paths (yours) --------
# PROJ="$HOME/LLMDiversity"
# REPO="$PROJ/hybrid_sft"
# MODEL_DIR="$PROJ/models/Qwen2-7B"                             # ALWAYS the base model
# DATA_DIR="$PROJ/datasets/ultrafeedback_tokenized_qwen2-7b"   # train.jsonl / test.jsonl
# OUT_ROOT="$PROJ/results/qwen2-7b"
# mkdir -p "$OUT_ROOT"

# DS_CFG="$REPO/scripts/deepspeed_config_qwen.json"

# # Quick sanity
# [ -d "$MODEL_DIR" ] || { echo "Missing MODEL_DIR: $MODEL_DIR"; exit 1; }
# [ -f "$DATA_DIR/train.jsonl" ] || { echo "Missing $DATA_DIR/train.jsonl"; exit 1; }

# # -------- Caches on scratch --------
# export PRJ=prj0000000224
# export SCRATCH="/scratch/$PRJ"
# export HF_HOME="$SCRATCH/LLMDiversity_work/cache/huggingface"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
# mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# # -------- Weights & Biases (optional but recommended) --------
# export WANDB_PROJECT="${WANDB_PROJECT:-llm_hybrid_sft}"
# export WANDB_ENTITY="${WANDB_ENTITY:-}"   # set if you use a team/org
# export WANDB_DIR="$SCRATCH/LLMDiversity_work/wandb"
# mkdir -p "$WANDB_DIR"
# # pull key from secret file if present (avoids printing key)
# if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.secrets/wandb_api_key" ]; then
#   export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_api_key")"
# fi

# # -------- Runtime / perf knobs --------
# export TOKENIZERS_PARALLELISM=true
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_TIMEOUT=1800
# # helps with large-alloc fragmentation on long ctx
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# # -------- Training knobs (tunable) --------
# : "${GPUS_PER_NODE:=4}"        # 4x H200
# : "${MICRO_BS:=8}"             # per-GPU micro-batch (~4096 ctx works on H200 w/ ZeRO-3)
# : "${GA:=4}"                   # gradient accumulation
# : "${EPOCHS:=3}"               # per phase
# : "${SAVE_STRATEGY:=steps}"    # "steps" gives quicker restart points
# : "${SAVE_STEPS:=1000}"        # raise to reduce I/O; lower for more frequent checkpoints
# : "${EVAL_STRATEGY:=no}"       # "no" for max throughput
# : "${EVAL_STEPS:=0}"
# : "${LOGGING_STEPS:=10}"
# : "${LR:=2e-6}"
# : "${MAX_STEPS:=0}"            # 0 = use epochs; >0 overrides epochs
# : "${MAX_TRAIN_SAMPLES:=0}"    # 0 = full dataset
# : "${REPORT_TO:=wandb}"        # "wandb" or "none"

# # Phase toggles (default off; set exactly one to 1 in your sbatch):
# : "${RUN_CE:=0}"
# : "${RUN_GEM:=0}"
# : "${RUN_HYB75:=0}"
# : "${RUN_HYB50:=0}"

# # Hugging Face throughput metrics
# INCLUDE_TOKENS_FLAGS=(--include_tokens_per_second True --include_num_input_tokens_seen True)

# LAUNCH() {
#   local outdir="$1" ; shift
#   local phase="$1"  ; shift
#   local run_name="${phase}_bs${MICRO_BS}x${GA}x${GPUS_PER_NODE}"

#   local overrides=()
#   if [ "${MAX_STEPS}" -gt 0 ]; then
#     overrides+=(--max_steps "$MAX_STEPS")
#   fi
#   if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
#     overrides+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
#   fi

#   # IMPORTANT: always start from base model for every independent run
#   local BASE_MODEL="$MODEL_DIR"

#   deepspeed --num_gpus "$GPUS_PER_NODE" "$REPO/train.py" \
#     --deepspeed "$DS_CFG" \
#     --seed 1234 \
#     --model_name_or_path "$BASE_MODEL" \
#     --train_tokenized_file "$DATA_DIR/train.jsonl" \
#     --test_tokenized_file  "$DATA_DIR/test.jsonl" \
#     --output_dir "$outdir" \
#     --overwrite_output_dir True \
#     --num_train_epochs "$EPOCHS" \
#     --per_device_train_batch_size "$MICRO_BS" \
#     --gradient_accumulation_steps "$GA" \
#     --save_strategy "$SAVE_STRATEGY" \
#     --save_steps "$SAVE_STEPS" \
#     --save_total_limit 2 \
#     --learning_rate "$LR" \
#     --max_grad_norm 0.5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_steps "$LOGGING_STEPS" \
#     --gradient_checkpointing True \
#     --evaluation_strategy "$EVAL_STRATEGY" \
#     --eval_steps "$EVAL_STEPS" \
#     --per_device_eval_batch_size "$MICRO_BS" \
#     --prediction_loss_only True \
#     --eval_accumulation_steps 2 \
#     --bf16 True \
#     --use_flash_attn True \
#     --report_to "$REPORT_TO" \
#     --run_name "$run_name" \
#     "${INCLUDE_TOKENS_FLAGS[@]}" \
#     "$@" \
#     "${overrides[@]}"
# }

# # -------- PHASES (independent) --------

# if [ "$RUN_CE" -eq 1 ]; then
#   OUT_CE="$OUT_ROOT/phase1_ce"
#   mkdir -p "$OUT_CE"
#   LAUNCH "$OUT_CE" phase1_ce --loss ce
# fi

# if [ "$RUN_GEM" -eq 1 ]; then
#   OUT_GEM="$OUT_ROOT/phase2_gem"
#   mkdir -p "$OUT_GEM"
#   LAUNCH "$OUT_GEM" phase2_gem --loss gem --gem_beta 0.7 --gem_h linear
# fi

# if [ "$RUN_HYB75" -eq 1 ]; then
#   OUT_HYB75="$OUT_ROOT/phase3_hybrid_alpha0.75"
#   mkdir -p "$OUT_HYB75"
#   LAUNCH "$OUT_HYB75" phase3_hybrid_a075 \
#     --loss hybrid --ns_type support_set --ns_alpha 0.75 --ns_temperature 1.0
# fi

# if [ "$RUN_HYB50" -eq 1 ]; then
#   OUT_HYB50="$OUT_ROOT/phase4_hybrid_alpha0.5"
#   mkdir -p "$OUT_HYB50"
#   LAUNCH "$OUT_HYB50" phase4_hybrid_a050 \
#     --loss hybrid --ns_type support_set --ns_alpha 0.5 --ns_temperature 1.0
# fi

# echo "Selected phase(s) finished."



#!/usr/bin/env bash
# Qwen2-7B training launcher (independent phases)
# Enable exactly one phase per Slurm job via env flags:
#   RUN_CE=1  or  RUN_GEM=1  or  RUN_HYB75=1  or  RUN_HYB50=1
set -euo pipefail
set -x

# -------- Paths (yours) --------
PROJ="$HOME/LLMDiversity"
REPO="$PROJ/hybrid_sft"
MODEL_DIR="$PROJ/models/Qwen2-7B"                             # ALWAYS the base model
DATA_DIR="$PROJ/datasets/ultrafeedback_tokenized_qwen2-7b"   # train.jsonl / test.jsonl
OUT_ROOT="$PROJ/results/qwen2-7b"
mkdir -p "$OUT_ROOT"

DS_CFG="$REPO/scripts/deepspeed_config_qwen.json"

# Quick sanity
[ -d "$MODEL_DIR" ] || { echo "Missing MODEL_DIR: $MODEL_DIR"; exit 1; }
[ -f "$DATA_DIR/train.jsonl" ] || { echo "Missing $DATA_DIR/train.jsonl"; exit 1; }
[ -f "$DATA_DIR/test.jsonl" ]  || { echo "Missing $DATA_DIR/test.jsonl";  exit 1; }

# -------- Caches on scratch --------
export PRJ=prj0000000224
export SCRATCH="/scratch/$PRJ"
export HF_HOME="$SCRATCH/LLMDiversity_work/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Clean possibly stale HF datasets cache for local JSON loads
CACHE_ROOT="${HF_DATASETS_CACHE:-/scratch/$PRJ/LLMDiversity_work/cache/huggingface/datasets}"
echo "[preflight] clearing cached JSON datasets under: $CACHE_ROOT/json"
rm -rf "$CACHE_ROOT/json" || true

# -------- Weights & Biases --------
export WANDB_PROJECT="${WANDB_PROJECT:-llm_hybrid_sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="$SCRATCH/LLMDiversity_work/wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.secrets/wandb_api_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_api_key")"
fi

# -------- Runtime / perf knobs --------
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# -------- Training knobs --------
: "${GPUS_PER_NODE:=4}"
: "${MICRO_BS:=8}"
: "${GA:=4}"
: "${EPOCHS:=3}"
: "${SAVE_STRATEGY:=steps}"
: "${SAVE_STEPS:=1000}"
: "${EVAL_STRATEGY:=no}"
: "${EVAL_STEPS:=0}"
: "${LOGGING_STEPS:=10}"
: "${LR:=2e-6}"
: "${MAX_STEPS:=0}"
: "${MAX_TRAIN_SAMPLES:=0}"
: "${REPORT_TO:=wandb}"

# Phase toggles
: "${RUN_CE:=0}"
: "${RUN_GEM:=0}"
: "${RUN_HYB75:=0}"
: "${RUN_HYB50:=0}"
: "${RUN_SPARSEMAX:=0}"
: "${RUN_CE_WD:=0}"
: "${RUN_NEFT:=0}"

# HF throughput metrics
INCLUDE_TOKENS_FLAGS=(--include_tokens_per_second True --include_num_input_tokens_seen True)

LAUNCH() {
  local outdir="$1" ; shift
  local phase="$1"  ; shift

  # unique run name per job to avoid W&B collisions
  local run_suffix="${SLURM_JOB_ID:-$RANDOM}"
  local run_name="${phase}_bs${MICRO_BS}x${GA}x${GPUS_PER_NODE}_j${run_suffix}"

  local overrides=()
  if [ "${MAX_STEPS}" -gt 0 ]; then
    overrides+=(--max_steps "$MAX_STEPS")
  fi
  if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
    overrides+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
  fi

  # Use a custom master port if provided by the sbatch wrapper
  local ds_extra=()
  if [ -n "${DS_MASTER_PORT:-}" ]; then
    ds_extra+=(--master_port "${DS_MASTER_PORT}")
  elif [ -n "${MASTER_PORT:-}" ]; then
    ds_extra+=(--master_port "${MASTER_PORT}")
  fi

  # Always start from the base model for independent runs
  local BASE_MODEL="$MODEL_DIR"

  # Optional resume support (RESUME_FROM=last or a path)
  if [ -n "${RESUME_FROM:-}" ]; then
    if [ "$RESUME_FROM" = "last" ]; then
      last_ckpt="$(ls -1d "$outdir"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
      if [ -n "$last_ckpt" ]; then
        overrides+=(--resume_from_checkpoint "$last_ckpt")
        echo "Resuming from last checkpoint: $last_ckpt"
      else
        echo "No checkpoints found in $outdir; starting fresh."
      fi
    else
      overrides+=(--resume_from_checkpoint "$RESUME_FROM")
      echo "Resuming from: $RESUME_FROM"
    fi
  fi

  deepspeed "${ds_extra[@]}" --num_gpus "$GPUS_PER_NODE" "$REPO/train.py" \
    --deepspeed "$DS_CFG" \
    --seed 1234 \
    --model_name_or_path "$BASE_MODEL" \
    --train_tokenized_file "$DATA_DIR/train.jsonl" \
    --test_tokenized_file  "$DATA_DIR/test.jsonl" \
    --output_dir "$outdir" \
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
    --eval_steps "$EVAL_STEPS" \
    --per_device_eval_batch_size "$MICRO_BS" \
    --prediction_loss_only True \
    --eval_accumulation_steps 2 \
    --bf16 True \
    --use_flash_attn True \
    --report_to "$REPORT_TO" \
    --run_name "$run_name" \
    "${INCLUDE_TOKENS_FLAGS[@]}" \
    "$@" \
    "${overrides[@]}"
}

# -------- PHASES (independent) --------
if [ "$RUN_CE" -eq 1 ]; then
  OUT_CE="$OUT_ROOT/phase1_ce"
  mkdir -p "$OUT_CE"
  LAUNCH "$OUT_CE" phase1_ce --loss ce
fi

if [ "$RUN_GEM" -eq 1 ]; then
  OUT_GEM="$OUT_ROOT/phase2_gem"
  mkdir -p "$OUT_GEM"
  LAUNCH "$OUT_GEM" phase2_gem --loss gem --gem_beta 0.7 --gem_h linear
fi

if [ "$RUN_HYB75" -eq 1 ]; then
  OUT_HYB75="$OUT_ROOT/phase3_hybrid_alpha0.75"
  mkdir -p "$OUT_HYB75"
  LAUNCH "$OUT_HYB75" phase3_hybrid_a075 \
    --loss hybrid --ns_type support_set --ns_alpha 0.75 --ns_temperature 1.0
fi

if [ "$RUN_HYB50" -eq 1 ]; then
  OUT_HYB50="$OUT_ROOT/phase4_hybrid_alpha0.5"
  mkdir -p "$OUT_HYB50"
  LAUNCH "$OUT_HYB50" phase4_hybrid_a050 \
    --loss hybrid --ns_type support_set --ns_alpha 0.5 --ns_temperature 1.0
fi

if [ "$RUN_SPARSEMAX" -eq 1 ]; then
  OUT_SPARSEMAX="$OUT_ROOT/phase5_sparsemax"
  mkdir -p "$OUT_SPARSEMAX"
  LAUNCH "$OUT_SPARSEMAX" phase5_sparsemax \
    --loss hybrid --ns_type support_set --ns_alpha 0 --ns_temperature 1.0
fi

if [ "$RUN_CE_WD" -eq 1 ]; then
  OUT_CE_WD="$OUT_ROOT/phase6_ce_wd"
  mkdir -p "$OUT_CE_WD"
  LAUNCH "$OUT_CE_WD" phase6_ce_wd \
    --loss ce \
    --weight_decay 0.1
fi

if [ "$RUN_NEFT" -eq 1 ]; then
  OUT_NEFT="$OUT_ROOT/phase7_neft_alpha5"
  mkdir -p "$OUT_NEFT"
  LAUNCH "$OUT_NEFT" phase7_neft_a5 \
    --loss ce \
    --neft_alpha 5
fi

echo "Selected phase(s) finished."