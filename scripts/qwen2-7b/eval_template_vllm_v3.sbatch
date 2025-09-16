#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 18:00:00
#SBATCH -J EVAL_TEMPLATE
#SBATCH -o EVAL_TEMPLATE-%j.out
#SBATCH -e EVAL_TEMPLATE-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Inputs ---
MODEL_DIR="${MODEL_DIR:?set path to trained model dir}"
RUN_NAME="${RUN_NAME:?set human-readable run name}"
OUT_DIR="${OUT_DIR:?set where to write eval outputs}"

# Optional inputs
USE_VLLM="${USE_VLLM:-True}"
N_RESP="${N_RESP:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"          # generation batch (vLLM)
RM_BATCH_SIZE="${RM_BATCH_SIZE:-2}"    # reward model batch (Transformers)
MAX_SIZE="${MAX_SIZE:-}"
BASE_TOKENIZER_DIR="${BASE_TOKENIZER_DIR:-$MODEL_DIR}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"

# Paths
mkdir -p "$OUT_DIR"
EVAL_DIR_ALP="$OUT_DIR/evaluation_chat_alpaca"
EVAL_DIR_POEM="$OUT_DIR/evaluation_poem"
EVAL_DIR_STORY="$OUT_DIR/evaluation_story"
mkdir -p "$EVAL_DIR_ALP" "$EVAL_DIR_POEM" "$EVAL_DIR_STORY"

export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

GEN_EXTRA=()
if [[ -n "${MAX_SIZE}" ]]; then
  GEN_EXTRA+=(--max_size "$MAX_SIZE")
fi

# 1) Generate responses (AlpacaEval)
python ~/LLMDiversity/hybrid_sft/evaluation/generate_response.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --dataset_path "tatsu-lab/alpaca_eval" \
  --split eval \
  --n "$N_RESP" \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens 1024 \
  --save_path "$EVAL_DIR_ALP/responses.json" \
  --remove_old True \
  "${GEN_EXTRA[@]}"

# 2) Reward scoring (+ summary)
rm -f "$EVAL_DIR_ALP/rewards.json" "$EVAL_DIR_ALP/reward_summary.json"
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --data_path "$EVAL_DIR_ALP/responses.json" \
  --batch_size "$RM_BATCH_SIZE" \
  --save_path "$EVAL_DIR_ALP/rewards.json" \
  --summary_path "$EVAL_DIR_ALP/reward_summary.json" \
  --tokenizer_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --baseline_json "$BASELINE_JSON"

# 3) Bradley–Terry vs GPT-4
if [[ -f "$BASELINE_JSON" ]]; then
  python ~/LLMDiversity/hybrid_sft/evaluation/bt_winrate_by_index.py \
    --candidate_json "$EVAL_DIR_ALP/responses.json" \
    --baseline_json "$BASELINE_JSON" \
    --reward_model "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --dtype bfloat16 \
    --batch_size "$RM_BATCH_SIZE" \
    --max_len 4096 \
    --out_file "$EVAL_DIR_ALP/bt_winrate_vs_gpt.json"
else
  echo "[warn] Baseline not found at '$BASELINE_JSON'; skipping BT step."
fi

# 4) Diversity metrics
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_ALP/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_ALP/diversity_summary.json" \
  > "$EVAL_DIR_ALP/diversity_metrics.log" 2>&1 \
  || echo "[warn] Alpaca diversity failed (see $EVAL_DIR_ALP/diversity_metrics.log); continuing."

# 5) Poem generation + diversity
python ~/LLMDiversity/hybrid_sft/evaluation/generate_response.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/poem_generation" \
  --split test \
  --n 16 \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens 512 \
  --save_path "$EVAL_DIR_POEM/responses.json" \
  --remove_old True \
  "${GEN_EXTRA[@]}"

python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_POEM/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_POEM/diversity_summary.json" \
  > "$EVAL_DIR_POEM/diversity_metrics.log" 2>&1 \
  || echo "[warn] Poem diversity failed (see $EVAL_DIR_POEM/diversity_metrics.log); continuing."

# 6) Story generation + diversity
python ~/LLMDiversity/hybrid_sft/evaluation/generate_response.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/story_generation" \
  --split test \
  --n 16 \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens 512 \
  --save_path "$EVAL_DIR_STORY/responses.json" \
  --remove_old True \
  "${GEN_EXTRA[@]}"

python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_STORY/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_STORY/diversity_summary.json" \
  > "$EVAL_DIR_STORY/diversity_metrics.log" 2>&1 \
  || echo "[warn] Story diversity failed (see $EVAL_DIR_STORY/diversity_metrics.log); continuing."