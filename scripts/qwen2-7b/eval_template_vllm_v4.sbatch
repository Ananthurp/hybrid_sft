#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 20:00:00
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

# --- Use NODE-LOCAL ephemeral storage for all heavy caches + tmp ---
# This avoids the shared /scratch which is currently full.
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"

export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
export VLLM_CACHE_DIR="$LOCAL_SCRATCH/vllm"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$VLLM_CACHE_DIR"

# Optional: show free space on the node-local disk
df -h "$LOCAL_SCRATCH" || true

# Clear stale cached schemas for local JSON loads (avoids 'dataclass' TypeError)
CACHE_ROOT="$HF_DATASETS_CACHE"
echo "[preflight] clearing cached JSON datasets under: $CACHE_ROOT/json"
rm -rf "$CACHE_ROOT/json" || true

# Misc runtime env
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Inputs (required) ---
MODEL_DIR="${MODEL_DIR:?set path to trained model dir}"
RUN_NAME="${RUN_NAME:?set human-readable run name}"
OUT_DIR="${OUT_DIR:?set where to write eval outputs (choose a location with space, e.g. under \$HOME)}"

# --- Optional inputs with sane defaults ---
USE_VLLM="${USE_VLLM:-True}"                 # "True"/"False"
N_RESP="${N_RESP:-32}"                        # responses per prompt (Alpaca)
BATCH_SIZE="${BATCH_SIZE:-8}"                 # generation batch (for vLLM or HF)
RM_BATCH_SIZE="${RM_BATCH_SIZE:-2}"           # reward model batch (Transformers)
MAX_SIZE="${MAX_SIZE:-}"                      # leave empty to run full Alpaca
BASE_TOKENIZER_DIR="${BASE_TOKENIZER_DIR:-$MODEL_DIR}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"

# Caps for Poem/Story diversity sets (keeps runtime reasonable)
POEM_MAX="${POEM_MAX:-300}"
STORY_MAX="${STORY_MAX:-300}"
POEM_N="${POEM_N:-8}"
STORY_N="${STORY_N:-8}"
GEN_TOK_SHORT="${GEN_TOK_SHORT:-256}"

# Paths
mkdir -p "$OUT_DIR"
EVAL_DIR_ALP="$OUT_DIR/evaluation_chat_alpaca"
EVAL_DIR_POEM="$OUT_DIR/evaluation_poem"
EVAL_DIR_STORY="$OUT_DIR/evaluation_story"
mkdir -p "$EVAL_DIR_ALP" "$EVAL_DIR_POEM" "$EVAL_DIR_STORY"

# Ensure we start from clean JSON files (previous ENOSPC can leave them truncated)
rm -f "$EVAL_DIR_ALP/responses.json" "$EVAL_DIR_POEM/responses.json" "$EVAL_DIR_STORY/responses.json"

# vLLM port scaffolding (harmless for HF fallback)
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

# Optional args for generation (used only when MAX_SIZE is set; Alpaca stays full)
GEN_EXTRA=()
if [[ -n "${MAX_SIZE}" ]]; then
  GEN_EXTRA+=(--max_size "$MAX_SIZE")
fi

# --------------------------------------------------------
# 1) Generate responses (AlpacaEval) — FULL 805 prompts
# --------------------------------------------------------
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
  --remove_old

# --------------------------------------------------------
# 2) Reward scoring (+ summary of winrate>0 / best-of-n / mean-of-n)
# --------------------------------------------------------
rm -f "$EVAL_DIR_ALP/rewards.json" "$EVAL_DIR_ALP/reward_summary.json"
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --tokenizer_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --data_path "$EVAL_DIR_ALP/responses.json" \
  --batch_size "$RM_BATCH_SIZE" \
  --save_path "$EVAL_DIR_ALP/rewards.json" \
  --summary_path "$EVAL_DIR_ALP/reward_summary.json" \
  --baseline_json "$BASELINE_JSON"

# --------------------------------------------------------
# 3) Bradley–Terry vs GPT-4 (AlpacaEval) — force full compare
# --------------------------------------------------------
if [[ -f "$BASELINE_JSON" ]]; then
  python ~/LLMDiversity/hybrid_sft/evaluation/bt_winrate_by_index.py \
    --candidate_json "$EVAL_DIR_ALP/responses.json" \
    --baseline_json "$BASELINE_JSON" \
    --reward_model "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --dtype bfloat16 \
    --batch_size "$RM_BATCH_SIZE" \
    --max_len 4096 \
    --budgets "2,4,8,16,32" \
    --limit 1000000 \
    --out_file "$EVAL_DIR_ALP/bt_winrate_vs_gpt.json" \
    --debug_file "$EVAL_DIR_ALP/bt_mismatches_debug.json"
else
  echo "[warn] Baseline not found at '$BASELINE_JSON'; skipping BT step."
fi

# --------------------------------------------------------
# 4) Diversity metrics on Alpaca prompts
# --------------------------------------------------------
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_ALP/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_ALP/diversity_summary.json" \
  > "$EVAL_DIR_ALP/diversity_metrics.log" 2>&1 \
  || echo "[warn] Alpaca diversity failed (see $EVAL_DIR_ALP/diversity_metrics.log); continuing."

# --------------------------------------------------------
# 5) Poem generation + diversity (CAPPED)
# --------------------------------------------------------
python ~/LLMDiversity/hybrid_sft/evaluation/generate_response.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/poem_generation" \
  --split test \
  --n "$POEM_N" \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens "$GEN_TOK_SHORT" \
  --save_path "$EVAL_DIR_POEM/responses.json" \
  --remove_old \
  --max_size "$POEM_MAX"

python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_POEM/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_POEM/diversity_summary.json" \
  > "$EVAL_DIR_POEM/diversity_metrics.log" 2>&1 \
  || echo "[warn] Poem diversity failed (see $EVAL_DIR_POEM/diversity_metrics.log); continuing."

# --------------------------------------------------------
# 6) Story generation + diversity (CAPPED)
# --------------------------------------------------------
python ~/LLMDiversity/hybrid_sft/evaluation/generate_response.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --dataset_path "$HOME/LLMDiversity/hybrid_sft/data/story_generation" \
  --split test \
  --n "$STORY_N" \
  --batch_size "$BATCH_SIZE" \
  --use_vllm "$USE_VLLM" \
  --max_new_tokens "$GEN_TOK_SHORT" \
  --save_path "$EVAL_DIR_STORY/responses.json" \
  --remove_old \
  --max_size "$STORY_MAX"

python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_STORY/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_STORY/diversity_summary.json" \
  > "$EVAL_DIR_STORY/diversity_metrics.log" 2>&1 \
  || echo "[warn] Story diversity failed (see $EVAL_DIR_STORY/diversity_metrics.log); continuing."