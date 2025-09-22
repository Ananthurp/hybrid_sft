#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 20:00:00
#SBATCH -J EVAL_STD
#SBATCH -o EVAL_STD-%j.out
#SBATCH -e EVAL_STD-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# --- Node-local caches ---
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
export VLLM_CACHE_DIR="$LOCAL_SCRATCH/vllm"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$VLLM_CACHE_DIR"
df -h "$LOCAL_SCRATCH" || true

# Clear stale cached JSON schemas (defensive)
CACHE_ROOT="$HF_DATASETS_CACHE"
rm -rf "$CACHE_ROOT/json" || true

# Misc runtime env
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

# --- Inputs (pass via --export) ---
: "${MODEL_DIR:?set path to trained model dir}"
: "${RUN_NAME:?set human-readable run name}"
: "${OUT_DIR:?set where to write eval outputs}"

# --- Optional with defaults ---
USE_VLLM="${USE_VLLM:-True}"
N_RESP="${N_RESP:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-2}"
MAX_SIZE="${MAX_SIZE:-}"  # empty = full Alpaca
BASE_TOKENIZER_DIR="${BASE_TOKENIZER_DIR:-$MODEL_DIR}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"

POEM_MAX="${POEM_MAX:-300}"
STORY_MAX="${STORY_MAX:-300}"
POEM_N="${POEM_N:-8}"
STORY_N="${STORY_N:-8}"
GEN_TOK_SHORT="${GEN_TOK_SHORT:-256}"

# --- Paths ---
mkdir -p "$OUT_DIR"
EVAL_DIR_ALP="$OUT_DIR/evaluation_chat_alpaca"
EVAL_DIR_POEM="$OUT_DIR/evaluation_poem"
EVAL_DIR_STORY="$OUT_DIR/evaluation_story"
mkdir -p "$EVAL_DIR_ALP" "$EVAL_DIR_POEM" "$EVAL_DIR_STORY"

# Ensure clean JSONs
rm -f "$EVAL_DIR_ALP/responses.json" "$EVAL_DIR_POEM/responses.json" "$EVAL_DIR_STORY/responses.json"

# Optional max-size for quick runs
GEN_EXTRA=()
if [[ -n "${MAX_SIZE}" ]]; then
  GEN_EXTRA+=(--max_size "$MAX_SIZE")
fi

# --- 1) AlpacaEval generation (vLLM) ---
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
  --remove_old "${GEN_EXTRA[@]}"

# --- 2) Reward scoring (+ winrate>0, best/mean-of-n); force SDPA for RM ---
REWARD_PY="$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py"
if ! grep -q 'attn_implementation="sdpa"' "$REWARD_PY"; then
  cp "$REWARD_PY" "${REWARD_PY}.bak.$SLURM_JOB_ID" || true
  python - <<'PY'
from pathlib import Path, re
p = Path("/home/users/astar/cfar/stuananthu/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py")
s = p.read_text()
pat = r"AutoModelForSequenceClassification\.from_pretrained\(\s*args\.model_name_or_path\s*,"
rep = 'AutoModelForSequenceClassification.from_pretrained(\n        args.model_name_or_path,\n        attn_implementation="sdpa",'
s2, n = re.subn(pat, rep, s, count=1)
if n != 1:
    print("[warn] SDPA patch may already be present or pattern changed; proceeding.")
else:
    p.write_text(s2)
    print("[ok] Patched evaluation_reward.py to use attn_implementation='sdpa'")
PY
fi

rm -f "$EVAL_DIR_ALP/rewards.json" "$EVAL_DIR_ALP/reward_summary.json"
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --tokenizer_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --data_path "$EVAL_DIR_ALP/responses.json" \
  --batch_size "$RM_BATCH_SIZE" \
  --save_path "$EVAL_DIR_ALP/rewards.json" \
  --summary_path "$EVAL_DIR_ALP/reward_summary.json" \
  --baseline_json "$BASELINE_JSON"

# --- 3) Bradley–Terry winrate vs GPT-4 ---
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

# --- 4) Diversity metrics on Alpaca ---
python ~/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py \
  --response_path "$EVAL_DIR_ALP/responses.json" \
  --tokenizer_path "$BASE_TOKENIZER_DIR" \
  --detokenizer_path "$BASE_TOKENIZER_DIR" \
  --out_path "$EVAL_DIR_ALP/diversity_summary.json" \
  > "$EVAL_DIR_ALP/diversity_metrics.log" 2>&1 || \
  echo "[warn] Alpaca diversity failed; see $EVAL_DIR_ALP/diversity_metrics.log"

# --- 5) Poem gen + diversity (capped) ---
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
  > "$EVAL_DIR_POEM/diversity_metrics.log" 2>&1 || \
  echo "[warn] Poem diversity failed; see $EVAL_DIR_POEM/diversity_metrics.log"

# --- 6) Story gen + diversity (capped) ---
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
  > "$EVAL_DIR_STORY/diversity_metrics.log" 2>&1 || \
  echo "[warn] Story diversity failed; see $EVAL_DIR_STORY/diversity_metrics.log"

echo "[done] Outputs under: $OUT_DIR"