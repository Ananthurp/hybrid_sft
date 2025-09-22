#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J ALP_POST
#SBATCH -o ALP_POST-%j.out
#SBATCH -e ALP_POST-%j.err
set -euo pipefail
set -x

# --- Env (sparsemax-patched vLLM lives here, but we only need HF for RM/diversity) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120_spmx
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# Node-local caches
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" || true
df -h "$LOCAL_SCRATCH" || true

# --- Inputs from --export ---
: "${RESP_JSON:?set RESP_JSON=/path/to/evaluation_chat_alpaca/responses.json}"
: "${OUT_DIR:?set OUT_DIR=/path/to/evaluation_chat_alpaca}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH=/path/to/tokenizer (usually your base model tokenizer)}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-2}"

mkdir -p "$OUT_DIR"

# Files to write
REWARD_JSON="$OUT_DIR/rewards.json"
REWARD_SUMMARY="$OUT_DIR/reward_summary.json"
BT_JSON="$OUT_DIR/bt_winrate_vs_gpt.json"
BT_DEBUG="$OUT_DIR/bt_mismatches_debug.json"
DIV_SUMMARY="$OUT_DIR/diversity_summary.json"

# --- Force SDPA for the reward model (avoid flash-attn dependency) ---
REWARD_PY="$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py"
if ! grep -q 'attn_implementation="sdpa"' "$REWARD_PY"; then
  cp "$REWARD_PY" "${REWARD_PY}.bak.$SLURM_JOB_ID"
  python - <<'PY'
from pathlib import Path, re
p = Path("/home/users/astar/cfar/stuananthu/LLMDiversity/hybrid_sft/evaluation/evaluation_reward.py")
s = p.read_text()
pat = r"AutoModelForSequenceClassification\.from_pretrained\(\s*args\.model_name_or_path\s*,"
rep = 'AutoModelForSequenceClassification.from_pretrained(\n        args.model_name_or_path,\n        attn_implementation="sdpa",'
s2, n = re.subn(pat, rep, s, count=1)
assert n == 1, "Could not patch evaluation_reward.py (autoset SDPA)"
p.write_text(s2)
print("[ok] evaluation_reward.py patched to use attn_implementation='sdpa'")
PY
fi

# --- 2) Reward scoring ---
rm -f "$REWARD_JSON" "$REWARD_SUMMARY" || true
python -u "$REWARD_PY" \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --tokenizer_path   "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --data_path   "$RESP_JSON" \
  --batch_size  "$RM_BATCH_SIZE" \
  --save_path   "$REWARD_JSON" \
  --summary_path "$REWARD_SUMMARY" \
  --baseline_json "$BASELINE_JSON" \
  2>&1 | tee "$OUT_DIR/reward_eval.log"

# --- 3) Bradley–Terry winrate vs GPT-4 (uses RM above) ---
if [[ -f "$BASELINE_JSON" ]]; then
  python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/bt_winrate_by_index.py" \
    --candidate_json "$RESP_JSON" \
    --baseline_json  "$BASELINE_JSON" \
    --reward_model   "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --dtype bfloat16 \
    --batch_size "$RM_BATCH_SIZE" \
    --max_len 4096 \
    --budgets "2,4,8,16,32" \
    --limit 1000000 \
    --out_file "$BT_JSON" \
    --debug_file "$BT_DEBUG" \
    2>&1 | tee "$OUT_DIR/bt_eval.log"
else
  echo "[warn] Baseline not found at '$BASELINE_JSON'; skipping BT step."
fi

# --- 4) Diversity metrics on Alpaca prompts ---
python -u "$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py" \
  --response_path  "$RESP_JSON" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --detokenizer_path "$TOKENIZER_PATH" \
  --out_path "$DIV_SUMMARY" \
  > "$OUT_DIR/diversity_metrics.log" 2>&1 || \
  echo "[warn] Diversity step failed; see $OUT_DIR/diversity_metrics.log"

echo "[done] wrote:"
ls -lh "$REWARD_JSON" "$REWARD_SUMMARY" "$BT_JSON" "$DIV_SUMMARY" 2>/dev/null || true