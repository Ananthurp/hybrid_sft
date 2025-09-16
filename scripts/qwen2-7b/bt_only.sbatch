#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p testqueue                 # override with -p l40sn on submit if needed
#SBATCH --gres=gpu:h200:1            # override on submit if needed
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J BT_ONLY
#SBATCH -o BT_ONLY-%j.out
#SBATCH -e BT_ONLY-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- Inputs (all overridable via --export=ALL,VAR=...) ---
CAND_JSON="${CAND_JSON:?set path to candidate responses.json}"
OUT_JSON="${OUT_JSON:?set path to write BT summary json}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"
DEBUG_JSON="${DEBUG_JSON:-}"                     # optional
REWARD_MODEL="${REWARD_MODEL:-sfairXC/FsfairX-LLaMA3-RM-v0.1}"
DTYPE="${DTYPE:-bfloat16}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-4096}"
BUDGETS="${BUDGETS:-2,4,8,16,32}"
LIMIT="${LIMIT:-1000000}"

# --- Sanity checks ---
[[ -f "$CAND_JSON" ]] || { echo "Missing CAND_JSON: $CAND_JSON"; exit 1; }
[[ -f "$BASELINE_JSON" ]] || { echo "Missing BASELINE_JSON: $BASELINE_JSON"; exit 1; }
mkdir -p "$(dirname "$OUT_JSON")"
if [[ -n "$DEBUG_JSON" ]]; then mkdir -p "$(dirname "$DEBUG_JSON")"; fi

# Optional guard: warn if tokenizer call in BT script still adds a generation prompt
if grep -q "add_generation_prompt=True" "$HOME/LLMDiversity/hybrid_sft/evaluation/bt_winrate_by_index.py"; then
  echo "[WARN] bt_winrate_by_index.py appears to use add_generation_prompt=True; this will corrupt RM scores."
fi

# --- Run BT comparison (reward-model only; no generation) ---
CMD=( python "$HOME/LLMDiversity/hybrid_sft/evaluation/bt_winrate_by_index.py"
  --candidate_json "$CAND_JSON"
  --baseline_json "$BASELINE_JSON"
  --reward_model "$REWARD_MODEL"
  --dtype "$DTYPE"
  --batch_size "$RM_BATCH_SIZE"
  --max_len "$MAX_LEN"
  --budgets "$BUDGETS"
  --limit "$LIMIT"
  --out_file "$OUT_JSON"
)
if [[ -n "$DEBUG_JSON" ]]; then
  CMD+=( --debug_file "$DEBUG_JSON" )
fi

"${CMD[@]}"