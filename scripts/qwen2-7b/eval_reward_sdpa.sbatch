#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J REWARD_SDPA
#SBATCH -o REWARD_SDPA-%j.out
#SBATCH -e REWARD_SDPA-%j.err
set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120_spmx
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# --- Node-local caches ---
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

# --- Required inputs (from --export) ---
: "${DATA_JSON:?path to Alpaca responses.json}"
: "${SAVE_JSON:?where to write rewards.json}"
: "${SUMMARY_JSON:?where to write reward_summary.json}"
BASELINE_JSON="${BASELINE_JSON:-$HOME/LLMDiversity/assets/gpt4_alpacaeval_responses.json}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-2}"

# --- Force SDPA for the reward model (patch once, safe to re-run) ---
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
assert n == 1, "Could not patch evaluation_reward.py"
p.write_text(s2)
print("[ok] evaluation_reward.py patched to use attn_implementation='sdpa'")
PY
fi

# --- Run reward scoring ---
python -u "$REWARD_PY" \
  --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --tokenizer_path   "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
  --data_path   "$DATA_JSON" \
  --batch_size  "$RM_BATCH_SIZE" \
  --save_path   "$SAVE_JSON" \
  --summary_path "$SUMMARY_JSON" \
  --baseline_json "$BASELINE_JSON" \
  2>&1 | tee "${SAVE_JSON%.json}.log"

echo "[done] wrote:"
ls -lh "$SAVE_JSON" "$SUMMARY_JSON" || true