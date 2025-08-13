#!/bin/bash
set -euo pipefail
set -x

# Usage:
#   bash scripts/qwen2-1.5b/eval/run_bt_winrate.sh <MODEL_DIR> <GPT_BASELINE_JSON>
# Requires: MODEL_DIR/evaluation_chat_alpaca/generated_responses.json

MODEL_DIR="$1"
GPT_BASELINE_JSON="$2"
OUTDIR="${MODEL_DIR}/evaluation_chat_alpaca"
CAND_JSON="${OUTDIR}/generated_responses.json"
OUTFILE="${OUTDIR}/bt_winrate_vs_gpt.json"

if [ ! -f "${CAND_JSON}" ]; then
  echo "Error: ${CAND_JSON} not found. Run Alpaca generation first."
  exit 1
fi

python scripts/qwen2-1.5b/eval/bt_winrate.py \
  --candidate_json "${CAND_JSON}" \
  --baseline_json "${GPT_BASELINE_JSON}" \
  --out_file "${OUTFILE}"
