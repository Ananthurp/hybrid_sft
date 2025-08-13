
#!/bin/bash
# Compute Bradley–Terry winrate vs GPT-4 for already-generated Alpaca eval outputs.
# Usage:
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/qwen2-1.5b/eval/run_bt_winrate_for_existing.sh \
#     <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON> <RUN_1> [RUN_2 ...]
#
# Optional ENV overrides:
#   BT_DTYPE=bfloat16|float16|float32    (default: bfloat16)
#   BT_BATCH_SIZE=<int>                  (default: 8)
#   BT_MAX_LEN=<int>                     (default: 4096)

set -euo pipefail
set -x

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON> <RUN_1> [RUN_2 ...]" >&2
  exit 1
fi

BASE_RESULTS_DIR="$1"; shift
BASELINE_JSON="$1"; shift
RUNS=("$@")

BT_SCRIPT="scripts/qwen2-1.5b/eval/bt_winrate_v2.py"
[ -f "${BT_SCRIPT}" ] || { echo "!!! Missing ${BT_SCRIPT}"; exit 2; }
[ -f "${BASELINE_JSON}" ] || { echo "!!! Missing baseline ${BASELINE_JSON}"; exit 3; }

# Defaults (can be overridden by env)
BT_DTYPE="${BT_DTYPE:-bfloat16}"
BT_BATCH_SIZE="${BT_BATCH_SIZE:-8}"
BT_MAX_LEN="${BT_MAX_LEN:-4096}"

for RUN in "${RUNS[@]}"; do
  CAND="${BASE_RESULTS_DIR}/${RUN}/output_dir_consolidated/evaluation_chat_alpaca/generated_responses.json"
  OUT_DIR="$(dirname "${CAND}")"
  OUT="${OUT_DIR}/bt_winrate_vs_gpt.json"
  DBG="${OUT_DIR}/bt_debug_unmatched.json"

  if [ ! -f "${CAND}" ]; then
    echo "--- Skip ${RUN}: not found ${CAND}"
    continue
  fi

  echo "--- BT winrate for ${RUN} ---"
  python "${BT_SCRIPT}" \
    --candidate_json "${CAND}" \
    --baseline_json "${BASELINE_JSON}" \
    --out_file "${OUT}" \
    --debug_unmatched "${DBG}" \
    --dtype "${BT_DTYPE}" \
    --batch_size "${BT_BATCH_SIZE}" \
    --max_len "${BT_MAX_LEN}"
done

echo "--- Done computing BT winrates ---"
