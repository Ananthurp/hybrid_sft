#!/bin/bash
# Full eval: Alpaca (BT win-rate vs GPT) + Poem/Story diversity (GEM evaluators only)
# Usage:
#   bash scripts/qwen2-1.5b/eval/run_all_evaluations.sh \
#     /data/ananthu/gem_project/results \
#     /data/ananthu/gem_project/assets/gpt4_alpacaeval_responses.json \
#     qwen_1.5b_ce_run qwen_1.5b_gem_run hybrid_alpha0.5_epoch3_topk3 ...

set -euo pipefail
set -x

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <BASE_RESULTS_DIR> <GPT_BASELINE_JSON> <RUN_NAME_1> [RUN_NAME_2 ...]"
  exit 1
fi

BASE_RESULTS_DIR="$1"; shift
GPT_BASELINE_JSON="$1"; shift
RUNS=("$@")

# Pick a free GPU; change as needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# Poem/Story datasets (GEM creative-writing pattern)
POEM_DATA="data/poem_generation/test.jsonl"
STORY_DATA="data/story_generation/test.jsonl"
PROMPT_KEY="instruction"

# Generation settings (GEM-like)
ALPACA_N=32
ALPACA_TEMP=0.6
VLLM_UTIL=0.7
MAX_SIZE=200

POEM_N=16
POEM_TEMP=1.0
STORY_N=16
STORY_TEMP=1.0

consolidate_checkpoint () {
  local EXP_DIR="$1"
  local OUT_DIR="${EXP_DIR}/output_dir_consolidated"
  mkdir -p "${OUT_DIR}"

  # Find latest checkpoint-* dir if present
  local FINAL_CKPT=""
  FINAL_CKPT=$(ls -d "${EXP_DIR}"/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1 || true)

  if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
    >&2 echo "--- Consolidating ${FINAL_CKPT} -> ${OUT_DIR} ---"
    # Redirect python output to stderr so stdout only contains the path we echo below
    python "${CONVERSION_SCRIPT_PATH}" "${FINAL_CKPT}" "${OUT_DIR}" 1>&2
  else
    >&2 echo "--- No checkpoint-* in ${EXP_DIR}; copying directory to ${OUT_DIR} as fallback ---"
    rsync -a --delete "${EXP_DIR}/" "${OUT_DIR}/" 1>&2 || true
  fi

  # Ensure tokenizer/config present
  cp -n ${BASE_MODEL_PATH}/*.json "${OUT_DIR}/" 2>/dev/null || true
  cp -n ${BASE_MODEL_PATH}/*.txt  "${OUT_DIR}/" 2>/dev/null || true

  # IMPORTANT: echo ONLY the path to stdout
  echo "${OUT_DIR}"
}

generate_alpaca () {
  local MODEL_DIR="$1"
  local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
  mkdir -p "${SAVE_DIR}"
  python evaluation/generate_response.py \
    --model_name_or_path "${MODEL_DIR}" \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --dataset_path "tatsu-lab/alpaca_eval" \
    --dataset_split "eval" \
    --prompt_key "instruction" \
    --max_size ${MAX_SIZE} \
    --n ${ALPACA_N} \
    --temperature ${ALPACA_TEMP} \
    --use_vllm True \
    --vllm_gpu_memory_utilization ${VLLM_UTIL} \
    --save_path "${SAVE_DIR}/generated_responses.json"
}

generate_creative () {
  local MODEL_DIR="$1"
  local DATA_PATH="$2"
  local TAG="$3"   # poem | story
  local N_SAMPLES="$4"
  local TEMP="$5"
  local SAVE_DIR="${MODEL_DIR}/evaluation_${TAG}"
  mkdir -p "${SAVE_DIR}"
  python evaluation/generate_response.py \
    --model_name_or_path "${MODEL_DIR}" \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --dataset_path "${DATA_PATH}" \
    --dataset_split "train" \
    --prompt_key "${PROMPT_KEY}" \
    --max_size ${MAX_SIZE} \
    --n ${N_SAMPLES} \
    --temperature ${TEMP} \
    --use_vllm True \
    --vllm_gpu_memory_utilization ${VLLM_UTIL} \
    --save_path "${SAVE_DIR}/generated_responses.json"
  python evaluation/evaluation_diversity.py \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --detokenizer_path "${BASE_MODEL_PATH}" \
    --response_path "${SAVE_DIR}/generated_responses.json" \
    2>&1 | tee "${SAVE_DIR}/diversity_metrics.log"
}

for RUN in "${RUNS[@]}"; do
  EXP_DIR="${BASE_RESULTS_DIR}/${RUN}"
  if [ ! -d "${EXP_DIR}" ]; then
    echo "Skip: ${EXP_DIR} not found"; continue
  fi

  # Capture ONLY the echoed path
  CONSOLIDATED="$(consolidate_checkpoint "${EXP_DIR}")"

  # Alpaca generations (GEM)
  generate_alpaca "${CONSOLIDATED}"

  # Bradley–Terry win-rate vs GPT baseline (paper method)
  bash scripts/qwen2-1.5b/eval/run_bt_winrate.sh "${CONSOLIDATED}" "${GPT_BASELINE_JSON}"

  # Poem & Story diversity (GEM creative-writing style)
  generate_creative "${CONSOLIDATED}" "${POEM_DATA}"  "poem"  ${POEM_N}  ${POEM_TEMP}
  generate_creative "${CONSOLIDATED}" "${STORY_DATA}" "story" ${STORY_N} ${STORY_TEMP}
done

echo "--- All evaluations complete ---"

