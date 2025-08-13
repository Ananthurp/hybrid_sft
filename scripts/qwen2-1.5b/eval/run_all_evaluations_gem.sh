# #!/bin/bash
# # GEM-only evaluation for multiple runs.
# # Usage:
# #   bash scripts/qwen2-1.5b/eval/run_all_evaluations_gem.sh \
# #     /data/ananthu/gem_project/results \
# #     qwen_1.5b_ce_run qwen_1.5b_gem_run \
# #     hybrid_alpha0.5_epoch3_topk3 hybrid_alpha0.5_epoch3_topk5 hybrid_alpha0.5_epoch3_topk10

# set -euo pipefail
# set -x

# if [ "$#" -lt 2 ]; then
#   echo "Usage: $0 <BASE_RESULTS_DIR> <RUN_1> [RUN_2 ...]"
#   exit 1
# fi

# BASE_RESULTS_DIR="$1"; shift
# RUNS=("$@")

# # Choose a free GPU (override by exporting CUDA_VISIBLE_DEVICES before running)
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# # Tokenizer / base model assets (unchanged)
# BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# # Your converted JSONL datasets (from Parquet)
# POEM_DATA="/data/ananthu/gem_project/code/GEM/data/poem_generation/hfds"
# STORY_DATA="/data/ananthu/gem_project/code/GEM/data/story_generation/hfds"

# # Generation settings (match GEM defaults you’ve used elsewhere)
# ALPACA_N=32
# ALPACA_TEMP=0.6
# VLLM_UTIL=0.7
# MAX_SIZE=200

# POEM_N=16
# POEM_TEMP=1.0
# STORY_N=16
# STORY_TEMP=1.0

# consolidate_checkpoint () {
#   local EXP_DIR="$1"
#   local OUT_DIR="${EXP_DIR}/output_dir_consolidated"
#   mkdir -p "${OUT_DIR}"

#   local FINAL_CKPT=""
#   FINAL_CKPT=$(ls -d "${EXP_DIR}"/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1 || true)

#   if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
#     echo "--- Consolidating ${FINAL_CKPT} -> ${OUT_DIR} ---"
#     python "${CONVERSION_SCRIPT_PATH}" "${FINAL_CKPT}" "${OUT_DIR}"
#   else
#     echo "--- No checkpoint-* in ${EXP_DIR}; copying as-is -> ${OUT_DIR} ---"
#     rsync -a --delete "${EXP_DIR}/" "${OUT_DIR}/" || true
#   fi

#   # Ensure tokenizer/config are present for generation
#   cp -n ${BASE_MODEL_PATH}/*.json "${OUT_DIR}/" 2>/dev/null || true
#   cp -n ${BASE_MODEL_PATH}/*.txt  "${OUT_DIR}/" 2>/dev/null || true

#   echo "${OUT_DIR}"
# }

# run_alpaca_eval () {
#   local MODEL_DIR="$1"
#   # Uses your existing GEM scripts (unchanged logic)
#   bash scripts/qwen2-1.5b/eval/run_generation.sh "${MODEL_DIR}"
#   bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh "${MODEL_DIR}"
# }

# run_creative_diversity () {
#   local MODEL_DIR="$1"
#   local DATA_PATH="$2"
#   local TAG="$3"      # poem | story
#   local N_SAMPLES="$4"
#   local TEMP="$5"

#   local SAVE_DIR="${MODEL_DIR}/evaluation_${TAG}"
#   mkdir -p "${SAVE_DIR}"

#   # Use GEM's generate_response.py exactly as in repo
#   # (explicitly pass --split train and --column_name instruction for local JSONL)
#   python evaluation/generate_response.py \
#     --model_name_or_path "${MODEL_DIR}" \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --dataset_path "${DATA_PATH}" \
#     --split "train" \
#     --column_name "instruction" \
#     --load_from_disk True \
#     --max_size ${MAX_SIZE} \
#     --n ${N_SAMPLES} \
#     --temperature ${TEMP} \
#     --use_vllm True \
#     --vllm_gpu_memory_utilization ${VLLM_UTIL} \
#     --save_path "${SAVE_DIR}/generated_responses.json"

#   python evaluation/evaluation_diversity.py \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --detokenizer_path "${BASE_MODEL_PATH}" \
#     --response_path "${SAVE_DIR}/generated_responses.json" \
#     2>&1 | tee "${SAVE_DIR}/diversity_metrics.log"
# }

# for RUN in "${RUNS[@]}"; do
#   EXP_DIR="${BASE_RESULTS_DIR}/${RUN}"
#   if [ ! -d "${EXP_DIR}" ]; then
#     echo "Skip: ${EXP_DIR} not found"
#     continue
#   fi

#   MODEL_DIR="$(consolidate_checkpoint "${EXP_DIR}" | tail -n 1)"

#   # 1) AlpacaEval (generation + reward → prints Best-of-n / Mean-of-n)
#   run_alpaca_eval "${MODEL_DIR}"

#   # 2) Poem / Story diversity (repo method)
#   run_creative_diversity "${MODEL_DIR}" "${POEM_DATA}"  "poem"  ${POEM_N}  ${POEM_TEMP}
#   run_creative_diversity "${MODEL_DIR}" "${STORY_DATA}" "story" ${STORY_N} ${STORY_TEMP}
# done

# echo "--- All GEM-style evaluations done ---"



#!/bin/bash
# GEM-style evaluation runner for multiple experiments.
# Usage:
#   bash scripts/qwen2-1.5b/eval/run_all_evaluations_gem.sh \
#     /data/ananthu/gem_project/results \
#     hybrid_alpha0.5_epochs3_topk3 \
#     hybrid_alpha0.5_epochs3_topk5 \
#     hybrid_alpha0.5_epochs3_topk10
#
# Expects GEM repo eval scripts:
#   - evaluation/generate_response.py
#   - evaluation/evaluation_reward.py (wrapped by run_reward_scoring.sh)
#   - evaluation/evaluation_diversity.py
#
# Ensure run_generation.sh uses: --split "eval" and --column_name "instruction".

# set -euo pipefail
# set -x

# if [ "$#" -lt 2 ]; then
#   echo "Usage: $0 <BASE_RESULTS_DIR> <RUN_1> [RUN_2 ...]" >&2
#   exit 1
# fi

# BASE_RESULTS_DIR="$1"; shift
# RUNS=("$@")

# # Respect caller; default to GPU 0 if not set
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# # Paths
# BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# # Saved HF datasets (made via datasets.save_to_disk)
# POEM_DATA="/data/ananthu/gem_project/code/GEM/data/poem_generation/hfds"
# STORY_DATA="/data/ananthu/gem_project/code/GEM/data/story_generation/hfds"

# # Sanity
# [ -d "$POEM_DATA" ]  || { echo "Missing POEM_DATA dir: $POEM_DATA"  >&2; exit 2; }
# [ -d "$STORY_DATA" ] || { echo "Missing STORY_DATA dir: $STORY_DATA" >&2; exit 2; }

# # Generation settings (GEM-ish)
# ALPACA_N=32
# ALPACA_TEMP=0.6
# VLLM_UTIL=0.7
# MAX_SIZE=200

# POEM_N=16
# POEM_TEMP=1.0
# STORY_N=16
# STORY_TEMP=1.0

# consolidate_checkpoint () {
#   local EXP_DIR="$1"
#   local OUT_DIR="${EXP_DIR}/output_dir_consolidated"
#   mkdir -p "${OUT_DIR}"

#   # 1) Top-level HF weights?
#   if ls "${EXP_DIR}"/pytorch_model*.bin "${EXP_DIR}"/*.safetensors >/dev/null 2>&1; then
#     >&2 echo "--- Found HF weights at ${EXP_DIR}; copying -> ${OUT_DIR} ---"
#     rsync -a --include="pytorch_model*.bin" --include="*.safetensors" --exclude="*" "${EXP_DIR}/" "${OUT_DIR}/"
#   else
#     # 2) Prefer newest checkpoint-* if present
#     local FINAL_CKPT=""
#     FINAL_CKPT=$(ls -d "${EXP_DIR}"/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1 || true)
#     if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
#       >&2 echo "--- Consolidating ${FINAL_CKPT} -> ${OUT_DIR} ---"
#       python "${CONVERSION_SCRIPT_PATH}" "${FINAL_CKPT}" "${OUT_DIR}" 1>&2
#     else
#       # 3) Epoch-* directories containing HF weights?
#       local EPOCH_DIR=""
#       EPOCH_DIR=$(ls -d "${EXP_DIR}"/epoch-*/ 2>/dev/null | sort -V | tail -n 1 || true)
#       if [ -n "${EPOCH_DIR}" ] && ls "${EPOCH_DIR}"/pytorch_model*.bin "${EPOCH_DIR}"/*.safetensors >/dev/null 2>&1; then
#         >&2 echo "--- Found HF weights in ${EPOCH_DIR}; copying -> ${OUT_DIR} ---"
#         rsync -a --include="pytorch_model*.bin" --include="*.safetensors" --exclude="*" "${EPOCH_DIR}/" "${OUT_DIR}/"
#       else
#         # 4) Any nested folder containing HF weight files?
#         local ANY_WEIGHT_FILE=""
#         ANY_WEIGHT_FILE=$(find "${EXP_DIR}" -type f \( -name 'pytorch_model*.bin' -o -name '*.safetensors' \) 2>/dev/null | sort | tail -n 1 || true)
#         if [ -n "${ANY_WEIGHT_FILE}" ]; then
#           local ANY_WEIGHT_DIR
#           ANY_WEIGHT_DIR="$(dirname "${ANY_WEIGHT_FILE}")"
#           >&2 echo "--- Found nested HF weights in ${ANY_WEIGHT_DIR}; copying -> ${OUT_DIR} ---"
#           rsync -a --include="pytorch_model*.bin" --include="*.safetensors" --exclude="*" "${ANY_WEIGHT_DIR}/" "${OUT_DIR}/"
#         else
#           # 5) Any folder with deepspeed stage-3 shards (global_step*/mp_rank_*)?
#           local DS_DIR=""
#           DS_DIR=$(find "${EXP_DIR}" -type f -name 'global_step*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
#           if [ -z "${DS_DIR}" ]; then
#             # also try presence of mp_rank_* only
#             DS_DIR=$(find "${EXP_DIR}" -maxdepth 2 -type d -name 'mp_rank_*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
#           fi
#           if [ -n "${DS_DIR}" ]; then
#             >&2 echo "--- Consolidating Deepspeed shards in ${DS_DIR} -> ${OUT_DIR} ---"
#             python "${CONVERSION_SCRIPT_PATH}" "${DS_DIR}" "${OUT_DIR}" 1>&2
#           else
#             >&2 echo "!!! No HF weights, no checkpoint-*, no epoch-* weights, and no deepspeed shards found in ${EXP_DIR}."
#             >&2 echo "    Contents:"
#             >&2 ls -lah "${EXP_DIR}" || true
#             exit 3
#           fi
#         fi
#       fi
#     fi
#   fi

#   # Ensure tokenizer/config present for vLLM/HF loaders
#   cp -n ${BASE_MODEL_PATH}/*.json "${OUT_DIR}/" 2>/dev/null || true
#   cp -n ${BASE_MODEL_PATH}/*.txt  "${OUT_DIR}/" 2>/dev/null || true

#   # Return path
#   echo "${OUT_DIR}"
# }

# run_alpaca_eval () {
#   local MODEL_DIR="$1"
#   bash scripts/qwen2-1.5b/eval/run_generation.sh "${MODEL_DIR}"
#   bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh "${MODEL_DIR}"
# }

# run_creative_diversity () {
#   local MODEL_DIR="$1"
#   local DATA_PATH="$2"   # hfds dir (poem/story)
#   local TAG="$3"         # "poem" | "story"
#   local N_SAMPLES="$4"   # 16
#   local TEMP="$5"        # 1.0

#   local SAVE_DIR="${MODEL_DIR}/evaluation_${TAG}"
#   mkdir -p "${SAVE_DIR}"

#   # Generate n samples per prompt (GEM script, loaded from disk)
#   python evaluation/generate_response.py \
#     --model_name_or_path "${MODEL_DIR}" \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --dataset_path "${DATA_PATH}" \
#     --load_from_disk True \
#     --split "train" \
#     --column_name "instruction" \
#     --max_size ${MAX_SIZE} \
#     --n ${N_SAMPLES} \
#     --temperature ${TEMP} \
#     --use_vllm True \
#     --vllm_gpu_memory_utilization ${VLLM_UTIL} \
#     --save_path "${SAVE_DIR}/generated_responses.json"

#   # Compute diversity metrics (n-gram avg, Self-BLEU, SBERT)
#   python evaluation/evaluation_diversity.py \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --detokenizer_path "${BASE_MODEL_PATH}" \
#     --response_path "${SAVE_DIR}/generated_responses.json" \
#     2>&1 | tee "${SAVE_DIR}/diversity_metrics.log"
# }

# # ----------------- main loop -----------------
# for RUN in "${RUNS[@]}"; do
#   EXP_DIR="${BASE_RESULTS_DIR}/${RUN}"
#   if [ ! -d "${EXP_DIR}" ]; then
#     echo "Skip: ${EXP_DIR} not found" >&2
#     continue
#   fi

#   MODEL_DIR="$(consolidate_checkpoint "${EXP_DIR}")"

#   # 1) AlpacaEval (generation + reward Best-of-n/Mean-of-n; prints to stdout & saves JSON)
#   run_alpaca_eval "${MODEL_DIR}"

#   # 2) Poem/Story diversity (GEM method)
#   run_creative_diversity "${MODEL_DIR}" "${POEM_DATA}"  "poem"  ${POEM_N}  ${POEM_TEMP}
#   run_creative_diversity "${MODEL_DIR}" "${STORY_DATA}" "story" ${STORY_N} ${STORY_TEMP}
# done

# echo "--- All GEM-style evaluations done ---"


#!/bin/bash
# GEM-style evaluation runner with Alpaca diversity + two winrate strategies.
# Usage:
#   bash scripts/qwen2-1.5b/eval/run_all_evaluations_gem.sh \
#     <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON or ""> <RUN_1> [RUN_2 ...]
#
# Produces under each RUN's output_dir_consolidated/:
#   - evaluation_chat_alpaca/generated_responses.json
#   - evaluation_chat_alpaca/reward_scores.json
#   - evaluation_chat_alpaca/winrate_gt0.json
#   - evaluation_chat_alpaca/bt_winrate_vs_gpt.json   (if baseline provided)
#   - evaluation_chat_alpaca/alpaca_diversity_metrics.log
#   - evaluation_poem/generated_responses.json + diversity_metrics.log
#   - evaluation_story/generated_responses.json + diversity_metrics.log

# set -euo pipefail
# set -x

# if [ "$#" -lt 3 ]; then
#   echo "Usage: $0 <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON or \"\"> <RUN_1> [RUN_2 ...]" >&2
#   exit 1
# fi

# BASE_RESULTS_DIR="$1"; shift
# GPT_BASELINE_JSON="$1"; shift
# RUNS=("$@")

# # Respect caller's GPU selection; default to 0
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# # Paths
# BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# # Poem/Story in HF saved-dataset form (hfds). If you switched to JSONL, change the flags below to match.
# POEM_DATA="/data/ananthu/gem_project/code/GEM/data/poem_generation/hfds"
# STORY_DATA="/data/ananthu/gem_project/code/GEM/data/story_generation/hfds"

# # Generation settings (GEM-like)
# ALPACA_N=32
# ALPACA_TEMP=0.6
# VLLM_UTIL=0.7
# MAX_SIZE=200

# POEM_N=16
# POEM_TEMP=1.0
# STORY_N=16
# STORY_TEMP=1.0

# # -------- helpers --------

# consolidate_checkpoint () {
#   local EXP_DIR="$1"
#   local OUT_DIR="${EXP_DIR}/output_dir_consolidated"

#   # Always overwrite as requested
#   rm -rf "${OUT_DIR}"
#   mkdir -p "${OUT_DIR}"

#   # Prefer newest checkpoint-* if present
#   local FINAL_CKPT=""
#   FINAL_CKPT=$(ls -d "${EXP_DIR}"/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1 || true)
#   if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
#     >&2 echo "--- Consolidating ${FINAL_CKPT} -> ${OUT_DIR} ---"
#     python "${CONVERSION_SCRIPT_PATH}" "${FINAL_CKPT}" "${OUT_DIR}" 1>&2
#   else
#     # If no checkpoint-* try direct HF weights or deepspeed shards
#     if ls "${EXP_DIR}"/pytorch_model*.bin "${EXP_DIR}"/*.safetensors >/dev/null 2>&1; then
#       >&2 echo "--- Found HF weights at ${EXP_DIR}; copying -> ${OUT_DIR} ---"
#       rsync -a --include="pytorch_model*.bin" --include="*.safetensors" --exclude="*" "${EXP_DIR}/" "${OUT_DIR}/"
#     else
#       local DS_DIR=""
#       DS_DIR=$(find "${EXP_DIR}" -type f -name 'global_step*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
#       if [ -z "${DS_DIR}" ]; then
#         DS_DIR=$(find "${EXP_DIR}" -maxdepth 2 -type d -name 'mp_rank_*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
#       fi
#       if [ -n "${DS_DIR}" ]; then
#         >&2 echo "--- Consolidating Deepspeed shards in ${DS_DIR} -> ${OUT_DIR} ---"
#         python "${CONVERSION_SCRIPT_PATH}" "${DS_DIR}" "${OUT_DIR}" 1>&2
#       else
#         >&2 echo "!!! No weights found in ${EXP_DIR}"
#         >&2 ls -lah "${EXP_DIR}" || true
#         exit 3
#       fi
#     fi
#   fi

#   # Ensure tokenizer/config present for vLLM/HF loaders
#   cp -n ${BASE_MODEL_PATH}/*.json "${OUT_DIR}/" 2>/dev/null || true
#   cp -n ${BASE_MODEL_PATH}/*.txt  "${OUT_DIR}/" 2>/dev/null || true

#   echo "${OUT_DIR}"
# }

# generate_alpaca () {
#   local MODEL_DIR="$1"
#   local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
#   mkdir -p "${SAVE_DIR}"

#   # Force fresh generations & scores each run (per your request to overwrite)
#   rm -f "${SAVE_DIR}/generated_responses.json" \
#         "${SAVE_DIR}/reward_scores.json" \
#         "${SAVE_DIR}/alpaca_diversity_metrics.log" \
#         "${SAVE_DIR}/winrate_gt0.json" \
#         "${SAVE_DIR}/bt_winrate_vs_gpt.json"

#   python evaluation/generate_response.py \
#     --model_name_or_path "${MODEL_DIR}" \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --dataset_path "tatsu-lab/alpaca_eval" \
#     --split "eval" \
#     --column_name "instruction" \
#     --max_size ${MAX_SIZE} \
#     --n ${ALPACA_N} \
#     --temperature ${ALPACA_TEMP} \
#     --use_vllm True \
#     --vllm_gpu_memory_utilization ${VLLM_UTIL} \
#     --save_path "${SAVE_DIR}/generated_responses.json"
# }

# reward_and_winrate_gt0 () {
#   local MODEL_DIR="$1"
#   local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
#   local RESP="${SAVE_DIR}/generated_responses.json"
#   local OUT="${SAVE_DIR}/reward_scores.json"
#   # evaluation_reward.py (your local copy) should have calculate_winrate() (>0) printing to stdout.
#   python evaluation/evaluation_reward.py \
#     --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
#     --batch_size 2 \
#     --detokenizer_path "${BASE_MODEL_PATH}" \
#     --data_path "${RESP}" \
#     --save_path "${OUT}" \
#     2>&1 | tee "${SAVE_DIR}/reward_eval.log"

#   # Save a small JSON with the thresholded winrate (computed from rewards)
  
#   "${SAVE_DIR}"
# }

# alpaca_diversity () {
#   local MODEL_DIR="$1"
#   local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
#   python evaluation/evaluation_diversity.py \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --detokenizer_path "${BASE_MODEL_PATH}" \
#     --response_path "${SAVE_DIR}/generated_responses.json" \
#     2>&1 | tee "${SAVE_DIR}/alpaca_diversity_metrics.log"
# }

# bt_winrate_vs_gpt4 () {
#   local MODEL_DIR="$1"
#   local BASELINE="$2"
#   [ -z "${BASELINE}" ] && { echo "--- Skipping BT winrate (no baseline provided) ---"; return 0; }
#   local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
#   local CAND_JSON="${SAVE_DIR}/generated_responses.json"
#   local OUTFILE="${SAVE_DIR}/bt_winrate_vs_gpt.json"

#   if [ ! -f "${BASELINE}" ]; then
#     echo "!!! Baseline JSON not found: ${BASELINE}" >&2
#     return 0
#   fi

#   # Assumes you have scripts/qwen2-1.5b/eval/bt_winrate.py in your repo.
#   python scripts/qwen2-1.5b/eval/bt_winrate.py \
#     --candidate_json "${CAND_JSON}" \
#     --baseline_json "${BASELINE}" \
#     --out_file "${OUTFILE}"
# }

# creative_diversity () {
#   local MODEL_DIR="$1"
#   local DATA_PATH="$2"   # hfds dir for poem/story
#   local TAG="$3"         # "poem" | "story"
#   local N_SAMPLES="$4"   # 16
#   local TEMP="$5"        # 1.0

#   local SAVE_DIR="${MODEL_DIR}/evaluation_${TAG}"
#   mkdir -p "${SAVE_DIR}"
#   rm -f "${SAVE_DIR}/generated_responses.json" "${SAVE_DIR}/diversity_metrics.log"

#   python evaluation/generate_response.py \
#     --model_name_or_path "${MODEL_DIR}" \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --dataset_path "${DATA_PATH}" \
#     --load_from_disk True \
#     --split "train" \
#     --column_name "instruction" \
#     --max_size ${MAX_SIZE} \
#     --n ${N_SAMPLES} \
#     --temperature ${TEMP} \
#     --use_vllm True \
#     --vllm_gpu_memory_utilization ${VLLM_UTIL} \
#     --save_path "${SAVE_DIR}/generated_responses.json"

#   python evaluation/evaluation_diversity.py \
#     --tokenizer_path "${BASE_MODEL_PATH}" \
#     --detokenizer_path "${BASE_MODEL_PATH}" \
#     --response_path "${SAVE_DIR}/generated_responses.json" \
#     2>&1 | tee "${SAVE_DIR}/diversity_metrics.log"
# }

# # -------- main loop --------
# for RUN in "${RUNS[@]}"; do
#   EXP_DIR="${BASE_RESULTS_DIR}/${RUN}"
#   if [ ! -d "${EXP_DIR}" ]; then
#     echo "Skip: ${EXP_DIR} not found" >&2
#     continue
#   fi

#   MODEL_DIR="$(consolidate_checkpoint "${EXP_DIR}")"

#   # Alpaca: gen + reward/mean/best-of + (>0) winrate + diversity
#   generate_alpaca "${MODEL_DIR}"
#   reward_and_winrate_gt0 "${MODEL_DIR}"
#   alpaca_diversity "${MODEL_DIR}"

#   # Bradley–Terry vs GPT-4
#   bt_winrate_vs_gpt4 "${MODEL_DIR}" "${GPT_BASELINE_JSON}"

#   # Poem & Story diversity
#   creative_diversity "${MODEL_DIR}" "${POEM_DATA}"  "poem"  ${POEM_N}  ${POEM_TEMP}
#   creative_diversity "${MODEL_DIR}" "${STORY_DATA}" "story" ${STORY_N} ${STORY_TEMP}
# done

# echo "--- All evaluations complete ---"

# # Aggregate a one-row-per-model table (CSV + optional XLSX/JSON)
# python scripts/qwen2-1.5b/eval/summarize_eval.py \
#   --base_dir "${BASE_RESULTS_DIR}" \
#   $(printf -- "--run %s " "${RUNS[@]}") \
#   --csv_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.csv" \
#   --xlsx_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.xlsx" \
#   --json_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.json"



#!/bin/bash
# GEM-style evaluation runner with Alpaca diversity + two winrate strategies.
# Usage:
#   bash scripts/qwen2-1.5b/eval/run_all_evaluations_gem.sh \
#     <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON or ""> <RUN_1> [RUN_2 ...]
#
# Produces under each RUN's output_dir_consolidated/:
#   - evaluation_chat_alpaca/generated_responses.json
#   - evaluation_chat_alpaca/reward_scores.json
#   - evaluation_chat_alpaca/winrate_gt0.json
#   - evaluation_chat_alpaca/bt_winrate_vs_gpt.json   (if baseline provided)
#   - evaluation_chat_alpaca/alpaca_diversity_metrics.log
#   - evaluation_poem/generated_responses.json + diversity_metrics.log
#   - evaluation_story/generated_responses.json + diversity_metrics.log

set -euo pipefail
set -x

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <BASE_RESULTS_DIR> <GPT4_BASELINE_JSON or \"\"> <RUN_1> [RUN_2 ...]" >&2
  exit 1
fi

BASE_RESULTS_DIR="$1"; shift
GPT_BASELINE_JSON="$1"; shift
RUNS=("$@")

# Respect caller's GPU selection; default to 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Paths
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# Poem/Story in HF saved-dataset form (hfds). If you switched to JSONL, change flags accordingly.
POEM_DATA="/data/ananthu/gem_project/code/GEM/data/poem_generation/hfds"
STORY_DATA="/data/ananthu/gem_project/code/GEM/data/story_generation/hfds"

# Generation settings (GEM-like)
ALPACA_N=32
ALPACA_TEMP=0.6
VLLM_UTIL=0.7
MAX_SIZE=200

POEM_N=16
POEM_TEMP=1.0
STORY_N=16
STORY_TEMP=1.0

# -------- helpers --------

consolidate_checkpoint () {
  local EXP_DIR="$1"
  local OUT_DIR="${EXP_DIR}/output_dir_consolidated"

  # Always overwrite as requested
  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  # Prefer newest checkpoint-* if present
  local FINAL_CKPT=""
  FINAL_CKPT=$(ls -d "${EXP_DIR}"/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
    >&2 echo "--- Consolidating ${FINAL_CKPT} -> ${OUT_DIR} ---"
    python "${CONVERSION_SCRIPT_PATH}" "${FINAL_CKPT}" "${OUT_DIR}" 1>&2
  else
    # If no checkpoint-* try direct HF weights or deepspeed shards
    if ls "${EXP_DIR}"/pytorch_model*.bin "${EXP_DIR}"/*.safetensors >/dev/null 2>&1; then
      >&2 echo "--- Found HF weights at ${EXP_DIR}; copying -> ${OUT_DIR} ---"
      rsync -a \
        --include="pytorch_model*.bin" \
        --include="*.safetensors" \
        --exclude="*" \
        "${EXP_DIR}/" "${OUT_DIR}/"
    else
      local DS_DIR=""
      DS_DIR=$(find "${EXP_DIR}" -type f -name 'global_step*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
      if [ -z "${DS_DIR}" ]; then
        DS_DIR=$(find "${EXP_DIR}" -maxdepth 2 -type d -name 'mp_rank_*' -printf '%h\n' 2>/dev/null | sort -u | tail -n 1 || true)
      fi
      if [ -n "${DS_DIR}" ]; then
        >&2 echo "--- Consolidating Deepspeed shards in ${DS_DIR} -> ${OUT_DIR} ---"
        python "${CONVERSION_SCRIPT_PATH}" "${DS_DIR}" "${OUT_DIR}" 1>&2
      else
        >&2 echo "!!! No weights found in ${EXP_DIR}"
        >&2 ls -lah "${EXP_DIR}" || true
        exit 3
      fi
    fi
  fi

  # Ensure tokenizer/config present for vLLM/HF loaders
  cp -n ${BASE_MODEL_PATH}/*.json "${OUT_DIR}/" 2>/dev/null || true
  cp -n ${BASE_MODEL_PATH}/*.txt  "${OUT_DIR}/" 2>/dev/null || true

  echo "${OUT_DIR}"
}

generate_alpaca () {
  local MODEL_DIR="$1"
  local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
  mkdir -p "${SAVE_DIR}"

  # Force fresh generations & scores each run
  rm -f "${SAVE_DIR}/generated_responses.json" \
        "${SAVE_DIR}/reward_scores.json" \
        "${SAVE_DIR}/alpaca_diversity_metrics.log" \
        "${SAVE_DIR}/winrate_gt0.json" \
        "${SAVE_DIR}/bt_winrate_vs_gpt.json"

  python evaluation/generate_response.py \
    --model_name_or_path "${MODEL_DIR}" \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --dataset_path "tatsu-lab/alpaca_eval" \
    --split "eval" \
    --column_name "instruction" \
    --max_size ${MAX_SIZE} \
    --n ${ALPACA_N} \
    --temperature ${ALPACA_TEMP} \
    --use_vllm True \
    --vllm_gpu_memory_utilization ${VLLM_UTIL} \
    --save_path "${SAVE_DIR}/generated_responses.json"
}

reward_and_winrate_gt0 () {
  local MODEL_DIR="$1"
  local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
  local RESP="${SAVE_DIR}/generated_responses.json"
  local OUT="${SAVE_DIR}/reward_scores.json"

  python evaluation/evaluation_reward.py \
    --model_name_or_path "sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    --batch_size 2 \
    --detokenizer_path "${BASE_MODEL_PATH}" \
    --data_path "${RESP}" \
    --save_path "${OUT}" \
    2>&1 | tee "${SAVE_DIR}/reward_eval.log"

  # (Optional) leave a breadcrumb
  echo "--- Reward + winrate(>0) done in: ${SAVE_DIR}"
}

alpaca_diversity () {
  local MODEL_DIR="$1"
  local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
  python evaluation/evaluation_diversity.py \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --detokenizer_path "${BASE_MODEL_PATH}" \
    --response_path "${SAVE_DIR}/generated_responses.json" \
    2>&1 | tee "${SAVE_DIR}/alpaca_diversity_metrics.log"
}

bt_winrate_vs_gpt4 () {
  local MODEL_DIR="$1"
  local BASELINE="$2"
  [ -z "${BASELINE}" ] && { echo "--- Skipping BT winrate (no baseline provided) ---"; return 0; }
  local SAVE_DIR="${MODEL_DIR}/evaluation_chat_alpaca"
  local CAND_JSON="${SAVE_DIR}/generated_responses.json"
  local OUTFILE="${SAVE_DIR}/bt_winrate_vs_gpt.json"

  if [ ! -f "${BASELINE}" ]; then
    echo "!!! Baseline JSON not found: ${BASELINE}" >&2
    return 0
  fi

  python scripts/qwen2-1.5b/eval/bt_winrate.py \
    --candidate_json "${CAND_JSON}" \
    --baseline_json "${BASELINE}" \
    --out_file "${OUTFILE}"
}

creative_diversity () {
  local MODEL_DIR="$1"
  local DATA_PATH="$2"   # hfds dir for poem/story
  local TAG="$3"         # "poem" | "story"
  local N_SAMPLES="$4"   # 16
  local TEMP="$5"        # 1.0

  local SAVE_DIR="${MODEL_DIR}/evaluation_${TAG}"
  mkdir -p "${SAVE_DIR}"
  rm -f "${SAVE_DIR}/generated_responses.json" "${SAVE_DIR}/diversity_metrics.log"

  python evaluation/generate_response.py \
    --model_name_or_path "${MODEL_DIR}" \
    --tokenizer_path "${BASE_MODEL_PATH}" \
    --dataset_path "${DATA_PATH}" \
    --load_from_disk True \
    --split "train" \
    --column_name "instruction" \
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

# -------- main loop --------
for RUN in "${RUNS[@]}"; do
  EXP_DIR="${BASE_RESULTS_DIR}/${RUN}"
  if [ ! -d "${EXP_DIR}" ]; then
    echo "Skip: ${EXP_DIR} not found" >&2
    continue
  fi

  MODEL_DIR="$(consolidate_checkpoint "${EXP_DIR}")"

  # Alpaca: gen + reward/mean/best-of + (>0) winrate + diversity
  generate_alpaca "${MODEL_DIR}"
  reward_and_winrate_gt0 "${MODEL_DIR}"
  alpaca_diversity "${MODEL_DIR}"

  # Bradley–Terry vs GPT-4
  bt_winrate_vs_gpt4 "${MODEL_DIR}" "${GPT_BASELINE_JSON}"

  # Poem & Story diversity
  creative_diversity "${MODEL_DIR}" "${POEM_DATA}"  "poem"  ${POEM_N}  ${POEM_TEMP}
  creative_diversity "${MODEL_DIR}" "${STORY_DATA}" "story" ${STORY_N} ${STORY_TEMP}
done

echo "--- All evaluations complete ---"

# Aggregate a one-row-per-model table (CSV + optional XLSX/JSON)
python scripts/qwen2-1.5b/eval/summarize_eval.py \
  --base_dir "${BASE_RESULTS_DIR}" \
  $(printf -- "--run %s " "${RUNS[@]}") \
  --csv_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.csv" \
  --xlsx_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.xlsx" \
  --json_out "${BASE_RESULTS_DIR}/aggregate_eval_summary.json"