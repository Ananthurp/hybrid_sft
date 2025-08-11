# #!/bin/bash
# set -e; set -x;

# # --- Step 0: Configuration ---
# if [ -z "$1" ]; then
#     echo "Error: You must provide the path to the experiment results directory."
#     echo "Usage: $0 /path/to/your/results/directory"
#     exit 1
# fi
# EXPERIMENT_RESULTS_DIR=$1

# # Define which GPU to use for this evaluation run.
# # Make sure this is a GPU that is NOT being used by your training job.
# export CUDA_VISIBLE_DEVICES="2"

# BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"

# # This logic finds the checkpoint with the highest number (the final one)
# FINAL_CHECKPOINT_DIR=$(ls -d ${EXPERIMENT_RESULTS_DIR}/checkpoint-*/ | sort -V | tail -n 1)
# CONSOLIDATED_MODEL_DIR="${EXPERIMENT_RESULTS_DIR}/output_dir_consolidated"

# # --- Step 1: Consolidate Checkpoint ---
# echo "--- Consolidating checkpoint from ${FINAL_CHECKPOINT_DIR} ---"
# mkdir -p $CONSOLIDATED_MODEL_DIR
# python ${CONVERSION_SCRIPT_PATH} ${FINAL_CHECKPOINT_DIR} ${CONSOLIDATED_MODEL_DIR}/pytorch_model.bin
# cp ${BASE_MODEL_PATH}/*.json ${CONSOLIDATED_MODEL_DIR}/
# cp ${BASE_MODEL_PATH}/*.txt ${CONSOLIDATED_MODEL_DIR}/

# # --- Step 2: Run Full Evaluation Suite ---
# echo "--- Starting evaluation on ${CONSOLIDATED_MODEL_DIR} ---"
# bash scripts/qwen2-1.5b/eval/run_generation.sh ${CONSOLIDATED_MODEL_DIR}
# bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh ${CONSOLIDATED_MODEL_DIR}
# bash scripts/qwen2-1.5b/eval/run_diversity_eval.sh ${CONSOLIDATED_MODEL_DIR}
# bash scripts/qwen2-1.5b/eval/gsm8k_eval.sh ${CONSOLIDATED_MODEL_DIR}
# bash scripts/qwen2-1.5b/eval/gsm8k_voting_eval.sh ${CONSOLIDATED_MODEL_DIR}

# echo "--- All evaluations for ${EXPERIMENT_RESULTS_DIR} are complete ---"


#!/bin/bash
# This master script consolidates a trained checkpoint and then runs the
# full evaluation suite on it.

set -e; set -x;

# --- Step 0: Configuration and Validation ---
echo "--- CONFIGURATION ---"
if [ -z "$1" ]; then
    echo "Error: You must provide the path to the experiment results directory."
    echo "Usage: $0 /path/to/your/results/directory"
    exit 1
fi
EXPERIMENT_RESULTS_DIR=$1

# --- THIS IS THE NEW CHECK ---
# Check if the provided directory actually exists
if [ ! -d "$EXPERIMENT_RESULTS_DIR" ]; then
    echo "Error: The directory '${EXPERIMENT_RESULTS_DIR}' does not exist."
    exit 1
fi

# Define which GPU to use for this evaluation run.
export CUDA_VISIBLE_DEVICES="0"

# Define other paths
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"
FINAL_CHECKPOINT_DIR=$(ls -d ${EXPERIMENT_RESULTS_DIR}/checkpoint-*/ | sort -V | tail -n 1)
CONSOLIDATED_MODEL_DIR="${EXPERIMENT_RESULTS_DIR}/output_dir_consolidated"

# --- Step 1: Consolidate Checkpoint ---
echo "--- Consolidating checkpoint from ${FINAL_CHECKPOINT_DIR} ---"
mkdir -p $CONSOLIDATED_MODEL_DIR
python ${CONVERSION_SCRIPT_PATH} ${FINAL_CHECKPOINT_DIR} ${CONSOLIDATED_MODEL_DIR}
cp ${BASE_MODEL_PATH}/*.json ${CONSOLIDATED_MODEL_DIR}/
cp ${BASE_MODEL_PATH}/*.txt ${CONSOLIDATED_MODEL_DIR}/

# --- Step 2: Run Full Evaluation Suite ---
echo "--- Starting evaluation on ${CONSOLIDATED_MODEL_DIR} ---"
bash scripts/qwen2-1.5b/eval/run_generation.sh ${CONSOLIDATED_MODEL_DIR}
bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh ${CONSOLIDATED_MODEL_DIR}
bash scripts/qwen2-1.5b/eval/run_diversity_eval.sh ${CONSOLIDATED_MODEL_DIR}
bash scripts/qwen2-1.5b/eval/gsm8k_eval.sh ${CONSOLIDATED_MODEL_DIR}
bash scripts/qwen2-1.5b/eval/gsm8k_voting_eval.sh ${CONSOLIDATED_MODEL_DIR}

echo "--- All evaluations for ${EXPERIMENT_RESULTS_DIR} are complete ---"