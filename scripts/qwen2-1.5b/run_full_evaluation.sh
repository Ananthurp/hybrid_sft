#!/bin/bash
# This master script consolidates a trained checkpoint and then runs the
# full evaluation suite on it.

set -e # Exit immediately if any command fails.
set -x # Print each command before it is executed.

# --- Section 0: Configuration ---
echo "--- CONFIGURATION ---"
# This script takes ONE argument: the path to the main results directory
# for the experiment you want to evaluate.
if [ -z "$1" ]; then
    echo "Error: You must provide the path to the experiment results directory."
    echo "Usage: $0 /path/to/your/results/directory"
    exit 1
fi
EXPERIMENT_RESULTS_DIR=$1

# Define paths to other necessary files and directories
BASE_MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
CONVERSION_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/zero_to_fp32.py"
# Assuming the final checkpoint is always at step 1392 for 3 epochs
FINAL_CHECKPOINT_DIR="${EXPERIMENT_RESULTS_DIR}/checkpoint-1392"
CONSOLIDATED_MODEL_DIR="${EXPERIMENT_RESULTS_DIR}/output_dir_consolidated"

# --- Section 1: Consolidate Checkpoint ---
echo "--- STARTING CHECKPOINT CONSOLIDATION for ${EXPERIMENT_RESULTS_DIR} ---"
mkdir -p $CONSOLIDATED_MODEL_DIR

python ${CONVERSION_SCRIPT_PATH} ${FINAL_CHECKPOINT_DIR} ${CONSOLIDATED_MODEL_DIR}/pytorch_model.bin

# Copy the necessary config/tokenizer files to make the directory self-contained
cp ${BASE_MODEL_PATH}/*.json ${CONSOLIDATED_MODEL_DIR}/
cp ${BASE_MODEL_PATH}/*.txt ${CONSOLIDATED_MODEL_DIR}/

echo "--- CONSOLIDATION COMPLETE ---"
echo "Consolidated model is ready at: ${CONSOLIDATED_MODEL_DIR}"


# --- Section 2: Run Full Evaluation Suite ---
echo "--- STARTING EVALUATION on ${CONSOLIDATED_MODEL_DIR} ---"

# 2.1: AlpacaEval Generation
bash scripts/qwen2-1.5b/eval/run_generation.sh ${CONSOLIDATED_MODEL_DIR}

# 2.2: AlpacaEval Reward Scoring
bash scripts/qwen2-1.5b/eval/run_reward_scoring.sh ${CONSOLIDATED_MODEL_DIR}

# 2.3: Diversity Evaluation
bash scripts/qwen2-1.5b/eval/run_diversity_eval.sh ${CONSOLIDATED_MODEL_DIR}

# 2.4: GSM8K Standard Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_eval.sh ${CONSOLIDATED_MODEL_DIR}

# 2.5: GSM8K Voting Evaluation
bash scripts/qwen2-1.5b/eval/gsm8k_voting_eval.sh ${CONSOLIDATED_MODEL_DIR}

echo "--- ALL EVALUATIONS FOR ${EXPERIMENT_NAME} ARE COMPLETE ---"