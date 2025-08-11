# #!/bin/sh

# set -e 
# set -x

# # This script runs the majority voting evaluation on the GSM8K benchmark.

# # --- Step 0: Configuration ---
# export TRANSFORMERS_OFFLINE=1
# # Allow downloading the GSM8K dataset if it's not cached
# # export HF_DATASETS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES="0" # Runs on a single GPU

# # Define all paths clearly
# MODEL_CHECKPOINT_PATH="/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
# TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# DATASET_NAME="gsm8k"

# # Define a clean save directory for the results
# SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_gsm8k_voting" 
# mkdir -p $SAVE_DIR

# # Define generation parameters for self-consistency/voting
# SEED=42
# N=32  # Generate 32 responses per prompt for robust voting
# T=0.6
# K=50
# P=0.9

# # --- Step 1: Generate 32 responses and evaluate with voting ---
# echo "Running GSM8K evaluation with N=32 majority voting..."

# # Corrected path to the python script
# python ../../evaluation/evaluation_gsm8k_voting.py \
#     --model_name_or_path $MODEL_CHECKPOINT_PATH \
#     --tokenizer_name_or_path $TOKENIZER_PATH \
#     --dataset_name_or_path $DATASET_NAME \
#     --dataset_split "test" \
#     --dtype bf16 \
#     --batch_size 128 \
#     --max_new_tokens 512 \
#     --seed $SEED \
#     --n $N \
#     --temperature $T \
#     --top_k $K \
#     --top_p $P \
#     --use_vllm True \
#     --remove_old True \
#     --save_path "${SAVE_DIR}/gsm8k_voting_generations.json" \
#     2>&1 | tee ${SAVE_DIR}/gsm8k_voting_eval.log

# echo "GSM8K voting evaluation complete. Results saved in ${SAVE_DIR}"


# #!/bin/sh
# set -e 
# set -x

# # This script runs the majority voting evaluation on the GSM8K benchmark.

# # --- Step 0: Configuration ---
# # Allow downloading the GSM8K dataset if it's not cached
# # export HF_DATASETS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES="0" # Use a single GPU

# # Define paths. The model path comes from the first command-line argument ($1).
# MODEL_CHECKPOINT_PATH=$1
# TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# DATASET_NAME="gsm8k"

# # Define a clean save directory for the results
# SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_gsm8k_voting" 
# mkdir -p $SAVE_DIR

# # Define generation parameters for self-consistency/voting
# SEED=42
# N=32

# # --- Step 1: Generate 32 responses and evaluate with voting ---
# echo "Running GSM8K voting evaluation on model: ${MODEL_CHECKPOINT_PATH}"

# # Corrected path to the python script
# python evaluation/evaluation_gsm8k_voting.py \
#     --model_name_or_path $MODEL_CHECKPOINT_PATH \
#     --tokenizer_name_or_path $TOKENIZER_PATH \
#     --dataset_name_or_path $DATASET_NAME \
#     --dataset_split "test" \
#     --dtype bf16 \
#     --batch_size 128 \
#     --max_new_tokens 512 \
#     --seed $SEED \
#     --n $N \
#     --temperature 0.6 \
#     --top_k 50 \
#     --top_p 0.9 \
#     --use_vllm True \
#     --remove_old True \
#     --save_path "${SAVE_DIR}/gsm8k_voting_generations.json" \
#     2>&1 | tee ${SAVE_DIR}/gsm8k_voting_eval.log

# echo "GSM8K voting evaluation complete. Results saved in ${SAVE_DIR}"

#!/bin/sh
set -e; set -x;

# This is a template script. It expects the model path as the first argument.
MODEL_CHECKPOINT_PATH=$1

# --- Configuration ---
export CUDA_VISIBLE_DEVICES="0"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_gsm8k_voting"
mkdir -p $SAVE_DIR

echo "--- Running GSM8K voting evaluation on model: ${MODEL_CHECKPOINT_PATH} ---"

python evaluation/evaluation_gsm8k_voting.py \
    --model_name_or_path $MODEL_CHECKPOINT_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --dataset_name_or_path "gsm8k" \
    --dataset_split "test" \
    --n 32 \
    --temperature 0.6 \
    --use_vllm True \
    --save_path "${SAVE_DIR}/gsm8k_voting_generations.json" \
    2>&1 | tee ${SAVE_DIR}/gsm8k_voting_eval.log