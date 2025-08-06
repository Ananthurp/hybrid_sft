#!/bin/sh

set -e 
set -x

# This script runs the AlpacaEval benchmark using a reward model.

# --- Step 0: Configuration ---
# The following line MUST be commented out for the first run to download the dataset and reward model.
# export TRANSFORMERS_OFFLINE=1 
export CUDA_VISIBLE_DEVICES="2" # Use a single free GPU (e.g., 1, 2, or 3)

# Define all paths clearly
MODEL_CHECKPOINT_PATH="/data/ananthu/gem_project/results/qwen_1.5b_gem_run/output_dir"
TOKENIZER_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
DATASET_PATH="tatsu-lab/alpaca_eval"
REWARD_MODEL_PATH="sfairXC/FsfairX-LLaMA3-RM-v0.1" 

# Define a clean save directory for the results
SAVE_DIR="${MODEL_CHECKPOINT_PATH}/evaluation_chat_alpaca" 
mkdir -p $SAVE_DIR

# Define generation parameters (from paper's Appendix)
SEED=42
T=0.6
K=50
P=0.9
# N is set to 32 to match the paper's "BON@32" (Best-of-N) evaluation
N=32

# --- Step 1: Generate 32 responses for each AlpacaEval prompt ---
echo "Generating ${N} responses for each prompt in AlpacaEval..."

# Using corrected paths to the python scripts
python evaluation/generate_response.py \
    --model_name_or_path $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATASET_PATH \
    --split "eval" \
    --column_name "instruction" \
    --max_size 100 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 2048 \
    --n $N \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.7 \
    --do_sample True \
    --remove_old True \
    --save_path "${SAVE_DIR}/generated_responses.json"


# --- Step 2: Evaluate the generated responses with the reward model ---
echo "Scoring generated responses with the reward model..."

# Using corrected paths to the python scripts
python evaluation/evaluation_reward.py \
    --model_name_or_path $REWARD_MODEL_PATH \
    --batch_size 8 \
    --detokenizer_path $TOKENIZER_PATH \
    --data_path "${SAVE_DIR}/generated_responses.json" \
    --save_path "${SAVE_DIR}/reward_scores.json"  \
    2>&1 | tee ${SAVE_DIR}/reward_eval.log

echo "AlpacaEval evaluation complete. Results saved in ${SAVE_DIR}"