# #!/bin/bash
# set -e; set -x;

# # --- Environment & Paths ---
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_TIMEOUT=1800
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_DISTRIBUTED_DEBUG=INFO

# MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
# TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
# OUTPUT_DIR="/data/ananthu/gem_project/results/qwen_1.5b_hybrid_2gpu_run"
# TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
# DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"

# mkdir -p $OUTPUT_DIR

# # --- Launch on 2 GPUs (0 and 3) with per-device batch size of 1 ---
# deepspeed --include localhost:0,3 ${TRAIN_SCRIPT_PATH} \
#     --deepspeed ${DEEPSPEED_CONFIG_PATH} \
#     --seed 1234 \
#     --model_name_or_path ${MODEL_PATH} \
#     --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
#     --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
#     --output_dir ${OUTPUT_DIR} \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 66 \
#     --num_train_epochs 3 \
#     --save_strategy "epoch" \
#     --loss "hybrid" \
#     --ns_alpha 0.5 \
#     --ns_tau 0.065 \
#     --learning_rate 2e-6 \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --logging_steps 10 \
#     --gradient_checkpointing True \
#     --overwrite_output_dir \
#     --bf16 True \
#     --use_flash_attn True \
#     2>&1 | tee $OUTPUT_DIR/training.log

#!/bin/bash
set -e; set -x;

# --- Environment & Paths ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO

MODEL_PATH="/data/ananthu/gem_project/models/Qwen2-1.5B-Instruct"
TOKENIZED_DATA_DIR="/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b"
OUTPUT_DIR="/data/ananthu/gem_project/results/qwen_1.5b_hybrid_2gpu_run"
TRAIN_SCRIPT_PATH="/data/ananthu/gem_project/code/GEM/train.py"
DEEPSPEED_CONFIG_PATH="/data/ananthu/gem_project/code/GEM/scripts/deepspeed_config_qwen.json"

mkdir -p $OUTPUT_DIR

# --- Launch on 2 GPUs with stability-focused hyperparameters ---
deepspeed --include localhost:0,3 ${TRAIN_SCRIPT_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    --seed 1234 \
    --model_name_or_path ${MODEL_PATH} \
    --train_tokenized_file "${TOKENIZED_DATA_DIR}/train.jsonl" \
    --test_tokenized_file "${TOKENIZED_DATA_DIR}/test.jsonl" \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 66 \
    --num_train_epochs 3 \
    --save_strategy "epoch" \
    --loss "hybrid" \
    --ns_alpha 0.5 \
    --ns_tau 0.065 \
    --learning_rate 2e-6 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    --use_flash_attn True \
    2>&1 | tee $OUTPUT_DIR/training.log