#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=qwen2-7b-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1

set -euxo pipefail

module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

# ---- Load a valid CUDA toolkit so DeepSpeed import won't error ----
# You have: cuda/11.2.2  cuda/11.8.0  cuda/12.4.1  cuda/12.6.2
module load cuda/12.4.1 || module load cuda/12.6.2 || module load cuda/11.8.0

# If CUDA_HOME wasn't exported by the module, derive it from nvcc
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"

# ---- Never build DS CUDA ops on the node ----
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_SKIP_CUDA_CHECK=1

# (optional) show versions
python - <<'PY'
import os, deepspeed, torch, transformers, datasets
print("CUDA_HOME   :", os.environ.get("CUDA_HOME"))
print("nvcc exists :", os.system("command -v nvcc >/dev/null 2>&1")==0)
print("deepspeed   :", deepspeed.__version__)
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets    :", datasets.__version__)
PY

# ---- Smoke settings (short & cheap) ----
export GPUS_PER_NODE=1
export MICRO_BS=1
export GA=4
export EPOCHS=1
export SAVE_STRATEGY=epoch
export EVAL_STRATEGY=no
export MAX_STEPS=80
export MAX_TRAIN_SAMPLES=2000
export REPORT_TO=none

bash "$HOME/LLMDiversity/hybrid_sft/scripts/qwen2-7b/train_qwen2_7b_pipeline.sh"

