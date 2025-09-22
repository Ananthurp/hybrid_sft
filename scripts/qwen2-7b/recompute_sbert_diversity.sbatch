#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1          # GPU not required, but harmless; SBERT can use it
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J SBERT_DIV
#SBATCH -o SBERT_DIV-%j.out
#SBATCH -e SBERT_DIV-%j.err

set -euo pipefail
set -x

# --- Env (sentence-transformers is in this env) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120

# Light CUDA is fine; SBERT will use GPU if available
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# Caches
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

# --- Inputs (via --export) ---
: "${BASE_TOKENIZER_DIR:?set BASE_TOKENIZER_DIR=/abs/path/to/base_tokenizer (e.g. $HOME/LLMDiversity/models/Qwen2-7B)}"
PHASE6_DIR="${PHASE6_DIR:-$HOME/LLMDiversity/results/qwen2-7b/phase6_ce_wd}"
PHASE7_DIR="${PHASE7_DIR:-$HOME/LLMDiversity/results/qwen2-7b/phase7_neft_alpha5}"
OUT_SUBDIR="${OUT_SUBDIR:-output_diversity}"  # where your earlier outputs live

# Targets
runs=("$PHASE6_DIR" "$PHASE7_DIR")
subsets=(evaluation_chat_alpaca evaluation_poem evaluation_story)

div_py="$HOME/LLMDiversity/hybrid_sft/evaluation/evaluation_diversity.py"

for run_dir in "${runs[@]}"; do
  for sub in "${subsets[@]}"; do
    d="$run_dir/$OUT_SUBDIR/$sub"
    resp="$d/responses.json"
    out_json="$d/diversity_summary.json"
    log="$d/diversity_metrics_sbert_recompute.log"

    if [[ -f "$resp" ]]; then
      echo "[info] Recomputing diversity for: $resp"
      # We simply rerun the script; now that sentence-transformers is installed,
      # SBERT diversity will be included and file will be refreshed.
      python -u "$div_py" \
        --response_path  "$resp" \
        --tokenizer_path "$BASE_TOKENIZER_DIR" \
        --detokenizer_path "$BASE_TOKENIZER_DIR" \
        --out_path "$out_json" \
        > "$log" 2>&1 || echo "[warn] diversity step failed for $resp (see $log)"
    else
      echo "[skip] Missing responses.json at: $resp"
    fi
  done
done

echo "[done] Recomputed diversity (incl. SBERT) for phase6_ce_wd and phase7_neft_alpha5."