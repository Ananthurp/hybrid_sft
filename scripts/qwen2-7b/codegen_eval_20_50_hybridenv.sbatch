#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J codegen_eval_20_50
#SBATCH -o codegen_eval_20_50-%j.out
#SBATCH -e codegen_eval_20_50-%j.err
set -euo pipefail
set -x

# --- Env (use the known-good one) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss
module load cuda/12.4.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.4.1}"

# Node-local tmp (optional)
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

BASE="$HOME/LLMDiversity/evals_codegen"
RUNS=(
  "codegen_phase1_ce"
  "codegen_phase2_gem_ckpt1434"
  "codegen_phase3_hybrid_a0.75_ckpt1434"
  "codegen_phase4_hybrid_a0.5_ckpt1000"
  "codegen_phase5_sparsemax_ckpt1434"
)

truncate_and_eval () {
  local src_jsonl="$1"
  local out_dir; out_dir="$(dirname "$src_jsonl")"

  for B in 20 50; do
    local tgt_jsonl="${src_jsonl%.jsonl}_top${B}.jsonl"
    echo "[info] Truncating to top ${B}: $tgt_jsonl"
    python ~/LLMDiversity/hybrid_sft/evaluation/truncate_evalplus_jsonl.py \
      "$src_jsonl" "$tgt_jsonl" "$B"

    echo "[info] Evaluating Humaneval (B=${B})"
    # Use module invocation to avoid console-script PATH issues
    python -m evalplus.evaluate --dataset humaneval --samples "$tgt_jsonl" \
      2>&1 | tee "${out_dir}/humaneval_top${B}.log"
  done
}

for RUN in "${RUNS[@]}"; do
  OUT_DIR="$BASE/$RUN"
  SRC="$OUT_DIR/evalplus-humaneval.jsonl"
  if [[ ! -f "$SRC" ]]; then
    echo "[warn] Missing $SRC — skipping $RUN"
    continue
  fi
  echo "[info] Processing: $RUN"
  truncate_and_eval "$SRC"
done

echo "[done] Created _top20/_top50 JSONLs and *_eval_results.json in each run dir."