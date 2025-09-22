#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 20:00:00
#SBATCH -J CODE_EVAL_HE
#SBATCH -o CODE_EVAL_HE-%j.out
#SBATCH -e CODE_EVAL_HE-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# Ensure our evaluation modules are importable
export PYTHONPATH="$HOME/LLMDiversity/hybrid_sft:${PYTHONPATH:-}"

# Node-local caches (faster + avoids HF cache contention)
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_DIR="$LOCAL_SCRATCH/vllm"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" "$TMPDIR"
df -h "$LOCAL_SCRATCH" || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

# --- Inputs (via --export) ---
: "${MODEL_DIR:?set MODEL_DIR=/abs/path/to/weights}"
: "${RUN_NAME:?set RUN_NAME=short_tag_for_outputs}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH=/abs/path/to/base_tokenizer}"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/codegen_latest}"

# generation settings
N_SAMPLES="${N_SAMPLES:-100}"   # per problem (needed for pass@100)
MAX_NEW="${MAX_NEW:-512}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.9}"
USE_VLLM="${USE_VLLM:-True}"

# text2code needs a model key only for preset/template choices
MODEL_KEY="${MODEL_KEY:-bigcode/starcoder}"

# vLLM port scaffolding (harmless for HF fallback)
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

mkdir -p "$OUT_DIR"

# Common args to your text2code entrypoint
gen_common_args=(
  --model_key "$MODEL_KEY"
  --model_name_or_path "$MODEL_DIR"
  --tokenizer_name_or_path "$TOKENIZER_PATH"
  --temperature "$TEMP"
  --top_p "$TOP_P"
  --max_new_tokens "$MAX_NEW"
  --n_problems_per_batch 100
  --n_samples_per_problem "$N_SAMPLES"
  --n_batches 1
  --use_vllm "$USE_VLLM"
  --template normal
)

# Helper: compute pass@{1,10,20,50,100} programmatically (compatible with evalplus 0.3.1)
run_extra_k () {
python - "$@" <<'PY'
import json, sys, os, glob
from evalplus import evaluate as E

dataset, samples_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
k_list = [1, 10, 20, 50, 100]

try:
    # Programmatic path (works even if CLI lacks --k)
    E.evaluate(dataset=dataset, samples=samples_path, k=k_list)
except TypeError:
    # Very old versions without 'k' param: fall back to defaults (1/10/100)
    E.evaluate(dataset=dataset, samples=samples_path)

# Try to collect metrics from the usual *_eval_results.json written by evalplus
prefix = os.path.splitext(os.path.basename(samples_path))[0]
cand = glob.glob(os.path.join(os.path.dirname(samples_path), f"{prefix}_eval_results.json"))
metrics = {}
if cand:
    try:
        with open(cand[0], "r") as f:
            metrics = json.load(f)
    except Exception:
        pass

with open(out_json, "w") as f:
    json.dump(metrics or {"note": "see stdout logs for metrics"}, f, indent=2)

print(f"[extra-k] wrote metrics JSON to {out_json}")
PY
}

# -------------------------
# HumanEval generation
# -------------------------
DATASET=humaneval
SAVE_PATH="$OUT_DIR/evalplus-${DATASET}.jsonl"

python -m evaluation.text2code \
  "${gen_common_args[@]}" \
  --dataset "$DATASET" \
  --save_path "$SAVE_PATH"

# CLI evaluation (no --k; prints table and writes *_eval_results.json)
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$SAVE_PATH" \
  2>&1 | tee "$OUT_DIR/${DATASET}.log"

# Extra pass@{1,10,20,50,100}
run_extra_k "$DATASET" "$SAVE_PATH" "$OUT_DIR/${DATASET}_metrics_extra_k.json" \
  2>&1 | tee -a "$OUT_DIR/${DATASET}.log"

echo "[done] Wrote outputs under: $OUT_DIR"
ls -lh "$OUT_DIR" || true