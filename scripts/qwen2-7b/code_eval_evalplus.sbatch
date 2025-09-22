#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 20:00:00
#SBATCH -J CODE_EVAL
#SBATCH -o CODE_EVAL-%j.out
#SBATCH -e CODE_EVAL-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/11.8.0}"

# Make sure our evaluation modules are importable
export PYTHONPATH="$HOME/LLMDiversity/hybrid_sft:${PYTHONPATH:-}"

# Node-local caches
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
N_SAMPLES="${N_SAMPLES:-100}"           # per problem (needed for pass@100)
MAX_NEW="${MAX_NEW:-512}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.9}"
USE_VLLM="${USE_VLLM:-True}"

# text2code needs a "model key" for prompt/FIM presets
MODEL_KEY="${MODEL_KEY:-bigcode/starcoder}"

# Optional: path to a local evalplus repo if you prefer its tools.sanitize
EVALPLUS_SRC="${EVALPLUS_SRC:-}"

# vLLM port scaffolding (harmless for HF fallback)
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

mkdir -p "$OUT_DIR"

# Common args for your text2code entrypoint
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

# Helper: run extra-k evaluation in-process to get 1/10/20/50/100 on evalplus 0.3.1
run_extra_k () {
python - "$@" <<'PY'
import json, sys, os, glob
from evalplus import evaluate as E

dataset, samples_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
k_list = [1, 10, 20, 50, 100]
try:
    E.evaluate(dataset=dataset, samples=samples_path, k=k_list)
except TypeError:
    # Older API without 'k': falls back to 1/10/100
    E.evaluate(dataset=dataset, samples=samples_path)

# Try to pick up the *_eval_results.json written by evalplus for machine-readable output
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
    json.dump(metrics or {"note": "see stdout logs for full table"}, f, indent=2)
print(f"[extra-k] wrote metrics JSON to {out_json}")
PY
}

# -------------------------
# 1) HumanEval generation
# -------------------------
DATASET=humaneval
SAVE_PATH="$OUT_DIR/evalplus-${DATASET}.jsonl"
python -m evaluation.text2code \
  "${gen_common_args[@]}" \
  --dataset "$DATASET" \
  --save_path "$SAVE_PATH"

# Evaluate (CLI prints table with default ks)
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$SAVE_PATH" \
  2>&1 | tee "$OUT_DIR/${DATASET}.log"

# Extra pass@{1,10,20,50,100}
run_extra_k "$DATASET" "$SAVE_PATH" "$OUT_DIR/${DATASET}_metrics_extra_k.json" \
  2>&1 | tee -a "$OUT_DIR/${DATASET}.log"

# -------------------------
# 2) MBPP generation
# -------------------------
DATASET=mbpp
SAVE_PATH="$OUT_DIR/evalplus-${DATASET}.jsonl"
SANITIZED_PATH="$OUT_DIR/evalplus-${DATASET}-sanitized.jsonl"

python -m evaluation.text2code \
  "${gen_common_args[@]}" \
  --dataset "$DATASET" \
  --save_path "$SAVE_PATH"

# Sanitize (evalplus 0.3.1 uses positional args)
if python - <<'PY'
try:
    import evalplus.sanitize  # noqa: F401
    print("ok")
except Exception:
    print("no")
PY
then
  python -m evalplus.sanitize "$DATASET" "$SAVE_PATH"
else
  if [[ -n "$EVALPLUS_SRC" ]]; then
    PYTHONPATH="$EVALPLUS_SRC${PYTHONPATH:+:$PYTHONPATH}" python -m tools.sanitize "$DATASET" "$SAVE_PATH"
  else
    echo "[warn] evalplus.sanitize not found and EVALPLUS_SRC not set; copying unsanitized -> sanitized."
    cp "$SAVE_PATH" "$SANITIZED_PATH"
  fi
fi

# Ensure sanitized path exists
if [[ ! -f "$SANITIZED_PATH" ]]; then
  if [[ -f "${SAVE_PATH%.jsonl}-sanitized.jsonl" ]]; then
    mv "${SAVE_PATH%.jsonl}-sanitized.jsonl" "$SANITIZED_PATH"
  else
    cp "$SAVE_PATH" "$SANITIZED_PATH"
  fi
fi

# Evaluate MBPP
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$SANITIZED_PATH" \
  2>&1 | tee "$OUT_DIR/${DATASET}.log"

# Extra pass@{1,10,20,50,100}
run_extra_k "$DATASET" "$SANITIZED_PATH" "$OUT_DIR/${DATASET}_metrics_extra_k.json" \
  2>&1 | tee -a "$OUT_DIR/${DATASET}.log"

echo "[done] Wrote outputs under: $OUT_DIR"
ls -lh "$OUT_DIR" || true