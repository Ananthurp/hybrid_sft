#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -J PASSK_HE
#SBATCH -o PASSK_HE-%j.out
#SBATCH -e PASSK_HE-%j.err

set -euo pipefail
set -x

# --- Env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer120
module load cuda/11.8.0 || true

# --- Inputs (via --export) ---
: "${SAMPLES:?set SAMPLES=/abs/path/evalplus-humaneval.jsonl}"   # original 100-sample-per-task file
DATASET="${DATASET:-humaneval}"                                  # keep default
OUT_PREFIX="${OUT_PREFIX:-${SAMPLES%.jsonl}}"

# Paths
FULL_JSON="${OUT_PREFIX}_eval_results.json"          # produced by evalplus on full file
S20="${OUT_PREFIX}-pt20.jsonl"
S50="${OUT_PREFIX}-pt50.jsonl"
R20="${OUT_PREFIX}-pt20_eval_results.json"
R50="${OUT_PREFIX}-pt50_eval_results.json"
MERGED="${OUT_PREFIX}_passk_1_10_20_50_100.json"

# 1) Evaluate the full file (gives pass@1/10/100)
evalplus.evaluate --dataset "$DATASET" --samples "$SAMPLES" 2>&1 | tee "${OUT_PREFIX}_reval_full.log"

# 2) Make per-task truncated files: 20 and 50
python "$HOME/LLMDiversity/hybrid_sft/evaluation/truncate_evalplus_jsonl.py" "$SAMPLES" "$S20" 20
python "$HOME/LLMDiversity/hybrid_sft/evaluation/truncate_evalplus_jsonl.py" "$SAMPLES" "$S50" 50

# 3) Evaluate the truncated files (their pass@100 equals original pass@20 / pass@50)
evalplus.evaluate --dataset "$DATASET" --samples "$S20" 2>&1 | tee "${OUT_PREFIX}_reval_pt20.log"
evalplus.evaluate --dataset "$DATASET" --samples "$S50" 2>&1 | tee "${OUT_PREFIX}_reval_pt50.log"

# 4) Merge metrics into one clean printout + compact JSON
python - <<'PY'
import json, os, sys, glob

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def grab(res, suite, ks=(1,10,100)):
    obj = res.get(suite, {})
    return {f"pass@{k}": obj.get(f"pass@{k}") for k in ks}

# Inputs from env (expanded by bash above)
FULL_JSON = os.environ["FULL_JSON"]
R20 = os.environ["R20"]
R50 = os.environ["R50"]
DATASET = os.environ.get("DATASET","humaneval")
MERGED = os.environ["MERGED"]

# Load result JSONs
full = load_json(FULL_JSON)
pt20 = load_json(R20)
pt50 = load_json(R50)

def fmt(d):
    return ", ".join(f"@{k}={d.get(f'pass@{k}','NA')}" for k in [1,10,20,50,100])

def combine_suite(name):
    # Base values from the full run
    b = grab(full, name, ks=(1,10,100))
    # pass@20 := pass@100 on pt20
    b20 = grab(pt20, name, ks=(100,)).get("pass@100")
    # pass@50 := pass@100 on pt50
    b50 = grab(pt50, name, ks=(100,)).get("pass@100")
    merged = {
        "pass@1": b.get("pass@1"),
        "pass@10": b.get("pass@10"),
        "pass@20": b20,
        "pass@50": b50,
        "pass@100": b.get("pass@100"),
    }
    return merged

base = combine_suite(DATASET)
plus = combine_suite(DATASET + "+")

print(f"{DATASET} (base):       " + fmt(base))
print(f"{DATASET}+ (base+extra): " + fmt(plus))

with open(MERGED, "w") as f:
    json.dump({DATASET: base, DATASET+"+": plus,
               "sources": {"full": FULL_JSON, "pt20": R20, "pt50": R50}}, f, indent=2)
print(f"[info] wrote merged metrics to {MERGED}")
PY