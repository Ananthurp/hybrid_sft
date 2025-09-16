#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p cpun
#SBATCH -N 1
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J codegen-passk
#SBATCH -o codegen-passk-%j.out
#SBATCH -e codegen-passk-%j.err

set -euo pipefail
set -x

# --- Env (EvalPlus lives here already on your cluster) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

# Node-local caches
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

BASE="${BASE:-$HOME/LLMDiversity/evals_codegen}"

# Comma-separated run folders under $BASE (override with --export RUNS="a,b,c")
RUNS_CSV="${RUNS:-codegen_phase1_ce,codegen_phase2_gem_ckpt1434,codegen_phase3_hybrid_a0.75_ckpt1434,codegen_phase4_hybrid_a0.5_ckpt1000,codegen_phase5_sparsemax_ckpt1434,codegen_phase6_ce_wd,codegen_phase7_neft_a5}"
IFS=',' read -ra RUNS_ARR <<< "$RUNS_CSV"

# Helper: robustly find the eval results JSON produced by evalplus
find_results_json() {
  local out_dir="$1" ; local dataset="$2"
  # typical names produced by evalplus:
  # evalplus-humaneval.eval_results.json OR evalplus-humaneval_eval_results.json
  local a="$out_dir/evalplus-${dataset}.eval_results.json"
  local b="$out_dir/evalplus-${dataset}_eval_results.json"
  if [[ -f "$a" ]]; then echo "$a"; return 0; fi
  if [[ -f "$b" ]]; then echo "$b"; return 0; fi
  # fallback: newest match
  ls -1t "$out_dir"/evalplus-${dataset}*.eval_results.json "$out_dir"/evalplus-${dataset}*_eval_results.json 2>/dev/null | head -n1
}

# Tiny Python to compute pass@k directly from the single 100-sample eval file
PASSK_PY="$LOCAL_SCRATCH/compute_passk.py"
cat > "$PASSK_PY" <<'PY'
import json, argparse, math
from collections import defaultdict
def pass_at_k(n, c, k):
    if c == 0: return 0.0
    if k > n: k = n
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
def extract_samples(obj):
    out=[]
    if isinstance(obj, dict):
        if "task_id" in obj and ("base_status" in obj or "plus_status" in obj):
            out.append(obj)
        for v in obj.values(): out += extract_samples(v)
    elif isinstance(obj, list):
        for it in obj: out += extract_samples(it)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1,10,20,50,100])
    args = ap.parse_args()
    with open(args.file,"r") as f: data=json.load(f)
    samples = extract_samples(data)
    if not samples: raise SystemExit("No per-sample records found.")
    base=defaultdict(lambda:[0,0]); plus=defaultdict(lambda:[0,0])
    def ok(x): return str(x).lower()=="pass"
    for s in samples:
        tid=s["task_id"]
        if "base_status" in s:
            base[tid][0]+=1; base[tid][1]+= ok(s["base_status"])
        if "plus_status" in s:
            plus[tid][0]+=1; plus[tid][1]+= ok(s["plus_status"])
    def summarize(d, ks):
        tids=list(d.keys()); out={}
        for k in ks:
            vals=[pass_at_k(d[t][0], d[t][1], k) for t in tids]
            out[f"pass@{k}"]=sum(vals)/len(vals) if vals else float("nan")
        return out
    ks=args.k
    b=summarize(base,ks); p=summarize(plus,ks)
    print("== Humaneval base ==")
    for k in ks: print(f"pass@{k}:\t{b[f'pass@{k}']:.3f}")
    if p:
        print("\n== Humaneval+ (extra) ==")
        for k in ks: print(f"pass@{k}:\t{p[f'pass@{k}']:.3f}")
PY

SUMMARY_CSV="$BASE/_summary_passk.csv"
echo "run,split,pass@1,pass@10,pass@20,pass@50,pass@100" > "$SUMMARY_CSV"

for RUN in "${RUNS_ARR[@]}"; do
  OUT_DIR="$BASE/$RUN"
  JSONL="$OUT_DIR/evalplus-humaneval.jsonl"
  if [[ ! -f "$JSONL" ]]; then
    echo "[warn] $RUN: missing $JSONL — skip"
    continue
  fi

  # Run evalplus once to produce per-sample results (if not already there)
  RES_JSON="$(find_results_json "$OUT_DIR" humaneval || true)"
  if [[ -z "${RES_JSON}" || ! -f "${RES_JSON}" ]]; then
    echo "[info] $RUN: running evalplus on 100-sample file"
    evalplus.evaluate --dataset humaneval --samples "$JSONL" 2>&1 | tee "$OUT_DIR/humaneval.log"
    RES_JSON="$(find_results_json "$OUT_DIR" humaneval || true)"
  fi

  if [[ -z "${RES_JSON}" || ! -f "${RES_JSON}" ]]; then
    echo "[error] $RUN: could not find eval results JSON after evaluation"
    ls -l "$OUT_DIR" || true
    exit 1
  fi

  echo
  echo "====================  $RUN  ===================="
  python "$PASSK_PY" --file "$RES_JSON" --k 1 10 20 50 100 | tee "$OUT_DIR/passk_summary.txt"

  # Append to CSV (base & plus)
  BASE_=$(python "$PASSK_PY" --file "$RES_JSON" --k 1 10 20 50 100 | awk '/^== Humaneval base ==/{f=1;next} /^==/{f=0} f' | awk '{print $2}' | paste -sd, -)
  PLUS_=$(python "$PASSK_PY" --file "$RES_JSON" --k 1 10 20 50 100 | awk '/^\=\= Humaneval\+/{f=1;next} /^$/{f=0} f' | awk '{print $2}' | paste -sd, -)

  if [[ -n "$BASE_" ]]; then echo "$RUN,base,$BASE_" >> "$SUMMARY_CSV"; fi
  if [[ -n "$PLUS_" ]]; then echo "$RUN,plus,$PLUS_" >> "$SUMMARY_CSV"; fi
done

echo
echo "[done] Summary CSV -> $SUMMARY_CSV"