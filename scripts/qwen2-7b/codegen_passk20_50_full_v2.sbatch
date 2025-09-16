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

# --- Env (CPU-only) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid_loss

# Node-local caches (optional)
export LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"
export HF_HOME="$LOCAL_SCRATCH/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

# Where your prior Humaneval generations live
BASE="${BASE:-$HOME/LLMDiversity/evals_codegen}"

# Comma-separated run folders (override with --export RUNS="a,b,c")
RUNS_CSV="${RUNS:-codegen_phase1_ce,codegen_phase2_gem_ckpt1434,codegen_phase3_hybrid_a0.75_ckpt1434,codegen_phase4_hybrid_a0.5_ckpt1000,codegen_phase5_sparsemax_ckpt1434}"
IFS=',' read -ra RUNS_ARR <<< "$RUNS_CSV"

# --- tiny helper to compute pass@k from results json ---
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
    ap.add_argument("--k", type=int, nargs="+", default=[1,10,20,50])
    args = ap.parse_args()
    with open(args.file,"r") as f: data=json.load(f)
    samples = extract_samples(data)
    if not samples: raise SystemExit("No per-sample records found.")
    base = defaultdict(lambda:[0,0]); plus=defaultdict(lambda:[0,0])
    def ok(x): return str(x).lower()=="pass"
    for s in samples:
        tid=s["task_id"]
        if "base_status" in s:
            base[tid][0]+=1; base[tid][1]+= ok(s["base_status"])
        if "plus_status" in s:
            plus[tid][0]+=1; plus[tid][1]+= ok(s["plus_status"])
    def summarize(d, ks):
        tids = list(d.keys()); out={}
        for k in ks:
            vals=[pass_at_k(d[t][0], d[t][1], k) for t in tids]
            out[f"pass@{k}"]=sum(vals)/len(vals) if vals else float("nan")
        return out
    b=summarize(base,args.k); p=summarize(plus,args.k)
    print("== Humaneval base ==")
    for k in args.k: print(f"pass@{k}:\t{b[f'pass@{k}']:.3f}")
    if p:
        print("\n== Humaneval+ (extra) ==")
        for k in args.k: print(f"pass@{k}:\t{p[f'pass@{k}']:.3f}")
if __name__=="__main__": main()
PY

# --- function: truncate + evaluate + compute pass@k (robust to file naming) ---
truncate_and_eval() {
  local src_jsonl="$1"     # .../evalplus-humaneval.jsonl
  local out_dir; out_dir="$(dirname "$src_jsonl")"
  local stem="${src_jsonl%.jsonl}"

  for B in 20 50; do
    local tgt_jsonl="${stem}_top${B}.jsonl"
    echo "[info] Ensuring truncated file exists: $tgt_jsonl"
    if [[ ! -f "$tgt_jsonl" ]]; then
      python "$HOME/LLMDiversity/hybrid_sft/evaluation/truncate_evalplus_jsonl.py" "$src_jsonl" "$tgt_jsonl" "$B"
    fi

    echo "[info] Evaluating Humaneval (B=${B})"
    # Only re-run if neither naming style exists
    local res_json_underscore="${tgt_jsonl%.jsonl}_eval_results.json"
    local res_json_dot="${tgt_jsonl%.jsonl}.eval_results.json"
    if [[ ! -f "$res_json_underscore" && ! -f "$res_json_dot" ]]; then
      evalplus.evaluate --dataset humaneval --samples "$tgt_jsonl" \
        2>&1 | tee "${out_dir}/humaneval_top${B}.log"
    fi

    # Resolve whichever file EvalPlus created
    local RES_JSON=""
    if [[ -f "$res_json_underscore" ]]; then
      RES_JSON="$res_json_underscore"
    elif [[ -f "$res_json_dot" ]]; then
      RES_JSON="$res_json_dot"
    else
      # last resort: any match starting with the stem
      RES_JSON="$(ls -1 "${tgt_jsonl%.jsonl}"*.eval_results.json 2>/dev/null | head -n 1 || true)"
    fi

    if [[ -z "${RES_JSON}" || ! -f "${RES_JSON}" ]]; then
      echo "[error] Could not find evalplus results JSON for $tgt_jsonl"
      ls -l "${out_dir}" || true
      exit 1
    fi

    echo "[info] Computing pass@{1,10,${B}} for ${RES_JSON}"
    python "$PASSK_PY" --file "$RES_JSON" --k 1 10 "$B" \
      | tee "${out_dir}/passk_top${B}.txt"
  done
}

# --- process runs ---
for RUN in "${RUNS_ARR[@]}"; do
  OUT_DIR="$BASE/$RUN"
  SRC="$OUT_DIR/evalplus-humaneval.jsonl"
  if [[ ! -f "$SRC" ]]; then
    echo "[warn] Skipping $RUN (missing $SRC)"
    continue
  fi
  echo "[info] Processing: $RUN"
  truncate_and_eval "$SRC"
done

echo "[done] Check each run folder for passk_top20.txt and passk_top50.txt"