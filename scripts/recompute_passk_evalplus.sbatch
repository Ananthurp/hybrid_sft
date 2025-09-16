#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p l40sn
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 05:00:00
#SBATCH -J passk_evalplus
#SBATCH -o passk_evalplus-%j.out
#SBATCH -e passk_evalplus-%j.err

set -euo pipefail

# --- env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118

# dirs to process (edit if you want a subset)
RUN_DIRS=(
  "$HOME/LLMDiversity/evals_codegen/codegen_phase1_ce"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase2_gem_ckpt1434"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase3_hybrid_a0.75_ckpt1434"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase4_hybrid_a0.5_ckpt1000"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase5_sparsemax_ckpt1434"
)

KS="1,10,20,50,100"

# Tiny helper to compute pass@k from detailed per-sample outcomes, if present
cat > /tmp/calc_passk_from_details.py <<'PY'
import json, os, sys, math
from collections import defaultdict
from math import comb

def unbiased_passk(n, c, k):
    if k > n: return None
    # 1 - C(n - c, k) / C(n, k)
    return 1.0 - (comb(n - c, k) / comb(n, k)) if c>0 else 0.0

def main(run_dir):
    # Prefer detailed JSONL if present (per-sample results)
    details = os.path.join(run_dir, "evalplus-humaneval_eval_details.jsonl")
    results  = os.path.join(run_dir, "evalplus-humaneval_eval_results.json")
    ks = [1,10,20,50,100]
    base = {k: None for k in ks}
    plus = {k: None for k in ks}

    if os.path.exists(details):
        # Build per-task counts n (samples) and c (number of passed) for base and plus
        per_task_counts = defaultdict(lambda: {"n":0,"c_base":0,"c_plus":0})
        with open(details, "r") as f:
            for line in f:
                rec = json.loads(line)
                tid = rec.get("task_id") or rec.get("task") or rec.get("taskname")
                if not tid: continue
                per_task_counts[tid]["n"] += 1
                if rec.get("base_passed") is True: per_task_counts[tid]["c_base"] += 1
                if rec.get("plus_passed") is True: per_task_counts[tid]["c_plus"] += 1
        # Average unbiased estimator over tasks
        def avg_passk(which):
            vals=[]
            for _,d in per_task_counts.items():
                n = d["n"]; c = d[which]
                for k in ks:
                    pass
            out={}
            for k in ks:
                sc=[]
                for _,d in per_task_counts.items():
                    n = d["n"]; c = d[which]
                    v = unbiased_passk(n, c, k)
                    if v is not None: sc.append(v)
                out[k] = (sum(sc)/len(sc)) if sc else None
            return out
        base = avg_passk("c_base")
        plus = avg_passk("c_plus")

    else:
        # Fall back: try aggregated results JSON (may only have 1/10/100)
        if os.path.exists(results):
            with open(results,"r") as f: r = json.load(f)
            # Try multiple possible layouts
            # 1) flat keys like {"base": {"pass@1": x, "pass@10": y, ...}, "plus": {...}}
            b = r.get("base") or r.get("humaneval_base") or {}
            p = r.get("plus") or r.get("humaneval_plus") or {}
            for k in ks:
                base[k] = b.get(f"pass@{k}")
                plus[k] = p.get(f"pass@{k}")
        # else: nothing available

    out = {
        "run_dir": run_dir,
        "K": ks,
        "base_pass@k": {str(k): (None if base[k] is None else round(base[k], 3)) for k in ks},
        "plus_pass@k": {str(k): (None if plus[k] is None else round(plus[k], 3)) for k in ks},
        "source": "details" if os.path.exists(details) else "results_json"
    }
    with open(os.path.join(run_dir, "passk_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {os.path.join(run_dir, 'passk_summary.json')} ({out['source']})")

if __name__ == "__main__":
    main(sys.argv[1])
PY

for RD in "${RUN_DIRS[@]}"; do
  echo "[info] run_dir: $RD"
  # If we *don't* have detailed JSONL, re-run evalplus to produce it **from the saved completions**.
  # This DOES NOT resample; it just (re)tests the existing samples.
  if [[ ! -f "$RD/evalplus-humaneval_eval_details.jsonl" ]]; then
    if [[ -f "$RD/evalplus-humaneval.jsonl" ]]; then
      echo "[info] re-evaluating saved samples to produce per-sample details..."
      # The following is the common pattern for evalplus; if your local CLI differs,
      # you can adjust; this step only (re)creates eval_details, not new generations.
      python - <<'PY'
import evalplus, os, json
from evalplus.evaluate import evaluate  # part of evalplus
import sys
rd = os.environ.get("RD")
inp = os.path.join(rd, "evalplus-humaneval.jsonl")
# This call reads completions JSONL and writes *_eval_results.json and *_eval_details.jsonl in the same dir.
evaluate(
    samples_jsonl=inp,
    parallelism=8,
    use_cache=True,
    # below options keep outputs next to the input file
    tmp_dir=None,
    out_dir=os.path.dirname(inp),
    report_all=True,
)
print("[ok] evalplus.evaluate finished for", inp)
PY
    else
      echo "[warn] No evalplus-humaneval.jsonl in $RD; cannot re-evaluate."
    fi
  fi

  RD="$RD" python /tmp/calc_passk_from_details.py "$RD"
done

echo "[all done]"