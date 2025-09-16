#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH -p cpun
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J passk_evalplus
#SBATCH -o passk_evalplus-%j.out
#SBATCH -e passk_evalplus-%j.err

set -euo pipefail

# --- Conda env (same as your other jobs) ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118

# --- Hardcoded inputs (edit if your layout changes) ---
KS_CSV="1,10,20,50,100"
RUN_DIRS=(
  "$HOME/LLMDiversity/evals_codegen/codegen_phase1_ce"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase2_gem_ckpt1434"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase3_hybrid_a0.75_ckpt1434"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase4_hybrid_a0.5_ckpt1000"
  "$HOME/LLMDiversity/evals_codegen/codegen_phase5_sparsemax_ckpt1434"
)

echo "[info] KS: ${KS_CSV}"
echo "[info] RUN_DIRS (${#RUN_DIRS[@]}):"
for r in "${RUN_DIRS[@]}"; do echo "  - ${r}"; done

WORKDIR="$(mktemp -d)"
SCRIPT="$WORKDIR/calc_passk_from_evalplus.py"
cat > "$SCRIPT" << 'PY'
import json, sys, os, re
from math import comb

def passk_from_counts(n, c, ks):
    out = {}
    for k in ks:
        if k <= 0 or n <= 0 or k > n:
            out[k] = 0.0
            continue
        if c <= 0:
            out[k] = 0.0
            continue
        if c >= n:
            out[k] = 1.0
            continue
        out[k] = 1.0 - (comb(n - c, k) / comb(n, k))
    return out

def agg_passk(pairs, ks):
    if not pairs:
        return {k: None for k in ks}
    sums = {k: 0.0 for k in ks}
    for (n, c) in pairs:
        pk = passk_from_counts(n, c, ks)
        for k in ks:
            sums[k] += pk[k]
    return {k: (sums[k] / len(pairs)) for k in ks}

def _maybe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def parse_counts(obj):
    """
    Try multiple likely shapes coming from evalplus caches.
    Returns: (base_pairs, plus_pairs) where each is a list of (n, c).
    """
    base, plus = [], []

    # 1) per-task dict with explicit n/c
    for key in ("per_task", "per_problem", "by_task", "details"):
        node = obj.get(key)
        if isinstance(node, dict):
            for _name, rec in node.items():
                for name, bag in (("base", base), ("plus", plus), ("humaneval", base), ("humaneval_plus", plus)):
                    sub = rec.get(name) or {}
                    n = sub.get("n") or sub.get("total") or sub.get("N") or sub.get("n_total")
                    c = sub.get("c") or sub.get("correct") or sub.get("C") or sub.get("n_correct")
                    n = _maybe_int(n); c = _maybe_int(c)
                    if n is not None and c is not None:
                        bag.append((n, c))
            if base or plus:
                return base, plus

    # 2) results dict with per-problem boolean arrays (e.g., "base_passed": [True/False,...])
    results = obj.get("results") or obj.get("data") or obj.get("items")
    if isinstance(results, dict):
        for rec in results.values():
            for key, bag in (("base_passed", base), ("plus_passed", plus), ("base", base), ("plus", plus)):
                arr = rec.get(key)
                if isinstance(arr, list) and arr:
                    n = len(arr)
                    c = sum(bool(x) for x in arr)
                    bag.append((n, c))
        if base or plus:
            return base, plus

    # 3) top-level list of objects with boolean arrays
    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict): 
                continue
            for key, bag in (("base_passed", base), ("plus_passed", plus), ("base", base), ("plus", plus)):
                arr = rec.get(key)
                if isinstance(arr, list) and arr:
                    n = len(arr)
                    c = sum(bool(x) for x in arr)
                    bag.append((n, c))
        if base or plus:
            return base, plus

    return base, plus

def parse_log(log_path):
    vals = {"base": {}, "plus": {}}
    if not os.path.exists(log_path):
        return vals
    mode = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().lower()
            if s.startswith("humaneval (base"):
                mode = "base"; continue
            if s.startswith("humaneval+"):
                mode = "plus"; continue
            m = re.search(r"pass@(\d+):\s*([0-9.]+)", s)
            if m and mode in ("base", "plus"):
                vals[mode][int(m.group(1))] = float(m.group(2))
    return vals

def process(run_dir, ks):
    run_dir = os.path.abspath(run_dir)
    out = {"run_dir": run_dir, "K": ks, "source": None,
           "base_pass@k": {}, "plus_pass@k": {}}

    # Prefer cached JSON if present
    for fname in ("evalplus-humaneval_eval_results.json", "human_eval_results.json"):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            try:
                data = json.load(open(fpath, "r", encoding="utf-8"))
            except Exception:
                continue
            base_pairs, plus_pairs = parse_counts(data)
            if base_pairs or plus_pairs:
                out["source"] = f"counts:{fname}"
                b = agg_passk(base_pairs, ks) if base_pairs else {k: None for k in ks}
                p = agg_passk(plus_pairs, ks) if plus_pairs else {k: None for k in ks}
                out["base_pass@k"] = {str(k): (None if b[k] is None else round(b[k], 3)) for k in ks}
                out["plus_pass@k"] = {str(k): (None if p[k] is None else round(p[k], 3)) for k in ks}
                return out

    # Fallback to parsing the log (gives pass@1/10/100 only if present)
    lv = parse_log(os.path.join(run_dir, "humaneval.log"))
    if lv["base"] or lv["plus"]:
        out["source"] = "log"
        for k in ks:
            out["base_pass@k"][str(k)] = (round(lv["base"][k], 3) if k in lv["base"] else None)
            out["plus_pass@k"][str(k)] = (round(lv["plus"][k], 3) if k in lv["plus"] else None)
        return out

    out["source"] = "none"
    out["base_pass@k"] = {str(k): None for k in ks}
    out["plus_pass@k"] = {str(k): None for k in ks}
    return out

def main():
    ks = [int(x) for x in os.environ.get("KS_CSV", "1,10,20,50,100").split(",") if x.strip()]
    run_dirs = sys.argv[1:]
    if not run_dirs:
        print("No run directories passed; exiting.", file=sys.stderr)
        sys.exit(1)
    for rd in run_dirs:
        res = process(rd, ks)
        op = os.path.join(rd, "passk_summary.json")
        json.dump(res, open(op, "w", encoding="utf-8"), indent=2)
        print(f"[ok] wrote {op}")
if __name__ == "__main__":
    main()
PY

# Execute once across all run dirs
python "$SCRIPT" "${RUN_DIRS[@]}"

echo "[all done]"