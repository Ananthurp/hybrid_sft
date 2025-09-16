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

# --- env ---
module load miniforge/24.11.3-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infer118

# --- inputs via --export ---
# RUN_DIRS (comma-separated), KS (comma-separated, default 1,10,20,50,100)
IFS=',' read -r -a RUNS <<< "${RUN_DIRS}"
KS_CSV="${KS:-1,10,20,50,100}"

# stage a tiny script
WORKDIR="$(mktemp -d)"
SCRIPT="$WORKDIR/calc_passk_from_evalplus.py"
cat > "$SCRIPT" << 'PY'
import json, sys, os, re, math
from math import comb

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def passk_from_counts(n, c, ks):
    # HumanEval estimator: p_k = 1 - C(n-c, k) / C(n, k)
    out = {}
    for k in ks:
        if k <= 0 or n <= 0 or k > n:
            out[k] = 0.0 if k > 0 else 0.0
            continue
        if c <= 0:
            out[k] = 0.0
            continue
        if c >= n:
            out[k] = 1.0
            continue
        out[k] = 1.0 - (comb(n - c, k) / comb(n, k))
    return out

def agg_passk(per_prob_counts, ks):
    # per_prob_counts: list of tuples (n, c)
    # aggregate by averaging across problems
    if not per_prob_counts:
        return {k: None for k in ks}
    totals = {k: 0.0 for k in ks}
    for (n, c) in per_prob_counts:
        pk = passk_from_counts(n, c, ks)
        for k in ks:
            totals[k] += pk[k]
    return {k: totals[k] / len(per_prob_counts) for k in ks}

def try_parse_counts_from_json(obj):
    """
    Tries many plausible evalplus shapes to extract per-problem (n, c) counts.
    Returns: dict(base=[(n,c),...], plus=[(n,c),...]) or None
    """
    base_counts, plus_counts = [], []

    # Common shape 1: {"per_task": {"HumanEval/0": {"base":{"n":N,"c":C},"plus":{"n":N,"c":C}}}}
    per_task = obj.get("per_task") or obj.get("per_problem") or obj.get("by_task")
    if isinstance(per_task, dict):
        for _, rec in per_task.items():
            b = rec.get("base") or {}
            p = rec.get("plus") or {}
            for src, lst in (("base", base_counts), ("plus", plus_counts)):
                node = rec.get(src) or {}
                # accept multiple key variants
                n = node.get("n") or node.get("n_total") or node.get("total") or node.get("N")
                c = node.get("c") or node.get("n_correct") or node.get("correct") or node.get("C")
                if isinstance(n, int) and isinstance(c, int):
                    lst.append((n, c))
        if base_counts or plus_counts:
            return {"base": base_counts, "plus": plus_counts}

    # Common shape 2: {"results": {"HumanEval/0": {"base_passed":[bool,bool,...], "plus_passed":[...]}}}
    results = obj.get("results") or obj.get("samples") or obj.get("data") or obj.get("items")
    if isinstance(results, dict):
        for _, rec in results.items():
            for src, lst, key_candidates in (
                ("base", base_counts, ["base_passed", "base", "passed_base", "passed"]),
                ("plus", plus_counts, ["plus_passed", "plus", "passed_plus"]),
            ):
                arr = None
                for k in key_candidates:
                    if k in rec:
                        arr = rec[k]
                        break
                if isinstance(arr, list) and arr and isinstance(arr[0], (bool, int)):
                    n = len(arr)
                    c = sum(bool(x) for x in arr)
                    lst.append((n, c))
        if base_counts or plus_counts:
            return {"base": base_counts, "plus": plus_counts}

    # Common shape 3: top-level lists of records: [{"task_id":..., "base_passed":[...], "plus_passed":[...]}]
    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict): 
                continue
            for src, lst, key_candidates in (
                ("base", base_counts, ["base_passed", "base", "passed_base", "passed"]),
                ("plus", plus_counts, ["plus_passed", "plus", "passed_plus"]),
            ):
                arr = None
                for k in key_candidates:
                    if k in rec:
                        arr = rec[k]
                        break
                if isinstance(arr, list) and arr and isinstance(arr[0], (bool, int)):
                    n = len(arr)
                    c = sum(bool(x) for x in arr)
                    lst.append((n, c))
        if base_counts or plus_counts:
            return {"base": base_counts, "plus": plus_counts}

    return None

def parse_log_for_passk(log_path):
    # returns dict {"base":{1:..,10:..,100:..}, "plus":{...}} with floats or None
    out = {"base": {}, "plus": {}}
    if not os.path.exists(log_path):
        return out
    pat = re.compile(r'^(humaneval\+|humaneval)\s+\(.*\)\s*$', re.I)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]
    mode = None  # 'base' or 'plus'
    for ln in lines:
        # detect sections
        if ln.lower().startswith("humaneval (base"):
            mode = "base"
            continue
        if ln.lower().startswith("humaneval+"):
            mode = "plus"
            continue
        m = re.match(r'pass@(\d+):\s*([0-9.]+)', ln)
        if m and mode in ("base", "plus"):
            k = int(m.group(1))
            val = safe_float(m.group(2))
            out[mode][k] = val
    return out

def process_run(run_dir, ks):
    run_dir = os.path.abspath(run_dir)
    out = {
        "run_dir": run_dir,
        "K": ks,
        "source": None,
        "base_pass@k": {},
        "plus_pass@k": {},
    }
    # prefer the evalplus cache json
    candidates = [
        os.path.join(run_dir, "evalplus-humaneval_eval_results.json"),
        os.path.join(run_dir, "human_eval_results.json"),
    ]
    data = None
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    out["cache_file"] = os.path.basename(c)
                    break
                except Exception:
                    pass

    if data is not None:
        counts = try_parse_counts_from_json(data)
        if counts is not None and (counts["base"] or counts["plus"]):
            out["source"] = "counts"
            base = agg_passk(counts["base"], ks) if counts["base"] else {k: None for k in ks}
            plus = agg_passk(counts["plus"], ks) if counts["plus"] else {k: None for k in ks}
            out["base_pass@k"] = {str(k): (None if base[k] is None else round(base[k], 3)) for k in ks}
            out["plus_pass@k"] = {str(k): (None if plus[k] is None else round(plus[k], 3)) for k in ks}
            return out

    # fallback: parse log for pass@1/10/100
    log_path = os.path.join(run_dir, "humaneval.log")
    logvals = parse_log_for_passk(log_path)
    if logvals["base"] or logvals["plus"]:
        out["source"] = "log"
        for k in ks:
            out["base_pass@k"][str(k)] = (round(logvals["base"][k], 3) 
                                          if k in logvals["base"] else None)
            out["plus_pass@k"][str(k)] = (round(logvals["plus"][k], 3) 
                                          if k in logvals["plus"] else None)
        return out

    out["source"] = "none"
    out["base_pass@k"] = {str(k): None for k in ks}
    out["plus_pass@k"] = {str(k): None for k in ks}
    return out

def main():
    if len(sys.argv) < 3:
        print("Usage: calc_passk_from_evalplus.py KS_CSV RUN_DIR [RUN_DIR ...]")
        sys.exit(2)
    ks = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    for run_dir in sys.argv[2:]:
        res = process_run(run_dir, ks)
        out_path = os.path.join(run_dir, "passk_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[ok] wrote {out_path}")

if __name__ == "__main__":
    main()
PY

echo "[info] Using script: $SCRIPT"
echo "[info] KS: $KS_CSV"

python "$SCRIPT" "$KS_CSV" "${RUNS[@]}"
echo "[all done]"