#!/usr/bin/env python
import json, argparse, math
from collections import defaultdict

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if k > n:
        k = n
    # If picking k distinct samples, if incorrect pool < k, success is guaranteed
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_samples(obj):
    """Recursively find dicts with per-sample fields (task_id + base/plus_status)."""
    found = []
    if isinstance(obj, dict):
        # a sample looks like: {"task_id": "...", "base_status": "pass"/"fail", ...}
        if "task_id" in obj and ("base_status" in obj or "plus_status" in obj):
            found.append(obj)
        for v in obj.values():
            found.extend(extract_samples(v))
    elif isinstance(obj, list):
        for it in obj:
            found.extend(extract_samples(it))
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="*_eval_results.json produced by evalplus.evaluate")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 20, 50],
                    help="Budgets to compute")
    args = ap.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

    samples = extract_samples(data)
    if not samples:
        raise SystemExit("No per-sample records found in the JSON. Re-run eval or point to the _eval_results.json.")

    # Group per task
    base_counts = defaultdict(lambda: [0,0])  # task_id -> [n, c]
    plus_counts = defaultdict(lambda: [0,0])

    def is_pass(x):
        if isinstance(x, str):
            return x.lower() == "pass"
        if isinstance(x, bool):
            return x
        return False

    for s in samples:
        tid = s["task_id"]
        # Some files contain both base_status and plus_status; count whichever exists
        if "base_status" in s:
            base_counts[tid][0] += 1
            base_counts[tid][1] += int(is_pass(s["base_status"]))
        if "plus_status" in s:
            plus_counts[tid][0] += 1
            plus_counts[tid][1] += int(is_pass(s["plus_status"]))

    def summarize(counts, ks):
        tids = sorted(counts.keys())
        out = {}
        for k in ks:
            vals = []
            for tid in tids:
                n, c = counts[tid]
                vals.append(pass_at_k(n, c, k))
            out[f"pass@{k}"] = sum(vals) / len(vals) if vals else float("nan")
        return out

    base_summary = summarize(base_counts, args.k)
    plus_summary = summarize(plus_counts, args.k)

    print("== Humaneval base ==")
    for k, v in base_summary.items():
        print(f"{k}:\t{v:.3f}")
    if plus_counts:
        print("\n== Humaneval+ (extra) ==")
        for k, v in plus_summary.items():
            print(f"{k}:\t{v:.3f}")

if __name__ == "__main__":
    main()