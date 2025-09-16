#!/usr/bin/env python3
import json, math, argparse, sys
from statistics import mean

def est_pass_at_k(c: int, n: int, k: int) -> float:
    """Codex pass@k estimator: 1 - C(n-c, k) / C(n, k) (cap k at n)."""
    if n <= 0:
        return 0.0
    k = min(k, n)
    if c <= 0 or k == 0:
        return 0.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def load_evalplus_results(jsonl_path: str):
    """
    Tries to parse the evalplus cached results JSONL.
    Returns dicts: per_task['base'][task_id]=(c, n), per_task['plus'][task_id]=(c, n)
    and the set of k values we can compare against (inferred from n).
    """
    per_task = {"base": {}, "plus": {}}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # task id
            tid = obj.get("task_id") or obj.get("task") or obj.get("id")

            # Common shapes seen in evalplus caches:
            #  - obj["base"] and obj["plus"] are lists of booleans
            #  - or dicts with key "passed" -> list of booleans
            def extract_pass_list(val):
                if isinstance(val, list):
                    return val
                if isinstance(val, dict) and "passed" in val and isinstance(val["passed"], list):
                    return val["passed"]
                return None

            base_list = extract_pass_list(obj.get("base"))
            plus_list = extract_pass_list(obj.get("plus"))

            # Some versions store a single list "passed" (base only)
            if base_list is None and "passed" in obj and isinstance(obj["passed"], list):
                base_list = obj["passed"]

            if base_list is not None and tid is not None:
                c = sum(bool(x) for x in base_list)
                n = len(base_list)
                per_task["base"][tid] = (c, n)
            if plus_list is not None and tid is not None:
                c = sum(bool(x) for x in plus_list)
                n = len(plus_list)
                per_task["plus"][tid] = (c, n)

    if not per_task["base"] and not per_task["plus"]:
        raise RuntimeError(
            f"Could not find per-sample pass lists in {jsonl_path}. "
            "Double-check you pointed to the *_eval_results.jsonl cache emitted by evalplus."
        )
    return per_task

def report(group_name, per_task_counts, ks):
    if not per_task_counts:
        return
    print(f"\n{group_name}")
    Ns = sorted({n for (_, n) in per_task_counts.values()})
    print(f"  problems: {len(per_task_counts)} | sample sizes seen (n): {Ns}")
    for k in ks:
        vals = [est_pass_at_k(c, n, k) for (c, n) in per_task_counts.values()]
        print(f"  pass@{k}: {mean(vals):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_jsonl", help="Path to evalplus cached results JSONL (e.g., samples_eval_results.jsonl)")
    ap.add_argument("--ks", default="1,10,20,50,100", help="Comma-separated k's to report")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    per_task = load_evalplus_results(args.results_jsonl)
    report("humaneval (base tests)", per_task.get("base", {}), ks)
    report("humaneval+ (base + extra tests)", per_task.get("plus", {}), ks)

if __name__ == "__main__":
    main()