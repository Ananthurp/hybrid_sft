import json, sys, pathlib, re, csv

# Usage: python merge_llm_eval.py /path/to/OUT_BASE
base = pathlib.Path(sys.argv[1])

rows = []
for summ in base.rglob("results-*.json"):
    run_name = summ.parent.name
    with open(summ) as f:
        data = json.load(f)
    # lm-eval dumps task metrics under "results"
    res = data.get("results", {})
    row = {"run_name": run_name, "file": str(summ)}
    # pull the headline metrics we care about
    for task in ["arc_challenge","hellaswag","winogrande","truthfulqa_mc2","mmlu","gsm8k"]:
        for key in ["acc","acc_norm","em","exact_match","acc,normalized","acc_per_category"]:  # different tasks expose different keys
            val = res.get(task, {}).get(key)
            if isinstance(val, (int,float)):
                row[f"{task}"] = float(val)
                break
    # simple macro-average over whatever was found
    vals = [v for k,v in row.items() if k not in ("run_name","file") and isinstance(v,(int,float))]
    row["average"] = sum(vals)/len(vals) if vals else None
    rows.append(row)

rows.sort(key=lambda r: r.get("average") or -1, reverse=True)

out_csv = base / "openllm_eval_summary.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"[done] Wrote {out_csv}")