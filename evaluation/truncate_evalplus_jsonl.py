import json, sys, collections
inp, outp, per_task = sys.argv[1], sys.argv[2], int(sys.argv[3])
by_task = collections.defaultdict(list)
with open(inp, "r") as f:
    for line in f:
        rec = json.loads(line)
        by_task[rec["task_id"]].append(rec)
with open(outp, "w") as f:
    for tid, recs in by_task.items():
        for rec in recs[:per_task]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Wrote {outp} with ≤{per_task} samples per task.")