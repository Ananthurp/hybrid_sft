#!/usr/bin/env python3
import os, json, glob, argparse
from evalplus.evaluate import evaluate  # works for evalplus==0.3.1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="path to evalplus-<dataset>.jsonl")
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval","mbpp"])
    ap.add_argument("--out_json", default=None, help="optional compact metrics JSON")
    args = ap.parse_args()

    # 1) (Re)compute with the exact k we need; this will also populate the results JSON.
    evaluate(dataset=args.dataset, samples=args.samples, k=[1,10,20,50,100])

    # 2) Locate the standard results file EvalPlus writes next to samples.
    prefix = os.path.splitext(os.path.basename(args.samples))[0]
    res_path = glob.glob(os.path.join(os.path.dirname(args.samples), f"{prefix}_eval_results.json"))
    if not res_path:
        raise SystemExit(f"Could not find results JSON next to {args.samples}")
    res_path = res_path[0]

    data = json.load(open(res_path))

    def pick(suite_key):
        suite = data.get(suite_key, {})
        return {f"pass@{k}": suite.get(f"pass@{k}") for k in [1,10,20,50,100] if f"pass@{k}" in suite}

    base = pick(args.dataset)
    plus = pick(args.dataset + "+")  # humaneval+ or mbpp+
    # Pretty print
    def fmt(d):
        keys = [1,10,20,50,100]
        return ", ".join([f"@{k}={d.get(f'pass@{k}', 'NA')}" for k in keys])

    print(f"{args.dataset} (base):      {fmt(base)}")
    print(f"{args.dataset}+ (base+extra): {fmt(plus)}")
    print(f"[info] source results: {res_path}")

    # Optional compact JSON
    if args.out_json:
        payload = {args.dataset: base, args.dataset + "+": plus, "source": res_path}
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[info] wrote compact metrics to {args.out_json}")

if __name__ == "__main__":
    main()