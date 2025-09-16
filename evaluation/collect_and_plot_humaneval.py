import json
from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt

def comb(n, k):
    if k<0 or k>n: return 0
    return math.comb(n, k)

def pass_at_k(n, c, k):
    if k>n or c==0: return 0.0
    return 1.0 - comb(n-c, k)/comb(n, k)

def load_results(res_path: Path):
    data = json.load(open(res_path))
    # Try to find per-problem stats across possible schema variants
    per_problem = (
        data.get("eval_plus", {}).get("humaneval", {}).get("problems")
        or data.get("problems")
        or None
    )
    provided = (data.get("results", {}).get("humaneval", {}) or data.get("results", {}))
    # Normalize keys like "pass@1"
    provided = {k.replace("pass@", ""): v for k, v in provided.items() if k.startswith("pass@")}
    return per_problem, provided

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Parent dir with one subdir per model result")
    ap.add_argument("--out_csv", default="humaneval_passk.csv")
    ap.add_argument("--out_png", default="humaneval_passk.png")
    ap.add_argument("--budgets", default="1,10,20,50,100")
    args = ap.parse_args()
    ks = [int(x) for x in args.budgets.split(",")]

    rows = []
    plt.figure()
    for model_dir in sorted(Path(args.root).glob("*")):
        res_json = None
        # evalplus typically writes *_eval_results.json; search for humaneval one
        for cand in model_dir.glob("*humaneval*_eval_results.json"):
            res_json = cand
            break
        if res_json is None:
            print(f"[skip] {model_dir} (no eval_results json found)")
            continue

        per_problem, provided = load_results(res_json)
        series = {}
        if per_problem:
            # Expect each entry has n (samples) and n_correct (count of passing samples)
            for k in ks:
                vals = []
                for p in per_problem:
                    n = int(p.get("n", 0))
                    c = int(p.get("n_correct", 0))
                    vals.append(pass_at_k(n, c, k) if n>=k else 0.0)
                series[k] = sum(vals)/len(vals)
        else:
            # Fall back to provided global metrics (likely 1,10,100 only)
            for k in ks:
                if str(k) in provided:
                    series[k] = provided[str(k)]

        # Append CSV row (percent)
        rows.append([model_dir.name] + [round(series.get(k, float("nan"))*100, 2) for k in ks])

        # Plot
        plt.plot(ks, [series.get(k, float("nan"))*100 for k in ks], marker="o", label=model_dir.name)

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.write_text(",".join(["model"]+[f"pass@{k} (%)" for k in ks])+"\n" + "\n".join(
        ",".join(map(str, r)) for r in rows
    ))
    # Save plot
    plt.xlabel("Sampling budget (k)")
    plt.ylabel("Pass@k (%)")
    plt.title("HumanEval pass@k vs sampling budget")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print(f"[saved] {out_csv}")
    print(f"[saved] {args.out_png}")

if __name__ == "__main__":
    main()