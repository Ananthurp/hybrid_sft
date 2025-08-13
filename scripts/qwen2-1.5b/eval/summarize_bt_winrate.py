import argparse, json, csv, os

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        bt_path = os.path.join(args.base_dir, run,
                               "output_dir_consolidated",
                               "evaluation_chat_alpaca",
                               "bt_winrate_vs_gpt.json")
        if not os.path.isfile(bt_path):
            print(f"[warn] Missing: {bt_path}")
            continue
        bt = load(bt_path)
        rows.append({
            "run": run,
            "n": bt.get("n", 0),
            "wins": bt.get("wins", 0),
            "losses": bt.get("losses", 0),
            "ties": bt.get("ties", 0),
            "skipped": bt.get("skipped", 0),
            "bt_winrate": bt.get("bt_winrate", 0.0),
            "mean_best_reward": bt.get("mean_best_reward", 0.0),
            "mean_baseline_reward": bt.get("mean_baseline_reward", 0.0),
        })

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    fields = ["run","n","wins","losses","ties","skipped","bt_winrate","mean_best_reward","mean_baseline_reward"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote:\n  {args.csv_out}\n  {args.json_out}")

if __name__ == "__main__":
    main()
