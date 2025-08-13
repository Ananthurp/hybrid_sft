#!/usr/bin/env python3
import argparse, json, os, re, sys
from typing import Dict, Any, List, Optional
from glob import glob

def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_diversity_log(log_path: str) -> Dict[str, Optional[float]]:
    """
    Try hard to extract:
      - averaged_ngram_diversity_score
      - bleu_diversity_score
      - sentbert_diversity_score
    Works with either a JSON dumped by evaluation_diversity.py (if you added one),
    the printed dict at the end, or individual '... score:' lines.
    """
    out = {"ngram": None, "bleu": None, "sbert": None}
    if not os.path.isfile(log_path):
        return out
    try:
        txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return out

    # 1) Try to find a dict literal
    m = re.search(r"\{[^{}]*'sentbert_diversity_score'[^{}]*\}", txt)
    if m:
        try:
            # Replace single quotes with double for json parsing
            dict_txt = m.group(0).replace("'", '"')
            d = json.loads(dict_txt)
            out["ngram"] = float(d.get("averaged_ngram_diversity_score", None))
            out["bleu"] = float(d.get("bleu_diversity_score", None))
            out["sbert"] = float(d.get("sentbert_diversity_score", None))
            return out
        except Exception:
            pass

    # 2) Try individual lines
    # Examples printed by evaluation_diversity.py in your repo:
    # "N-gram diversity score: X"
    # "BLEU similarity score: Y" (we actually store 100 - selfbleu inside the script)
    # "Bert diversity score: Z"
    pat_num = r"([-+]?\d+(?:\.\d+)?)"
    m1 = re.search(r"N-gram diversity score:\s*" + pat_num, txt)
    m2 = re.search(r"BLEU similarity score:\s*" + pat_num, txt)
    m3 = re.search(r"Bert diversity score:\s*" + pat_num, txt)
    if m1: out["ngram"] = float(m1.group(1))
    if m2: out["bleu"]  = float(m2.group(1))
    if m3: out["sbert"] = float(m3.group(1))
    return out

def pick_first_existing(*paths) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def collect_for_model(model_root: str, run_name: str) -> Dict[str, Any]:
    """
    model_root = .../<RUN>/output_dir_consolidated
    """
    res: Dict[str, Any] = {
        "run": run_name,
        "model_root": model_root,
        # high-level metrics we’ll fill if found
        "alpaca": {},
        "poem": {},
        "story": {},
    }

    # ---------- Alpaca ----------
    alp_dir = os.path.join(model_root, "evaluation_chat_alpaca")
    if os.path.isdir(alp_dir):
        # reward summary (best/mean/winrate>0 of n)
        reward_sum = read_json(os.path.join(alp_dir, "reward_summary.json"))
        if reward_sum:
            res["alpaca"]["n_grid"]         = reward_sum.get("n_grid")
            res["alpaca"]["best_of_n"]      = reward_sum.get("best_of_n")
            res["alpaca"]["mean_of_n"]      = reward_sum.get("mean_of_n")
            res["alpaca"]["winrate_gt0_of_n"] = reward_sum.get("winrate_gt0_of_n")
            res["alpaca"]["num_items"]      = reward_sum.get("num_items")

        # BT win-rate vs GPT-4
        bt = read_json(os.path.join(alp_dir, "bt_winrate_vs_gpt.json"))
        if bt:
            res["alpaca"]["bt_winrate"] = bt.get("bt_winrate")
            res["alpaca"]["wins"] = bt.get("wins")
            res["alpaca"]["losses"] = bt.get("losses")
            res["alpaca"]["ties"] = bt.get("ties")
            res["alpaca"]["n_bt"] = bt.get("n")

        # diversity on Alpaca prompts (if you ran it)
        # Common locations: evaluation_diversity_alpaca/diversity_metrics.log
        alp_div_dir = os.path.join(model_root, "evaluation_diversity_alpaca")
        if os.path.isdir(alp_div_dir):
            logp = pick_first_existing(
                os.path.join(alp_div_dir, "diversity_metrics.json"),
                os.path.join(alp_div_dir, "diversity_metrics.log"),
            )
            if logp and logp.endswith(".json"):
                d = read_json(logp) or {}
                res["alpaca"]["div_ngram"] = d.get("averaged_ngram_diversity_score")
                res["alpaca"]["div_bleu"]  = d.get("bleu_diversity_score")
                res["alpaca"]["div_sbert"] = d.get("sentbert_diversity_score")
            elif logp:
                d = parse_diversity_log(logp)
                res["alpaca"]["div_ngram"] = d["ngram"]
                res["alpaca"]["div_bleu"]  = d["bleu"]
                res["alpaca"]["div_sbert"] = d["sbert"]

    # ---------- Poem / Story diversity ----------
    for tag in ["poem", "story"]:
        tdir = os.path.join(model_root, f"evaluation_{tag}")
        if os.path.isdir(tdir):
            logp = pick_first_existing(
                os.path.join(tdir, "diversity_metrics.json"),
                os.path.join(tdir, "diversity_metrics.log"),
            )
            if logp and logp.endswith(".json"):
                d = read_json(logp) or {}
                res[tag]["div_ngram"] = d.get("averaged_ngram_diversity_score")
                res[tag]["div_bleu"]  = d.get("bleu_diversity_score")
                res[tag]["div_sbert"] = d.get("sentbert_diversity_score")
            elif logp:
                d = parse_diversity_log(logp)
                res[tag]["div_ngram"] = d["ngram"]
                res[tag]["div_bleu"]  = d["bleu"]
                res[tag]["div_sbert"] = d["sbert"]

    # write a per-model summary for convenience
    try:
        with open(os.path.join(model_root, "model_summary.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return res

def flatten_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Make a single CSV row with consistent columns."""
    row: Dict[str, Any] = {"run": summary.get("run")}
    # BT win-rate
    row["bt_winrate"] = summary.get("alpaca", {}).get("bt_winrate")
    row["wins"] = summary.get("alpaca", {}).get("wins")
    row["losses"] = summary.get("alpaca", {}).get("losses")
    row["ties"] = summary.get("alpaca", {}).get("ties")
    # Winrate >0 best-of-n (we lay out standard n’s if present)
    wr = summary.get("alpaca", {}).get("winrate_gt0_of_n") or []
    ngrid = summary.get("alpaca", {}).get("n_grid") or []
    for want in [1,2,4,8,16,32]:
        try:
            idx = ngrid.index(want)
            row[f"wr_gt0_n{want}"] = wr[idx]
        except Exception:
            row[f"wr_gt0_n{want}"] = None
    # Best-of-n and Mean-of-n
    best = summary.get("alpaca", {}).get("best_of_n") or []
    mean = summary.get("alpaca", {}).get("mean_of_n") or []
    for want in [1,2,4,8,16,32]:
        try:
            idx = ngrid.index(want)
            row[f"best_of_n{want}"] = best[idx]
            row[f"mean_of_n{want}"] = mean[idx]
        except Exception:
            row[f"best_of_n{want}"] = None
            row[f"mean_of_n{want}"] = None
    # Alpaca diversity
    row["alpaca_div_ngram"] = summary.get("alpaca", {}).get("div_ngram")
    row["alpaca_div_bleu"]  = summary.get("alpaca", {}).get("div_bleu")
    row["alpaca_div_sbert"] = summary.get("alpaca", {}).get("div_sbert")
    # Poem/Story diversity
    for tag in ["poem","story"]:
        row[f"{tag}_div_ngram"] = summary.get(tag, {}).get("div_ngram")
        row[f"{tag}_div_bleu"]  = summary.get(tag, {}).get("div_bleu")
        row[f"{tag}_div_sbert"] = summary.get(tag, {}).get("div_sbert")
    return row

def write_table(rows: List[Dict[str, Any]], csv_out: str, xlsx_out: Optional[str]):
    import csv
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    # collect headers
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # optional xlsx
    if xlsx_out:
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)
            df.to_excel(xlsx_out, index=False)
        except Exception as e:
            print(f"[warn] could not write xlsx ({e}); CSV is still saved.", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Base results dir containing RUN subdirs")
    ap.add_argument("--run", action="append", required=True, help="Run name (repeatable)")
    ap.add_argument("--csv_out", required=True, help="Path to write combined CSV")
    ap.add_argument("--xlsx_out", default="", help="Optional XLSX path")
    ap.add_argument("--json_out", default="", help="Optional combined JSON path")
    args = ap.parse_args()

    summaries: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for run in args.run:
        model_root = os.path.join(args.base_dir, run, "output_dir_consolidated")
        if not os.path.isdir(model_root):
            print(f"[skip] {run}: {model_root} not found", file=sys.stderr)
            continue
        summ = collect_for_model(model_root, run)
        summaries.append(summ)
        rows.append(flatten_row(summ))

    write_table(rows, args.csv_out, args.xlsx_out or None)

    if args.json_out:
        try:
            os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"[ok] wrote CSV:  {args.csv_out}")
    if args.xlsx_out:
        print(f"[ok] wrote XLSX: {args.xlsx_out}")
    if args.json_out:
        print(f"[ok] wrote JSON: {args.json_out}")

if __name__ == "__main__":
    main()
