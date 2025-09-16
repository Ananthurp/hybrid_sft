#!/usr/bin/env python3
"""
Roll up per-task evaluation artifacts into one CSV/XLSX/JSON.

Inputs expected (if present) under:
  <base_dir>/output_dir_consolidated/
    evaluation_chat_alpaca/
      - rewards.json (optional)
      - reward_summary.json
      - bt_winrate_vs_gpt.json (optional)
      - diversity_summary.json
    evaluation_poem/
      - responses.json
      - diversity_summary.json
    evaluation_story/
      - responses.json
      - diversity_summary.json

Usage (matches your sbatch):
  python summarize_eval.py \
    --base_dir "<RUN_DIR>" \
    --run "<RUN_NAME>" \
    --csv_out "<OUT_DIR>/summary_<RUN_NAME>.csv" \
    --xlsx_out "<OUT_DIR>/summary_<RUN_NAME>.xlsx" \
    --json_out "<OUT_DIR>/summary_<RUN_NAME>.json"
"""
import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd


def jload(p: str) -> Any:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_task_rows(out_dir: str) -> List[Dict[str, Any]]:
    tasks = [
        ("alpaca", "evaluation_chat_alpaca"),
        ("poem", "evaluation_poem"),
        ("story", "evaluation_story"),
    ]
    rows: List[Dict[str, Any]] = []

    for domain, subdir in tasks:
        dpath = os.path.join(out_dir, subdir)
        if not os.path.isdir(dpath):
            # Task folder missing: still emit an empty row (helps compare runs)
            rows.append({"domain": domain})
            continue

        reward_summary = jload(os.path.join(dpath, "reward_summary.json"))
        bt_summary = jload(os.path.join(dpath, "bt_winrate_vs_gpt.json"))
        div_summary = jload(os.path.join(dpath, "diversity_summary.json"))

        row: Dict[str, Any] = {"domain": domain}

        # ---- Rewards (Alpaca only; others don’t produce reward_summary.json by design)
        if isinstance(reward_summary, dict):
            row.update(
                {
                    "num_items": reward_summary.get("num_items"),
                    "n_grid": reward_summary.get("n_grid"),
                    "best_of_n": reward_summary.get("best_of_n"),
                    "mean_of_n": reward_summary.get("mean_of_n"),
                    "winrate_gt0_of_n_%": reward_summary.get("winrate_gt0_of_n"),
                }
            )

        # ---- Bradley–Terry vs GPT (Alpaca only if file exists)
        if isinstance(bt_summary, dict):
            row.update(
                {
                    "bt_n": bt_summary.get("n"),
                    "bt_wins": bt_summary.get("wins"),
                    "bt_losses": bt_summary.get("losses"),
                    "bt_ties": bt_summary.get("ties"),
                    "bt_skipped": bt_summary.get("skipped"),
                    "bt_winrate_prob": bt_summary.get("bt_winrate"),
                    "mean_best_reward": bt_summary.get("mean_best_reward"),
                    "mean_baseline_reward": bt_summary.get("mean_baseline_reward"),
                    "candidate_total": bt_summary.get("candidate_total"),
                    "baseline_total": bt_summary.get("baseline_total"),
                }
            )

        # ---- Diversity (all three tasks)
        if isinstance(div_summary, dict):
            row.update(
                {
                    "div_ngram_%": div_summary.get("averaged_ngram_diversity_score"),
                    "div_selfbleu_%": div_summary.get("bleu_diversity_score"),
                    "div_sentbert_%": div_summary.get("sentbert_diversity_score"),
                }
            )

        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Run directory (dirname of OUT_DIR).")
    ap.add_argument("--run", required=True, help="Run name (for metadata only).")
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--xlsx_out", required=True)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    # OUT_DIR inferred as <base_dir>/output_dir_consolidated
    out_dir = os.path.join(args.base_dir, "output_dir_consolidated")
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    rows = collect_task_rows(out_dir)
    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(args.csv_out, index=False)

    # Save XLSX (optional if openpyxl installed)
    try:
        df.to_excel(args.xlsx_out, index=False)
    except Exception as e:
        # Fallback: write a note and still continue
        with open(args.xlsx_out + ".txt", "w", encoding="utf-8") as f:
            f.write(f"XLSX export skipped: {e}\n")
            f.write(f"CSV available at: {args.csv_out}\n")

    # Save JSON (rows + a small header)
    payload = {
        "run": args.run,
        "base_dir": args.base_dir,
        "out_dir": out_dir,
        "rows": rows,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote\n  CSV : {args.csv_out}\n  XLSX: {args.xlsx_out}\n  JSON: {args.json_out}")


if __name__ == "__main__":
    main()