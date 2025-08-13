#!/bin/bash
set -euo pipefail
set -x

# Usage:
#   bash scripts/qwen2-1.5b/eval/run_winrate.sh <BASE_MODEL_DIR> <OTHER_MODEL_DIR>
BASE="$1"
OTHER="$2"

BASE_JSON="${BASE}/evaluation_chat_alpaca/reward_scores.json"
OTHER_JSON="${OTHER}/evaluation_chat_alpaca/reward_scores.json"
OUTDIR="${OTHER}/evaluation_chat_alpaca"
OUTFILE="${OUTDIR}/winrate_vs_baseline.json"
mkdir -p "$OUTDIR"

python - <<'PY'
import json, sys, os
base_json = os.environ["BASE_JSON"]
other_json = os.environ["OTHER_JSON"]
out_file  = os.environ["OUTFILE"]

with open(base_json, "r", encoding="utf-8") as f:
    A = json.load(f)
with open(other_json, "r", encoding="utf-8") as f:
    B = json.load(f)

# Expect a list of dicts, one per prompt, each with a reward "score" field.
# If your reward script uses a different key, adjust here (still a summary of GEM’s scores).
def extract_scores(x):
    scores=[]
    for r in x:
        # try common keys
        for k in ("score","reward","rm_score"):
            if k in r:
                scores.append(r[k]); break
    return scores

sa = extract_scores(A)
sb = extract_scores(B)
n  = min(len(sa), len(sb))
wins = sum(1 for i in range(n) if sb[i] > sa[i])
loss = sum(1 for i in range(n) if sb[i] < sa[i])
ties = n - wins - loss
winrate = wins / max(1, wins + loss)

res = {"n": n, "wins": wins, "losses": loss, "ties": ties, "win_rate": winrate}
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, indent=2))
PY
