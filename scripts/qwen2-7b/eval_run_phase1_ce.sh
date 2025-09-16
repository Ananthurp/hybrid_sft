#!/usr/bin/env bash
set -euo pipefail

PROJ=~/LLMDiversity
EVAL_DIR="$PROJ/hybrid_sft/evaluation"
BASE="$PROJ/models/Qwen2-7B"                       # tokenizer base
MODEL_DIR="$PROJ/results/qwen2-7b/phase1_ce"       # model to evaluate
OUT_ROOT="$PROJ/results_eval/qwen2-7b/phase1_ce"
ASSETS="$EVAL_DIR/assets"

# Make sure EVAL_DIR is exported (used inside sbatch templates)
export EVAL_DIR

mkdir -p "$OUT_ROOT"/{alpacaeval,poem,story}
cd "$PROJ/hybrid_sft/scripts/qwen2-7b"

# ---------------- AlpacaEval generation ----------------
export MODEL_DIR TOKENIZER_DIR="$BASE" DATASET_KEY="tatsu-lab/alpaca_eval" SPLIT="eval"
export SAVE_JSON="$OUT_ROOT/alpacaeval/responses.json" SHORTNAME="alp-ce"
jid_gen_alp=$(sbatch --parsable \
  --export=ALL,MODEL_DIR,TOKENIZER_DIR,DATASET_KEY,SPLIT,SAVE_JSON,EVAL_DIR,SHORTNAME \
  gen_generic.sbatch)

# ---------------- Poem / Story generation --------------
export DATASET_KEY="$PROJ/hybrid_sft/data/poem_generation" SPLIT="test"
export SAVE_JSON="$OUT_ROOT/poem/responses.json" SHORTNAME="poem-ce"
jid_gen_poem=$(sbatch --parsable \
  --export=ALL,MODEL_DIR,TOKENIZER_DIR,DATASET_KEY,SPLIT,SAVE_JSON,EVAL_DIR,SHORTNAME \
  gen_generic.sbatch)

export DATASET_KEY="$PROJ/hybrid_sft/data/story_generation" SPLIT="test"
export SAVE_JSON="$OUT_ROOT/story/responses.json" SHORTNAME="story-ce"
jid_gen_story=$(sbatch --parsable \
  --export=ALL,MODEL_DIR,TOKENIZER_DIR,DATASET_KEY,SPLIT,SAVE_JSON,EVAL_DIR,SHORTNAME \
  gen_generic.sbatch)

# ---------------- Reward (alpaca) ----------------------
export RESP_JSON="$OUT_ROOT/alpacaeval/responses.json"
export OUT_JSON="$OUT_ROOT/alpacaeval/rewards.json"
export SUM_JSON="$OUT_ROOT/alpacaeval/reward_summary.json"
export SHORTNAME="alp-ce"
jid_rwd_alp=$(sbatch --parsable --dependency=afterok:$jid_gen_alp \
  --export=ALL,EVAL_DIR,TOKENIZER_DIR,RESP_JSON,OUT_JSON,SUM_JSON,SHORTNAME \
  score_rewards.sbatch)

# ---------------- Bradley–Terry (alpaca) ---------------
export CAND_JSON="$OUT_ROOT/alpacaeval/responses.json"
export BASE_JSON="$ASSETS/gpt4_alpacaeval_responses.json"
export OUT_JSON="$OUT_ROOT/alpacaeval/bt_winrate_vs_gpt.json"
export SHORTNAME="alp-ce"
jid_bt_alp=$(sbatch --parsable --dependency=afterok:$jid_gen_alp \
  --export=ALL,EVAL_DIR,CAND_JSON,BASE_JSON,OUT_JSON,SHORTNAME \
  bt_eval.sbatch)

# ---------------- Diversity (alpaca/poem/story) --------
export RESP_JSON="$OUT_ROOT/alpacaeval/responses.json"
export DIVERSITY_LOG="$OUT_ROOT/alpacaeval/diversity_metrics.log"
export SHORTNAME="alp-ce"
jid_div_alp=$(sbatch --parsable --dependency=afterok:$jid_gen_alp \
  --export=ALL,EVAL_DIR,TOKENIZER_DIR,RESP_JSON,DIVERSITY_LOG,SHORTNAME \
  diversity_eval.sbatch)

export RESP_JSON="$OUT_ROOT/poem/responses.json"
export DIVERSITY_LOG="$OUT_ROOT/poem/diversity_metrics.log"
export SHORTNAME="poem-ce"
jid_div_poem=$(sbatch --parsable --dependency=afterok:$jid_gen_poem \
  --export=ALL,EVAL_DIR,TOKENIZER_DIR,RESP_JSON,DIVERSITY_LOG,SHORTNAME \
  diversity_eval.sbatch)

export RESP_JSON="$OUT_ROOT/story/responses.json"
export DIVERSITY_LOG="$OUT_ROOT/story/diversity_metrics.log"
export SHORTNAME="story-ce"
jid_div_story=$(sbatch --parsable --dependency=afterok:$jid_gen_story \
  --export=ALL,EVAL_DIR,TOKENIZER_DIR,RESP_JSON,DIVERSITY_LOG,SHORTNAME \
  diversity_eval.sbatch)

echo "[CE] submitted:
  gen alpaca: $jid_gen_alp
  gen poem:   $jid_gen_poem
  gen story:  $jid_gen_story
  rwd alpaca: $jid_rwd_alp
  bt  alpaca: $jid_bt_alp
  div alpaca: $jid_div_alp
  div poem:   $jid_div_poem
  div story:  $jid_div_story"