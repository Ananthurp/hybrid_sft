# import argparse, json, os
# from typing import List, Tuple, Dict
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# def load_json(p):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# def norm(s: str) -> str:
#     return (s or "").strip()

# def extract_candidate_prompts_and_responses(data) -> List[Tuple[str, List[str]]]:
#     """Support common formats produced by evaluation/generate_response.py"""
#     out = []
#     # expected: list of items; each item has a prompt and a list of responses
#     for obj in data:
#         prompt = obj.get("instruction") or obj.get("prompt") or obj.get("input") or ""
#         # responses may be under several keys
#         candidates = (
#             obj.get("responses") or obj.get("outputs") or obj.get("generations")
#             or obj.get("answers") or obj.get("output") or obj.get("response")
#         )
#         if isinstance(candidates, str):
#             candidates = [candidates]
#         if not isinstance(candidates, list):
#             # sometimes saved as {"responses":[{"text":...}, ...]}
#             if isinstance(candidates, dict) and "responses" in candidates:
#                 cand = candidates["responses"]
#                 if isinstance(cand, list) and cand and isinstance(cand[0], dict) and "text" in cand[0]:
#                     candidates = [c["text"] for c in cand]
#             if not isinstance(candidates, list):
#                 candidates = []
#         out.append((norm(prompt), [norm(x) for x in candidates]))
#     return out

# def extract_baseline_prompts_and_response(data) -> Dict[str, str]:
#     """Baseline (GPT) has 1 response per prompt."""
#     m = {}
#     for obj in data:
#         prompt = obj.get("instruction") or obj.get("prompt") or obj.get("input") or ""
#         resp   = obj.get("output") or obj.get("response") or obj.get("completion") or ""
#         m[norm(prompt)] = norm(resp)
#     return m

# def batch_score(tokenizer, model, pairs, max_len=2048) -> List[float]:
#     texts = [f"Prompt:\n{p}\n\nResponse:\n{r}" for (p,r) in pairs]
#     enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(model.device)
#     with torch.no_grad():
#         out = model(**enc)
#     logits = out.logits.squeeze(-1)
#     return logits.detach().float().cpu().tolist()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--candidate_json", required=True, help=".../evaluation_chat_alpaca/generated_responses.json")
#     ap.add_argument("--baseline_json", required=True, help="GPT baseline JSON (AlpacaEval GPT-4 outputs)")
#     ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
#     ap.add_argument("--out_file", required=True)
#     args = ap.parse_args()

#     dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
#     tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
#     rm  = AutoModelForSequenceClassification.from_pretrained(
#         args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
#     ).eval()

#     cand = load_json(args.candidate_json)
#     base = load_json(args.baseline_json)

#     cand_list = extract_candidate_prompts_and_responses(cand)
#     base_map  = extract_baseline_prompts_and_response(base)

#     n = wins = losses = ties = 0
#     bt_probs = []
#     mean_best = []
#     mean_base = []

#     for prompt, responses in cand_list:
#         if prompt not in base_map:
#             continue  # skip prompts not found in baseline file
#         baseline_resp = base_map[prompt]
#         # score candidate responses, take best-of-n
#         if not responses:
#             continue
#         # batch score: all candidates + baseline
#         pairs = [(prompt, r) for r in responses] + [(prompt, baseline_resp)]
#         scores = batch_score(tok, rm, pairs)
#         best_cand = max(scores[:-1])
#         base_score = scores[-1]

#         # Bradley–Terry probability of candidate beating baseline
#         # P = exp(rc) / (exp(rc) + exp(rb))
#         # Equivalent to sigmoid(rc - rb).
#         rc, rb = best_cand, base_score
#         import math
#         p_win = 1.0 / (1.0 + math.exp(rb - rc))
#         bt_probs.append(p_win)
#         mean_best.append(rc)
#         mean_base.append(rb)

#         if abs(rc - rb) < 1e-6:
#             ties += 1
#         elif rc > rb:
#             wins += 1
#         else:
#             losses += 1
#         n += 1

#     result = {
#         "n": n,
#         "wins": wins,
#         "losses": losses,
#         "ties": ties,
#         "bt_winrate": (sum(bt_probs) / n) if n > 0 else 0.0,
#         "mean_best_reward": (sum(mean_best)/n) if n>0 else 0.0,
#         "mean_baseline_reward": (sum(mean_base)/n) if n>0 else 0.0
#     }
#     os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
#     with open(args.out_file, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     main()


import argparse, json, os, math
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return (s or "").strip()

def extract_candidate_prompts_and_responses(data) -> List[Tuple[str, List[str]]]:
    """
    Supports the format from evaluation/generate_response.py:
      [{"prompt": ..., "answer": [resp1, resp2, ...]}, ...]
    Also tolerates a few alternate field names.
    """
    out = []
    for obj in data:
        prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""
        # candidates may be under several keys
        candidates = (
            obj.get("answer") or
            obj.get("answers") or
            obj.get("responses") or
            obj.get("outputs") or
            obj.get("generations") or
            obj.get("output") or
            obj.get("response")
        )
        if isinstance(candidates, str):
            candidates = [candidates]
        if not isinstance(candidates, list):
            # try nested [{"text":...}, ...]
            if isinstance(candidates, dict) and "responses" in candidates:
                cand = candidates["responses"]
                if (isinstance(cand, list) and cand and isinstance(cand[0], dict)
                        and "text" in cand[0]):
                    candidates = [c["text"] for c in cand]
            if not isinstance(candidates, list):
                candidates = []
        out.append((norm(prompt), [norm(x) for x in candidates]))
    return out

def extract_baseline_prompts_and_response(data) -> Dict[str, str]:
    """Baseline (GPT) has 1 response per prompt."""
    m = {}
    for obj in data:
        prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""
        resp   = obj.get("output") or obj.get("response") or obj.get("completion") or ""
        m[norm(prompt)] = norm(resp)
    return m

def batch_score_chat(tokenizer, model, pairs, max_len=4096, batch_size=8) -> List[float]:
    """
    Score (prompt,response) pairs with the RM using the chat template, returning scalar logits.
    """
    scores = []
    device = model.device
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        chats = [[{"role":"user","content":p},{"role":"assistant","content":r}] for (p,r) in chunk]
        enc = tokenizer.apply_chat_template(
            chats, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        s = out.logits.squeeze(-1).detach().float().cpu().tolist()
        if isinstance(s, float): s = [s]
        scores.extend(s)
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_json", required=True, help=".../evaluation_chat_alpaca/generated_responses.json")
    ap.add_argument("--baseline_json", required=True, help="GPT baseline JSON (AlpacaEval GPT-4 outputs)")
    ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
    rm  = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()

    cand = load_json(args.candidate_json)
    base = load_json(args.baseline_json)

    cand_list = extract_candidate_prompts_and_responses(cand)
    base_map  = extract_baseline_prompts_and_response(base)

    n = wins = losses = ties = 0
    bt_probs = []
    mean_best = []
    mean_base = []
    skipped = 0

    for prompt, responses in cand_list:
        if prompt not in base_map:
            skipped += 1
            continue
        if not responses:
            skipped += 1
            continue

        baseline_resp = base_map[prompt]
        # batch score: all candidates + baseline
        pairs = [(prompt, r) for r in responses] + [(prompt, baseline_resp)]
        scores = batch_score_chat(tok, rm, pairs, batch_size=args.batch_size)
        if not scores:
            skipped += 1
            continue
        best_cand = max(scores[:-1])
        base_score = scores[-1]

        # Bradley–Terry probability (sigmoid(rc - rb))
        p_win = 1.0 / (1.0 + math.exp(base_score - best_cand))
        bt_probs.append(p_win)
        mean_best.append(best_cand)
        mean_base.append(base_score)

        if abs(best_cand - base_score) < 1e-6:
            ties += 1
        elif best_cand > base_score:
            wins += 1
        else:
            losses += 1
        n += 1

    result = {
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "bt_winrate": (sum(bt_probs) / n) if n > 0 else 0.0,
        "mean_best_reward": (sum(mean_best)/n) if n>0 else 0.0,
        "mean_baseline_reward": (sum(mean_base)/n) if n>0 else 0.0
    }
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
