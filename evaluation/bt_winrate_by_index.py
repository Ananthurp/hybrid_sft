# import argparse, json, os, math, re, unicodedata
# from typing import List, Dict, Any, Tuple
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # ---------- utils ----------
# def load_json(p: str):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# _qwen_user_pat = re.compile(r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
# _llama_user_pat = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", re.DOTALL | re.IGNORECASE)
# def detemplate_prompt(s: str) -> str:
#     if not isinstance(s, str): return ""
#     q = _qwen_user_pat.findall(s)
#     if q: return q[-1]
#     l = _llama_user_pat.findall(s)
#     if l: return l[-1]
#     return s

# def squash_ws(s: str) -> str:
#     if s is None: return ""
#     s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def get_prompt(obj: Dict[str,Any]) -> str:
#     return obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""

# def get_candidates(obj: Dict[str,Any]) -> List[str]:
#     v = (obj.get("answer") or obj.get("answers") or obj.get("responses") or
#          obj.get("outputs") or obj.get("generations") or obj.get("output") or obj.get("response"))
#     outs: List[str] = []
#     if isinstance(v, str):
#         outs = [v]
#     elif isinstance(v, list):
#         for it in v:
#             if isinstance(it, str): outs.append(it)
#             elif isinstance(it, dict):
#                 for tkey in ("text","content","response","output"):
#                     tv = it.get(tkey)
#                     if isinstance(tv, str) and tv.strip():
#                         outs.append(tv); break
#     elif isinstance(v, dict) and isinstance(v.get("responses"), list):
#         for it in v["responses"]:
#             if isinstance(it, dict) and isinstance(it.get("text"), str):
#                 outs.append(it["text"])
#     return [squash_ws(x) for x in outs]

# def get_single_baseline_resp(obj: Dict[str,Any]) -> str:
#     for k in ("output","response","completion"):
#         v = obj.get(k)
#         if isinstance(v, str) and v.strip(): return squash_ws(v)
#     v = obj.get("outputs")
#     if isinstance(v, list) and v:
#         d = v[0] if isinstance(v[0], dict) else {}
#         for t in ("text","content","response","output"):
#             tv = d.get(t)
#             if isinstance(tv, str) and tv.strip(): return squash_ws(tv)
#     v = obj.get("choices")
#     if isinstance(v, list) and v:
#         d = v[0] if isinstance(v[0], dict) else {}
#         m = d.get("message") if isinstance(d.get("message"), dict) else {}
#         c = m.get("content")
#         if isinstance(c, str) and c.strip(): return squash_ws(c)
#         t = d.get("text")
#         if isinstance(t, str) and t.strip(): return squash_ws(t)
#     v = obj.get("response")
#     if isinstance(v, dict):
#         t = v.get("text")
#         if isinstance(t, str) and t.strip(): return squash_ws(t)
#     return ""

# def batch_score_chat(tokenizer, model, pairs: List[Tuple[str,str]], max_len=4096, batch_size=8) -> List[float]:
#     scores = []
#     device = model.device
#     for i in range(0, len(pairs), batch_size):
#         chunk = pairs[i:i+batch_size]
#         chats = [[{"role":"user","content":p},{"role":"assistant","content":r}] for (p,r) in chunk]
#         enc = tokenizer.apply_chat_template(
#             chats, padding=True, truncation=True, max_length=max_len,
#             return_tensors="pt", return_dict=True, add_generation_prompt=True
#         ).to(device)
#         with torch.no_grad():
#             out = model(**enc)
#         s = out.logits.squeeze(-1).detach().float().cpu().tolist()
#         if isinstance(s, float): s = [s]
#         scores.extend(s)
#     return scores

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--candidate_json", required=True)      # your model's generated_responses.json
#     ap.add_argument("--baseline_json", required=True)       # GPT-4 Alpaca baseline JSON
#     ap.add_argument("--budgets", default="2,4,8,16,32")     # comma-separated
#     ap.add_argument("--limit", type=int, default=200)       # first N items
#     ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--max_len", type=int, default=4096)
#     ap.add_argument("--out_file", required=True)
#     ap.add_argument("--debug_file", default="")
#     args = ap.parse_args()

#     budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
#     budgets = sorted(set([b for b in budgets if b > 0]))

#     dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
#     tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
#     rm  = AutoModelForSequenceClassification.from_pretrained(
#         args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
#     ).eval()

#     cand = load_json(args.candidate_json)
#     base = load_json(args.baseline_json)
#     if isinstance(cand, dict): cand = list(cand.values())
#     if isinstance(base, dict): base = list(base.values())
#     N = min(len(cand), len(base), args.limit)

#     # accumulators per K
#     eps = 1e-6
#     acc = {K: {"n":0,"wins":0,"losses":0,"ties":0,"skipped":0,"sum_bt":0.0,"sum_best":0.0,"sum_base":0.0} for K in budgets}
#     mismatches = []

#     for i in range(N):
#         c_obj = cand[i]; b_obj = base[i]
#         p_user = squash_ws(detemplate_prompt(get_prompt(c_obj)))
#         b_prompt = squash_ws(get_prompt(b_obj))
#         if p_user and b_prompt and p_user.lower()[:80] != b_prompt.lower()[:80] and len(mismatches) < 20:
#             mismatches.append({"i": i, "cand_prompt": p_user[:120], "base_prompt": b_prompt[:120]})

#         cands = get_candidates(c_obj)
#         b_resp = get_single_baseline_resp(b_obj)
#         if not cands or not b_resp or not p_user:
#             for K in budgets:
#                 acc[K]["skipped"] += 1
#             continue

#         pairs = [(p_user, r) for r in cands] + [(p_user, b_resp)]
#         scores = batch_score_chat(tok, rm, pairs, max_len=args.max_len, batch_size=args.batch_size)
#         if not scores:
#             for K in budgets:
#                 acc[K]["skipped"] += 1
#             continue

#         cand_scores = scores[:-1]
#         base_score  = scores[-1]

#         for K in budgets:
#             kk = min(K, len(cand_scores))
#             if kk <= 0:
#                 acc[K]["skipped"] += 1
#                 continue
#             best_k = max(cand_scores[:kk])

#             p_win = 1.0 / (1.0 + math.exp(base_score - best_k))  # sigmoid(best - base)

#             a = acc[K]
#             a["n"] += 1
#             a["sum_bt"] += p_win
#             a["sum_best"] += best_k
#             a["sum_base"] += base_score
#             if abs(best_k - base_score) < eps: a["ties"] += 1
#             elif best_k > base_score:          a["wins"] += 1
#             else:                               a["losses"] += 1

#     # finalize
#     out = {
#         "candidate_total": len(cand),
#         "baseline_total": len(base),
#         "index_compared": N,
#         "budgets": {}
#     }
#     for K in budgets:
#         a = acc[K]
#         n = max(a["n"], 1)
#         out["budgets"][str(K)] = {
#             "n": a["n"],
#             "wins": a["wins"],
#             "losses": a["losses"],
#             "ties": a["ties"],
#             "skipped": a["skipped"],
#             "bt_winrate": (a["sum_bt"] / a["n"]) if a["n"] > 0 else 0.0,
#             "mean_best_reward": (a["sum_best"] / n) if a["n"] > 0 else 0.0,
#             "mean_baseline_reward": (a["sum_base"] / n) if a["n"] > 0 else 0.0
#         }

#     os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
#     with open(args.out_file, "w", encoding="utf-8") as f:
#         json.dump(out, f, ensure_ascii=False, indent=2)
#     print(json.dumps(out, indent=2))

#     if args.debug_file:
#         with open(args.debug_file, "w", encoding="utf-8") as f:
#             json.dump(mismatches, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()


# import argparse, json, os, math, re, unicodedata
# from typing import List, Dict, Any, Tuple
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # ---------- utils ----------
# def load_json(p: str):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# _qwen_user_pat = re.compile(r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
# _llama_user_pat = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", re.DOTALL | re.IGNORECASE)
# def detemplate_prompt(s: str) -> str:
#     if not isinstance(s, str): return ""
#     q = _qwen_user_pat.findall(s)
#     if q: return q[-1]
#     l = _llama_user_pat.findall(s)
#     if l: return l[-1]
#     return s

# def squash_ws(s: str) -> str:
#     if s is None: return ""
#     s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def get_prompt(obj: Dict[str,Any]) -> str:
#     return obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""

# def get_candidates(obj: Dict[str,Any]) -> List[str]:
#     v = (obj.get("answer") or obj.get("answers") or obj.get("responses") or
#          obj.get("outputs") or obj.get("generations") or obj.get("output") or obj.get("response"))
#     outs: List[str] = []
#     if isinstance(v, str):
#         outs = [v]
#     elif isinstance(v, list):
#         for it in v:
#             if isinstance(it, str): outs.append(it)
#             elif isinstance(it, dict):
#                 for tkey in ("text","content","response","output"):
#                     tv = it.get(tkey)
#                     if isinstance(tv, str) and tv.strip():
#                         outs.append(tv); break
#     elif isinstance(v, dict) and isinstance(v.get("responses"), list):
#         for it in v["responses"]:
#             if isinstance(it, dict) and isinstance(it.get("text"), str):
#                 outs.append(it["text"])
#     return [squash_ws(x) for x in outs]

# def get_single_baseline_resp(obj: Dict[str,Any]) -> str:
#     for k in ("output","response","completion"):
#         v = obj.get(k)
#         if isinstance(v, str) and v.strip(): return squash_ws(v)
#     v = obj.get("outputs")
#     if isinstance(v, list) and v:
#         d = v[0] if isinstance(v[0], dict) else {}
#         for t in ("text","content","response","output"):
#             tv = d.get(t)
#             if isinstance(tv, str) and tv.strip(): return squash_ws(tv)
#     v = obj.get("choices")
#     if isinstance(v, list) and v:
#         d = v[0] if isinstance(v[0], dict) else {}
#         m = d.get("message") if isinstance(d.get("message"), dict) else {}
#         c = m.get("content")
#         if isinstance(c, str) and c.strip(): return squash_ws(c)
#         t = d.get("text")
#         if isinstance(t, str) and t.strip(): return squash_ws(t)
#     v = obj.get("response")
#     if isinstance(v, dict):
#         t = v.get("text")
#         if isinstance(t, str) and t.strip(): return squash_ws(t)
#     return ""

# # ---------- reward-model helper ----------
# def forward_value_fn(
#     self,
#     input_ids=None,
#     attention_mask=None,
#     past_key_values=None,
#     position_ids=None,
#     inputs_embeds=None,
#     return_value_only=False,
#     prompt_length=0,
#     use_cache=False,
#     **kwargs,
# ):
#     """
#     Matches the FsfairX RM "value head" behavior; returns per-token values and chosen_end_scores.
#     """
#     transformer_outputs = self.model(
#         input_ids,
#         past_key_values=past_key_values,
#         attention_mask=attention_mask,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         **kwargs,
#     )
#     hidden_states = transformer_outputs[0]         # [B, T, H]
#     values = self.score(hidden_states).squeeze(-1) # [B, T]
#     if return_value_only:
#         return values
#     if attention_mask is None:
#         chosen_end_scores = values[:, -1]
#     else:
#         # last non-pad index per row
#         last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
#         chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {"values": values, "chosen_end_scores": chosen_end_scores}

# def _model_device(model):
#     try:
#         return model.device
#     except Exception:
#         return next(model.parameters()).device

# def batch_score_chat(tokenizer, model, pairs: List[Tuple[str,str]], max_len=4096, batch_size=8) -> List[float]:
#     """
#     Scores (prompt, response) pairs with the reward model.
#     IMPORTANT: do NOT add a generation prompt; the assistant response is already present.
#     Uses forward_value if available; falls back to logits.
#     """
#     scores = []
#     device = _model_device(model)
#     for i in range(0, len(pairs), batch_size):
#         chunk = pairs[i:i+batch_size]
#         chats = [[{"role":"user","content":p},{"role":"assistant","content":r}] for (p,r) in chunk]
#         enc = tokenizer.apply_chat_template(
#             chats,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#             return_dict=True,
#             add_generation_prompt=False,  # <-- FIX: assistant already present
#         )
#         # Place on first device (works with device_map='auto' since first shard is there)
#         enc = {k: v.to(device) for k, v in enc.items()}

#         with torch.no_grad():
#             # Prefer the RM value head if available
#             if hasattr(model, "forward_value"):
#                 out = model.forward_value(**enc)["chosen_end_scores"]
#             else:
#                 out = model(**enc, use_cache=False).logits.squeeze(-1)

#         s = out.detach().float().cpu().tolist()
#         if isinstance(s, float): s = [s]
#         scores.extend(s)
#     return scores

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--candidate_json", required=True)      # your model's generated_responses.json
#     ap.add_argument("--baseline_json", required=True)       # GPT-4 Alpaca baseline JSON
#     ap.add_argument("--budgets", default="2,4,8,16,32")     # comma-separated
#     ap.add_argument("--limit", type=int, default=200)       # first N items
#     ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--max_len", type=int, default=4096)
#     ap.add_argument("--out_file", required=True)
#     ap.add_argument("--debug_file", default="")
#     args = ap.parse_args()

#     budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
#     budgets = sorted(set([b for b in budgets if b > 0]))

#     dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
#     tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
#     rm  = AutoModelForSequenceClassification.from_pretrained(
#         args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
#     ).eval()
#     # Attach value-head helper for FsfairX RM
#     try:
#         from types import MethodType
#         rm.forward_value = MethodType(forward_value_fn, rm)
#     except Exception:
#         pass

#     cand = load_json(args.candidate_json)
#     base = load_json(args.baseline_json)
#     if isinstance(cand, dict): cand = list(cand.values())
#     if isinstance(base, dict): base = list(base.values())
#     N = min(len(cand), len(base), args.limit)

#     # accumulators per K
#     eps = 1e-6
#     acc = {K: {"n":0,"wins":0,"losses":0,"ties":0,"skipped":0,"sum_bt":0.0,"sum_best":0.0,"sum_base":0.0} for K in budgets}
#     mismatches = []

#     for i in range(N):
#         c_obj = cand[i]; b_obj = base[i]
#         p_user = squash_ws(detemplate_prompt(get_prompt(c_obj)))
#         b_prompt = squash_ws(get_prompt(b_obj))
#         if p_user and b_prompt and p_user.lower()[:80] != b_prompt.lower()[:80] and len(mismatches) < 20:
#             mismatches.append({"i": i, "cand_prompt": p_user[:120], "base_prompt": b_prompt[:120]})

#         cands = get_candidates(c_obj)
#         b_resp = get_single_baseline_resp(b_obj)
#         if not cands or not b_resp or not p_user:
#             for K in budgets:
#                 acc[K]["skipped"] += 1
#             continue

#         pairs = [(p_user, r) for r in cands] + [(p_user, b_resp)]
#         scores = batch_score_chat(tok, rm, pairs, max_len=args.max_len, batch_size=args.batch_size)
#         if not scores:
#             for K in budgets:
#                 acc[K]["skipped"] += 1
#             continue

#         cand_scores = scores[:-1]
#         base_score  = scores[-1]

#         for K in budgets:
#             kk = min(K, len(cand_scores))
#             if kk <= 0:
#                 acc[K]["skipped"] += 1
#                 continue
#             best_k = max(cand_scores[:kk])

#             # Bradley–Terry (prob candidate best_k beats baseline)
#             p_win = 1.0 / (1.0 + math.exp(base_score - best_k))

#             a = acc[K]
#             a["n"] += 1
#             a["sum_bt"] += p_win
#             a["sum_best"] += best_k
#             a["sum_base"] += base_score
#             if abs(best_k - base_score) < eps: a["ties"] += 1
#             elif best_k > base_score:          a["wins"] += 1
#             else:                               a["losses"] += 1

#     # finalize
#     out = {
#         "candidate_total": len(cand),
#         "baseline_total": len(base),
#         "index_compared": N,
#         "budgets": {}
#     }
#     for K in budgets:
#         a = acc[K]
#         n = max(a["n"], 1)
#         out["budgets"][str(K)] = {
#             "n": a["n"],
#             "wins": a["wins"],
#             "losses": a["losses"],
#             "ties": a["ties"],
#             "skipped": a["skipped"],
#             "bt_winrate": (a["sum_bt"] / a["n"]) if a["n"] > 0 else 0.0,
#             "mean_best_reward": (a["sum_best"] / n) if a["n"] > 0 else 0.0,
#             "mean_baseline_reward": (a["sum_base"] / n) if a["n"] > 0 else 0.0
#         }

#     os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
#     with open(args.out_file, "w", encoding="utf-8") as f:
#         json.dump(out, f, ensure_ascii=False, indent=2)
#     print(json.dumps(out, indent=2))

#     if args.debug_file:
#         with open(args.debug_file, "w", encoding="utf-8") as f:
#             json.dump(mismatches, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()


import argparse, json, os, math, re, unicodedata
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- utils ----------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

_qwen_user_pat = re.compile(r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
_llama_user_pat = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", re.DOTALL | re.IGNORECASE)
def detemplate_prompt(s: str) -> str:
    if not isinstance(s, str): return ""
    q = _qwen_user_pat.findall(s)
    if q: return q[-1]
    l = _llama_user_pat.findall(s)
    if l: return l[-1]
    return s

def squash_ws(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_prompt(obj: Dict[str,Any]) -> str:
    return obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""

def get_candidates(obj: Dict[str,Any]) -> List[str]:
    v = (obj.get("answer") or obj.get("answers") or obj.get("responses") or
         obj.get("outputs") or obj.get("generations") or obj.get("output") or obj.get("response"))
    outs: List[str] = []
    if isinstance(v, str):
        outs = [v]
    elif isinstance(v, list):
        for it in v:
            if isinstance(it, str): outs.append(it)
            elif isinstance(it, dict):
                for tkey in ("text","content","response","output"):
                    tv = it.get(tkey)
                    if isinstance(tv, str) and tv.strip():
                        outs.append(tv); break
    elif isinstance(v, dict) and isinstance(v.get("responses"), list):
        for it in v["responses"]:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                outs.append(it["text"])
    return [squash_ws(x) for x in outs]

def get_single_baseline_resp(obj: Dict[str,Any]) -> str:
    for k in ("output","response","completion"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip(): return squash_ws(v)
    v = obj.get("outputs")
    if isinstance(v, list) and v:
        d = v[0] if isinstance(v[0], dict) else {}
        for t in ("text","content","response","output"):
            tv = d.get(t)
            if isinstance(tv, str) and tv.strip(): return squash_ws(tv)
    v = obj.get("choices")
    if isinstance(v, list) and v:
        d = v[0] if isinstance(v[0], dict) else {}
        m = d.get("message") if isinstance(d.get("message"), dict) else {}
        c = m.get("content")
        if isinstance(c, str) and c.strip(): return squash_ws(c)
        t = d.get("text")
        if isinstance(t, str) and t.strip(): return squash_ws(t)
    v = obj.get("response")
    if isinstance(v, dict):
        t = v.get("text")
        if isinstance(t, str) and t.strip(): return squash_ws(t)
    return ""

# ---------- reward-model helper ----------
def forward_value_fn(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    inputs_embeds=None,
    return_value_only=False,
    prompt_length=0,
    use_cache=False,
    **kwargs,
):
    transformer_outputs = self.model(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = transformer_outputs[0]
    values = self.score(hidden_states).squeeze(-1)
    if return_value_only:
        return values
    if attention_mask is None:
        chosen_end_scores = values[:, -1]
    else:
        last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
        chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
    return {"values": values, "chosen_end_scores": chosen_end_scores}

def _model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device

def str2bool(s: str) -> bool:
    return str(s).lower() in ("1","true","t","yes","y")

def batch_score_chat(tokenizer, model, pairs: List[Tuple[str,str]], max_len=4096, batch_size=8, add_gen_prompt=True) -> List[float]:
    """
    Scores (prompt, response) pairs with the reward model.
    GEM-style: add_generation_prompt is configurable (default True).
    """
    scores = []
    device = _model_device(model)
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        chats = [[{"role":"user","content":p},{"role":"assistant","content":r}] for (p,r) in chunk]
        enc = tokenizer.apply_chat_template(
            chats,
            padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", return_dict=True,
            add_generation_prompt=add_gen_prompt,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if hasattr(model, "forward_value"):
                out = model.forward_value(**enc)["chosen_end_scores"]
            else:
                out = model(**enc, use_cache=False).logits.squeeze(-1)
        s = out.detach().float().cpu().tolist()
        if isinstance(s, float): s = [s]
        scores.extend(s)
    return scores

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_json", required=True)
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--budgets", default="2,4,8,16,32")
    ap.add_argument("--limit", type=int, default=1000000)   # default: effectively "all"
    ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--add_gen_prompt", default="true")     # GEM-style default
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--debug_file", default="")
    args = ap.parse_args()

    add_gen_prompt = str2bool(args.add_gen_prompt)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted(set([b for b in budgets if b > 0]))

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
    rm  = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    try:
        from types import MethodType
        rm.forward_value = MethodType(forward_value_fn, rm)
    except Exception:
        pass

    cand = load_json(args.candidate_json)
    base = load_json(args.baseline_json)
    if isinstance(cand, dict): cand = list(cand.values())
    if isinstance(base, dict): base = list(base.values())
    N = min(len(cand), len(base), args.limit)

    eps = 1e-6
    acc = {K: {"n":0,"wins":0,"losses":0,"ties":0,"skipped":0,"sum_bt":0.0,"sum_best":0.0,"sum_base":0.0} for K in budgets}
    mismatches = []

    for i in range(N):
        c_obj = cand[i]; b_obj = base[i]
        p_user = squash_ws(detemplate_prompt(get_prompt(c_obj)))
        b_prompt = squash_ws(get_prompt(b_obj))
        if p_user and b_prompt and p_user.lower()[:80] != b_prompt.lower()[:80] and len(mismatches) < 20:
            mismatches.append({"i": i, "cand_prompt": p_user[:120], "base_prompt": b_prompt[:120]})

        cands = get_candidates(c_obj)
        b_resp = get_single_baseline_resp(b_obj)
        if not cands or not b_resp or not p_user:
            for K in budgets:
                acc[K]["skipped"] += 1
            continue

        pairs = [(p_user, r) for r in cands] + [(p_user, b_resp)]
        scores = batch_score_chat(tok, rm, pairs, max_len=args.max_len, batch_size=args.batch_size, add_gen_prompt=add_gen_prompt)
        if not scores:
            for K in budgets:
                acc[K]["skipped"] += 1
            continue

        cand_scores = scores[:-1]
        base_score  = scores[-1]

        for K in budgets:
            kk = min(K, len(cand_scores))
            if kk <= 0:
                acc[K]["skipped"] += 1
                continue
            best_k = max(cand_scores[:kk])
            p_win = 1.0 / (1.0 + math.exp(base_score - best_k))
            a = acc[K]
            a["n"] += 1
            a["sum_bt"] += p_win
            a["sum_best"] += best_k
            a["sum_base"] += base_score
            if abs(best_k - base_score) < eps: a["ties"] += 1
            elif best_k > base_score:          a["wins"] += 1
            else:                               a["losses"] += 1

    out = {
        "candidate_total": len(cand),
        "baseline_total": len(base),
        "index_compared": N,
        "budgets": {}
    }
    for K in budgets:
        a = acc[K]
        n = max(a["n"], 1)
        out["budgets"][str(K)] = {
            "n": a["n"],
            "wins": a["wins"],
            "losses": a["losses"],
            "ties": a["ties"],
            "skipped": a["skipped"],
            "bt_winrate": (a["sum_bt"] / a["n"]) if a["n"] > 0 else 0.0,
            "mean_best_reward": (a["sum_best"] / n) if a["n"] > 0 else 0.0,
            "mean_baseline_reward": (a["sum_base"] / n) if a["n"] > 0 else 0.0
        }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, indent=2))

    if args.debug_file:
        with open(args.debug_file, "w", encoding="utf-8") as f:
            json.dump(mismatches, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()