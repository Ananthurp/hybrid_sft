# import argparse, json, os, math, re, unicodedata
# from typing import List, Tuple, Dict, Any
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# def load_json(p: str):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)

# # ---------- prompt normalization ----------
# _qwen_user_pat = re.compile(r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
# _llama_user_pat = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", re.DOTALL | re.IGNORECASE)

# def detemplate_prompt(s: str) -> str:
#     """Return the last user turn from a chat-templated prompt, else the original string."""
#     if not isinstance(s, str):
#         return ""
#     # Qwen-style: <|im_start|>user ... <|im_end|>
#     q = _qwen_user_pat.findall(s)
#     if q:
#         return q[-1]
#     # Llama-3 style: <|start_header_id|>user<|end_header_id|> ... <|eot_id|>
#     l = _llama_user_pat.findall(s)
#     if l:
#         return l[-1]
#     # Fallback: try crude "User:" → "Assistant:" span
#     m = re.search(r"(?:^|\n)\s*user\s*:\s*(.*?)(?:\n\s*assistant\s*:|\Z)", s, re.IGNORECASE | re.DOTALL)
#     if m:
#         return m.group(1)
#     return s

# def squash_ws(s: str) -> str:
#     if s is None: return ""
#     s = unicodedata.normalize("NFKC", s)
#     s = s.replace("\u00A0", " ")
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def canon_prompt_key(s: str) -> str:
#     """Detemplate → normalize → lower → trim trailing punctuation."""
#     s = detemplate_prompt(s)
#     s = squash_ws(s)
#     s = re.sub(r"[ \t]*[?!.]+$", "", s)  # drop trailing ? ! .
#     return s.lower()

# def norm_text(s: str) -> str:
#     return squash_ws(s)

# # ---------- extraction ----------
# def get_prompt(obj: Dict[str, Any]) -> str:
#     return (
#         obj.get("prompt")
#         or obj.get("instruction")
#         or obj.get("input")
#         or ""
#     )

# def get_single_response_string(obj: Dict[str, Any]) -> str:
#     # direct string fields
#     for k in ("output", "response", "completion"):
#         v = obj.get(k)
#         if isinstance(v, str) and v.strip():
#             return v
#     # outputs: [{"text":...}] or [{"content":...}]
#     v = obj.get("outputs")
#     if isinstance(v, list) and v:
#         first = v[0]
#         if isinstance(first, dict):
#             for tkey in ("text", "content", "response", "output"):
#                 tv = first.get(tkey)
#                 if isinstance(tv, str) and tv.strip():
#                     return tv
#     # choices format (OpenAI-ish)
#     v = obj.get("choices")
#     if isinstance(v, list) and v:
#         c0 = v[0]
#         if isinstance(c0, dict):
#             msg = c0.get("message")
#             if isinstance(msg, dict):
#                 cont = msg.get("content")
#                 if isinstance(cont, str) and cont.strip():
#                     return cont
#             t = c0.get("text")
#             if isinstance(t, str) and t.strip():
#                 return t
#     # response dict with "text"
#     v = obj.get("response")
#     if isinstance(v, dict):
#         t = v.get("text")
#         if isinstance(t, str) and t.strip():
#             return t
#     return ""

# def extract_baseline_map(data: Any) -> Dict[str, str]:
#     m: Dict[str, str] = {}
#     if isinstance(data, dict):
#         data = list(data.values())
#     for obj in data:
#         p = get_prompt(obj)
#         r = get_single_response_string(obj)
#         kp = canon_prompt_key(p)
#         if kp and r:
#             m[kp] = norm_text(r)
#     return m

# def get_candidate_responses(obj: Dict[str, Any]) -> List[str]:
#     keys_try = ["answer","answers","responses","outputs","generations","output","response"]
#     v = None
#     for k in keys_try:
#         if k in obj:
#             v = obj[k]; break
#     if isinstance(v, str):
#         return [norm_text(v)]
#     if isinstance(v, list):
#         outs = []
#         for it in v:
#             if isinstance(it, str):
#                 outs.append(norm_text(it))
#             elif isinstance(it, dict):
#                 for tkey in ("text","content","response","output"):
#                     tv = it.get(tkey)
#                     if isinstance(tv, str) and tv.strip():
#                         outs.append(norm_text(tv)); break
#         return outs
#     if isinstance(v, dict) and "responses" in v and isinstance(v["responses"], list):
#         outs = []
#         for it in v["responses"]:
#             if isinstance(it, dict) and isinstance(it.get("text"), str):
#                 outs.append(norm_text(it["text"]))
#         return outs
#     return []

# def extract_candidate_list(data: Any) -> List[Tuple[str, List[str], str, str]]:
#     """
#     Returns list of (key_for_match, [responses], raw_prompt, user_prompt_text)
#     """
#     out = []
#     if isinstance(data, dict):
#         data = list(data.values())
#     for obj in data:
#         p_raw = get_prompt(obj)
#         p_user = detemplate_prompt(p_raw)
#         kp = canon_prompt_key(p_raw)
#         cands = get_candidate_responses(obj)
#         out.append((kp, cands, p_raw, squash_ws(p_user)))
#     return out

# # ---------- scoring ----------
# def batch_score_chat(tokenizer, model, pairs, max_len=4096, batch_size=8) -> List[float]:
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

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--candidate_json", required=True)
#     ap.add_argument("--baseline_json", required=True)
#     ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--max_len", type=int, default=4096)
#     ap.add_argument("--out_file", required=True)
#     ap.add_argument("--debug_unmatched", default="")
#     args = ap.parse_args()

#     dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
#     tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True, trust_remote_code=True)
#     rm  = AutoModelForSequenceClassification.from_pretrained(
#         args.reward_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
#     ).eval()

#     cand = load_json(args.candidate_json)
#     base = load_json(args.baseline_json)

#     cand_list = extract_candidate_list(cand)
#     base_map  = extract_baseline_map(base)

#     n = wins = losses = ties = 0
#     bt_probs = []
#     mean_best = []
#     mean_base = []
#     skipped = 0
#     unmatched_examples = []

#     for kp, responses, raw_p, user_p in cand_list:
#         if kp not in base_map:
#             skipped += 1
#             if len(unmatched_examples) < 10:
#                 unmatched_examples.append({"candidate_prompt_raw": raw_p, "user_prompt": user_p})
#             continue
#         if not responses:
#             skipped += 1
#             continue

#         baseline_resp = base_map[kp]
#         pairs = [(user_p, r) for r in responses] + [(user_p, baseline_resp)]
#         scores = batch_score_chat(tok, rm, pairs, max_len=args.max_len, batch_size=args.batch_size)
#         if not scores:
#             skipped += 1
#             continue

#         best_cand = max(scores[:-1])
#         base_score = scores[-1]
#         p_win = 1.0 / (1.0 + math.exp(base_score - best_cand))
#         bt_probs.append(p_win)
#         mean_best.append(best_cand)
#         mean_base.append(base_score)

#         if abs(best_cand - base_score) < 1e-6:
#             ties += 1
#         elif best_cand > base_score:
#             wins += 1
#         else:
#             losses += 1
#         n += 1

#     result = {
#         "n": n, "wins": wins, "losses": losses, "ties": ties, "skipped": skipped,
#         "bt_winrate": (sum(bt_probs) / n) if n > 0 else 0.0,
#         "mean_best_reward": (sum(mean_best)/n) if n>0 else 0.0,
#         "mean_baseline_reward": (sum(mean_base)/n) if n>0 else 0.0,
#         "candidate_total": len(cand_list),
#         "baseline_total": len(base_map),
#     }

#     os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
#     with open(args.out_file, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)
#     print(json.dumps(result, indent=2))

#     if args.debug_unmatched:
#         with open(args.debug_unmatched, "w", encoding="utf-8") as f:
#             json.dump(unmatched_examples, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()


import argparse, json, os, math, re, unicodedata
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- I/O helpers ----------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_load_tokenizer(name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True, trust_remote_code=True)

def safe_load_rm(name_or_path: str, dtype):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
    except Exception:
        # last-resort fp32
        return AutoModelForSequenceClassification.from_pretrained(
            name_or_path, torch_dtype=torch.float32, trust_remote_code=True
        )

# ---------------- prompt normalization ----------------
_qwen_user_pat  = re.compile(r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
_llama_user_pat = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", re.DOTALL | re.IGNORECASE)

def detemplate_prompt(s: str) -> str:
    """Return the last user turn from a chat-templated prompt, else the original string."""
    if not isinstance(s, str):
        return ""
    q = _qwen_user_pat.findall(s)
    if q:
        return q[-1]
    l = _llama_user_pat.findall(s)
    if l:
        return l[-1]
    # Fallback: try "user:" → "assistant:"
    m = re.search(r"(?:^|\n)\s*user\s*:\s*(.*?)(?:\n\s*assistant\s*:|\Z)", s, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    return s

def squash_ws(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_prompt_key(s: str) -> str:
    """Detemplate → normalize → lower → trim trailing punctuation, for robust alignment."""
    s = detemplate_prompt(s)
    s = squash_ws(s)
    s = re.sub(r"[ \t]*[?!.]+$", "", s)
    return s.lower()

def norm_text(s: str) -> str:
    return squash_ws(s)

# ---------------- extraction ----------------
def get_prompt(obj: Dict[str, Any]) -> str:
    return obj.get("prompt") or obj.get("instruction") or obj.get("input") or ""

def get_single_response_string(obj: Dict[str, Any]) -> str:
    # direct string fields
    for k in ("output", "response", "completion"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # outputs: [{"text":...}] or [{"content":...}]
    v = obj.get("outputs")
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, dict):
            for tkey in ("text", "content", "response", "output"):
                tv = first.get(tkey)
                if isinstance(tv, str) and tv.strip():
                    return tv
    # choices format
    v = obj.get("choices")
    if isinstance(v, list) and v:
        c0 = v[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict):
                cont = msg.get("content")
                if isinstance(cont, str) and cont.strip():
                    return cont
            t = c0.get("text")
            if isinstance(t, str) and t.strip():
                return t
    # response dict with "text"
    v = obj.get("response")
    if isinstance(v, dict):
        t = v.get("text")
        if isinstance(t, str) and t.strip():
            return t
    return ""

def extract_baseline_map(data: Any) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if isinstance(data, dict):
        data = list(data.values())
    for obj in data:
        p = get_prompt(obj)
        r = get_single_response_string(obj)
        kp = canon_prompt_key(p)
        if kp and r:
            m[kp] = norm_text(r)
    return m

def get_candidate_responses(obj: Dict[str, Any]) -> List[str]:
    keys_try = ["answer","answers","responses","outputs","generations","output","response"]
    v = None
    for k in keys_try:
        if k in obj:
            v = obj[k]; break
    if isinstance(v, str):
        return [norm_text(v)]
    if isinstance(v, list):
        outs = []
        for it in v:
            if isinstance(it, str):
                outs.append(norm_text(it))
            elif isinstance(it, dict):
                for tkey in ("text","content","response","output"):
                    tv = it.get(tkey)
                    if isinstance(tv, str) and tv.strip():
                        outs.append(norm_text(tv)); break
        return outs
    if isinstance(v, dict) and "responses" in v and isinstance(v["responses"], list):
        outs = []
        for it in v["responses"]:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                outs.append(norm_text(it["text"]))
        return outs
    return []

def extract_candidate_list(data: Any) -> List[Tuple[str, List[str], str, str]]:
    """
    Returns list of (key_for_match, [responses], raw_prompt, user_prompt_text)
    """
    out = []
    if isinstance(data, dict):
        data = list(data.values())
    for obj in data:
        p_raw = get_prompt(obj)
        p_user = detemplate_prompt(p_raw)
        kp = canon_prompt_key(p_raw)
        cands = get_candidate_responses(obj)
        out.append((kp, cands, p_raw, squash_ws(p_user)))
    return out

# ---------------- scoring ----------------
def batch_score_chat(tokenizer, model, pairs, max_len=4096, batch_size=8) -> List[float]:
    scores: List[float] = []
    device = next(model.parameters()).device
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        chats = [[{"role":"user","content":p},{"role":"assistant","content":r}] for (p,r) in chunk]
        # IMPORTANT: assistant content is present -> do NOT add generation prompt
        try:
            enc = tokenizer.apply_chat_template(
                chats,
                padding="longest",
                truncation=True,
                max_length=max_len,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            ).to(device)
        except Exception:
            # Fallback if no chat template is defined
            texts = [f"User: {p}\nAssistant: {r}" for (p, r) in chunk]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.squeeze(-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        scores.extend(logits.detach().float().cpu().tolist())
    return scores

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_json", required=True)
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--reward_model", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--debug_unmatched", default="")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = safe_load_tokenizer(args.reward_model)
    rm = safe_load_rm(args.reward_model, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm.to(device).eval()

    cand = load_json(args.candidate_json)
    base = load_json(args.baseline_json)

    cand_list = extract_candidate_list(cand)
    base_map  = extract_baseline_map(base)

    n = wins = losses = ties = 0
    bt_probs: List[float] = []
    mean_best: List[float] = []
    mean_base: List[float] = []
    skipped = 0
    unmatched_examples = []

    for kp, responses, raw_p, user_p in cand_list:
        if kp not in base_map:
            skipped += 1
            if len(unmatched_examples) < 10:
                unmatched_examples.append({"candidate_prompt_raw": raw_p, "user_prompt": user_p})
            continue
        if not responses:
            skipped += 1
            continue

        baseline_resp = base_map[kp]
        pairs = [(user_p, r) for r in responses] + [(user_p, baseline_resp)]
        scores = batch_score_chat(tokenizer, rm, pairs, max_len=args.max_len, batch_size=args.batch_size)
        if not scores:
            skipped += 1
            continue

        best_cand = max(scores[:-1])
        base_score = scores[-1]
        # Bradley–Terry probability that candidate > baseline
        p_win = 1.0 / (1.0 + math.exp(base_score - best_cand))
        bt_probs.append(p_win)
        mean_best.append(best_cand)
        mean_base.append(base_score)

        eps = 1e-6
        if abs(best_cand - base_score) < eps:
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
        "mean_best_reward": (sum(mean_best) / n) if n > 0 else 0.0,
        "mean_baseline_reward": (sum(mean_base) / n) if n > 0 else 0.0,
        "candidate_total": len(cand_list),
        "baseline_total": len(base_map),
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))

    if args.debug_unmatched:
        with open(args.debug_unmatched, "w", encoding="utf-8") as f:
            json.dump(unmatched_examples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()