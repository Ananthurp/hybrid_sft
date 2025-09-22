# #!/usr/bin/env python3
# import os, json
# from dataclasses import dataclass, field
# from pprint import pprint
# from types import MethodType
# from tqdm import tqdm
# import numpy as np

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser


# @dataclass
# class Arguments:
#     # Reward model + tokenizers
#     model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     tokenizer_path: str = field(default=None)
#     detokenizer_path: str = field(default=None)

#     # IO
#     data_path: str = field(default=None)         # candidate responses.json (with prompts + answers)
#     batch_size: int = field(default=2)
#     max_size: int = field(default=None)
#     save_path: str = field(default=None)         # per-item rewards json (will contain "reward" arrays)
#     summary_path: str = field(default=None)      # rollup with n-grid stats (auto-derived if None)

#     # Optional baseline for vs-baseline metrics
#     baseline_json: str = field(default=None)     # path to baseline responses (e.g., GPT-4) with instruction/prompt + output


# # ---------- Reward-model forward helper (FsfairX RM) ----------
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
#         # last non-pad position per row
#         last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
#         chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {"values": values, "chosen_end_scores": chosen_end_scores}


# # ---------- Tokenizer helpers ----------
# def _safe_load_tokenizer(path: str):
#     try:
#         return AutoTokenizer.from_pretrained(path, use_fast=True)
#     except Exception:
#         return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


# def _normalize_prompt(s: str) -> str:
#     # normalize whitespace + strip to make matching robust
#     return " ".join((s or "").split())


# # ---------- n-grid helpers ----------
# def _n_grid(n_max: int):
#     # standard grid capped by available n; ensure non-empty
#     base = [1, 2, 4, 8, 16, 32]
#     if n_max is None or n_max < 1:
#         return [1]
#     return [n for n in base if n <= n_max] or [1]


# def _detect_n_max(data):
#     n_max = 0
#     for x in data:
#         ans = x.get("answer")
#         if isinstance(ans, list):
#             n_max = max(n_max, len(ans))
#     return max(1, n_max)


# # ---------- Metrics ----------
# def calculation_best_of_n(data):
#     print("Calculating best of n / mean of n ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     best_n = np.zeros([len(data), len(grid)], dtype=float)
#     mean_n = np.zeros([len(data), len(grid)], dtype=float)

#     for i in tqdm(range(len(data))):
#         rewards = data[i].get("reward", [])
#         if not rewards:
#             continue
#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             best_n[i, gi] = float(np.max(rewards[:m]))
#             mean_n[i, gi] = float(np.mean(rewards[:m]))

#     best_n = np.mean(best_n, axis=0).tolist()
#     mean_n = np.mean(mean_n, axis=0).tolist()
#     print("Best of n:", np.round(best_n, 2))
#     print("Mean of n:", np.round(mean_n, 2))
#     return grid, best_n, mean_n


# def calculate_winrate_gt0(data):
#     """Legacy: win if max(reward[:n]) > 0."""
#     print("Calculating winrate (>0) best-of-n ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     wr = np.zeros([len(grid)], dtype=float)
#     total = 0
#     for x in tqdm(data):
#         rewards = x.get("reward", [])
#         if not rewards:
#             continue
#         total += 1
#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             wr[gi] += 1.0 if float(np.max(rewards[:m])) > 0.0 else 0.0

#     wr = (wr / total * 100.0).tolist() if total > 0 else wr.tolist()
#     print("Winrate (%), best-of-n (>0):", np.round(wr, 2))
#     return grid, wr, total


# def calculate_winrate_vs_baseline(data, baseline_reward_map):
#     """
#     Win if max(candidate_rewards[:n]) > baseline_reward (strict).
#     Only counts items where the baseline exists for the same normalized prompt.
#     """
#     print("Calculating winrate vs baseline (strict >) ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     wins = np.zeros([len(grid)], dtype=float)
#     total = np.zeros([len(grid)], dtype=float)

#     for x in tqdm(data):
#         # find normalized prompt string used for matching
#         ptxt = _normalize_prompt(x.get("prompt") or x.get("instruction", ""))
#         if not ptxt:
#             continue
#         if ptxt not in baseline_reward_map:
#             continue
#         b = float(baseline_reward_map[ptxt])

#         rewards = x.get("reward", [])
#         if not rewards:
#             continue

#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             best = float(np.max(rewards[:m]))
#             total[gi] += 1.0
#             if best > b:
#                 wins[gi] += 1.0

#     # avoid division by zero
#     wr = []
#     for w, t in zip(wins, total):
#         wr.append(float(w / t * 100.0) if t > 0 else 0.0)

#     print("Winrate vs baseline (%), best-of-n:", np.round(wr, 2))
#     return grid, wr, int(total.max() if len(total) else 0)


# # ---------- Baseline scoring ----------
# def _score_pairs(model, tokenizer, device, pairs, batch_size):
#     """
#     Score a list of (prompt, answer) pairs with the RM.
#     Returns list of floats (chosen_end_scores).
#     """
#     scores = []
#     for start in range(0, len(pairs), batch_size):
#         chunk = pairs[start : start + batch_size]
#         chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}] for p, a in chunk]
#         try:
#             inputs = tokenizer.apply_chat_template(
#                 chats,
#                 padding="longest",
#                 add_generation_prompt=False,   # assistant response already present
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
#         except Exception:
#             texts = [f"User: {p}\nAssistant: {a}" for p, a in chunk]
#             inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

#         with torch.no_grad():
#             try:
#                 out = model.forward_value(**inputs)["chosen_end_scores"]
#             except Exception:
#                 out = model(**inputs, use_cache=False).logits.squeeze(-1)
#         scores.extend([float(s) for s in out.detach().cpu().tolist()])
#     return scores


# def build_baseline_reward_map(
#     model,
#     tok_for_rm,
#     detok,
#     baseline_path,
#     device,
#     batch_size=8,
# ):
#     """
#     Load baseline JSON (list of {instruction/prompt, output}) and compute a mapping:
#        normalized_prompt -> baseline_reward
#     """
#     print(f"[baseline] scoring baseline answers from: {baseline_path}")
#     data = json.load(open(baseline_path, "r"))

#     pairs = []
#     key_prompts = []
#     for x in data:
#         # normalize prompt text to match candidate file
#         if detok:
#             ptxt_raw = (x.get("prompt") or x.get("instruction", "")).strip()
#             ptxt = detok.decode(detok.encode(ptxt_raw), skip_special_tokens=True)
#             ptxt = ptxt.replace("user\n\n", "").replace("assistant\n\n", "")
#         else:
#             ptxt = x.get("prompt") or x.get("instruction", "")

#         pnorm = _normalize_prompt(ptxt)
#         atxt = x.get("output") or (x.get("answer")[0] if isinstance(x.get("answer"), list) and x["answer"] else "")
#         if not pnorm or not isinstance(atxt, str) or not atxt.strip():
#             continue
#         pairs.append((pnorm, atxt))
#         key_prompts.append(pnorm)

#     # Score (prompt, output)
#     # NOTE: The RM tokenizer should be used (tok_for_rm).
#     pq = [(p, a) for p, a in pairs]
#     vals = _score_pairs(model, tok_for_rm, device, pq, batch_size=batch_size)

#     # Keep first occurrence per prompt (in case of duplicates)
#     reward_map = {}
#     for k, v in zip(key_prompts, vals):
#         if k not in reward_map:
#             reward_map[k] = v

#     print(f"[baseline] scored {len(reward_map)} unique prompts from baseline.")
#     return reward_map


# # ---------- Main ----------
# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)
#     assert args.data_path, "data_path is required"
#     assert args.save_path, "save_path is required"

#     if args.summary_path is None:
#         args.summary_path = os.path.join(os.path.dirname(args.save_path), "reward_summary.json")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Reward model
#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
#         trust_remote_code=True,
#     )
#     model.forward_value = MethodType(forward_value_fn, model)  # attach helper
#     model.eval().to(device)

#     # Tokenizers
#     tok_name = args.tokenizer_path or args.model_name_or_path
#     tokenizer = _safe_load_tokenizer(tok_name)
#     detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None

#     # Load candidate responses
#     response_data = json.load(open(args.data_path, "r"))
#     if args.max_size:
#         response_data = response_data[: args.max_size]

#     # Optional: build baseline reward map
#     baseline_reward_map = None
#     if args.baseline_json and os.path.isfile(args.baseline_json):
#         # IMPORTANT: For RM scoring we should use the RM tokenizer, not the model’s base tokenizer.
#         # We already loaded `tokenizer` from args.tokenizer_path or RM path.
#         baseline_reward_map = build_baseline_reward_map(
#             model=model,
#             tok_for_rm=tokenizer,
#             detok=detok,
#             baseline_path=args.baseline_json,
#             device=device,
#             batch_size=max(4, args.batch_size),
#         )

#     # If per-item file exists, re-summarize only (and compute vs-baseline if provided)
#     if os.path.exists(args.save_path):
#         print(f"{args.save_path} exists; loading and recomputing summaries...")
#         response_data = json.load(open(args.save_path, "r"))
#         grid, best_n, mean_n = calculation_best_of_n(response_data)
#         _, wr_gt0, used = calculate_winrate_gt0(response_data)

#         summary = {
#             "n_grid": grid,
#             "best_of_n": best_n,
#             "mean_of_n": mean_n,
#             "winrate_gt0_of_n": wr_gt0,
#             "num_items": len(response_data),
#             "num_items_scored": int(used),
#             "reward_model": args.model_name_or_path,
#         }

#         if baseline_reward_map is not None:
#             _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map)
#             summary["winrate_vs_baseline_of_n"] = wr_vs_base
#             summary["num_items_matched_baseline"] = int(matched)

#         json.dump(summary, open(args.summary_path, "w"), indent=2)
#         print(f"Saved summary to {args.summary_path}")
#         return

#     # --------- Compute candidate rewards ---------
#     for start in tqdm(range(0, len(response_data), args.batch_size)):
#         end = min(start + args.batch_size, len(response_data))
#         prompts, answers = [], []

#         # Flatten (prompt, answer) pairs from the candidate file
#         for x in response_data[start:end]:
#             # canonicalize prompt text similarly to how we save it
#             if detok:
#                 ptxt = detok.decode(detok.encode(x.get("prompt") or x.get("instruction", "")),
#                                     skip_special_tokens=True)
#                 ptxt = ptxt.replace("user\n\n", "").replace("assistant\n\n", "")
#             else:
#                 ptxt = x.get("prompt") or x.get("instruction", "")

#             ans = x.get("answer")
#             if isinstance(ans, list) and ans:
#                 for a in ans:
#                     answers.append(a)
#                     prompts.append(ptxt)
#             elif isinstance(x.get("output"), str):
#                 answers.append(x["output"])
#                 prompts.append(ptxt)
#             else:
#                 # skip rows with no answers
#                 pass

#         if not answers:
#             continue

#         # Build chat messages: user + assistant content already present
#         chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}]
#                  for p, a in zip(prompts, answers)]

#         # IMPORTANT: do NOT add generation prompt since assistant turn exists
#         try:
#             inputs = tokenizer.apply_chat_template(
#                 chats,
#                 padding="longest",
#                 add_generation_prompt=False,
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
#         except Exception:
#             texts = [f"User: {p}\nAssistant: {a}" for p, a in zip(prompts, answers)]
#             inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

#         with torch.no_grad():
#             try:
#                 outputs = model.forward_value(**inputs)["chosen_end_scores"]
#             except Exception:
#                 out = model(**inputs, use_cache=False)
#                 outputs = out.logits.squeeze(-1)

#         # write rewards back to the same structure
#         c = 0
#         for x in response_data[start:end]:
#             ans = x.get("answer")
#             if isinstance(ans, list) and ans:
#                 k = len(ans)
#                 x["reward"] = outputs[c : c + k].tolist()
#                 c += k
#             elif isinstance(x.get("output"), str):
#                 x["reward"] = [float(outputs[c].item())]
#                 c += 1

#             # If baseline provided, attach per-item baseline reward (if match)
#             if baseline_reward_map is not None:
#                 ptxt_norm = _normalize_prompt(x.get("prompt") or x.get("instruction", ""))
#                 if ptxt_norm in baseline_reward_map:
#                     x["baseline_reward"] = float(baseline_reward_map[ptxt_norm])

#     # --------- Summaries ---------
#     grid, best_n, mean_n = calculation_best_of_n(response_data)
#     _, wr_gt0, used = calculate_winrate_gt0(response_data)

#     summary = {
#         "n_grid": grid,
#         "best_of_n": best_n,
#         "mean_of_n": mean_n,
#         "winrate_gt0_of_n": wr_gt0,
#         "num_items": len(response_data),
#         "num_items_scored": int(used),
#         "reward_model": args.model_name_or_path,
#     }

#     if baseline_reward_map is not None:
#         _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map)
#         summary["winrate_vs_baseline_of_n"] = wr_vs_base
#         summary["num_items_matched_baseline"] = int(matched)

#     # --------- Save ---------
#     json.dump(response_data, open(args.save_path, "w"), indent=2)
#     json.dump(summary, open(args.summary_path, "w"), indent=2)
#     print(f"Saved per-item to {args.save_path}")
#     print(f"Saved summary to {args.summary_path}")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# import os, json
# from dataclasses import dataclass, field
# from pprint import pprint
# from types import MethodType
# from tqdm import tqdm
# import numpy as np

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser


# @dataclass
# class Arguments:
#     # Reward model + tokenizers
#     model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     tokenizer_path: str = field(default=None)
#     detokenizer_path: str = field(default=None)

#     # IO
#     data_path: str = field(default=None)         # candidate responses.json (with prompts + answers)
#     batch_size: int = field(default=2)
#     max_size: int = field(default=None)
#     save_path: str = field(default=None)         # per-item rewards json (will contain "reward" arrays)
#     summary_path: str = field(default=None)      # rollup with n-grid stats (auto-derived if None)

#     # Optional baseline for vs-baseline metrics
#     baseline_json: str = field(default=None)     # path to baseline responses (e.g., GPT-4) with instruction/prompt + output


# # ---------- Reward-model forward helper (FsfairX RM) ----------
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
#         # last non-pad position per row
#         last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
#         chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {"values": values, "chosen_end_scores": chosen_end_scores}


# # ---------- Tokenizer helpers ----------
# def _safe_load_tokenizer(path: str):
#     try:
#         return AutoTokenizer.from_pretrained(path, use_fast=True)
#     except Exception:
#         return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


# def _normalize_prompt(s: str) -> str:
#     # normalize whitespace + strip to make matching robust
#     return " ".join((s or "").split())


# def _extract_and_normalize_prompt(raw_prompt: str | None, raw_instruction: str | None, detok):
#     """
#     Use the same detokenization + normalization logic everywhere (baseline, candidates, metrics).
#     If detok is provided, we roundtrip encode/decode to drop special tokens/templates.
#     """
#     text = (raw_prompt or raw_instruction or "").strip()
#     if detok:
#         # roundtrip through the detokenizer to remove special tokens if any
#         text = detok.decode(detok.encode(text), skip_special_tokens=True)
#         # some templates may leave literal role tags as plain text
#         text = text.replace("user\n\n", "").replace("assistant\n\n", "")
#     return _normalize_prompt(text)


# # ---------- n-grid helpers ----------
# def _n_grid(n_max: int):
#     # standard grid capped by available n; ensure non-empty
#     base = [1, 2, 4, 8, 16, 32]
#     if n_max is None or n_max < 1:
#         return [1]
#     return [n for n in base if n <= n_max] or [1]


# def _detect_n_max(data):
#     n_max = 0
#     for x in data:
#         ans = x.get("answer")
#         if isinstance(ans, list):
#             n_max = max(n_max, len(ans))
#     return max(1, n_max)


# # ---------- Metrics ----------
# def calculation_best_of_n(data):
#     print("Calculating best of n / mean of n ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     best_n = np.zeros([len(data), len(grid)], dtype=float)
#     mean_n = np.zeros([len(data), len(grid)], dtype=float)

#     for i in tqdm(range(len(data))):
#         rewards = data[i].get("reward", [])
#         if not rewards:
#             continue
#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             best_n[i, gi] = float(np.max(rewards[:m]))
#             mean_n[i, gi] = float(np.mean(rewards[:m]))

#     best_n = np.mean(best_n, axis=0).tolist()
#     mean_n = np.mean(mean_n, axis=0).tolist()
#     print("Best of n:", np.round(best_n, 2))
#     print("Mean of n:", np.round(mean_n, 2))
#     return grid, best_n, mean_n


# def calculate_winrate_gt0(data):
#     """Legacy: win if max(reward[:n]) > 0."""
#     print("Calculating winrate (>0) best-of-n ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     wr = np.zeros([len(grid)], dtype=float)
#     total = 0
#     for x in tqdm(data):
#         rewards = x.get("reward", [])
#         if not rewards:
#             continue
#         total += 1
#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             wr[gi] += 1.0 if float(np.max(rewards[:m])) > 0.0 else 0.0

#     wr = (wr / total * 100.0).tolist() if total > 0 else wr.tolist()
#     print("Winrate (%), best-of-n (>0):", np.round(wr, 2))
#     return grid, wr, total


# def calculate_winrate_vs_baseline(data, baseline_reward_map, detok):
#     """
#     Win if max(candidate_rewards[:n]) > baseline_reward (strict).
#     Only counts items where the baseline exists for the same (detok+normalized) prompt.
#     """
#     print("Calculating winrate vs baseline (strict >) ...")
#     n_max = _detect_n_max(data)
#     grid = _n_grid(n_max)

#     wins = np.zeros([len(grid)], dtype=float)
#     total = np.zeros([len(grid)], dtype=float)

#     for x in tqdm(data):
#         # MATCH using the same detok+normalize function used when building the baseline map
#         ptxt_norm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
#         if not ptxt_norm or ptxt_norm not in baseline_reward_map:
#             continue
#         b = float(baseline_reward_map[ptxt_norm])

#         rewards = x.get("reward", [])
#         if not rewards:
#             continue

#         for gi, n in enumerate(grid):
#             m = min(n, len(rewards))
#             best = float(np.max(rewards[:m]))
#             total[gi] += 1.0
#             if best > b:
#                 wins[gi] += 1.0

#     # avoid division by zero
#     wr = []
#     for w, t in zip(wins, total):
#         wr.append(float(w / t * 100.0) if t > 0 else 0.0)

#     print("Winrate vs baseline (%), best-of-n:", np.round(wr, 2))
#     return grid, wr, int(total.max() if len(total) else 0)


# # ---------- Scoring helpers ----------
# def _score_pairs(model, tokenizer, device, pairs, batch_size):
#     """
#     Score a list of (prompt, answer) pairs with the RM.
#     Returns list of floats (chosen_end_scores).
#     """
#     scores = []
#     for start in range(0, len(pairs), batch_size):
#         chunk = pairs[start : start + batch_size]
#         chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}] for p, a in chunk]
#         try:
#             inputs = tokenizer.apply_chat_template(
#                 chats,
#                 padding="longest",
#                 add_generation_prompt=False,   # assistant response already present
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
#         except Exception:
#             texts = [f"User: {p}\nAssistant: {a}" for p, a in chunk]
#             inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

#         with torch.no_grad():
#             try:
#                 out = model.forward_value(**inputs)["chosen_end_scores"]
#             except Exception:
#                 out = model(**inputs, use_cache=False).logits.squeeze(-1)
#         scores.extend([float(s) for s in out.detach().cpu().tolist()])
#     return scores


# def build_baseline_reward_map(
#     model,
#     tok_for_rm,
#     detok,
#     baseline_path,
#     device,
#     batch_size=8,
# ):
#     """
#     Load baseline JSON (list of {instruction/prompt, output}) and compute a mapping:
#        normalized_prompt -> baseline_reward
#     """
#     print(f"[baseline] scoring baseline answers from: {baseline_path}")
#     data = json.load(open(baseline_path, "r"))

#     pairs = []
#     key_prompts = []
#     for x in data:
#         # detok + normalize prompt
#         pnorm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
#         atxt = x.get("output") or (x.get("answer")[0] if isinstance(x.get("answer"), list) and x["answer"] else "")
#         if not pnorm or not isinstance(atxt, str) or not atxt.strip():
#             continue
#         pairs.append((pnorm, atxt))
#         key_prompts.append(pnorm)

#     # Score (prompt, output)
#     vals = _score_pairs(model, tok_for_rm, device, pairs, batch_size=batch_size)

#     # Keep first occurrence per prompt (in case of duplicates)
#     reward_map = {}
#     for k, v in zip(key_prompts, vals):
#         if k not in reward_map:
#             reward_map[k] = v

#     print(f"[baseline] scored {len(reward_map)} unique prompts from baseline.")
#     return reward_map


# # ---------- Main ----------
# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)
#     assert args.data_path, "data_path is required"
#     assert args.save_path, "save_path is required"

#     if args.summary_path is None:
#         args.summary_path = os.path.join(os.path.dirname(args.save_path), "reward_summary.json")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Reward model
#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
#         trust_remote_code=True,
#     )
#     model.forward_value = MethodType(forward_value_fn, model)  # attach helper
#     model.eval().to(device)

#     # Tokenizers
#     tok_name = args.tokenizer_path or args.model_name_or_path
#     tokenizer = _safe_load_tokenizer(tok_name)
#     detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None

#     # Load candidate responses
#     response_data = json.load(open(args.data_path, "r"))
#     if args.max_size:
#         response_data = response_data[: args.max_size]

#     # Optional: build baseline reward map
#     baseline_reward_map = None
#     if args.baseline_json and os.path.isfile(args.baseline_json):
#         # IMPORTANT: For RM scoring we use the RM tokenizer (tokenizer loaded above).
#         baseline_reward_map = build_baseline_reward_map(
#             model=model,
#             tok_for_rm=tokenizer,
#             detok=detok,
#             baseline_path=args.baseline_json,
#             device=device,
#             batch_size=max(4, args.batch_size),
#         )

#     # If per-item file exists, re-summarize only (and compute vs-baseline if provided)
#     if os.path.exists(args.save_path):
#         print(f"{args.save_path} exists; loading and recomputing summaries...")
#         response_data = json.load(open(args.save_path, "r"))
#         grid, best_n, mean_n = calculation_best_of_n(response_data)
#         _, wr_gt0, used = calculate_winrate_gt0(response_data)

#         summary = {
#             "n_grid": grid,
#             "best_of_n": best_n,
#             "mean_of_n": mean_n,
#             "winrate_gt0_of_n": wr_gt0,
#             "num_items": len(response_data),
#             "num_items_scored": int(used),
#             "reward_model": args.model_name_or_path,
#         }

#         if baseline_reward_map is not None:
#             _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map, detok)
#             summary["winrate_vs_baseline_of_n"] = wr_vs_base
#             summary["num_items_matched_baseline"] = int(matched)

#         json.dump(summary, open(args.summary_path, "w"), indent=2)
#         print(f"Saved summary to {args.summary_path}")
#         return

#     # --------- Compute candidate rewards ---------
#     for start in tqdm(range(0, len(response_data), args.batch_size)):
#         end = min(start + args.batch_size, len(response_data))
#         prompts, answers = [], []

#         # Flatten (prompt, answer) pairs from the candidate file
#         for x in response_data[start:end]:
#             # use unified extractor (detok + normalize) for prompts
#             ptxt_norm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)

#             ans = x.get("answer")
#             if isinstance(ans, list) and ans:
#                 for a in ans:
#                     answers.append(a)
#                     prompts.append(ptxt_norm)
#             elif isinstance(x.get("output"), str):
#                 answers.append(x["output"])
#                 prompts.append(ptxt_norm)
#             else:
#                 # skip rows with no answers
#                 pass

#         if not answers:
#             continue

#         # Build chat messages: user + assistant content already present
#         chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}]
#                  for p, a in zip(prompts, answers)]

#         # IMPORTANT: do NOT add generation prompt since assistant turn exists
#         try:
#             inputs = tokenizer.apply_chat_template(
#                 chats,
#                 padding="longest",
#                 add_generation_prompt=False,
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
#         except Exception:
#             texts = [f"User: {p}\nAssistant: {a}" for p, a in zip(prompts, answers)]
#             inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

#         with torch.no_grad():
#             try:
#                 outputs = model.forward_value(**inputs)["chosen_end_scores"]
#             except Exception:
#                 out = model(**inputs, use_cache=False)
#                 outputs = out.logits.squeeze(-1)

#         # write rewards back to the same structure
#         c = 0
#         for x in response_data[start:end]:
#             ans = x.get("answer")
#             if isinstance(ans, list) and ans:
#                 k = len(ans)
#                 x["reward"] = outputs[c : c + k].tolist()
#                 c += k
#             elif isinstance(x.get("output"), str):
#                 x["reward"] = [float(outputs[c].item())]
#                 c += 1

#             # If baseline provided, attach per-item baseline reward (if match)
#             if baseline_reward_map is not None:
#                 ptxt_norm_item = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
#                 if ptxt_norm_item in baseline_reward_map:
#                     x["baseline_reward"] = float(baseline_reward_map[ptxt_norm_item])

#     # --------- Summaries ---------
#     grid, best_n, mean_n = calculation_best_of_n(response_data)
#     _, wr_gt0, used = calculate_winrate_gt0(response_data)

#     summary = {
#         "n_grid": grid,
#         "best_of_n": best_n,
#         "mean_of_n": mean_n,
#         "winrate_gt0_of_n": wr_gt0,
#         "num_items": len(response_data),
#         "num_items_scored": int(used),
#         "reward_model": args.model_name_or_path,
#     }

#     if baseline_reward_map is not None:
#         _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map, detok)
#         summary["winrate_vs_baseline_of_n"] = wr_vs_base
#         summary["num_items_matched_baseline"] = int(matched)

#     # --------- Save ---------
#     json.dump(response_data, open(args.save_path, "w"), indent=2)
#     json.dump(summary, open(args.summary_path, "w"), indent=2)
#     print(f"Saved per-item to {args.save_path}")
#     print(f"Saved summary to {args.summary_path}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os, json
from dataclasses import dataclass, field
from pprint import pprint
from types import MethodType
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser


@dataclass
class Arguments:
    # Reward model + tokenizers
    model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    tokenizer_path: str = field(default=None)
    detokenizer_path: str = field(default=None)

    # IO
    data_path: str = field(default=None)         # candidate responses.json (with prompts + answers)
    batch_size: int = field(default=2)
    max_size: int = field(default=None)
    save_path: str = field(default=None)         # per-item rewards json (will contain "reward" arrays)
    summary_path: str = field(default=None)      # rollup with n-grid stats (auto-derived if None)

    # Optional baseline for vs-baseline metrics
    baseline_json: str = field(default=None)     # path to baseline responses (e.g., GPT-4)

    # GEM-style toggle: append empty assistant header when formatting chats
    add_gen_prompt: bool = field(default=True)


# ---------- Reward-model forward helper (FsfairX RM) ----------
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
    hidden_states = transformer_outputs[0]         # [B, T, H]
    values = self.score(hidden_states).squeeze(-1) # [B, T]
    if return_value_only:
        return values
    if attention_mask is None:
        chosen_end_scores = values[:, -1]
    else:
        # last non-pad position per row
        last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
        chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
    return {"values": values, "chosen_end_scores": chosen_end_scores}


# ---------- Tokenizer helpers ----------
def _safe_load_tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


def _normalize_prompt(s: str) -> str:
    return " ".join((s or "").split())


def _extract_and_normalize_prompt(raw_prompt: str | None, raw_instruction: str | None, detok):
    """
    Use the same detokenization + normalization logic everywhere (baseline, candidates, metrics).
    If detok is provided, we roundtrip encode/decode to drop special tokens/templates.
    """
    text = (raw_prompt or raw_instruction or "").strip()
    if detok:
        text = detok.decode(detok.encode(text), skip_special_tokens=True)
        text = text.replace("user\n\n", "").replace("assistant\n\n", "")
    return _normalize_prompt(text)


# ---------- n-grid helpers ----------
def _n_grid(n_max: int):
    base = [1, 2, 4, 8, 16, 32]
    if n_max is None or n_max < 1:
        return [1]
    return [n for n in base if n <= n_max] or [1]


def _detect_n_max(data):
    n_max = 0
    for x in data:
        ans = x.get("answer")
        if isinstance(ans, list):
            n_max = max(n_max, len(ans))
    return max(1, n_max)


# ---------- Metrics ----------
def calculation_best_of_n(data):
    print("Calculating best of n / mean of n ...")
    n_max = _detect_n_max(data)
    grid = _n_grid(n_max)

    best_n = np.zeros([len(data), len(grid)], dtype=float)
    mean_n = np.zeros([len(data), len(grid)], dtype=float)

    for i in tqdm(range(len(data))):
        rewards = data[i].get("reward", [])
        if not rewards:
            continue
        for gi, n in enumerate(grid):
            m = min(n, len(rewards))
            best_n[i, gi] = float(np.max(rewards[:m]))
            mean_n[i, gi] = float(np.mean(rewards[:m]))

    best_n = np.mean(best_n, axis=0).tolist()
    mean_n = np.mean(mean_n, axis=0).tolist()
    print("Best of n:", np.round(best_n, 2))
    print("Mean of n:", np.round(mean_n, 2))
    return grid, best_n, mean_n


def calculate_winrate_gt0(data):
    """Legacy: win if max(reward[:n]) > 0."""
    print("Calculating winrate (>0) best-of-n ...")
    n_max = _detect_n_max(data)
    grid = _n_grid(n_max)

    wr = np.zeros([len(grid)], dtype=float)
    total = 0
    for x in tqdm(data):
        rewards = x.get("reward", [])
        if not rewards:
            continue
        total += 1
        for gi, n in enumerate(grid):
            m = min(n, len(rewards))
            wr[gi] += 1.0 if float(np.max(rewards[:m])) > 0.0 else 0.0

    wr = (wr / total * 100.0).tolist() if total > 0 else wr.tolist()
    print("Winrate (%), best-of-n (>0):", np.round(wr, 2))
    return grid, wr, total


def calculate_winrate_vs_baseline(data, baseline_reward_map, detok):
    """
    Win if max(candidate_rewards[:n]) > baseline_reward (strict).
    Only counts items where the baseline exists for the same (detok+normalized) prompt.
    """
    print("Calculating winrate vs baseline (strict >) ...")
    n_max = _detect_n_max(data)
    grid = _n_grid(n_max)

    wins = np.zeros([len(grid)], dtype=float)
    total = np.zeros([len(grid)], dtype=float)

    for x in tqdm(data):
        ptxt_norm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
        if not ptxt_norm or ptxt_norm not in baseline_reward_map:
            continue
        b = float(baseline_reward_map[ptxt_norm])

        rewards = x.get("reward", [])
        if not rewards:
            continue

        for gi, n in enumerate(grid):
            m = min(n, len(rewards))
            best = float(np.max(rewards[:m]))
            total[gi] += 1.0
            if best > b:
                wins[gi] += 1.0

    wr = []
    for w, t in zip(wins, total):
        wr.append(float(w / t * 100.0) if t > 0 else 0.0)

    print("Winrate vs baseline (%), best-of-n:", np.round(wr, 2))
    return grid, wr, int(total.max() if len(total) else 0)


# ---------- Scoring helpers ----------
def _score_pairs(model, tokenizer, device, pairs, batch_size, add_gen_prompt: bool):
    """
    Score a list of (prompt, answer) pairs with the RM.
    Returns list of floats (chosen_end_scores).
    GEM-style: add_generation_prompt=True (configurable).
    """
    scores = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}] for p, a in chunk]
        try:
            inputs = tokenizer.apply_chat_template(
                chats,
                padding="longest",
                add_generation_prompt=add_gen_prompt,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        except Exception:
            texts = [f"User: {p}\nAssistant: {a}" for p, a in chunk]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            try:
                out = model.forward_value(**inputs)["chosen_end_scores"]
            except Exception:
                out = model(**inputs, use_cache=False).logits.squeeze(-1)
        scores.extend([float(s) for s in out.detach().cpu().tolist()])
    return scores


def build_baseline_reward_map(
    model,
    tok_for_rm,
    detok,
    baseline_path,
    device,
    batch_size=8,
    add_gen_prompt=True,
):
    """
    Load baseline JSON (list of {instruction/prompt, output}) and compute a mapping:
       normalized_prompt -> baseline_reward
    """
    print(f"[baseline] scoring baseline answers from: {baseline_path}")
    data = json.load(open(baseline_path, "r"))

    pairs = []
    key_prompts = []
    for x in data:
        pnorm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
        atxt = x.get("output") or (x.get("answer")[0] if isinstance(x.get("answer"), list) and x["answer"] else "")
        if not pnorm or not isinstance(atxt, str) or not atxt.strip():
            continue
        pairs.append((pnorm, atxt))
        key_prompts.append(pnorm)

    vals = _score_pairs(model, tok_for_rm, device, pairs, batch_size=batch_size, add_gen_prompt=add_gen_prompt)

    reward_map = {}
    for k, v in zip(key_prompts, vals):
        if k not in reward_map:
            reward_map[k] = v

    print(f"[baseline] scored {len(reward_map)} unique prompts from baseline.")
    return reward_map


# ---------- Main ----------
def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)
    assert args.data_path, "data_path is required"
    assert args.save_path, "save_path is required"

    if args.summary_path is None:
        args.summary_path = os.path.join(os.path.dirname(args.save_path), "reward_summary.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        trust_remote_code=True,
    )
    model.forward_value = MethodType(forward_value_fn, model)  # attach helper
    model.eval().to(device)

    # Tokenizers
    tok_name = args.tokenizer_path or args.model_name_or_path
    tokenizer = _safe_load_tokenizer(tok_name)
    detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None

    # Load candidate responses
    response_data = json.load(open(args.data_path, "r"))
    if args.max_size:
        response_data = response_data[: args.max_size]

    # Optional: build baseline reward map
    baseline_reward_map = None
    if args.baseline_json and os.path.isfile(args.baseline_json):
        baseline_reward_map = build_baseline_reward_map(
            model=model,
            tok_for_rm=tokenizer,
            detok=detok,
            baseline_path=args.baseline_json,
            device=device,
            batch_size=max(4, args.batch_size),
            add_gen_prompt=args.add_gen_prompt,
        )

    # If per-item file exists, re-summarize only (and compute vs-baseline if provided)
    if os.path.exists(args.save_path):
        print(f"{args.save_path} exists; loading and recomputing summaries...")
        response_data = json.load(open(args.save_path, "r"))
        grid, best_n, mean_n = calculation_best_of_n(response_data)
        _, wr_gt0, used = calculate_winrate_gt0(response_data)

        summary = {
            "n_grid": grid,
            "best_of_n": best_n,
            "mean_of_n": mean_n,
            "winrate_gt0_of_n": wr_gt0,
            "num_items": len(response_data),
            "num_items_scored": int(used),
            "reward_model": args.model_name_or_path,
        }

        if baseline_reward_map is not None:
            _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map, detok)
            summary["winrate_vs_baseline_of_n"] = wr_vs_base
            summary["num_items_matched_baseline"] = int(matched)

        json.dump(summary, open(args.summary_path, "w"), indent=2)
        print(f"Saved summary to {args.summary_path}")
        return

    # --------- Compute candidate rewards ---------
    for start in tqdm(range(0, len(response_data), args.batch_size)):
        end = min(start + args.batch_size, len(response_data))
        prompts, answers = [], []

        # Flatten (prompt, answer) pairs from the candidate file
        for x in response_data[start:end]:
            ptxt_norm = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)

            ans = x.get("answer")
            if isinstance(ans, list) and ans:
                for a in ans:
                    answers.append(a)
                    prompts.append(ptxt_norm)
            elif isinstance(x.get("output"), str):
                answers.append(x["output"])
                prompts.append(ptxt_norm)

        if not answers:
            continue

        chats = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}]
                 for p, a in zip(prompts, answers)]

        try:
            inputs = tokenizer.apply_chat_template(
                chats,
                padding="longest",
                add_generation_prompt=args.add_gen_prompt,  # GEM-style
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        except Exception:
            texts = [f"User: {p}\nAssistant: {a}" for p, a in zip(prompts, answers)]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            try:
                outputs = model.forward_value(**inputs)["chosen_end_scores"]
            except Exception:
                out = model(**inputs, use_cache=False)
                outputs = out.logits.squeeze(-1)

        # write rewards back
        c = 0
        for x in response_data[start:end]:
            ans = x.get("answer")
            if isinstance(ans, list) and ans:
                k = len(ans)
                x["reward"] = outputs[c : c + k].tolist()
                c += k
            elif isinstance(x.get("output"), str):
                x["reward"] = [float(outputs[c].item())]
                c += 1

            if baseline_reward_map is not None:
                ptxt_norm_item = _extract_and_normalize_prompt(x.get("prompt"), x.get("instruction"), detok)
                if ptxt_norm_item in baseline_reward_map:
                    x["baseline_reward"] = float(baseline_reward_map[ptxt_norm_item])

    # --------- Summaries ---------
    grid, best_n, mean_n = calculation_best_of_n(response_data)
    _, wr_gt0, used = calculate_winrate_gt0(response_data)

    summary = {
        "n_grid": grid,
        "best_of_n": best_n,
        "mean_of_n": mean_n,
        "winrate_gt0_of_n": wr_gt0,
        "num_items": len(response_data),
        "num_items_scored": int(used),
        "reward_model": args.model_name_or_path,
    }

    if baseline_reward_map is not None:
        _, wr_vs_base, matched = calculate_winrate_vs_baseline(response_data, baseline_reward_map, detok)
        summary["winrate_vs_baseline_of_n"] = wr_vs_base
        summary["num_items_matched_baseline"] = int(matched)

    # --------- Save ---------
    json.dump(response_data, open(args.save_path, "w"), indent=2)
    json.dump(summary, open(args.summary_path, "w"), indent=2)
    print(f"Saved per-item to {args.save_path}")
    print(f"Saved summary to {args.summary_path}")


if __name__ == "__main__":
    main()