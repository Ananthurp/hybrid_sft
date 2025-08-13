# import os
# from dataclasses import dataclass, field
# from pprint import pprint
# import json
# from types import MethodType
# from tqdm import tqdm
# import numpy as np

# import torch
# from transformers import AutoTokenizer, pipeline
# from transformers import AutoModel, AutoModelForSequenceClassification, HfArgumentParser


# @dataclass
# class Arguments:
#     model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     tokenizer_path: str = field(default=None)

#     detokenizer_path: str = field(default=None)

#     data_path: str = field(default=None)
#     batch_size: int = field(default=1)
#     max_size: int = field(default=None)

#     save_path: str = field(default=None)


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
#     hidden_states = transformer_outputs[0]
#     values = self.score(hidden_states).squeeze(-1)
#     if return_value_only:
#         return values
#     else:
#         if attention_mask is None:
#             chosen_end_scores = values[:, -1]
#         else:
#             last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
#             chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {
#         "values": values,
#         "chosen_end_scores": chosen_end_scores,
#     }


# def calculation_best_of_n(data):
#     print("Calculating best of n reward ....")
#     best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
#     mean_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
#     for i in tqdm(range(len(data))):
#         rewards = data[i]["reward"]
#         best_n[i][0] = rewards[0]
#         best_n[i][1] = max(rewards[:2])
#         best_n[i][2] = max(rewards[:4])
#         best_n[i][3] = max(rewards[:8])
#         best_n[i][4] = max(rewards[:16])
#         best_n[i][5] = max(rewards[:32])

#         mean_n[i][0] = rewards[0]
#         mean_n[i][1] = np.mean(rewards[:2])
#         mean_n[i][2] = np.mean(rewards[:4])
#         mean_n[i][3] = np.mean(rewards[:8])
#         mean_n[i][4] = np.mean(rewards[:16])
#         mean_n[i][5] = np.mean(rewards[:32])
#     best_n = np.mean(best_n, axis=0)
#     print("Best of n: {}".format(np.round(best_n, 2)))
#     mean_n = np.mean(mean_n, axis=0)
#     print("Mean of n: {}".format(np.round(mean_n, 2)))
#     return best_n, mean_n


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)
#     assert args.data_path is not None
#     assert args.save_path is not None

#     device = torch.device("cuda")

#     model_class = AutoModelForSequenceClassification
#     flash_attn = True
#     model = model_class.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2" if flash_attn else "eager",
#         trust_remote_code=True,
#     )
#     # model.forward_value = forward_value_fn
#     model.forward_value = MethodType(forward_value_fn, model)
#     model.eval()
#     model.to(device)

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_path or args.model_name_or_path
#     )
#     tokenizer.padding_side = "right"
#     if args.detokenizer_path is not None:
#         detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
#     else:
#         detokenizer = None

#     response_data = json.load(open(args.data_path, "r"))

#     if args.max_size:
#         response_data = response_data[: args.max_size]
#     if os.path.exists(args.save_path):
#         response_data = json.load(open(args.save_path, "r"))
#         calculation_best_of_n(response_data)
#         return
#     for start in tqdm(range(0, len(response_data), args.batch_size)):
#         end = start + args.batch_size
#         prompts = []
#         answers = []
#         for x in response_data[start:end]:
#             if detokenizer:
#                 prompt_str = (
#                     detokenizer.decode(
#                         detokenizer.encode(x["prompt"]), skip_special_tokens=True
#                     )
#                     .replace("user\n\n", "")
#                     .replace("assistant\n\n", "")
#                 )
#             else:
#                 if "prompt" in x:
#                     prompt_str = x["prompt"]
#                 elif "instruction" in x:
#                     prompt_str = x["instruction"]
#                 else:
#                     raise ValueError(x)
#             if "answer" in x:
#                 for ans in x["answer"]:
#                     if detokenizer:
#                         ans_str = detokenizer.decode(
#                             detokenizer.encode(ans), skip_special_tokens=True
#                         )
#                     else:
#                         ans_str = ans
#                     prompts.append(prompt_str)
#                     answers.append(ans_str)
#             elif "output" in x:
#                 ans_str = x["output"]
#                 prompts.append(prompt_str)
#                 answers.append(ans_str)
#             else:
#                 raise ValueError(x)

#         chat = []
#         for i in range(len(prompts)):
#             chat.append(
#                 [
#                     {"role": "user", "content": prompts[i]},
#                     {"role": "assistant", "content": answers[i]},
#                 ]
#             )
#         inputs = tokenizer.apply_chat_template(
#             chat,
#             padding="longest",
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(device)

#         with torch.no_grad():
#             if "FsfairX-LLaMA3-RM-v0.1" in args.model_name_or_path:
#                 outputs = model.forward_value(**inputs)["chosen_end_scores"]
#             else:
#                 outputs = model(**inputs, use_cahe=False)

#         c_start = 0
#         for x in response_data[start:end]:
#             if "answer" in x:
#                 x["reward"] = outputs[c_start : c_start + len(x["answer"])].tolist()
#                 c_start += len(x["answer"])
#             elif "output" in x:
#                 x["reward"] = outputs[c_start].tolist()
#                 c_start += 1
#             else:
#                 raise ValueError(x)

#         print(chat[0])
#         print(outputs[0])

#     if "answer" in x:
#         calculation_best_of_n(response_data)

#     json.dump(response_data, open(args.save_path, "w"), indent=2)
#     print("saving result to {}".format(args.save_path))


# if __name__ == "__main__":
#     main()



# import os
# import json
# import numpy as np
# from dataclasses import dataclass, field
# from pprint import pprint
# from types import MethodType
# from typing import List, Dict, Any

# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     HfArgumentParser,
# )

# @dataclass
# class Arguments:
#     model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     tokenizer_path: str = field(default=None)
#     detokenizer_path: str = field(default=None)

#     data_path: str = field(default=None)
#     batch_size: int = field(default=1)
#     max_size: int = field(default=None)

#     save_path: str = field(default=None)


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
#     hidden_states = transformer_outputs[0]
#     values = self.score(hidden_states).squeeze(-1)  # [B, T]
#     if return_value_only:
#         return values
#     else:
#         if attention_mask is None:
#             chosen_end_scores = values[:, -1]
#         else:
#             # index of last non-padding token
#             last_index = attention_mask.sum(dim=1) - 1
#             last_index = last_index.clamp(min=0).long()
#             chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {"values": values, "chosen_end_scores": chosen_end_scores}


# def _normalize_answers_field(x: Dict[str, Any]) -> List[str]:
#     """Return a list of answers for an item, supporting 'answers' (list), 'answer' (list/str), or 'output' (str)."""
#     if "answers" in x and isinstance(x["answers"], list):
#         return x["answers"]
#     if "answer" in x:
#         if isinstance(x["answer"], list):
#             return x["answer"]
#         if isinstance(x["answer"], str):
#             return [x["answer"]]
#     if "output" in x and isinstance(x["output"], str):
#         return [x["output"]]
#     raise ValueError(f"Cannot find answers in item keys={list(x.keys())}")


# def calculation_best_of_n(data: List[Dict[str, Any]]):
#     """Compute Best-of-n and Mean-of-n for n in {1,2,4,8,16,32} (safely handles shorter lists)."""
#     print("Calculating best of n reward ....")
#     buckets = [1, 2, 4, 8, 16, 32]
#     best_mat = np.zeros([len(data), len(buckets)], dtype=np.float32)
#     mean_mat = np.zeros_like(best_mat)
#     for i, item in enumerate(data):
#         rewards = item.get("reward", [])
#         # ensure rewards is a list (could be scalar if single answer)
#         if not isinstance(rewards, list):
#             rewards = [rewards]
#         for j, n in enumerate(buckets):
#             r_slice = rewards[:n] if len(rewards) >= 1 else []
#             if len(r_slice) == 0:
#                 best_mat[i, j] = np.nan
#                 mean_mat[i, j] = np.nan
#             else:
#                 best_mat[i, j] = np.max(r_slice)
#                 mean_mat[i, j] = np.mean(r_slice)
#     # ignore rows where NaN occurred (e.g., empty prompts)
#     best_mean = np.nanmean(best_mat, axis=0)
#     mean_mean = np.nanmean(mean_mat, axis=0)

#     # print in the same style
#     print("Best of n: {}".format(np.round(best_mean, 2)))
#     print("Mean of n: {}".format(np.round(mean_mean, 2)))
#     return best_mean, mean_mean


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(vars(args))

#     assert args.data_path is not None, "--data_path is required"
#     assert args.save_path is not None, "--save_path is required"

#     # If results already exist, just summarize and exit (repo behavior)
#     if os.path.exists(args.save_path):
#         response_data = json.load(open(args.save_path, "r", encoding="utf-8"))
#         calculation_best_of_n(response_data)
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Reward model & tokenizer
#     rm = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
#         attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
#         trust_remote_code=True,
#     ).to(device).eval()

#     # Some RMs (e.g., FsfairX-LLaMA3-RM) use a custom forward_value; patch if needed
#     rm.forward_value = MethodType(forward_value_fn, rm)

#     rm_tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_name_or_path, trust_remote_code=True)
#     rm_tok.padding_side = "right"

#     detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None

#     # Load generations
#     response_data = json.load(open(args.data_path, "r", encoding="utf-8"))
#     if args.max_size:
#         response_data = response_data[: args.max_size]

#     # Build flat prompt/answer pairs for batched scoring
#     flat_prompts: List[str] = []
#     flat_answers: List[str] = []
#     per_item_counts: List[int] = []  # how many answers per item

#     for x in response_data:
#         # choose prompt field (repo uses "prompt" or "instruction")
#         if "prompt" in x:
#             prompt_str = x["prompt"]
#         elif "instruction" in x:
#             prompt_str = x["instruction"]
#         else:
#             raise ValueError(f"Missing prompt/instruction in item keys={list(x.keys())}")

#         # detokenize to strip template/specials if requested
#         if detok is not None:
#             prompt_str = detok.decode(detok.encode(prompt_str), skip_special_tokens=True)
#             prompt_str = prompt_str.replace("user\n\n", "").replace("assistant\n\n", "")

#         answers = _normalize_answers_field(x)
#         if detok is not None:
#             answers = [detok.decode(detok.encode(a), skip_special_tokens=True) for a in answers]

#         per_item_counts.append(len(answers))
#         for a in answers:
#             flat_prompts.append(prompt_str)
#             flat_answers.append(a)

#     # Convert to chat format for RM scoring (no generation prompt)
#     chats = [
#         [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
#         for p, a in zip(flat_prompts, flat_answers)
#     ]
#     enc = rm_tok.apply_chat_template(
#         chats,
#         padding="longest",
#         add_generation_prompt=False,  # IMPORTANT: we are scoring, not generating
#         return_tensors="pt",
#         return_dict=True,
#     )
#     enc = {k: v.to(device) for k, v in enc.items()}

#     # Run the reward model
#     rewards_all: List[float] = []
#     with torch.no_grad():
#         # If the RM exposes forward_value (patched above), use it to get end-of-seq score
#         try:
#             out = rm.forward_value(**enc)["chosen_end_scores"]
#             rewards_all = out.detach().float().cpu().tolist()
#         except Exception:
#             # Fallback: standard SequenceClassification logits
#             out = rm(**enc, use_cache=False)  # correct kwarg
#             logits = out.logits.squeeze(-1)
#             rewards_all = logits.detach().float().cpu().tolist()

#     # Stitch rewards back per item
#     idx = 0
#     for x, count in zip(response_data, per_item_counts):
#         if count == 1:
#             x["reward"] = float(rewards_all[idx])
#             idx += 1
#         else:
#             x["reward"] = [float(r) for r in rewards_all[idx: idx + count]]
#             idx += count

#     # Print Best-of-n / Mean-of-n summary like the repo
#     calculation_best_of_n(response_data)

#     # Save augmented results
#     os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
#     with open(args.save_path, "w", encoding="utf-8") as f:
#         json.dump(response_data, f, ensure_ascii=False, indent=2)
#     print(f"Saving result to {args.save_path}")


# if __name__ == "__main__":
#     main()





# import os
# from dataclasses import dataclass, field
# from pprint import pprint
# import json
# from types import MethodType
# from tqdm import tqdm
# import numpy as np

# import torch
# from transformers import AutoTokenizer, pipeline
# from transformers import AutoModel, AutoModelForSequenceClassification, HfArgumentParser


# @dataclass
# class Arguments:
#     model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
#     tokenizer_path: str = field(default=None)

#     detokenizer_path: str = field(default=None)

#     data_path: str = field(default=None)
#     batch_size: int = field(default=1)
#     max_size: int = field(default=None)

#     save_path: str = field(default=None)


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
#     hidden_states = transformer_outputs[0]
#     values = self.score(hidden_states).squeeze(-1)
#     if return_value_only:
#         return values
#     else:
#         if attention_mask is None:
#             chosen_end_scores = values[:, -1]
#         else:
#             last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
#             chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
#     return {
#         "values": values,
#         "chosen_end_scores": chosen_end_scores,
#     }


# def calculation_best_of_n(data):
#     print("Calculating best of n reward ....")
#     best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
#     mean_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
#     for i in tqdm(range(len(data))):
#         rewards = data[i]["reward"]
#         best_n[i][0] = rewards[0]
#         best_n[i][1] = max(rewards[:2])
#         best_n[i][2] = max(rewards[:4])
#         best_n[i][3] = max(rewards[:8])
#         best_n[i][4] = max(rewards[:16])
#         best_n[i][5] = max(rewards[:32])

#         mean_n[i][0] = rewards[0]
#         mean_n[i][1] = np.mean(rewards[:2])
#         mean_n[i][2] = np.mean(rewards[:4])
#         mean_n[i][3] = np.mean(rewards[:8])
#         mean_n[i][4] = np.mean(rewards[:16])
#         mean_n[i][5] = np.mean(rewards[:32])
#     best_n = np.mean(best_n, axis=0)
#     print("Best of n: {}".format(np.round(best_n, 2)))
#     mean_n = np.mean(mean_n, axis=0)
#     print("Mean of n: {}".format(np.round(mean_n, 2)))
#     return best_n, mean_n


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)
#     assert args.data_path is not None
#     assert args.save_path is not None

#     device = torch.device("cuda")

#     model_class = AutoModelForSequenceClassification
#     flash_attn = True
#     model = model_class.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2" if flash_attn else "eager",
#         trust_remote_code=True,
#     )
#     # model.forward_value = forward_value_fn
#     model.forward_value = MethodType(forward_value_fn, model)
#     model.eval()
#     model.to(device)

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_path or args.model_name_or_path
#     )
#     tokenizer.padding_side = "right"
#     if args.detokenizer_path is not None:
#         detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
#     else:
#         detokenizer = None

#     response_data = json.load(open(args.data_path, "r"))

#     if args.max_size:
#         response_data = response_data[: args.max_size]
#     if os.path.exists(args.save_path):
#         response_data = json.load(open(args.save_path, "r"))
#         calculation_best_of_n(response_data)
#         return
#     for start in tqdm(range(0, len(response_data), args.batch_size)):
#         end = start + args.batch_size
#         prompts = []
#         answers = []
#         for x in response_data[start:end]:
#             if detokenizer:
#                 prompt_str = (
#                     detokenizer.decode(
#                         detokenizer.encode(x["prompt"]), skip_special_tokens=True
#                     )
#                     .replace("user\n\n", "")
#                     .replace("assistant\n\n", "")
#                 )
#             else:
#                 if "prompt" in x:
#                     prompt_str = x["prompt"]
#                 elif "instruction" in x:
#                     prompt_str = x["instruction"]
#                 else:
#                     raise ValueError(x)
#             if "answer" in x:
#                 for ans in x["answer"]:
#                     if detokenizer:
#                         ans_str = detokenizer.decode(
#                             detokenizer.encode(ans), skip_special_tokens=True
#                         )
#                     else:
#                         ans_str = ans
#                     prompts.append(prompt_str)
#                     answers.append(ans_str)
#             elif "output" in x:
#                 ans_str = x["output"]
#                 prompts.append(prompt_str)
#                 answers.append(ans_str)
#             else:
#                 raise ValueError(x)

#         chat = []
#         for i in range(len(prompts)):
#             chat.append(
#                 [
#                     {"role": "user", "content": prompts[i]},
#                     {"role": "assistant", "content": answers[i]},
#                 ]
#             )
#         inputs = tokenizer.apply_chat_template(
#             chat,
#             padding="longest",
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(device)

#         with torch.no_grad():
#             if "FsfairX-LLaMA3-RM-v0.1" in args.model_name_or_path:
#                 outputs = model.forward_value(**inputs)["chosen_end_scores"]
#             else:
#                 outputs = model(**inputs, use_cahe=False)

#         c_start = 0
#         for x in response_data[start:end]:
#             if "answer" in x:
#                 x["reward"] = outputs[c_start : c_start + len(x["answer"])].tolist()
#                 c_start += len(x["answer"])
#             elif "output" in x:
#                 x["reward"] = outputs[c_start].tolist()
#                 c_start += 1
#             else:
#                 raise ValueError(x)

#         print(chat[0])
#         print(outputs[0])

#     if "answer" in x:
#         calculation_best_of_n(response_data)

#     json.dump(response_data, open(args.save_path, "w"), indent=2)
#     print("saving result to {}".format(args.save_path))


# if __name__ == "__main__":
#     main()


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
    model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    tokenizer_path: str = field(default=None)
    detokenizer_path: str = field(default=None)
    data_path: str = field(default=None)
    batch_size: int = field(default=2)
    max_size: int = field(default=None)
    save_path: str = field(default=None)  # per-item rewards json
    summary_path: str = field(default=None)  # optional; if None, auto derive

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
    else:
        if attention_mask is None:
            chosen_end_scores = values[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
    return {"values": values, "chosen_end_scores": chosen_end_scores}

def _n_grid(n_max):
    # cap at available n; standard grid up to 32
    grid = [1, 2, 4, 8, 16, 32]
    return [n for n in grid if n <= n_max]

def calculation_best_of_n(data):
    print("Calculating best of n / mean of n ...")
    # detect available n (answers length of first example that has it)
    n_max = 1
    for x in data:
        if isinstance(x.get("answer"), list) and x["answer"]:
            n_max = max(n_max, len(x["answer"]))
    grid = _n_grid(n_max)

    best_n = np.zeros([len(data), len(grid)], dtype=float)
    mean_n = np.zeros([len(data), len(grid)], dtype=float)

    for i in tqdm(range(len(data))):
        rewards = data[i].get("reward", [])
        if not rewards:
            continue
        for gi, n in enumerate(grid):
            m = min(n, len(rewards))
            best_n[i, gi] = np.max(rewards[:m])
            mean_n[i, gi] = np.mean(rewards[:m])

    best_n = np.mean(best_n, axis=0).tolist()
    mean_n = np.mean(mean_n, axis=0).tolist()
    print("Best of n:", np.round(best_n, 2))
    print("Mean of n:", np.round(mean_n, 2))
    return grid, best_n, mean_n

def calculate_winrate(data):
    print("Calculating winrate (>0) best-of-n ...")
    n_max = 1
    for x in data:
        if isinstance(x.get("answer"), list) and x["answer"]:
            n_max = max(n_max, len(x["answer"]))
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
            wr[gi] += 1.0 if np.max(rewards[:m]) > 0 else 0.0
    if total > 0:
        wr = (wr / total * 100.0).tolist()
    else:
        wr = wr.tolist()
    print("Winrate (%), best-of-n:", np.round(wr, 2))
    return grid, wr

def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)
    assert args.data_path is not None
    assert args.save_path is not None

    if args.summary_path is None:
        args.summary_path = os.path.join(os.path.dirname(args.save_path), "reward_summary.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        trust_remote_code=True,
    )
    # Attach forward_value if model supports it (FsfairX RM)
    model.forward_value = MethodType(forward_value_fn, model)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_name_or_path, use_fast=True, trust_remote_code=True)
    detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None
    response_data = json.load(open(args.data_path, "r"))

    if args.max_size:
        response_data = response_data[: args.max_size]

    # If file exists, recompute summaries from it (overwrite summary), but keep per-item rewards
    if os.path.exists(args.save_path):
        print(f"{args.save_path} exists; loading and recomputing summaries...")
        response_data = json.load(open(args.save_path, "r"))
        grid, best_n, mean_n = calculation_best_of_n(response_data)
        _, wr = calculate_winrate(response_data)
        json.dump(
            {"n_grid": grid, "best_of_n": best_n, "mean_of_n": mean_n, "winrate_gt0_of_n": wr, "num_items": len(response_data)},
            open(args.summary_path, "w"),
            indent=2,
        )
        print(f"Saved summary to {args.summary_path}")
        return

    # Compute rewards
    for start in tqdm(range(0, len(response_data), args.batch_size)):
        end = min(start + args.batch_size, len(response_data))
        prompts, answers = [], []
        for x in response_data[start:end]:
            # pick prompt text
            if detok:
                ptxt = detok.decode(detok.encode(x.get("prompt") or x.get("instruction", "")), skip_special_tokens=True)
                ptxt = ptxt.replace("user\n\n", "").replace("assistant\n\n", "")
            else:
                ptxt = x.get("prompt") or x.get("instruction", "")
            # gather answers
            if "answer" in x and isinstance(x["answer"], list):
                for a in x["answer"]:
                    answers.append(a)
                    prompts.append(ptxt)
            elif "output" in x and isinstance(x["output"], str):
                answers.append(x["output"])
                prompts.append(ptxt)
            else:
                # skip empty rows safely
                continue

        # Build chat template batch
        chats = []
        for p, a in zip(prompts, answers):
            chats.append([{"role": "user", "content": p}, {"role": "assistant", "content": a}])
        if not chats:
            continue

        inputs = tokenizer.apply_chat_template(
            chats,
            padding="longest",
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Prefer chosen_end_scores if model has forward_value
            try:
                outputs = model.forward_value(**inputs)["chosen_end_scores"]
            except Exception:
                out = model(**inputs, use_cache=False)
                outputs = out.logits.squeeze(-1)

        # write rewards back
        c = 0
        for x in response_data[start:end]:
            if "answer" in x and isinstance(x["answer"], list) and x["answer"]:
                k = len(x["answer"])
                x["reward"] = outputs[c : c + k].tolist()
                c += k
            elif "output" in x and isinstance(x["output"], str):
                x["reward"] = [outputs[c].item()]
                c += 1

    # Summaries
    grid, best_n, mean_n = calculation_best_of_n(response_data)
    _, wr = calculate_winrate(response_data)

    # Save per-item rewards and summary
    json.dump(response_data, open(args.save_path, "w"), indent=2)
    json.dump(
        {"n_grid": grid, "best_of_n": best_n, "mean_of_n": mean_n, "winrate_gt0_of_n": wr, "num_items": len(response_data)},
        open(args.summary_path, "w"),
        indent=2,
    )
    print(f"Saved per-item to {args.save_path}")
    print(f"Saved summary to {args.summary_path}")

if __name__ == "__main__":
    main()
