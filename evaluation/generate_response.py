# #!/usr/bin/env python3
# import os
# import json
# from pprint import pprint
# from dataclasses import dataclass, field
# from typing import List, Any

# # vLLM optional
# try:
#     import vllm  # type: ignore
#     from vllm import SamplingParams  # type: ignore
# except Exception:
#     vllm = None
#     SamplingParams = None

# import torch
# from tqdm import tqdm
# import pandas as pd
# from datasets import load_dataset, load_from_disk, Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, HfArgumentParser


# @dataclass
# class Arguments:
#     # model / tokenizer
#     model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat")
#     tokenizer_path: str = field(default="meta-llama/Llama-2-7b-chat")

#     # dataset
#     dataset_path: str = field(default="tatsu-lab/alpaca_eval")
#     split: str = field(default=None)
#     column_name: str = field(default=None)
#     standard_format: bool = field(default=None)
#     load_from_disk: bool = field(default=False)
#     max_size: int = field(default=None)

#     # vLLM
#     use_vllm: bool = field(default=False)
#     vllm_gpu_memory_utilization: float = field(default=0.9)

#     # generation
#     seed: int = field(default=42)
#     batch_size: int = field(default=64)
#     n: int = field(default=1)
#     # NEW: when NOT using vLLM, sample this many responses at a time to avoid OOM
#     hf_n_chunk: int = field(default=4)
#     do_sample: bool = field(default=True)
#     top_k: int = field(default=50)
#     top_p: float = field(default=0.9)
#     temperature: float = field(default=0.6)
#     max_new_tokens: int = field(default=1024)

#     # save
#     remove_old: bool = field(default=False)
#     save_path: str = field(default="responses.json")

#     def __post_init__(self):
#         # column defaults
#         if self.column_name is None:
#             if "tatsu-lab/alpaca_eval" in self.dataset_path:
#                 self.column_name = "instruction"
#             elif "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
#                 self.column_name = "prompt"
#             elif "if_eval" in self.dataset_path:
#                 self.column_name = "prompt"
#             else:
#                 self.column_name = "instruction"
#         # split defaults
#         if self.split is None:
#             if "tatsu-lab/alpaca_eval" in self.dataset_path:
#                 self.split = "eval"
#             elif "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
#                 self.split = "test_sft"
#             elif "if_eval" in self.dataset_path:
#                 self.split = "train"
#             else:
#                 self.split = "test"
#         # format default
#         if self.standard_format is None:
#             self.standard_format = False


# def _read_local_json_or_jsonl(path: str) -> List[Any]:
#     if path.endswith(".json"):
#         return json.load(open(path, "r", encoding="utf-8"))
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))
#     return rows


# def get_dataset(dataset_name: str, split: str = "test", from_disk: bool = False) -> Dataset:
#     """Load HF or local datasets. Supports directories with {split}.jsonl or {split}.json."""
#     if from_disk:
#         return load_from_disk(dataset_name)

#     # Local directory with json/jsonl files
#     if os.path.isdir(dataset_name):
#         # prefer split file if present
#         for ext in ("jsonl", "json"):
#             cand = os.path.join(dataset_name, f"{split}.{ext}")
#             if os.path.exists(cand):
#                 rows = _read_local_json_or_jsonl(cand)
#                 return Dataset.from_pandas(pd.DataFrame(rows))
#         # otherwise merge all json/jsonl
#         files = [
#             os.path.join(dataset_name, f)
#             for f in os.listdir(dataset_name)
#             if f.endswith((".jsonl", ".json"))
#         ]
#         if files:
#             rows = []
#             for p in files:
#                 rows.extend(_read_local_json_or_jsonl(p))
#             return Dataset.from_pandas(pd.DataFrame(rows))
#         raise FileNotFoundError(f"No split files under {dataset_name} (expected {split}.json[l]).")

#     # Single local file
#     if dataset_name.endswith((".jsonl", ".json")) and os.path.exists(dataset_name):
#         rows = _read_local_json_or_jsonl(dataset_name)
#         return Dataset.from_pandas(pd.DataFrame(rows))

#     # HF datasets
#     if "tatsu-lab/alpaca_eval" in dataset_name:
#         # IMPORTANT: this dataset uses a custom loader
#         ds = load_dataset(dataset_name, "alpaca_eval", trust_remote_code=True)
#     else:
#         ds = load_dataset(dataset_name)

#     # Split handling
#     if split in ds:
#         return ds[split]
#     assert "train" in ds, f"Split '{split}' not found and no 'train' split."
#     total = len(ds["train"])
#     eval_sz = min(1000, int(total * 0.1))
#     return ds["train"].shuffle(seed=42).select(range(total - eval_sz, total))


# def _safe_load_tokenizer(path: str):
#     """Try regular load; on failure, retry with trust_remote_code=True."""
#     try:
#         return AutoTokenizer.from_pretrained(path, use_fast=True)
#     except Exception:
#         return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


# def _build_hf_model(tok, model_name):
#     """
#     Build an HF model in a memory-safe way:
#     - multi-GPU: device_map="auto"
#     - single GPU/CPU: explicit .to(device)
#     """
#     common = dict(
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         trust_remote_code=True,
#     )
#     multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
#     if multi_gpu:
#         try:
#             m = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 attn_implementation="flash_attention_2",
#                 device_map="auto",  # shard across GPUs
#                 **common,
#             )
#         except Exception:
#             m = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 attn_implementation="eager",
#                 device_map="auto",
#                 **common,
#             )
#         # If we added a pad token, resize embeddings
#         try:
#             if m.get_input_embeddings().num_embeddings < len(tok):
#                 m.resize_token_embeddings(len(tok))
#         except Exception:
#             pass
#         m.eval()
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         try:
#             m = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 attn_implementation="flash_attention_2",
#                 **common,
#             )
#         except Exception:
#             m = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 attn_implementation="eager",
#                 **common,
#             )
#         try:
#             if m.get_input_embeddings().num_embeddings < len(tok):
#                 m.resize_token_embeddings(len(tok))
#         except Exception:
#             pass
#         m.to(device).eval()
#     return m


# def save_prompts_and_answers(model_name: str, prompts: List[str], answers: List[List[str]], file_path: str):
#     assert len(prompts) == len(answers), "prompts and answers must match in length"
#     recs = [
#         {"id": i, "model_name": model_name, "prompt": prompts[i], "answer": answers[i]}
#         for i in range(len(prompts))
#     ]
#     if not os.path.exists(file_path):
#         json.dump(recs, open(file_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
#         return
#     data = json.load(open(file_path, "r", encoding="utf-8"))
#     next_id = (data[-1]["id"] + 1) if data else 0
#     for k, r in enumerate(recs):
#         r["id"] = next_id + k
#     data.extend(recs)
#     json.dump(data, open(file_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)
#     set_seed(args.seed)

#     # handle existing save path
#     if os.path.exists(args.save_path):
#         if args.remove_old:
#             os.remove(args.save_path)
#         else:
#             print(f"{args.save_path} exists. Exit.")
#             return

#     # load data
#     ds = get_dataset(args.dataset_path, args.split, args.load_from_disk)
#     if args.max_size:
#         ds = ds.select(range(0, min(len(ds), args.max_size)))

#     # tokenizer
#     tok = _safe_load_tokenizer(args.tokenizer_path)

#     # ensure a pad token exists (left padding for batched generation)
#     if tok.pad_token_id is None:
#         try:
#             tok.pad_token = tok.eos_token
#             tok.pad_token_id = tok.eos_token_id
#         except Exception:
#             tok.add_special_tokens({"pad_token": "<|pad|>"})
#             tok.pad_token = "<|pad|>"
#             tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
#     tok.padding_side = "left"

#     # vLLM availability
#     if args.use_vllm and vllm is None:
#         print("[warn] --use_vllm True but vllm not importable; falling back to HF.")
#         args.use_vllm = False

#     # model init
#     llm = None
#     sparams = None
#     model = None
#     if args.use_vllm:
#         try:
#             llm = vllm.LLM(
#                 model=args.model_name_or_path,
#                 tokenizer=args.tokenizer_path,
#                 tensor_parallel_size=max(1, torch.cuda.device_count()),
#                 dtype="bfloat16",
#                 gpu_memory_utilization=args.vllm_gpu_memory_utilization,
#                 seed=args.seed,
#                 swap_space=16,
#                 trust_remote_code=True,
#             )
#             sparams = SamplingParams(
#                 n=args.n,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 top_k=args.top_k,
#                 max_tokens=args.max_new_tokens,
#             )
#             print("[info] Using vLLM backend.")
#         except Exception as e:
#             print(f"[warn] vLLM init failed: {e}\n[warn] Falling back to HF Transformers.")
#             args.use_vllm = False
#             model = _build_hf_model(tok, args.model_name_or_path)
#     else:
#         model = _build_hf_model(tok, args.model_name_or_path)
#         print("[info] Using HF Transformers backend.")

#     # buffers
#     p_buf: List[str] = []
#     a_buf: List[List[str]] = []
#     batches = 0

#     # generation loop
#     for i in tqdm(range(0, len(ds), args.batch_size)):
#         if args.standard_format:
#             # ds[column_name] is a list of conversations; save a templated string as prompt
#             chosen = ds[i : i + args.batch_size][args.column_name]
#             prompt_conv = [x[:-1] for x in chosen]  # drop final assistant turn if present
#             prompt_strs = tok.apply_chat_template(
#                 prompt_conv, tokenize=False, add_generation_prompt=True
#             )
#             saved_prompts = prompt_strs  # templated prompt text
#         else:
#             # standard string column (e.g., AlpacaEval "instruction")
#             prompts = ds[i : i + args.batch_size][args.column_name]
#             prompt_conv = [[{"role": "user", "content": x}] for x in prompts]
#             # build the chat template for the model input,
#             # but SAVE the RAW instruction like GEM
#             _ = tok.apply_chat_template(
#                 prompt_conv, tokenize=False, add_generation_prompt=True
#             )
#             saved_prompts = prompts

#         # tokenization (keep on CPU for now; move later as needed)
#         enc = tok.apply_chat_template(
#             prompt_conv,
#             padding="longest",
#             add_generation_prompt=True,
#             return_tensors="pt",
#             return_dict=True,
#         )
#         plen = enc.input_ids.size(-1)
#         B = enc.input_ids.size(0)

#         if args.use_vllm:
#             # vLLM takes per-example trimmed input ids (keep them on CPU)
#             ids = []
#             for j in range(B):
#                 mask = enc.attention_mask[j].bool()
#                 ids.append(enc.input_ids[j, mask].tolist())
#             try:
#                 with torch.no_grad():
#                     outs = llm.generate(prompt_token_ids=ids, sampling_params=sparams)
#                 batch_answers = [[o.outputs[k].text for k in range(len(o.outputs))] for o in outs]
#             except Exception as e:
#                 print(f"[warn] vLLM generate failed: {e}\n[warn] Switching to HF for the rest of the run.")
#                 args.use_vllm = False
#                 if model is None:
#                     model = _build_hf_model(tok, args.model_name_or_path)
#                 # fall through to HF path for this batch
#                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#                 enc = {k: v.to(device) for k, v in enc.items()}
#                 batch_answers = [[] for _ in range(B)]
#                 remaining = args.n
#                 while remaining > 0:
#                     k = min(args.hf_n_chunk, remaining)
#                     rep_ids  = enc["input_ids"].repeat_interleave(k, dim=0)
#                     rep_attn = enc["attention_mask"].repeat_interleave(k, dim=0)
#                     with torch.no_grad():
#                         gen = model.generate(
#                             rep_ids,
#                             attention_mask=rep_attn,
#                             do_sample=args.do_sample,
#                             top_k=args.top_k,
#                             top_p=args.top_p,
#                             temperature=args.temperature,
#                             max_new_tokens=args.max_new_tokens,
#                             pad_token_id=tok.pad_token_id,
#                             num_return_sequences=1,
#                         )
#                     dec = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
#                     for j in range(B):
#                         batch_answers[j].extend(dec[j*k : (j+1)*k])
#                     remaining -= k
#         else:
#             # HF path (move inputs once)
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             enc = {k: v.to(device) for k, v in enc.items()}
#             # chunk 'n' to avoid OOM
#             batch_answers = [[] for _ in range(B)]
#             remaining = args.n
#             while remaining > 0:
#                 k = min(args.hf_n_chunk, remaining)
#                 rep_ids  = enc["input_ids"].repeat_interleave(k, dim=0)
#                 rep_attn = enc["attention_mask"].repeat_interleave(k, dim=0)
#                 with torch.no_grad():
#                     gen = model.generate(
#                         rep_ids,
#                         attention_mask=rep_attn,
#                         do_sample=args.do_sample,
#                         top_k=args.top_k,
#                         top_p=args.top_p,
#                         temperature=args.temperature,
#                         max_new_tokens=args.max_new_tokens,
#                         pad_token_id=tok.pad_token_id,
#                         num_return_sequences=1,  # we already repeated inputs
#                     )
#                 dec = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
#                 for j in range(B):
#                     batch_answers[j].extend(dec[j*k : (j+1)*k])
#                 remaining -= k

#         # buffer & periodic save
#         p_buf.extend(saved_prompts)
#         a_buf.extend(batch_answers)
#         batches += 1
#         if batches % 10 == 0:
#             save_prompts_and_answers(args.model_name_or_path, p_buf, a_buf, args.save_path)
#             p_buf.clear()
#             a_buf.clear()

#     # final flush
#     if p_buf:
#         save_prompts_and_answers(args.model_name_or_path, p_buf, a_buf, args.save_path)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import json
from pprint import pprint
from dataclasses import dataclass, field
from typing import List, Any

# vLLM optional
try:
    import vllm  # type: ignore
    from vllm import SamplingParams  # type: ignore
except Exception:
    vllm = None
    SamplingParams = None

import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, HfArgumentParser


@dataclass
class Arguments:
    # model / tokenizer
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat")
    tokenizer_path: str = field(default="meta-llama/Llama-2-7b-chat")

    # dataset
    dataset_path: str = field(default="tatsu-lab/alpaca_eval")
    split: str = field(default=None)
    column_name: str = field(default=None)
    standard_format: bool = field(default=None)
    load_from_disk: bool = field(default=False)
    max_size: int = field(default=None)

    # vLLM
    use_vllm: bool = field(default=False)
    vllm_gpu_memory_utilization: float = field(default=0.9)

    # generation
    seed: int = field(default=42)
    batch_size: int = field(default=64)
    n: int = field(default=1)
    # when NOT using vLLM, sample this many responses at a time to avoid OOM
    hf_n_chunk: int = field(default=4)
    do_sample: bool = field(default=True)
    top_k: int = field(default=50)
    top_p: float = field(default=0.9)
    temperature: float = field(default=0.6)
    max_new_tokens: int = field(default=1024)

    # save
    remove_old: bool = field(default=False)
    save_path: str = field(default="responses.json")

    def __post_init__(self):
        # column defaults
        if self.column_name is None:
            if "tatsu-lab/alpaca_eval" in self.dataset_path:
                self.column_name = "instruction"
            elif "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
                self.column_name = "prompt"
            elif "if_eval" in self.dataset_path:
                self.column_name = "prompt"
            else:
                self.column_name = "instruction"
        # split defaults
        if self.split is None:
            if "tatsu-lab/alpaca_eval" in self.dataset_path:
                self.split = "eval"
            elif "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
                self.split = "test_sft"
            elif "if_eval" in self.dataset_path:
                self.split = "train"
            else:
                self.split = "test"
        # format default
        if self.standard_format is None:
            self.standard_format = False


def _read_local_json_or_jsonl(path: str) -> List[Any]:
    if path.endswith(".json"):
        return json.load(open(path, "r", encoding="utf-8"))
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_dataset(dataset_name: str, split: str = "test", from_disk: bool = False) -> Dataset:
    """Load HF or local datasets. Supports directories with {split}.jsonl or {split}.json."""
    if from_disk:
        return load_from_disk(dataset_name)

    # Local directory with json/jsonl files
    if os.path.isdir(dataset_name):
        # prefer split file if present
        for ext in ("jsonl", "json"):
            cand = os.path.join(dataset_name, f"{split}.{ext}")
            if os.path.exists(cand):
                rows = _read_local_json_or_jsonl(cand)
                return Dataset.from_pandas(pd.DataFrame(rows))
        # otherwise merge all json/jsonl
        files = [
            os.path.join(dataset_name, f)
            for f in os.listdir(dataset_name)
            if f.endswith((".jsonl", ".json"))
        ]
        if files:
            rows = []
            for p in files:
                rows.extend(_read_local_json_or_jsonl(p))
            return Dataset.from_pandas(pd.DataFrame(rows))
        raise FileNotFoundError(f"No split files under {dataset_name} (expected {split}.json[l]).")

    # Single local file
    if dataset_name.endswith((".jsonl", ".json")) and os.path.exists(dataset_name):
        rows = _read_local_json_or_jsonl(dataset_name)
        return Dataset.from_pandas(pd.DataFrame(rows))

    # HF datasets
    if "tatsu-lab/alpaca_eval" in dataset_name:
        # this dataset uses a custom loader
        ds = load_dataset(dataset_name, "alpaca_eval", trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name)

    # Split handling
    if split in ds:
        return ds[split]
    assert "train" in ds, f"Split '{split}' not found and no 'train' split."
    total = len(ds["train"])
    eval_sz = min(1000, int(total * 0.1))
    return ds["train"].shuffle(seed=42).select(range(total - eval_sz, total))


def _safe_load_tokenizer(path: str):
    """Try regular load; on failure, retry with trust_remote_code=True."""
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


def _build_hf_model(tok, model_name):
    """
    Build an HF model in a memory-safe way:
    - multi-GPU: device_map="auto"
    - single GPU/CPU: explicit .to(device)
    """
    common = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if multi_gpu:
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                device_map="auto",  # shard across GPUs
                **common,
            )
        except Exception:
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                device_map="auto",
                **common,
            )
        try:
            if m.get_input_embeddings().num_embeddings < len(tok):
                m.resize_token_embeddings(len(tok))
        except Exception:
            pass
        m.eval()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **common,
            )
        except Exception:
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                **common,
            )
        try:
            if m.get_input_embeddings().num_embeddings < len(tok):
                m.resize_token_embeddings(len(tok))
        except Exception:
            pass
        m.to(device).eval()
    return m


def save_prompts_and_answers(model_name: str, prompts: List[str], answers: List[List[str]], file_path: str):
    assert len(prompts) == len(answers), "prompts and answers must match in length"
    recs = [
        {"id": i, "model_name": model_name, "prompt": prompts[i], "answer": answers[i]}
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        json.dump(recs, open(file_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return
    data = json.load(open(file_path, "r", encoding="utf-8"))
    next_id = (data[-1]["id"] + 1) if data else 0
    for k, r in enumerate(recs):
        r["id"] = next_id + k
    data.extend(recs)
    json.dump(data, open(file_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)
    set_seed(args.seed)

    # handle existing save path
    if os.path.exists(args.save_path):
        if args.remove_old:
            os.remove(args.save_path)
        else:
            print(f"{args.save_path} exists. Exit.")
            return

    # load data
    ds = get_dataset(args.dataset_path, args.split, args.load_from_disk)
    if args.max_size:
        ds = ds.select(range(0, min(len(ds), args.max_size)))

    # tokenizer
    tok = _safe_load_tokenizer(args.tokenizer_path)

    # ensure a pad token exists (left padding for batched generation)
    if tok.pad_token_id is None:
        try:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        except Exception:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token = "<|pad|>"
            tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
    tok.padding_side = "left"

    # vLLM availability
    if args.use_vllm and vllm is None:
        print("[warn] --use_vllm True but vllm not importable; falling back to HF.")
        args.use_vllm = False

    # model init
    llm = None
    sparams = None
    model = None
    if args.use_vllm:
        try:
            llm = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_path,
                tensor_parallel_size=max(1, torch.cuda.device_count()),
                dtype="bfloat16",
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                seed=args.seed,
                swap_space=16,
                trust_remote_code=True,
            )
            sparams = SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_new_tokens,
            )
            print("[info] Using vLLM backend.")
        except Exception as e:
            print(f"[warn] vLLM init failed: {e}\n[warn] Falling back to HF Transformers.")
            args.use_vllm = False
            model = _build_hf_model(tok, args.model_name_or_path)
    else:
        model = _build_hf_model(tok, args.model_name_or_path)
        print("[info] Using HF Transformers backend.")

    # buffers
    p_buf: List[str] = []
    a_buf: List[List[str]] = []
    batches = 0

    # generation loop
    for i in tqdm(range(0, len(ds), args.batch_size)):
        if args.standard_format:
            # ds[column_name] is a list of conversations; save templated string as prompt (GEM-compatible)
            chosen = ds[i : i + args.batch_size][args.column_name]
            prompt_conv = [x[:-1] for x in chosen]  # drop final assistant turn if present
            prompt_strs = tok.apply_chat_template(
                prompt_conv, tokenize=False, add_generation_prompt=True
            )
            saved_prompts = prompt_strs
        else:
            # standard string column (e.g., AlpacaEval "instruction")
            prompts = ds[i : i + args.batch_size][args.column_name]
            prompt_conv = [[{"role": "user", "content": x}] for x in prompts]
            # Build the chat template for the model input,
            # and SAVE the templated prompt text (GEM-compatible)
            prompt_strs = tok.apply_chat_template(
                prompt_conv, tokenize=False, add_generation_prompt=True
            )
            saved_prompts = prompt_strs  # <-- CHANGED to match GEM

        # tokenization (keep on CPU for now; move later as needed)
        enc = tok.apply_chat_template(
            prompt_conv,
            padding="longest",
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        plen = enc.input_ids.size(-1)
        B = enc.input_ids.size(0)

        if args.use_vllm:
            # vLLM takes per-example trimmed input ids (keep them on CPU)
            ids = []
            for j in range(B):
                mask = enc.attention_mask[j].bool()
                ids.append(enc.input_ids[j, mask].tolist())
            try:
                with torch.no_grad():
                    outs = llm.generate(prompt_token_ids=ids, sampling_params=sparams)
                batch_answers = [[o.outputs[k].text for k in range(len(o.outputs))] for o in outs]
            except Exception as e:
                print(f"[warn] vLLM generate failed: {e}\n[warn] Switching to HF for the rest of the run.")
                args.use_vllm = False
                if model is None:
                    model = _build_hf_model(tok, args.model_name_or_path)
                # fall through to HF path for this batch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                enc = {k: v.to(device) for k, v in enc.items()}
                batch_answers = [[] for _ in range(B)]
                remaining = args.n
                while remaining > 0:
                    k = min(args.hf_n_chunk, remaining)
                    rep_ids  = enc["input_ids"].repeat_interleave(k, dim=0)
                    rep_attn = enc["attention_mask"].repeat_interleave(k, dim=0)
                    with torch.no_grad():
                        gen = model.generate(
                            rep_ids,
                            attention_mask=rep_attn,
                            do_sample=args.do_sample,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tok.pad_token_id,
                            num_return_sequences=1,
                        )
                    dec = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
                    for j in range(B):
                        batch_answers[j].extend(dec[j*k : (j+1)*k])
                    remaining -= k
        else:
            # HF path (move inputs once)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            enc = {k: v.to(device) for k, v in enc.items()}
            # chunk 'n' to avoid OOM
            batch_answers = [[] for _ in range(B)]
            remaining = args.n
            while remaining > 0:
                k = min(args.hf_n_chunk, remaining)
                rep_ids  = enc["input_ids"].repeat_interleave(k, dim=0)
                rep_attn = enc["attention_mask"].repeat_interleave(k, dim=0)
                with torch.no_grad():
                    gen = model.generate(
                        rep_ids,
                        attention_mask=rep_attn,
                        do_sample=args.do_sample,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tok.pad_token_id,
                        num_return_sequences=1,  # we already repeated inputs
                    )
                dec = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
                for j in range(B):
                    batch_answers[j].extend(dec[j*k : (j+1)*k])
                remaining -= k

        # buffer & periodic save
        p_buf.extend(saved_prompts)
        a_buf.extend(batch_answers)
        batches += 1
        if batches % 10 == 0:
            save_prompts_and_answers(args.model_name_or_path, p_buf, a_buf, args.save_path)
            p_buf.clear()
            a_buf.clear()

    # final flush
    if p_buf:
        save_prompts_and_answers(args.model_name_or_path, p_buf, a_buf, args.save_path)


if __name__ == "__main__":
    main()