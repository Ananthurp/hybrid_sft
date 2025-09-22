#!/usr/bin/env python
# coding: utf-8
import os, sys, json, math
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer
from tqdm import tqdm

# Optional import from your repo; we also include a fallback.
# (Works whether you run from LLMDiversity repo root or any cwd.)
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(ROOT)
    from evaluation.utils.gsm8k import extract_answer_number as _ext_ans  # type: ignore
    HAVE_UTIL = True
except Exception:
    HAVE_UTIL = False

def extract_answer_number_fallback(text: str):
    """
    Fallback extraction:
    - looks for the last integer/float in the text
    - or a pattern "The answer is: <number>"
    """
    import re
    m = re.search(r"The answer is:\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            v = m.group(1)
            return int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else float(v)
        except Exception:
            pass
    # last number anywhere
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        v = nums[-1]
        try:
            return int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else float(v)
        except Exception:
            return None
    return None

def extract_answer_number(text: str):
    if HAVE_UTIL:
        try:
            return _ext_ans(text)
        except Exception:
            return extract_answer_number_fallback(text)
    return extract_answer_number_fallback(text)

def majority_and_bestof(reference, candidates: List, depths=(1,4,8,16,32)) -> Tuple[List[int], List[int]]:
    from collections import Counter
    maj, best = [], []
    for d in depths:
        d = min(d, len(candidates))
        cur = candidates[:d]
        # majority vote
        most_common = Counter(cur).most_common(1)[0][0] if len(cur) else None
        maj.append(int(most_common == reference))
        # best-of-n (any correct)
        best.append(int(reference in cur))
    return maj, best

@dataclass
class Args:
    # model
    model_name_or_path: str = field(metadata={"help": "HF path to your fine-tuned model"})
    tokenizer_name_or_path: str = field(metadata={"help": "HF path to tokenizer (e.g., base Qwen2-7B)"})
    dtype: str = field(default="bf16", metadata={"choices":["bf16","fp16"]})

    # generation (multi-response)
    n: int = field(default=32, metadata={"help":"responses per question"})
    temperature: float = field(default=0.6)
    top_p: float = field(default=0.9)
    top_k: int = field(default=-1)
    max_new_tokens: int = field(default=512)

    # vLLM
    use_vllm: bool = field(default=True)
    vllm_gpu_memory_utilization: float = field(default=0.9)
    seed: int = field(default=42)

    # batching
    batch_size: int = field(default=16)

    # paths
    save_path: str = field(default="gsm8k_voting_dump.json", metadata={"help":"Per-sample dump (json)"})
    summary_path: str = field(default="gsm8k_voting_summary.json", metadata={"help":"Summary metrics (json)"})
    remove_old: bool = field(default=False)

def main():
    parser = HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if args.remove_old:
        for p in (args.save_path, args.summary_path):
            if os.path.exists(p):
                os.remove(p)

    # dataset: HF "gsm8k", config "main", split "test"
    ds = load_dataset("gsm8k", "main", split="test")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    # Pad-token safety for Qwen/Llama families
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token":"<|pad|>"})
    tok.padding_side = "left"

    # dtype
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    # vLLM
    if args.use_vllm:
        import vllm
        from vllm import SamplingParams
        llm = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch_dtype,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
        )
        # sampling = SamplingParams(
        #     n=args.n,
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     top_k=(args.top_k if args.top_k>0 else None),
        #     max_tokens=args.max_new_tokens,
        # )
        # vLLM expects top_k >= 1 or -1 (disabled)
        _effective_top_k = args.top_k if (args.top_k and args.top_k > 0) else -1
        sampling = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=_effective_top_k,
            max_tokens=args.max_new_tokens,
        )
    else:
        raise NotImplementedError("For speed/stability we require vLLM here.")

    # template -> simple CoT + explicit answer tag
    SYS_PROMPT = ""
    USER_TMPL = (
        "Your task is to solve the math word problem below. "
        "Show your reasoning, and at the end reply exactly with \"The answer is: <number>\".\n\n"
        "Question: {q}\n"
    )

    # storage
    depths = [d for d in (1,4,8,16,32) if d <= args.n]
    maj_acc = np.zeros((len(ds), len(depths)), dtype=np.int32)
    bon_acc = np.zeros((len(ds), len(depths)), dtype=np.int32)

    dump_rows = []

    def flush_dump(rows):
        if not rows: return
        if not os.path.exists(args.save_path):
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        else:
            with open(args.save_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old.extend(rows)
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(old, f, indent=2)
        rows.clear()

    # iterate batches
    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[i:i+args.batch_size]
        qs = batch["question"]
        labels = [f"The answer is: {x.split('####')[-1].strip()}" for x in batch["answer"]]

        # chat templates (works for Qwen2/LLaMA families via HF)
        conversations = [
            [{"role":"user","content": USER_TMPL.format(q=q)}] for q in qs
        ]
        # prompt-token-ids per row (trim left pads for vLLM)
        prompt_token = tok.apply_chat_template(
            conversations, padding="longest", add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        ).to("cuda")
        prompt_ids_list = [
            prompt_token.input_ids[j, prompt_token.attention_mask[j].bool()].tolist()
            for j in range(len(conversations))
        ]

        # generate with vLLM
        outputs = llm.generate(prompt_token_ids=prompt_ids_list, sampling_params=sampling)

        for j, out in enumerate(outputs):
            gens = [out.outputs[k].text for k in range(len(out.outputs))]
            # numeric extractions
            extracted = [extract_answer_number(t) for t in gens]
            true_ans = extract_answer_number(labels[j])

            m, b = majority_and_bestof(true_ans, extracted, depths=depths)
            maj_acc[i+j] = np.array(m, dtype=np.int32)
            bon_acc[i+j] = np.array(b, dtype=np.int32)

            dump_rows.append({
                "id": i+j,
                "prompt_chat": conversations[j],
                "label": labels[j],
                "generations": gens,
                "extracted": extracted,
                "evaluation": {
                    "depths": depths,
                    "majority_eval": m,
                    "best_of_n_eval": b
                }
            })

        # incremental save every ~128 items
        if ((i // max(1,args.batch_size)) % 8) == 0:
            flush_dump(dump_rows)

    # final save + summary
    flush_dump(dump_rows)

    majority = (maj_acc.mean(axis=0) * 100.0).round(2).tolist()
    bestof   = (bon_acc.mean(axis=0) * 100.0).round(2).tolist()
    summary = {
        "model": args.model_name_or_path,
        "tokenizer": args.tokenizer_name_or_path,
        "n": args.n,
        "depths": depths,
        "majority_voting_accuracy_percent": majority,
        "best_of_n_accuracy_percent": bestof
    }
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pprint(summary)

if __name__ == "__main__":
    main()