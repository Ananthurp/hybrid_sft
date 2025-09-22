import os, re, json, sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

import vllm
from vllm import SamplingParams

# ----------------------
# Prompt & parsing utils
# ----------------------

MC_TEMPLATE = """You will be given a question with multiple-choice answers.
Think step by step briefly, then give ONLY the final choice letter.

Question:
{question}

Choices:
{choices_str}

When you are ready, answer strictly in the format:
The answer is: X
(where X is one of A, B, C, D or E).
"""

LETTER_RE = re.compile(
    r"(?:The\s+answer\s+is\s*[:\-]?\s*([A-E]))\b|(?:\b([A-E])\b(?=[\s\.\)\:,$]))",
    flags=re.IGNORECASE
)

def choices_to_str(choices: List[str]) -> str:
    letters = "ABCDE"
    lines = []
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}. {ch}".strip())
    return "\n".join(lines)

def extract_choice_letter(text: str, choices: List[str]) -> str | None:
    # 1) Try to find an explicit letter.
    found = None
    for m in LETTER_RE.finditer(text):
        g = m.group(1) or m.group(2)
        if g:
            found = g.upper()
    if found:
        return found

    # 2) Weak fallback: if it literally quotes one choice string, map it.
    text_low = text.lower()
    for i, ch in enumerate(choices):
        if ch.lower() in text_low:
            return "ABCDE"[i]

    return None

def majority_and_best_of_n(reference_letter: str,
                           candidate_letters: List[str],
                           depths=(1, 4, 8, 16, 32)) -> Tuple[List[int], List[int]]:
    maj, bon = [], []
    for d in depths:
        if d > len(candidate_letters):
            break
        window = candidate_letters[:d]
        # Majority vote
        votes: Dict[str,int] = {}
        for c in window:
            if c is not None:
                votes[c] = votes.get(c, 0) + 1
        majority_choice = max(votes.items(), key=lambda x: x[1])[0] if votes else None
        maj.append(1 if majority_choice == reference_letter else 0)
        # Best-of-n
        bon.append(1 if reference_letter in window else 0)
    return maj, bon

# ---------------
# Dataset loaders
# ---------------

def load_arc(split="validation"):
    # ARC-Challenge
    ds = load_dataset("ai2_arc", "ARC-Challenge")[split]
    items = []
    for r in ds:
        q = r["question"]
        ch = r["choices"]["text"]            # list[str]
        ans_label = r["answerKey"].strip().upper()  # 'A'...'E'
        items.append((q, ch, ans_label))
    return items

def load_hellaswag(split="validation"):
    ds = load_dataset("hellaswag")[split]
    items = []
    # HellaSwag: context + endings; label is int 0..3
    for r in ds:
        q = (r["ctx"] + " " + r["ctx_a"]).strip()
        ch = r["endings"]                    # 4 endings
        ans_idx = int(r["label"])
        ans_letter = "ABCD"[ans_idx]
        items.append((q, ch, ans_letter))
    return items

def load_winogrande(split="validation"):
    # Use debiased version, labels available in validation
    ds = load_dataset("winogrande", "winogrande_debiased")[split]
    items = []
    for r in ds:
        q = r["sentence"].replace("_", "_____")
        ch = [r["option1"], r["option2"]]
        ans_idx = 0 if r["answer"].strip() == "1" else 1
        ans_letter = "AB"[ans_idx]
        items.append((q, ch, ans_letter))
    return items

def load_mmlu():
    from datasets import load_dataset

    def _rows_to_triples(rows):
        items = []
        for r in rows:
            # Choices may be a list or separate A/B/C/D columns
            if "choices" in r and isinstance(r["choices"], (list, tuple)):
                choices = list(r["choices"])
            else:
                choices = [r.get("A"), r.get("B"), r.get("C"), r.get("D")]

            # Normalize ground truth to a letter "A"/"B"/"C"/"D"
            ans = r.get("answer")
            if isinstance(ans, str):
                gt_letter = ans.strip().upper()  # e.g., "A"
            else:
                gt_letter = "ABCD"[int(ans)]

            # Append exactly a 3-tuple: (question, choices, letter)
            items.append((r["question"], choices, gt_letter))
        return items

    try:
        ds = load_dataset("lukaemon/mmlu")
        split = "test" if "test" in ds else ("validation" if "validation" in ds else "dev")
        return _rows_to_triples(ds[split])
    except Exception as e:
        print(f"[warn] lukaemon/mmlu failed ({e}). Falling back to cais/mmlu.")
        ds = load_dataset("cais/mmlu", "all")
        return _rows_to_triples(ds["test"])

def load_truthfulqa_mc1(split="validation"):
    # TruthfulQA is a bit irregular across versions. Try a robust path:
    # 'truthful_qa' with field mc1_targets having 'choices' & 'labels'
    ds = load_dataset("truthful_qa", "multiple_choice")[split]
    items = []
    for r in ds:
        q = r["question"]
        # Try MC1 first
        tgt = r.get("mc1_targets", None) or r.get("mc1_targets_scores", None)
        if tgt and "choices" in tgt:
            ch = tgt["choices"]
            # labels might be missing; fallback to 'correct' field if available
            labels = tgt.get("labels", None)
            if labels:
                # choose any label==1 as correct (usually one)
                idxs = [i for i, v in enumerate(labels) if int(v) == 1]
                if idxs:
                    ans_letter = "ABCDE"[idxs[0]]
                else:
                    # fallback: if 'correct' exists
                    correct = r.get("mc1_correct", None)
                    if correct and correct in ch:
                        ans_letter = "ABCDE"[ch.index(correct)]
                    else:
                        continue
            else:
                correct = r.get("mc1_correct", None)
                if correct and correct in ch:
                    ans_letter = "ABCDE"[ch.index(correct)]
                else:
                    continue
            # Cap at 5 choices if more exist (rare)
            ch = ch[:5]
            items.append((q, ch, ans_letter))
    return items

TASK_LOADERS = {
    "arc": load_arc,
    "hellaswag": load_hellaswag,
    "winogrande": load_winogrande,
    "mmlu": load_mmlu,
    "truthfulqa": load_truthfulqa_mc1,
}

# -----------
# CLI config
# -----------

@dataclass
class Arguments:
    task: str = field(default="arc",
                      metadata={"help": "one of: arc, hellaswag, winogrande, mmlu, truthfulqa"})
    # model
    model_name_or_path: str = field(default="")
    tokenizer_name_or_path: str = field(default="")
    dtype: str = field(default="bf16", metadata={"choices": ["fp16", "bf16"]})

    # generation
    temperature: float = 0.6
    top_k: int = -1   # -1 disables top-k in vLLM
    top_p: float = 0.9
    n: int = 32
    max_new_tokens: int = 256
    batch_size: int = 16
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    seed: int = 42

    # save
    remove_old: bool = False
    save_path: str = field(default="")
    summary_path: str = field(default="")
    depths_csv: str = field(default="1,4,8,16,32")

def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if args.remove_old:
        for p in [args.save_path, args.summary_path]:
            if p and os.path.exists(p):
                os.remove(p)

    # load dataset
    task = args.task.lower()
    if task not in TASK_LOADERS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASK_LOADERS.keys())}")
    items = TASK_LOADERS[task]()
    print(f"[info] Loaded {len(items)} items for task: {task}")

    # tokenizer & vLLM
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Some chat tokenizers need a pad token
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.decode(len(tokenizer) - 1)
        tokenizer.pad_token_id = len(tokenizer) - 1
    elif "llama-2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if not args.use_vllm:
        raise NotImplementedError("This script currently supports vLLM only.")

    llm = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=True,
        enable_chunked_prefill=False,
        swap_space=16,
    )

    depths = tuple(int(x) for x in args.depths_csv.split(",") if x.strip())

    save_rows: List[Dict[str, Any]] = []
    max_depth = max(depths)
    maj_all = np.zeros((len(items), len(depths)), dtype=int)
    bon_all = np.zeros((len(items), len(depths)), dtype=int)

    # batching
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        qs = []
        choices_batch = []
        gts = []
        for q, ch, gt_letter in batch:
            qs.append(q)
            choices_batch.append(ch)
            gts.append(gt_letter)

        # build chat prompts
        conv = []
        for q, ch in zip(qs, choices_batch):
            prompt = MC_TEMPLATE.format(question=q, choices_str=choices_to_str(ch))
            conv.append([{"role": "user", "content": prompt}])

        # left-padding helps batching long prompts
        tokenizer.padding_side = "left"
        prompt_token = tokenizer.apply_chat_template(
            conv,
            padding="longest",
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        prompt_token_ids = [
            prompt_token.input_ids[j, prompt_token.attention_mask[j].bool()].tolist()
            for j in range(len(conv))
        ]

        # vLLM sampling params
        eff_top_k = args.top_k if (args.top_k is not None and args.top_k > 0) else -1
        sampling = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=eff_top_k,
            max_tokens=args.max_new_tokens,
        )

        with torch.no_grad():
            outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                                   sampling_params=sampling)

        # collect
        for j, out in enumerate(outputs):
            all_texts = [out.outputs[k].text for k in range(len(out.outputs))]
            letters = [extract_choice_letter(t, choices_batch[j]) for t in all_texts]
            maj, bon = majority_and_best_of_n(gts[j], letters, depths=depths)

            # write row
            save_rows.append({
                "id": i + j,
                "prompt": conv[j],
                "choices": choices_batch[j],
                "gold": gts[j],
                "responses": all_texts,
                "parsed_letters": letters,
                "majority_eval": maj,
                "best_of_n_eval": bon,
            })

            # aggregate
            maj_all[i + j, :len(maj)] = np.array(maj, dtype=int)
            bon_all[i + j, :len(bon)] = np.array(bon, dtype=int)

        # periodic flush
        if (i // args.batch_size) % 8 == 0 and args.save_path:
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(save_rows, f, indent=2)

    # final save
    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(save_rows, f, indent=2)

    # summary
    maj_acc = np.mean(maj_all, axis=0).tolist()
    bon_acc = np.mean(bon_all, axis=0).tolist()
    summary = {
        "task": task,
        "depths": list(depths),
        "majority_vote_accuracy": maj_acc,
        "best_of_n_accuracy": bon_acc,
        "n_samples": args.n,
        "count": len(items),
    }
    pprint(summary)
    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()