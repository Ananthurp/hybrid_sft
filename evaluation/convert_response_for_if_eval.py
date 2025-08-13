# import os
# import json
# from pprint import pprint
# from tqdm import tqdm
# import pandas as pd

# from dataclasses import dataclass, field
# from transformers import AutoTokenizer, HfArgumentParser
# from datasets import load_dataset, Dataset


# @dataclass
# class Arguments:
#     response_path: str = field(
#         default=None,
#         metadata={"help": "Response path (json file) to convert."},
#     )
#     tokenizer_path: str = field(
#         default="meta-llama/Meta-Llama-3-8B-Instruct",
#         metadata={"help": "Tokenizer path to help clean str."},
#     )
#     save_path: str = field(default="alpaca_eval_response.json")


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()

#     pprint(args.__dict__)

#     old_data = json.load(open(args.response_path, "r"))

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

#     dataset = []
#     with open("./instruction_following_eval/data/input_data.jsonl") as f:
#         for line in f.readlines():
#             dataset.append(json.loads(line))
#     if_eval_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

#     new_data = []

#     for i in tqdm(range(len(old_data))):
#         prompt = old_data[i]["prompt"]

#         prompt_clean = (
#             tokenizer.decode(
#                 tokenizer(prompt.replace(tokenizer.bos_token, "")).input_ids,
#                 skip_special_tokens=True,
#             )
#             .replace("user\n\n", "")
#             .replace("assistant\n\n", "")
#         )
#         prompt_ref = if_eval_dataset[i]["prompt"]

#         if prompt_clean.strip()[:10] != prompt_ref.strip()[:10]:
#             import ipdb

#             ipdb.set_trace()

#         new_data.append(
#             {
#                 "id": i,
#                 "prompt": prompt_ref,
#                 "response": (
#                     old_data[i]["answer"]
#                     if isinstance(old_data[i]["answer"], str)
#                     else old_data[i]["answer"][0]
#                     .replace("<|eot_id|>", "")
#                     .replace(tokenizer.eos_token, "")
#                     .strip()
#                 ),
#                 "generator": old_data[i]["model_name"],
#             }
#         )
#     os.makedirs(
#         args.save_path.replace(args.save_path.split("/")[-1], ""), exist_ok=True
#     )

#     with open(args.save_path, "w") as outfile:
#         for entry in new_data:
#             json.dump(entry, outfile)
#             outfile.write("\n")
#     print(f"Save response to {args.save_path}")


# if __name__ == "__main__":
#     main()



# import os
# import json
# from pprint import pprint
# from dataclasses import dataclass, field
# from tqdm import tqdm

# from transformers import AutoTokenizer, HfArgumentParser


# @dataclass
# class Arguments:
#     response_path: str = field(
#         default=None, metadata={"help": "Path to generated responses JSON (from generate_response.py)."}
#     )
#     tokenizer_path: str = field(
#         default="meta-llama/Meta-Llama-3-8B-Instruct",
#         metadata={"help": "Tokenizer for cleaning prompts/answers."},
#     )
#     if_eval_data_path: str = field(
#         default="./instruction_following_eval/data/input_data.jsonl",
#         metadata={"help": "IF-Eval input JSONL (prompts)."},
#     )
#     save_path: str = field(
#         default="alpaca_eval_response.jsonl",
#         metadata={"help": "Output JSONL with fields {id, prompt, response, generator}."},
#     )


# def _load_json(p):
#     with open(p, "r", encoding="utf-8") as f:
#         return json.load(f)


# def _load_jsonl(p):
#     rows = []
#     with open(p, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))
#     return rows


# def _clean_text_with_tok(tok, s: str) -> str:
#     # Robustly strip specials via encode/decode if possible
#     try:
#         ids = tok(s).input_ids
#         s2 = tok.decode(ids, skip_special_tokens=True)
#     except Exception:
#         s2 = s
#     # Remove chat markers if present
#     s2 = s2.replace("user\n\n", "").replace("assistant\n\n", "")
#     return s2


# def _pick_first_answer(item) -> str:
#     """
#     Generator may save:
#       - "answers": List[str]
#       - "answer": List[str] or str
#     We return the first string.
#     """
#     if "answers" in item and isinstance(item["answers"], list) and item["answers"]:
#         val = item["answers"][0]
#     elif "answer" in item:
#         if isinstance(item["answer"], list) and item["answer"]:
#             val = item["answer"][0]
#         elif isinstance(item["answer"], str):
#             val = item["answer"]
#         else:
#             raise ValueError("Unrecognized 'answer' format.")
#     elif "output" in item and isinstance(item["output"], str):
#         val = item["output"]
#     else:
#         raise ValueError(f"No answer/answers/output in item keys={list(item.keys())}")
#     return val


# def main():
#     parser = HfArgumentParser((Arguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(vars(args))

#     if not args.response_path or not os.path.exists(args.response_path):
#         raise FileNotFoundError(f"response_path not found: {args.response_path}")
#     if not os.path.exists(args.if_eval_data_path):
#         raise FileNotFoundError(f"if_eval_data_path not found: {args.if_eval_data_path}")

#     tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

#     old_data = _load_json(args.response_path)
#     if_eval_rows = _load_jsonl(args.if_eval_data_path)

#     if len(old_data) != len(if_eval_rows):
#         print(f"⚠️ Length mismatch: generated={len(old_data)} vs if_eval={len(if_eval_rows)}. Will map by index.")

#     out_rows = []
#     for i in tqdm(range(min(len(old_data), len(if_eval_rows)))):
#         gen_item = old_data[i]
#         ref_item = if_eval_rows[i]

#         # Clean prompt from generator file
#         prompt_raw = gen_item.get("prompt") or gen_item.get("instruction") or ""
#         prompt_clean = _clean_text_with_tok(tok, prompt_raw)

#         # IF-Eval reference prompt (trusted)
#         prompt_ref = ref_item.get("prompt", "")

#         # Soft check; warn but don't crash
#         if prompt_clean.strip()[:20] != prompt_ref.strip()[:20]:
#             print(f"⚠️ Prompt header mismatch at id={i}")

#         # Pick first answer (repo’s converters typically require one response per prompt)
#         ans = _pick_first_answer(gen_item)
#         # Strip specials/EOT/EOS if they exist
#         eos = tok.eos_token or ""
#         ans = ans.replace("<|eot_id|>", "")
#         if eos:
#             ans = ans.replace(eos, "")
#         ans = ans.strip()

#         out_rows.append({
#             "id": i,
#             "prompt": prompt_ref,
#             "response": ans,
#             "generator": gen_item.get("model_name", "unknown"),
#         })

#     # Ensure directory exists
#     out_dir = os.path.dirname(args.save_path)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)

#     # Write JSONL
#     with open(args.save_path, "w", encoding="utf-8") as f:
#         for row in out_rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     print(f"✅ Saved {len(out_rows)} rows to {args.save_path}")


# if __name__ == "__main__":
#     main()




import os
import json
from pprint import pprint
from tqdm import tqdm
import pandas as pd

from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset, Dataset


@dataclass
class Arguments:
    response_path: str = field(
        default=None,
        metadata={"help": "Response path (json file) to convert."},
    )
    tokenizer_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Tokenizer path to help clean str."},
    )
    save_path: str = field(default="alpaca_eval_response.json")


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()

    pprint(args.__dict__)

    old_data = json.load(open(args.response_path, "r"))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = []
    with open("./instruction_following_eval/data/input_data.jsonl") as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    if_eval_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

    new_data = []

    for i in tqdm(range(len(old_data))):
        prompt = old_data[i]["prompt"]

        prompt_clean = (
            tokenizer.decode(
                tokenizer(prompt.replace(tokenizer.bos_token, "")).input_ids,
                skip_special_tokens=True,
            )
            .replace("user\n\n", "")
            .replace("assistant\n\n", "")
        )
        prompt_ref = if_eval_dataset[i]["prompt"]

        if prompt_clean.strip()[:10] != prompt_ref.strip()[:10]:
            import ipdb # type: ignore

            ipdb.set_trace()

        new_data.append(
            {
                "id": i,
                "prompt": prompt_ref,
                "response": (
                    old_data[i]["answer"]
                    if isinstance(old_data[i]["answer"], str)
                    else old_data[i]["answer"][0]
                    .replace("<|eot_id|>", "")
                    .replace(tokenizer.eos_token, "")
                    .strip()
                ),
                "generator": old_data[i]["model_name"],
            }
        )
    os.makedirs(
        args.save_path.replace(args.save_path.split("/")[-1], ""), exist_ok=True
    )

    with open(args.save_path, "w") as outfile:
        for entry in new_data:
            json.dump(entry, outfile)
            outfile.write("\n")
    print(f"Save response to {args.save_path}")


if __name__ == "__main__":
    main()
