# # #!/usr/bin/env python3
# # # Simplified GEM-style diversity evaluation on raw answers
# # # Metrics: Distinct-n (avg over n=1..3), Self-BLEU, Sentence-BERT diversity

# import os
# import json
# from dataclasses import dataclass, field
# from pprint import pprint
# from typing import List, Optional

# import numpy as np
# from tqdm import tqdm

# # Optional CUDA acceleration for sentence-transformers
# import torch

# # BLEU
# import sacrebleu

# # Tokenization (robust fallback if NLTK data isn't installed)
# try:
#     from nltk import word_tokenize as _nltk_word_tokenize  # requires punkt
#     _HAVE_NLTK = True
# except Exception:
#     _HAVE_NLTK = False

# # Sentence-BERT
# try:
#     import sentence_transformers
# except Exception as e:
#     sentence_transformers = None

# @dataclass
# class AllArguments:
#     # Path to responses.json created by generate_response.py
#     response_path: str = field(default="responses.json")

#     # Optional: we don't actually need chat tokenizer here; kept for parity with older scripts
#     tokenizer_path: Optional[str] = field(default=None)
#     detokenizer_path: Optional[str] = field(default=None)

#     # Optional: write a JSON summary next to logs
#     out_path: Optional[str] = field(default=None)


# # ---------------------- helpers ----------------------
# def word_tokenize_safe(text: str) -> List[str]:
#     """Tokenize with NLTK if available; otherwise fall back to simple whitespace split."""
#     if not isinstance(text, str):
#         return []
#     text = text.strip()
#     if not text:
#         return []
#     if _HAVE_NLTK:
#         try:
#             return _nltk_word_tokenize(text)
#         except Exception:
#             pass
#     # simple fallback
#     return text.split()


# def build_answer_matrix(data: list) -> List[List[str]]:
#     """
#     Build a rectangular matrix of answers:
#       - Let P = number of prompts with at least 1 answer list
#       - Let K = min number of answers among those prompts
#     Returns a list-of-lists with shape [K][P], i.e.,
#       answers_by_sample[j][i] = j-th answer for prompt i
#     This mirrors GEM's layout (sample-major), enabling pairwise comparisons across samples.
#     """
#     per_prompt: List[List[str]] = []
#     for x in data:
#         ans = None
#         # Primary format produced by generate_response.py
#         if isinstance(x.get("answer"), list) and x["answer"]:
#             ans = [a for a in x["answer"] if isinstance(a, str) and a.strip()]
#         # Fallback: single-string fields (won't give diversity across n>1 but keeps code robust)
#         elif isinstance(x.get("output"), str) and x["output"].strip():
#             ans = [x["output"].strip()]

#         if ans:
#             per_prompt.append(ans)

#     if not per_prompt:
#         raise ValueError("No usable answers found in response file.")

#     # Choose a common K across prompts
#     K = min(len(a) for a in per_prompt)
#     if K < 2:
#         # You can still compute distinct/self-BLEU per prompt if K==1, but pairwise SentBERT needs >=2.
#         # We'll allow K==1 and handle SentBERT gracefully.
#         pass

#     # Trim to rectangular [num_prompts x K]
#     per_prompt = [a[:K] for a in per_prompt]

#     # Re-layout to sample-major [K][num_prompts]
#     answers_by_sample: List[List[str]] = [[] for _ in range(K)]
#     for i in range(len(per_prompt)):
#         for j in range(K):
#             answers_by_sample[j].append(per_prompt[i][j])

#     return answers_by_sample  # shape [K][P]


# # ---------------------- metrics ----------------------
# class AveragedNgramDiversityMetric:
#     """Average distinct-n over n in [n_min, n_max], averaged across prompts."""

#     def __init__(self, n_min: int = 1, n_max: int = 3):
#         self.n_min = n_min
#         self.n_max = n_max

#     def _distinct_n(self, responses: List[str], n: int) -> float:
#         all_ngrams = []
#         for resp in responses:
#             toks = word_tokenize_safe(resp)
#             if len(toks) < n:
#                 continue
#             ngrams = [tuple(toks[k:k+n]) for k in range(len(toks) - n + 1)]
#             all_ngrams.extend(ngrams)
#         if not all_ngrams:
#             return 0.0
#         return len(set(all_ngrams)) / float(len(all_ngrams))

#     def __call__(self, by_sample: List[List[str]]) -> float:
#         """
#         by_sample: [K][P]
#           K = responses per prompt (min across prompts)
#           P = number of prompts
#         For each prompt i, we gather all K responses at i: [by_sample[j][i] for j in range(K)].
#         """
#         if not by_sample:
#             return 0.0
#         K, P = len(by_sample), len(by_sample[0])
#         scores = []
#         for i in range(P):
#             texts_i = [by_sample[j][i] for j in range(K)]
#             for n in range(self.n_min, self.n_max + 1):
#                 scores.append(self._distinct_n(texts_i, n))
#         return float(np.mean(scores)) if scores else 0.0


# class SelfBLEUMetric:
#     """Average Self-BLEU across prompts (lower BLEU => higher diversity)."""

#     def __call__(self, by_sample: List[List[str]]) -> float:
#         if not by_sample:
#             return 0.0
#         K, P = len(by_sample), len(by_sample[0])
#         bleu_scores = []
#         for i in range(P):
#             texts = [by_sample[j][i] for j in range(K)]
#             if len(texts) < 2:
#                 continue
#             # Average leave-one-out BLEU at this prompt
#             per_i = []
#             for h in range(len(texts)):
#                 hyp = [texts[h]]
#                 refs = [texts[:h] + texts[h+1:]]  # sacrebleu expects list of reference sets
#                 try:
#                     score = sacrebleu.corpus_bleu(hyp, refs).score
#                 except Exception:
#                     score = 0.0
#                 per_i.append(score)
#             if per_i:
#                 bleu_scores.append(np.mean(per_i))
#         return float(np.mean(bleu_scores)) if bleu_scores else 0.0


# class SentBertSimilarity:
#     """Pairwise cosine similarity via Sentence-BERT; runs on GPU if available."""

#     def __init__(self, model_name: str = "bert-large-nli-stsb-mean-tokens", batch_size: int = 512):
#         if sentence_transformers is None:
#             raise RuntimeError(
#                 "sentence-transformers is not installed. Please install it "
#                 "(e.g., pip install sentence-transformers) to compute SentBERT diversity."
#             )
#         self.model = sentence_transformers.SentenceTransformer(model_name)
#         self.batch_size = batch_size
#         if torch.cuda.is_available():
#             self.model.to(torch.device("cuda"))

#     def __call__(self, a: List[str], b: List[str]) -> np.ndarray:
#         # a and b are lists with equal length (number of prompts P)
#         emb_a = self.model.encode(a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
#         emb_b = self.model.encode(b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
#         if torch.cuda.is_available():
#             emb_a = emb_a.to(torch.device("cuda"))
#             emb_b = emb_b.to(torch.device("cuda"))
#         # cosine similarity per row
#         dot = (emb_a * emb_b).sum(dim=1)
#         sim = dot / (emb_a.norm(dim=1) * emb_b.norm(dim=1) + 1e-12)
#         return sim.detach().cpu().numpy()


# class SentBertDiversity:
#     """
#     Diversity = 1 - mean cosine similarity, averaged over all pairs of samples (i<j) and prompts.
#     Input is sample-major [K][P]; compares rows pairwise as in GEM.
#     """

#     def __init__(self, model_name: str = "bert-large-nli-stsb-mean-tokens"):
#         self.sim = SentBertSimilarity(model_name=model_name)

#     def __call__(self, by_sample: List[List[str]]) -> float:
#         if not by_sample or len(by_sample) < 2:
#             # Not enough samples to compute pairwise similarity
#             return 0.0
#         K = len(by_sample)
#         sims = []
#         for i in range(K):
#             for j in range(i):
#                 s = self.sim(by_sample[i], by_sample[j])  # vector over prompts
#                 sims.append(s)
#         if not sims:
#             return 0.0
#         mean_sim = float(np.mean(np.concatenate([s.reshape(-1) for s in sims], axis=0)))
#         return 1.0 - mean_sim


# # ---------------------- main ----------------------
# def main():
#     from transformers import HfArgumentParser
#     parser = HfArgumentParser((AllArguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)

#     # Load responses.json
#     data = json.load(open(args.response_path, "r", encoding="utf-8"))

#     # Build K x P matrix of raw answers (sample-major)
#     by_sample = build_answer_matrix(data)  # [K][P]
#     K, P = len(by_sample), (len(by_sample[0]) if by_sample else 0)
#     print(f"[info] Using {P} prompts with K={K} responses per prompt.")

#     results = {
#         "averaged_ngram_diversity_score": None,
#         "bleu_diversity_score": None,
#         "sentbert_diversity_score": None,
#         "num_prompts": P,
#         "num_responses_per_prompt": K,
#     }

#     # Distinct-n averaged (n=1..3)
#     print("Calculating N-gram diversity score...")
#     ngram_metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
#     ngram_div = ngram_metric(by_sample)
#     results["averaged_ngram_diversity_score"] = round(ngram_div * 100.0, 2)
#     print(f"N-gram diversity score: {ngram_div:.6f}")

#     # Self-BLEU (lower = less similarity); we report diversity = 100 - BLEU
#     print("Calculating Self-BLEU similarity score...")
#     sb_metric = SelfBLEUMetric()
#     self_bleu = sb_metric(by_sample)
#     results["bleu_diversity_score"] = round(100.0 - self_bleu, 2)
#     print(f"Self-BLEU similarity (diversity = 100 - score): {self_bleu:.6f}")

#     # SentBERT diversity (optional if sentence-transformers installed)
#     try:
#         print("Calculating Sentence-BERT diversity score...")
#         sb_div_metric = SentBertDiversity()
#         sb_div = sb_div_metric(by_sample)
#         results["sentbert_diversity_score"] = round(sb_div * 100.0, 2)
#         print(f"SentBERT diversity score: {sb_div:.6f}")
#     except RuntimeError as e:
#         print(f"[warn] {e}")
#         results["sentbert_diversity_score"] = None

#     pprint(results)

#     # Optional: save summary JSON
#     if args.out_path:
#         os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
#         with open(args.out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2)
#         print(f"[ok] wrote diversity summary to {args.out_path}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# Simplified GEM-style diversity evaluation.
# Metrics: Distinct-n (avg over n=1..3), Self-BLEU (lower -> more diverse, we report 100-SelfBLEU),
#          Sentence-BERT diversity (1 - mean cosine sim across sample pairs).
#
# Behavior aligned with GEM:
# - Build a [K][P] "response_set" of chat-templated strings (prompt+answer) via apply_chat_template.
# - Cache to <responses-cleaned.json> and reuse if present.

import os
import json
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Optional

import numpy as np
from tqdm import tqdm

# Optional CUDA acceleration for sentence-transformers
import torch

# BLEU
import sacrebleu

# Tokenization (robust fallback if NLTK data isn't installed)
try:
    from nltk import word_tokenize as _nltk_word_tokenize  # requires punkt
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

# Sentence-BERT
try:
    import sentence_transformers
except Exception:
    sentence_transformers = None


@dataclass
class AllArguments:
    # Path to responses.json created by generate_response.py
    response_path: str = field(default="responses.json")

    # Tokenizers (used only during "cleaning" to reproduce GEM inputs)
    tokenizer_path: Optional[str] = field(default=None)
    detokenizer_path: Optional[str] = field(default=None)

    # Optional: write a JSON summary next to logs
    out_path: Optional[str] = field(default=None)


# ---------------------- helpers ----------------------
def word_tokenize_safe(text: str) -> List[str]:
    """Tokenize with NLTK if available; otherwise fall back to simple whitespace split."""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    if _HAVE_NLTK:
        try:
            return _nltk_word_tokenize(text)
        except Exception:
            pass
    # simple fallback
    return text.split()


def _maybe_build_cleaned_response_set(resp_path: str,
                                      tokenizer_path: Optional[str],
                                      detok_path: Optional[str]) -> List[List[str]]:
    """
    GEM-style data prep:
      - If <resp_path> has a sibling <resp_path.replace('.json', '-cleaned.json')>, load and return it.
      - Else: load responses.json (list of items with {prompt/instruction, answer:list or output}),
              reconstruct chat-templated strings with apply_chat_template, lay them out as [K][P],
              dump -cleaned.json, and return it.
    """
    cleaned_path = resp_path.replace(".json", "-cleaned.json")
    if os.path.exists(cleaned_path):
        return json.load(open(cleaned_path, "r", encoding="utf-8"))

    data = json.load(open(resp_path, "r", encoding="utf-8"))

    # We only need tokenizer(s) here to reproduce the same text GEM evaluates.
    from transformers import AutoTokenizer
    if tokenizer_path is None:
        raise ValueError("--tokenizer_path is required to build the cleaned response set like GEM.")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    detok = AutoTokenizer.from_pretrained(detok_path) if detok_path else None

    response_set: List[List[str]] = []
    for i in tqdm(range(len(data)), desc="Preparing cleaned response_set"):
        x = data[i]
        # Determine number of answers for this prompt
        if isinstance(x.get("answer"), list) and x["answer"]:
            n = len(x["answer"])
        elif isinstance(x.get("output"), str) and x["output"].strip():
            n = 1
        else:
            # skip items without usable answers
            continue

        # Initialize rows [K] on the first usable item
        if not response_set:
            response_set = [[] for _ in range(n)]
        else:
            # GEM assumes same K across prompts; if not, we trim later to min K across prompts.
            # Here we enforce equality to match GEM behavior as closely as possible.
            assert len(response_set) == n, (
                f"Found different number of answers across prompts: "
                f"expected {len(response_set)} but got {n} at item {i}."
            )

        # Build prompt string (optionally detokenize to strip specials)
        if detok:
            raw_prompt = (x.get("prompt") or x.get("instruction") or "").strip()
            prompt_str = detok.decode(detok.encode(raw_prompt), skip_special_tokens=True)
            prompt_str = prompt_str.replace("user\n\n", "").replace("assistant\n\n", "")
        else:
            prompt_str = (x.get("prompt") or x.get("instruction") or "").strip()

        # Answers list
        if n == 1:
            answers = [x["output"].strip()]
        else:
            answers = [a for a in x["answer"] if isinstance(a, str)]

        # Compose chat-templated strings (prompt+answer) like GEM
        for j in range(n):
            ans_str = answers[j]
            # Align with GEM's light cleanup on certain templates
            ans_str = ans_str.replace("<|eot_id|>", "")
            chat = [
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": ans_str},
            ]
            res = tok.apply_chat_template(chat, tokenize=False)
            response_set[j].append(res)

    # Persist for reuse
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(response_set, f, indent=2)
    return response_set


def build_answer_matrix_raw_answers(data: list) -> List[List[str]]:
    """
    (Alternative) Build a rectangular matrix of RAW answers only:
      answers_by_sample[j][i] = j-th answer for prompt i
    Kept for reference; we now prefer GEM-style cleaned set above.
    """
    per_prompt: List[List[str]] = []
    for x in data:
        ans = None
        if isinstance(x.get("answer"), list) and x["answer"]:
            ans = [a for a in x["answer"] if isinstance(a, str) and a.strip()]
        elif isinstance(x.get("output"), str) and x["output"].strip():
            ans = [x["output"].strip()]
        if ans:
            per_prompt.append(ans)

    if not per_prompt:
        raise ValueError("No usable answers found in response file.")

    K = min(len(a) for a in per_prompt)
    per_prompt = [a[:K] for a in per_prompt]

    answers_by_sample: List[List[str]] = [[] for _ in range(K)]
    for i in range(len(per_prompt)):
        for j in range(K):
            answers_by_sample[j].append(per_prompt[i][j])

    return answers_by_sample  # [K][P]


# ---------------------- metrics ----------------------
class AveragedNgramDiversityMetric:
    """Average distinct-n over n in [n_min, n_max], averaged across prompts."""

    def __init__(self, n_min: int = 1, n_max: int = 3):
        self.n_min = n_min
        self.n_max = n_max

    def _distinct_n(self, responses: List[str], n: int) -> float:
        all_ngrams = []
        for resp in responses:
            toks = word_tokenize_safe(resp)
            if len(toks) < n:
                continue
            ngrams = [tuple(toks[k:k+n]) for k in range(len(toks) - n + 1)]
            all_ngrams.extend(ngrams)
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / float(len(all_ngrams))

    def __call__(self, by_sample: List[List[str]]) -> float:
        """
        by_sample: [K][P]
          K = responses per prompt
          P = number of prompts
        For each prompt i, gather all K responses at i: [by_sample[j][i] for j in range(K)].
        """
        if not by_sample:
            return 0.0
        K, P = len(by_sample), len(by_sample[0])
        scores = []
        for i in range(P):
            texts_i = [by_sample[j][i] for j in range(K)]
            for n in range(self.n_min, self.n_max + 1):
                scores.append(self._distinct_n(texts_i, n))
        return float(np.mean(scores)) if scores else 0.0


class SelfBLEUMetric:
    """Average Self-BLEU across prompts (lower BLEU => higher diversity)."""

    def __call__(self, by_sample: List[List[str]]) -> float:
        if not by_sample:
            return 0.0
        K, P = len(by_sample), len(by_sample[0])
        bleu_scores = []
        for i in range(P):
            texts = [by_sample[j][i] for j in range(K)]
            if len(texts) < 2:
                continue
            # Average leave-one-out BLEU at this prompt
            per_i = []
            for h in range(len(texts)):
                hypothesis = [texts[h]]
                references = texts[:h] + texts[h+1:]
                # FIX: sacrebleu expects ref streams as list of corpora, each aligned with sys_stream
                # Single hypothesis -> each reference must be a list of length 1
                ref_streams = [[r] for r in references]
                try:
                    score = sacrebleu.corpus_bleu(hypothesis, ref_streams).score
                except Exception:
                    score = 0.0
                per_i.append(score)
            if per_i:
                bleu_scores.append(np.mean(per_i))
        return float(np.mean(bleu_scores)) if bleu_scores else 0.0


class SentBertSimilarity:
    """Pairwise cosine similarity via Sentence-BERT; runs on GPU if available."""

    def __init__(self, model_name: str = "bert-large-nli-stsb-mean-tokens", batch_size: int = 512):
        if sentence_transformers is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Please install it "
                "(e.g., pip install sentence-transformers) to compute SentBERT diversity."
            )
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

    def __call__(self, a: List[str], b: List[str]) -> np.ndarray:
        # a and b are lists with equal length (number of prompts P)
        emb_a = self.model.encode(a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
        emb_b = self.model.encode(b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
        if torch.cuda.is_available():
            emb_a = emb_a.to(torch.device("cuda"))
            emb_b = emb_b.to(torch.device("cuda"))
        # cosine similarity per row
        dot = (emb_a * emb_b).sum(dim=1)
        sim = dot / (emb_a.norm(dim=1) * emb_b.norm(dim=1) + 1e-12)
        return sim.detach().cpu().numpy()


class SentBertDiversity:
    """
    Diversity = 1 - mean cosine similarity, averaged over all pairs of samples (i<j) and prompts.
    Input is sample-major [K][P]; compares rows pairwise as in GEM.
    """

    def __init__(self, model_name: str = "bert-large-nli-stsb-mean-tokens"):
        self.sim = SentBertSimilarity(model_name=model_name)

    def __call__(self, by_sample: List[List[str]]) -> float:
        if not by_sample or len(by_sample) < 2:
            # Not enough samples to compute pairwise similarity
            return 0.0
        K = len(by_sample)
        sims = []
        for i in range(K):
            for j in range(i):
                s = self.sim(by_sample[i], by_sample[j])  # vector over prompts
                sims.append(s)
        if not sims:
            return 0.0
        mean_sim = float(np.mean(np.concatenate([s.reshape(-1) for s in sims], axis=0)))
        return 1.0 - mean_sim


# ---------------------- main ----------------------
def main():
    from transformers import HfArgumentParser
    parser = HfArgumentParser((AllArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    # GEM-style: prefer the "-cleaned.json" cache if present; else build it
    try:
        by_sample = _maybe_build_cleaned_response_set(
            resp_path=args.response_path,
            tokenizer_path=args.tokenizer_path,
            detok_path=args.detokenizer_path,
        )
    except ValueError as e:
        # If user really wants raw-answer diversity and didn't provide tokenizer_path,
        # fall back to raw answers (will NOT match GEM exactly).
        print(f"[warn] {e} Falling back to raw answers only.")
        data = json.load(open(args.response_path, "r", encoding="utf-8"))
        by_sample = build_answer_matrix_raw_answers(data)

    # [K][P]
    K = len(by_sample)
    P = len(by_sample[0]) if K > 0 else 0
    print(f"[info] Using {P} prompts with K={K} responses per prompt.")

    results = {
        "averaged_ngram_diversity_score": None,
        "bleu_diversity_score": None,
        "sentbert_diversity_score": None,
        "num_prompts": P,
        "num_responses_per_prompt": K,
    }

    # Distinct-n averaged (n=1..3)
    print("Calculating N-gram diversity score...")
    ngram_metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
    ngram_div = ngram_metric(by_sample)
    results["averaged_ngram_diversity_score"] = round(ngram_div * 100.0, 2)
    print(f"N-gram diversity score: {ngram_div:.6f}")

    # Self-BLEU (lower = less similarity); we report diversity = 100 - BLEU
    print("Calculating Self-BLEU similarity score...")
    sb_metric = SelfBLEUMetric()
    self_bleu = sb_metric(by_sample)
    results["bleu_diversity_score"] = round(100.0 - self_bleu, 2)
    print(f"Self-BLEU similarity (diversity = 100 - score): {self_bleu:.6f}")

    # SentBERT diversity (optional if sentence-transformers installed)
    try:
        print("Calculating Sentence-BERT diversity score...")
        sb_div_metric = SentBertDiversity()
        sb_div = sb_div_metric(by_sample)
        results["sentbert_diversity_score"] = round(sb_div * 100.0, 2)
        print(f"SentBERT diversity score: {sb_div:.6f}")
    except RuntimeError as e:
        print(f"[warn] {e}")
        results["sentbert_diversity_score"] = None

    pprint(results)

    # Optional: save summary JSON
    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[ok] wrote diversity summary to {args.out_path}")


if __name__ == "__main__":
    main()