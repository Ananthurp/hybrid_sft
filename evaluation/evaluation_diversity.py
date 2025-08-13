# #################
# # This code is modified from https://github.com/facebookresearch/rlfh-gen-div
# #################
# import os
# from dataclasses import dataclass, field
# import json
# from pprint import pprint

# import torch
# import numpy as np
# import sentence_transformers # type: ignore
# from tqdm import tqdm

# # from sklearn.metrics.pairwise import cosine_similarity
# from transformers import set_seed, HfArgumentParser, AutoTokenizer

# from nltk.util import ngrams # type: ignore
# from nltk import word_tokenize # type: ignore
# from collections import Counter

# import sacrebleu # type: ignore

# @dataclass
# class AllArguments:
#     response_path: str = field(
#         default="./results/responses", metadata={"help": "Response path (json file)."}
#     )

#     tokenizer_path: str = field(default=None)
#     detokenizer_path: str = field(default=None)


# class SentBertSimilarity:
#     def __init__(self):

#         self.model_name = "bert-large-nli-stsb-mean-tokens"  # FIXME - hard coded
#         self.model = sentence_transformers.SentenceTransformer(self.model_name)
#         if torch.cuda.is_available():
#             self.model.to(torch.device("cuda"))

#     # @functools.cache
#     def embed(self, sentence):
#         return self.model.encode(sentence)

#     # @functools.cache
#     def sent_bert_cosine_similarity(self, resps_1, resps_2):
#         embeds_1 = self.model.encode(
#             resps_1, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
#         )
#         embeds_2 = self.model.encode(
#             resps_2, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
#         )

#         if torch.cuda.is_available():
#             embeds_1 = embeds_1.to(torch.device("cuda"))
#             embeds_2 = embeds_2.to(torch.device("cuda"))

#         dot_product = (embeds_1 * embeds_2).sum(dim=1)

#         # Calculate cosine similarity
#         cosine_similarity = dot_product / (embeds_1.norm(dim=1) * embeds_2.norm(dim=1))

#         return cosine_similarity.detach().cpu().numpy()

#     def __call__(self, resp_a, resp_b):
#         return self.sent_bert_cosine_similarity(resp_a, resp_b)


# class SentBertDiversity:
#     """
#     Implements the diversity to similarity reduction specified on section 5 in the paper
#     (https://arxiv.org/pdf/2004.02990.pdf)
#     for any similarity metric.

#     config:
#         shared with the original similarity metric.

#     usage:
#         metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
#         metric(response_set)

#     inheritance guidelines:
#         implement __init__ only

#     inheritance example:
#         see CosineSimilarity2Diversity
#     """

#     def __init__(self):
#         self.similarity_metric = SentBertSimilarity()

#     def __call__(self, response_set):
#         similarity_list = []
#         for i in tqdm(range(len(response_set))):
#             for j in range(i):
#                 similarity_list.append(
#                     self.similarity_metric(response_set[i], response_set[j])
#                 )
#         diversity_score = 1 - np.mean(similarity_list)
#         return diversity_score


# class AveragedNgramDiversityMetric:
#     """
#     Calculates the mean values of an n-gram based diversity metric in range n in [n_min, n_max].

#     config:
#         shared with the original n-gram metric.
#         n_min(int) > 0 - Specify the lowest n-gram value to be averaged
#         n_max(int) > 0 - Specify the highest n-gram value to be averaged

#     usage:
#         metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
#         metric(response_set)

#     inheritance guidelines:
#         implement __init__ only

#     inheritance example:
#         see AveragedDistinctNgrams
#     """

#     def __init__(self, n_min, n_max):
#         # add n field
#         self.n_min = n_min
#         self.n_max = n_max

#     def __call__(self, response_set):
#         ngrams_results = []
#         num_set = len(response_set)
#         for i in range(len(response_set[0])):
#             for n in range(self.n_min, self.n_max + 1):
#                 result = self.calculate_distinct_n(
#                     [response_set[j][i] for j in range(num_set)], n
#                 )
#                 ngrams_results.append(result)
#         return np.mean(ngrams_results)

#     def calculate_distinct_n(self, responses, n):
#         all_ngrams = []
#         for response in responses:
#             tokens = word_tokenize(response)
#             response_ngrams = list(ngrams(tokens, n))
#             all_ngrams.extend(response_ngrams)
#         unique_ngrams = len(set(all_ngrams))
#         total_ngrams = len(all_ngrams)

#         return unique_ngrams / total_ngrams if total_ngrams > 0 else 0


# class SelfBLEUMetric:
#     def __call__(self, response_set):
#         """Calculate the average Self-BLEU score for a list of texts."""
#         bleu_scores = []
#         k = len(response_set)
#         for i in range(len(response_set[0])):
#             texts = [response_set[j][i] for j in range(k)]
#             bleu_scores.append(self.calculate_bleu_score(texts))

#         return np.mean(bleu_scores)

#     def calculate_bleu_score(self, texts):
#         bleu_scores = []
#         for i in range(len(texts)):
#             # Treat the current text as the hypothesis
#             hypothesis = texts[i]
#             # Treat all other texts as references
#             references = texts[:i] + texts[i + 1 :]

#             if references:  # Ensure there are references to compare against
#                 bleu_score = sacrebleu.corpus_bleu([hypothesis], [references])
#                 bleu_scores.append(bleu_score.score)

#         # Compute the average BLEU score
#         average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
#         return average_bleu


# def main():
#     parser = HfArgumentParser((AllArguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(args.__dict__)

#     if os.path.exists(args.response_path.replace(".json", "-cleaned.json")):
#         args.response_path = args.response_path.replace(".json", "-cleaned.json")

#     if args.response_path.endswith("-cleaned.json"):
#         response_set = json.load(open(args.response_path, "r"))
#     else:
#         data = json.load(open(args.response_path, "r"))

#         tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
#         if args.detokenizer_path is not None:
#             detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
#         else:
#             detokenizer = None

#         response_set = []
#         for i in tqdm(range(len(data))):
#             n = len(data[i]["answer"])
#             if len(response_set) == 0:
#                 response_set = [[] for _ in range(n)]
#             else:
#                 assert len(response_set) == n
#             for j in range(n):
#                 x = data[i]
#                 if detokenizer:
#                     prompt_str = (
#                         detokenizer.decode(
#                             detokenizer.encode(x["prompt"]), skip_special_tokens=True
#                         )
#                         .replace("user\n\n", "")
#                         .replace("assistant\n\n", "")
#                     )
#                 else:
#                     prompt_str = x["prompt"]
#                 if detokenizer:
#                     # ans_str = detokenizer.decode(
#                     #     detokenizer.encode(data[i]["answer"][j]), skip_special_tokens=True
#                     # )
#                     ans_str = data[i]["answer"][j].replace("<|eot_id|>", "")
#                 else:
#                     ans_str = data[i]["answer"][j]
#                 chat = [
#                     {
#                         "role": "user",
#                         "content": prompt_str,
#                     },
#                     {"role": "assistant", "content": ans_str},
#                 ]
#                 res = tokenizer.apply_chat_template(chat, tokenize=False)
#                 response_set[j].append(res)
#         json.dump(
#             response_set,
#             open(args.response_path.replace(".json", "-cleaned.json"), "w"),
#             indent=2,
#         )

#         response_set = json.load(
#             open(args.response_path.replace(".json", "-cleaned.json"), "r")
#         )
#         print("Finished Data Preparation.")

#     evaluation_results = {
#         "sentbert_diversity_score": None,
#         "bleu_diversity_score": None,
#         "averaged_ngram_diversity_score": None,
#     }

#     print("Calculating N-gram diversity score...")
#     metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
#     diversity_score = metric(response_set)
#     evaluation_results["averaged_ngram_diversity_score"] = np.round(
#         diversity_score * 100, 2
#     )
#     print("N-gram diversity score: {}".format(diversity_score))

#     print("Calculating BLEU similarity score...")
#     metric = SelfBLEUMetric()
#     similarity_score = metric(response_set)
#     evaluation_results["bleu_diversity_score"] = np.round(100 - similarity_score, 2)
#     print("BLEU similarity score: {}".format(100 - similarity_score))

#     print("Calculating Bert diversity score...")
#     metric = SentBertDiversity()
#     diversity_score = metric(response_set)
#     evaluation_results["sentbert_diversity_score"] = np.round(diversity_score * 100, 2)
#     print("Bert diversity score: {}".format(diversity_score))

#     pprint(evaluation_results)


# if __name__ == "__main__":
#     main()



# #################
# # This code is modified from https://github.com/facebookresearch/rlfh-gen-div
# #################
# import os
# import json
# from dataclasses import dataclass, field
# from pprint import pprint
# from typing import List

# import numpy as np
# import torch
# import sacrebleu  # type: ignore
# import sentence_transformers  # type: ignore
# from tqdm import tqdm

# from transformers import HfArgumentParser, AutoTokenizer

# from nltk.util import ngrams  # type: ignore
# from nltk import word_tokenize  # type: ignore
# from collections import Counter


# @dataclass
# class AllArguments:
#     response_path: str = field(
#         default="./results/responses.json",
#         metadata={"help": "Response path (json file)."},
#     )
#     tokenizer_path: str = field(default=None)
#     detokenizer_path: str = field(default=None)


# def _safe_tokenize(text: str) -> List[str]:
#     try:
#         return word_tokenize(text)
#     except LookupError:
#         # Fallback if NLTK punkt is not available
#         return text.split()


# class SentBertSimilarity:
#     def __init__(self):
#         self.model_name = "bert-large-nli-stsb-mean-tokens"  # follow repo
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = sentence_transformers.SentenceTransformer(
#             self.model_name, device=device
#         )

#     def sent_bert_cosine_similarity(self, resps_1, resps_2):
#         embeds_1 = self.model.encode(
#             resps_1, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
#         )
#         embeds_2 = self.model.encode(
#             resps_2, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
#         )
#         # cosine on normalized tensors
#         embeds_1 = torch.nn.functional.normalize(embeds_1, dim=1)
#         embeds_2 = torch.nn.functional.normalize(embeds_2, dim=1)
#         cosine = (embeds_1 * embeds_2).sum(dim=1)
#         return cosine.detach().cpu().numpy()

#     def __call__(self, resp_a, resp_b):
#         return self.sent_bert_cosine_similarity(resp_a, resp_b)


# class SentBertDiversity:
#     """
#     1 - average pairwise SBERT cosine across the N samples (averaged over prompts).
#     Follows the reduction described in (Ghazvininejad et al., 2020) Section 5.
#     """

#     def __init__(self):
#         self.similarity_metric = SentBertSimilarity()

#     def __call__(self, response_set):
#         similarity_list = []
#         # response_set is a list of length N (samples),
#         # each entry is a list of strings over prompts.
#         for i in tqdm(range(len(response_set))):
#             for j in range(i):
#                 similarity_list.append(
#                     self.similarity_metric(response_set[i], response_set[j])
#                 )
#         # similarity_list is a list of arrays (per-prompt similarities)
#         diversity_score = 1 - np.mean(similarity_list) if similarity_list else 0.0
#         return diversity_score


# class AveragedNgramDiversityMetric:
#     """
#     Mean of Distinct-n across n in [n_min, n_max], averaged over prompts.
#     """

#     def __init__(self, n_min: int, n_max: int):
#         self.n_min = n_min
#         self.n_max = n_max

#     def __call__(self, response_set):
#         ngrams_results = []
#         num_set = len(response_set)          # N (samples)
#         num_prompts = len(response_set[0])   # M (prompts)
#         for i in range(num_prompts):
#             # collect N responses for prompt i
#             per_prompt = [response_set[j][i] for j in range(num_set)]
#             for n in range(self.n_min, self.n_max + 1):
#                 ngrams_results.append(self.calculate_distinct_n(per_prompt, n))
#         return float(np.mean(ngrams_results)) if ngrams_results else 0.0

#     def calculate_distinct_n(self, responses, n):
#         all_ngrams = []
#         for response in responses:
#             tokens = _safe_tokenize(response)
#             response_ngrams = list(ngrams(tokens, n)) if len(tokens) >= n else []
#             all_ngrams.extend(response_ngrams)
#         unique_ngrams = len(set(all_ngrams))
#         total_ngrams = len(all_ngrams)
#         return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0


# class SelfBLEUMetric:
#     """Average Self-BLEU over the N responses per prompt, then averaged over prompts."""

#     def __call__(self, response_set):
#         bleu_scores = []
#         k = len(response_set)               # N samples
#         num_prompts = len(response_set[0])  # M prompts
#         for i in range(num_prompts):
#             texts = [response_set[j][i] for j in range(k)]
#             bleu_scores.append(self.calculate_bleu_score(texts))
#         return float(np.mean(bleu_scores)) if bleu_scores else 0.0

#     def calculate_bleu_score(self, texts):
#         scores = []
#         for i in range(len(texts)):
#             hyp = [texts[i]]
#             refs = [texts[:i] + texts[i + 1 :]]  # sacrebleu expects list of ref lists
#             if refs[0]:  # there are references
#                 scores.append(sacrebleu.corpus_bleu(hyp, refs).score)
#         return sum(scores) / len(scores) if scores else 0.0


# def main():
#     parser = HfArgumentParser((AllArguments,))
#     (args,) = parser.parse_args_into_dataclasses()
#     pprint(vars(args))

#     # If we've already cleaned once, reuse it
#     cleaned_path = args.response_path.replace(".json", "-cleaned.json")
#     if os.path.exists(cleaned_path):
#         args.response_path = cleaned_path

#     if args.response_path.endswith("-cleaned.json"):
#         response_set = json.load(open(args.response_path, "r"))
#     else:
#         data = json.load(open(args.response_path, "r"))

#         tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
#         detok = AutoTokenizer.from_pretrained(args.detokenizer_path) if args.detokenizer_path else None

#         # Build response_set as a list of N lists, each length M prompts.
#         response_set = []
#         for i in tqdm(range(len(data))):
#             # Number of responses per prompt; accept 'answers' or 'answer'
#             if "answers" in data[i]:
#                 ans_list = data[i]["answers"]
#             else:
#                 ans_list = data[i]["answer"]
#             n = len(ans_list)

#             if len(response_set) == 0:
#                 response_set = [[] for _ in range(n)]
#             else:
#                 assert len(response_set) == n, "All items must have same N"

#             # choose prompt field
#             x = data[i]
#             if detok:
#                 prompt_str = detok.decode(detok.encode(x["prompt"]), skip_special_tokens=True)
#                 prompt_str = prompt_str.replace("user\n\n", "").replace("assistant\n\n", "")
#             else:
#                 prompt_str = x["prompt"]

#             for j in range(n):
#                 if detok:
#                     ans_str = detok.decode(detok.encode(ans_list[j]), skip_special_tokens=True)
#                 else:
#                     ans_str = ans_list[j]
#                 ans_str = ans_str.replace("<|eot_id|>", "")

#                 chat = [
#                     {"role": "user", "content": prompt_str},
#                     {"role": "assistant", "content": ans_str},
#                 ]
#                 # match repo behavior: compute metrics on chat-templated strings
#                 res = tokenizer.apply_chat_template(chat, tokenize=False)
#                 response_set[j].append(res)

#         json.dump(response_set, open(cleaned_path, "w"), indent=2, ensure_ascii=False)
#         print("Finished Data Preparation.")
#         response_set = json.load(open(cleaned_path, "r"))

#     evaluation_results = {
#         "averaged_ngram_diversity_score": None,
#         "bleu_diversity_score": None,
#         "sentbert_diversity_score": None,
#     }

#     print("Calculating N-gram diversity score...")
#     ngram_metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
#     ngram_div = ngram_metric(response_set)
#     evaluation_results["averaged_ngram_diversity_score"] = round(ngram_div * 100, 2)
#     print(f"N-gram diversity score: {ngram_div}")

#     print("Calculating BLEU similarity score...")
#     bleu_metric = SelfBLEUMetric()
#     bleu_sim = bleu_metric(response_set)
#     evaluation_results["bleu_diversity_score"] = round(100 - bleu_sim, 2)
#     print(f"BLEU diversity score: {100 - bleu_sim}")

#     print("Calculating Bert diversity score...")
#     sentbert_metric = SentBertDiversity()
#     sbert_div = sentbert_metric(response_set)
#     evaluation_results["sentbert_diversity_score"] = round(sbert_div * 100, 2)
#     print(f"Bert diversity score: {sbert_div}")

#     pprint(evaluation_results)


# if __name__ == "__main__":
#     main()



#################
# This code is modified from https://github.com/facebookresearch/rlfh-gen-div
#################
import os
from dataclasses import dataclass, field
import json
from pprint import pprint

import torch
import numpy as np
import sentence_transformers
from tqdm import tqdm

# from sklearn.metrics.pairwise import cosine_similarity
from transformers import set_seed, HfArgumentParser, AutoTokenizer

from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter

import sacrebleu

@dataclass
class AllArguments:
    response_path: str = field(
        default="./results/responses", metadata={"help": "Response path (json file)."}
    )

    tokenizer_path: str = field(default=None)
    detokenizer_path: str = field(default=None)


class SentBertSimilarity:
    def __init__(self):

        self.model_name = "bert-large-nli-stsb-mean-tokens"  # FIXME - hard coded
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

    # @functools.cache
    def embed(self, sentence):
        return self.model.encode(sentence)

    # @functools.cache
    def sent_bert_cosine_similarity(self, resps_1, resps_2):
        embeds_1 = self.model.encode(
            resps_1, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
        )
        embeds_2 = self.model.encode(
            resps_2, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
        )

        if torch.cuda.is_available():
            embeds_1 = embeds_1.to(torch.device("cuda"))
            embeds_2 = embeds_2.to(torch.device("cuda"))

        dot_product = (embeds_1 * embeds_2).sum(dim=1)

        # Calculate cosine similarity
        cosine_similarity = dot_product / (embeds_1.norm(dim=1) * embeds_2.norm(dim=1))

        return cosine_similarity.detach().cpu().numpy()

    def __call__(self, resp_a, resp_b):
        return self.sent_bert_cosine_similarity(resp_a, resp_b)


class SentBertDiversity:
    """
    Implements the diversity to similarity reduction specified on section 5 in the paper
    (https://arxiv.org/pdf/2004.02990.pdf)
    for any similarity metric.

    config:
        shared with the original similarity metric.

    usage:
        metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see CosineSimilarity2Diversity
    """

    def __init__(self):
        self.similarity_metric = SentBertSimilarity()

    def __call__(self, response_set):
        similarity_list = []
        for i in tqdm(range(len(response_set))):
            for j in range(i):
                similarity_list.append(
                    self.similarity_metric(response_set[i], response_set[j])
                )
        diversity_score = 1 - np.mean(similarity_list)
        return diversity_score


class AveragedNgramDiversityMetric:
    """
    Calculates the mean values of an n-gram based diversity metric in range n in [n_min, n_max].

    config:
        shared with the original n-gram metric.
        n_min(int) > 0 - Specify the lowest n-gram value to be averaged
        n_max(int) > 0 - Specify the highest n-gram value to be averaged

    usage:
        metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see AveragedDistinctNgrams
    """

    def __init__(self, n_min, n_max):
        # add n field
        self.n_min = n_min
        self.n_max = n_max

    def __call__(self, response_set):
        ngrams_results = []
        num_set = len(response_set)
        for i in range(len(response_set[0])):
            for n in range(self.n_min, self.n_max + 1):
                result = self.calculate_distinct_n(
                    [response_set[j][i] for j in range(num_set)], n
                )
                ngrams_results.append(result)
        return np.mean(ngrams_results)

    def calculate_distinct_n(self, responses, n):
        all_ngrams = []
        for response in responses:
            tokens = word_tokenize(response)
            response_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(response_ngrams)
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0


class SelfBLEUMetric:
    def __call__(self, response_set):
        """Calculate the average Self-BLEU score for a list of texts."""
        bleu_scores = []
        k = len(response_set)
        for i in range(len(response_set[0])):
            texts = [response_set[j][i] for j in range(k)]
            bleu_scores.append(self.calculate_bleu_score(texts))

        return np.mean(bleu_scores)

    def calculate_bleu_score(self, texts):
        bleu_scores = []
        for i in range(len(texts)):
            # Treat the current text as the hypothesis
            hypothesis = texts[i]
            # Treat all other texts as references
            references = texts[:i] + texts[i + 1 :]

            if references:  # Ensure there are references to compare against
                bleu_score = sacrebleu.corpus_bleu([hypothesis], [references])
                bleu_scores.append(bleu_score.score)

        # Compute the average BLEU score
        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        return average_bleu


def main():
    parser = HfArgumentParser((AllArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if os.path.exists(args.response_path.replace(".json", "-cleaned.json")):
        args.response_path = args.response_path.replace(".json", "-cleaned.json")

    if args.response_path.endswith("-cleaned.json"):
        response_set = json.load(open(args.response_path, "r"))
    else:
        data = json.load(open(args.response_path, "r"))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if args.detokenizer_path is not None:
            detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
        else:
            detokenizer = None

        response_set = []
        for i in tqdm(range(len(data))):
            n = len(data[i]["answer"])
            if len(response_set) == 0:
                response_set = [[] for _ in range(n)]
            else:
                assert len(response_set) == n
            for j in range(n):
                x = data[i]
                if detokenizer:
                    prompt_str = (
                        detokenizer.decode(
                            detokenizer.encode(x["prompt"]), skip_special_tokens=True
                        )
                        .replace("user\n\n", "")
                        .replace("assistant\n\n", "")
                    )
                else:
                    prompt_str = x["prompt"]
                if detokenizer:
                    # ans_str = detokenizer.decode(
                    #     detokenizer.encode(data[i]["answer"][j]), skip_special_tokens=True
                    # )
                    ans_str = data[i]["answer"][j].replace("<|eot_id|>", "")
                else:
                    ans_str = data[i]["answer"][j]
                chat = [
                    {
                        "role": "user",
                        "content": prompt_str,
                    },
                    {"role": "assistant", "content": ans_str},
                ]
                res = tokenizer.apply_chat_template(chat, tokenize=False)
                response_set[j].append(res)
        json.dump(
            response_set,
            open(args.response_path.replace(".json", "-cleaned.json"), "w"),
            indent=2,
        )

        response_set = json.load(
            open(args.response_path.replace(".json", "-cleaned.json"), "r")
        )
        print("Finished Data Preparation.")

    evaluation_results = {
        "sentbert_diversity_score": None,
        "bleu_diversity_score": None,
        "averaged_ngram_diversity_score": None,
    }

    print("Calculating N-gram diversity score...")
    metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
    diversity_score = metric(response_set)
    evaluation_results["averaged_ngram_diversity_score"] = np.round(
        diversity_score * 100, 2
    )
    print("N-gram diversity score: {}".format(diversity_score))

    print("Calculating BLEU similarity score...")
    metric = SelfBLEUMetric()
    similarity_score = metric(response_set)
    evaluation_results["bleu_diversity_score"] = np.round(100 - similarity_score, 2)
    print("BLEU similarity score: {}".format(100 - similarity_score))

    print("Calculating Bert diversity score...")
    metric = SentBertDiversity()
    diversity_score = metric(response_set)
    evaluation_results["sentbert_diversity_score"] = np.round(diversity_score * 100, 2)
    print("Bert diversity score: {}".format(diversity_score))

    pprint(evaluation_results)


if __name__ == "__main__":
    main()