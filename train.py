# #!/usr/bin/env python
# # coding=utf-8
# """
# This file is modified from the huggingface example for finetuning language models
# [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
# """

# import logging
# import os
# from sparsemax_trainer import SparsemaxSFTTrainer
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# import sys
# from typing import Optional
# from functools import partial
# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from dataclasses import dataclass, field
# from typing import Optional, List, Union

# import transformers
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint

# from packaging import version

# if version.parse(transformers.__version__) >= version.parse("4.46.0"):
#     from sft_trainer_v2 import SFTTrainer
# else:
#     from sft_trainer import SFTTrainer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(
#         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "gem_triton"]}
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM. A value between 0 and 1. A value close to 1.0 makes GEM behave more like CE, while a value close to 0.0 preserves more diversity."})
#     gem_h: str = field(
#         default="linear", metadata={"help": "Function $h$ in GEM. The 'logsigmoid' function is more adaptive, but the difference between 'logsigmoid' and 'linear' is usually negligible.", "choices": ["logsigmoid", "linear"]}
#     )
#     print_entropy: bool = field(
#         default=False, metadata={"help": "Print entropy during training"}
#     )


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_flash_attn: bool = field(
#         default=True,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     test_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         training_args,
#         data_args,
#         model_args,
#         train_tokenized_file,
#     ):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args

#         raw_datasets = load_dataset(
#             "json",
#             data_files=[train_tokenized_file],
#             cache_dir=self.model_args.cache_dir,
#         )
#         self.data = raw_datasets["train"]

#         if self.data_args.max_train_samples is not None:
#             max_samples = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(max_samples))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         example = self.data[item]
#         assert "input_ids" in example
#         assert "labels" in example
#         example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
#         return example


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(
#             json_file=os.path.abspath(sys.argv[1])
#         )
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         # The default of training_args.log_level is passive, so we set log level at info here to have that default.
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # Log on each process the small summary:
#     global_rank = dist.get_rank()
#     logger.warning(
#         f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#     )
#     logger.info(f"Training parameters {training_args}")

#     # Set seed before initializing model.
#     set_seed(training_args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=(
#             "flash_attention_2" if model_args.use_flash_attn else "eager"
#         ),
#     )

#     # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
#     # on a small vocab and want a smaller embedding size, remove this test.
#     # gather deepspeed to get "real" embedding size
#     embeddings = model.get_input_embeddings()
#     with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#         embedding_size = embeddings.weight.shape[0]
#     # resize does its own gather
#     if len(tokenizer) > embedding_size:
#         # pad to multiple for tensor cores.
#         logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     # set up datasets
#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     if data_args.test_tokenized_file:
#         test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
#     else:
#         test_dataset = None

#     # initalize a trainer
#     # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
#     # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         tokenizer=tokenizer,
#         data_collator=DataCollatorForSeq2Seq(
#             tokenizer=tokenizer, model=model, padding="longest"
#         ),
#         preprocess_logits_for_metrics=None,
#         compute_metrics=None,
#     )

#     # Training
#     logger.info("*** Train ***")
#     checkpoint = None
#     if training_args.resume_from_checkpoint is not None:
#         checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()  # Saves the tokenizer too for easy upload

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

# import logging
# import os

# # Import our new custom trainer
# from sparsemax_trainer import SparsemaxSFTTrainer

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# import sys
# from typing import Optional
# from functools import partial
# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from dataclasses import dataclass, field
# from typing import Optional, List, Union

# import transformers
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint

# from packaging import version

# # We will handle the SFTTrainer import inside the main function logic
# # if version.parse(transformers.__version__) >= version.parse("4.46.0"):
# #     from sft_trainer_v2 import SFTTrainer
# # else:
# #     from sft_trainer import SFTTrainer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     # --- CHANGE #1: Added "sparsemax" as a valid choice ---
#     loss: str = field(
#         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "gem_triton", "sparsemax"]}
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM. A value between 0 and 1. A value close to 1.0 makes GEM behave more like CE, while a value close to 0.0 preserves more diversity."})
#     gem_h: str = field(
#         default="linear", metadata={"help": "Function $h$ in GEM. The 'logsigmoid' function is more adaptive, but the difference between 'logsigmoid' and 'linear' is usually negligible.", "choices": ["logsigmoid", "linear"]}
#     )
#     print_entropy: bool = field(
#         default=False, metadata={"help": "Print entropy during training"}
#     )


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_flash_attn: bool = field(
#         default=True,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     test_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         training_args,
#         data_args,
#         model_args,
#         train_tokenized_file,
#     ):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args

#         raw_datasets = load_dataset(
#             "json",
#             data_files=[train_tokenized_file],
#             cache_dir=self.model_args.cache_dir,
#         )
#         self.data = raw_datasets["train"]

#         if self.data_args.max_train_samples is not None:
#             max_samples = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(max_samples))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         example = self.data[item]
#         assert "input_ids" in example
#         assert "labels" in example
#         example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
#         return example


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(
#             json_file=os.path.abspath(sys.argv[1])
#         )
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         # The default of training_args.log_level is passive, so we set log level at info here to have that default.
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # Log on each process the small summary:
#     global_rank = dist.get_rank()
#     logger.warning(
#         f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#     )
#     logger.info(f"Training parameters {training_args}")

#     # Set seed before initializing model.
#     set_seed(training_args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=(
#             "flash_attention_2" if model_args.use_flash_attn else "eager"
#         ),
#     )

#     # We resize the embeddings only when necessary to avoid index errors.
#     embeddings = model.get_input_embeddings()
#     with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#         embedding_size = embeddings.weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     # set up datasets
#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     if data_args.test_tokenized_file:
#         test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
#     else:
#         test_dataset = None

#     # --- CHANGE #2: Conditional Trainer Initialization ---
#     if training_args.loss == "sparsemax":
#         logger.info("Using SparsemaxSFTTrainer for Fenchel-Young loss.")
#         trainer = SparsemaxSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     else:
#         # Default behavior for "gem" and "ce"
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.46.0"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
        
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )
#     # --- END OF CHANGE ---


#     # Training
#     logger.info("*** Train ***")
#     checkpoint = None
#     if training_args.resume_from_checkpoint is not None:
#         checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()  # Saves the tokenizer too for easy upload

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python
# # coding=utf-8
# """
# This file is modified from the huggingface example for finetuning language models
# [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
# """

# import logging
# import os

# # Import our custom trainers
# from sparsemax_trainer import SparsemaxSFTTrainer
# from hybrid_trainer import HybridSFTTrainer

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# import sys
# from typing import Optional
# from functools import partial
# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from dataclasses import dataclass, field
# from typing import Optional, List, Union

# import transformers
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint

# from packaging import version

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     # --- CHANGE #1: Added "hybrid" as a valid choice ---
#     loss: str = field(
#         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "gem_triton", "sparsemax", "hybrid"]}
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(
#         default="linear", metadata={"help": "Function $h$ in GEM.", "choices": ["logsigmoid", "linear"]}
#     )
#     print_entropy: bool = field(
#         default=False, metadata={"help": "Print entropy during training"}
#     )
#     # --- CHANGE #2: Added new arguments for the Hybrid Loss ---
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for the Negative Sampling loss component."})
#     ns_tau: float = field(default=0.1, metadata={"help": "Threshold (tau) for Negative Sampling."})


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_flash_attn: bool = field(
#         default=True,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     test_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         training_args,
#         data_args,
#         model_args,
#         train_tokenized_file,
#     ):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args

#         raw_datasets = load_dataset(
#             "json",
#             data_files=[train_tokenized_file],
#             cache_dir=self.model_args.cache_dir,
#         )
#         self.data = raw_datasets["train"]

#         if self.data_args.max_train_samples is not None:
#             max_samples = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(max_samples))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         example = self.data[item]
#         assert "input_ids" in example
#         assert "labels" in example
#         example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
#         return example


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(
#             json_file=os.path.abspath(sys.argv[1])
#         )
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # Log on each process the small summary:
#     global_rank = dist.get_rank()
#     logger.warning(
#         f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#     )
#     logger.info(f"Training parameters {training_args}")

#     # Set seed before initializing model.
#     set_seed(training_args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=(
#             "flash_attention_2" if model_args.use_flash_attn else "eager"
#         ),
#     )

#     embeddings = model.get_input_embeddings()
#     with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#         embedding_size = embeddings.weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     if data_args.test_tokenized_file:
#         test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
#     else:
#         test_dataset = None

#     # --- CHANGE #3: Updated Conditional Trainer Initialization ---
#     if training_args.loss == "sparsemax":
#         logger.info("Using SparsemaxSFTTrainer for Fenchel-Young loss.")
#         trainer = SparsemaxSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     elif training_args.loss == "hybrid":
#         logger.info("Using HybridSFTTrainer for Fenchel-Young + Negative Sampling loss.")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     else:
#         # Default behavior for "gem" and "ce"
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.46.0"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
        
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )
#     # --- END OF CHANGE ---

#     # Training
#     logger.info("*** Train ***")
#     checkpoint = None
#     if training_args.resume_from_checkpoint is not None:
#         checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python
# # coding=utf-8
# """
# This file is modified from the huggingface example for finetuning language models
# [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
# """

# import logging
# import os

# # Import our custom trainers
# from sparsemax_trainer import SparsemaxSFTTrainer
# from hybrid_trainer import HybridSFTTrainer

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# import sys
# from typing import Optional
# from functools import partial
# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from dataclasses import dataclass, field
# from typing import Optional, List, Union

# import transformers
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint

# from packaging import version

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(
#         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "gem_triton", "sparsemax", "hybrid"]}
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(
#         default="linear", metadata={"help": "Function $h$ in GEM.", "choices": ["logsigmoid", "linear"]}
#     )
#     print_entropy: bool = field(
#         default=False, metadata={"help": "Print entropy during training"}
#     )
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for the Negative Sampling loss component."})
#     ns_tau: float = field(default=0.1, metadata={"help": "Threshold (tau) for Negative Sampling."})


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_flash_attn: bool = field(
#         default=True,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     test_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         training_args,
#         data_args,
#         model_args,
#         train_tokenized_file,
#     ):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args

#         raw_datasets = load_dataset(
#             "json",
#             data_files=[train_tokenized_file],
#             cache_dir=self.model_args.cache_dir,
#         )
#         self.data = raw_datasets["train"]

#         if self.data_args.max_train_samples is not None:
#             max_samples = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(max_samples))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         example = self.data[item]
#         assert "input_ids" in example
#         assert "labels" in example
#         example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
#         return example


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(
#             json_file=os.path.abspath(sys.argv[1])
#         )
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # --- THIS IS THE CORRECTED BLOCK ---
#     # Log on each process the small summary:
#     # Check if we are in a distributed environment before calling dist.get_rank()
#     is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
#     if is_distributed:
#         global_rank = dist.get_rank()
#         logger.warning(
#             f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         )
#     else:
#         logger.warning(
#             f"Running in single-GPU mode. Device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         )
#     # --- END OF CORRECTION ---
    
#     logger.info(f"Training parameters {training_args}")

#     # Set seed before initializing model.
#     set_seed(training_args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=(
#             "flash_attention_2" if model_args.use_flash_attn else "eager"
#         ),
#     )

#     embeddings = model.get_input_embeddings()
#     if is_distributed:
#         with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#             embedding_size = embeddings.weight.shape[0]
#     else:
#         embedding_size = embeddings.weight.shape[0]
        
#     if len(tokenizer) > embedding_size:
#         logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     if data_args.test_tokenized_file:
#         test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
#     else:
#         test_dataset = None

#     # Conditional Trainer Initialization
#     if training_args.loss == "sparsemax":
#         logger.info("Using SparsemaxSFTTrainer for Fenchel-Young loss.")
#         trainer = SparsemaxSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     elif training_args.loss == "hybrid":
#         logger.info("Using HybridSFTTrainer for Fenchel-Young + Negative Sampling loss.")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     else:
#         # Default behavior for "gem" and "ce"
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.46.0"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
        
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )

#     # Training
#     logger.info("*** Train ***")
#     checkpoint = None
#     if training_args.resume_from_checkpoint is not None:
#         checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# coding=utf-8
# """
# This file is modified from the huggingface example for finetuning language models
# [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
# """

# import logging
# import os

# # Import our custom trainers
# # We only need the hybrid trainer now
# from hybrid_trainer import HybridSFTTrainer

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# import sys
# from typing import Optional
# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from dataclasses import dataclass, field
# from typing import Optional, List, Union
# from typing import Literal 

# import transformers
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint

# from packaging import version

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # @dataclass
# # class TrainingArguments(transformers.TrainingArguments):
# #     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
# #     # --- Simplified loss choices ---
# #     loss: str = field(
# #         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "hybrid"]}
# #     )
# #     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
# #     gem_h: str = field(
# #         default="linear", metadata={"help": "Function $h$ in GEM.", "choices": ["logsigmoid", "linear"]}
# #     )
# #     print_entropy: bool = field(
# #         default=False, metadata={"help": "Print entropy during training"}
# #     )
    
# #     # --- Arguments for Hybrid Loss ---
# #     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for the Negative Sampling loss."})
# #     ns_type: str = field(
# #         default="top_k", metadata={"help": "Type of negative sampling.", "choices": ["top_k", "bottom_p"]}
# #     )
# #     ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
# #     ns_bottom_p: float = field(default=0.9, metadata={"help": "Percentage for bottom-p negative sampling."})


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(
#         default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "hybrid"]}
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(
#         default="linear", metadata={"help": "Function $h$ in GEM.", "choices": ["logsigmoid", "linear"]}
#     )
#     print_entropy: bool = field(
#         default=False, metadata={"help": "Print entropy during training"}
#     )
    
#     # --- Arguments for Hybrid Loss ---
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for the Negative Sampling loss."})
#     ns_type: str = field(
#         default="top_k", metadata={"help": "Type of negative sampling.", "choices": ["top_k", "bottom_p" ,"support_set"]}
#     )
#     ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
#     ns_bottom_p: float = field(default=0.9, metadata={"help": "Percentage for bottom-p negative sampling."})

#     # --- ADD THESE TWO LINES FOR EVALUATION ---
#     evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no")
#     eval_steps: Optional[int] = field(default=None)

    
# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     use_flash_attn: bool = field(
#         default=True,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     test_tokenized_file: str = field(
#         default=None, metadata={"help": "huggingface dataset name or local data path"}
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_seq_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False,
#         metadata={"help": "Overwrite the cached training and evaluation sets"},
#     )


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         training_args,
#         data_args,
#         model_args,
#         train_tokenized_file,
#     ):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args

#         raw_datasets = load_dataset(
#             "json",
#             data_files=[train_tokenized_file],
#             cache_dir=self.model_args.cache_dir,
#         )
#         self.data = raw_datasets["train"]

#         if self.data_args.max_train_samples is not None:
#             max_samples = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(max_samples))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         example = self.data[item]
#         assert "input_ids" in example
#         assert "labels" in example
#         example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
#         return example


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(
#             json_file=os.path.abspath(sys.argv[1])
#         )
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
#     if is_distributed:
#         global_rank = dist.get_rank()
#         logger.warning(
#             f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         )
#     else:
#         logger.warning(
#             f"Running in single-GPU mode. Device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         )
    
#     logger.info(f"Training parameters {training_args}")

#     set_seed(training_args.seed)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=(
#             "flash_attention_2" if model_args.use_flash_attn else "eager"
#         ),
#     )

#     embeddings = model.get_input_embeddings()
#     if is_distributed:
#         with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#             embedding_size = embeddings.weight.shape[0]
#     else:
#         embedding_size = embeddings.weight.shape[0]
        
#     if len(tokenizer) > embedding_size:
#         logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     if data_args.test_tokenized_file:
#         test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
#     else:
#         test_dataset = None

#     # --- Simplified Trainer Initialization ---
#     if training_args.loss == "hybrid":
#         logger.info("Using HybridSFTTrainer for Fenchel-Young + Negative Sampling loss.")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#         )
#     else:
#         # Default behavior for "gem" and "ce"
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.52.4"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
        
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(
#                 tokenizer=tokenizer, model=model, padding="longest"
#             ),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )

#     # Training
#     logger.info("*** Train ***")
#     checkpoint = None
#     if training_args.resume_from_checkpoint is not None:
#         checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python
# # coding=utf-8
# import logging, os, sys
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# from dataclasses import dataclass, field
# from typing import Optional, Literal

# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# import transformers
# from datasets import load_dataset
# from torch.utils.data import Dataset

# from packaging import version
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )

# # Our custom hybrid trainer
# from hybrid_trainer import HybridSFTTrainer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(default="gem", metadata={"help": "Loss name", "choices": ["gem", "ce", "hybrid"]})
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(default="linear", metadata={"help": "Function h in GEM.", "choices": ["logsigmoid", "linear"]})
#     print_entropy: bool = field(default=False, metadata={"help": "Print entropy during training"})

#     # Hybrid loss args
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for Negative Sampling loss."})
#     ns_type: str = field(default="top_k", metadata={"help": "Negative sampling type.", "choices": ["top_k", "bottom_p", "support_set"]})
#     ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
#     ns_bottom_p: float = field(default=0.9, metadata={"help": "BOTTOM-p by COUNT (fraction of vocab to suppress)."})
#     ns_temperature: float = field(default=1.0, metadata={"help": "Temperature applied to sparsemax path (like colleague)."})

#     # Evaluation cadence
#     evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no")
#     eval_steps: Optional[int] = field(default=None)


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(metadata={"help": "HF model path or identifier"})
#     cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir"})
#     use_flash_attn: bool = field(default=True, metadata={"help": "Use FlashAttention-2 when available"})


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(default=None, metadata={"help": "Path to tokenized train jsonl"})
#     test_tokenized_file: Optional[str] = field(default=None, metadata={"help": "Path to tokenized eval jsonl"})
#     max_train_samples: Optional[int] = field(default=None)
#     max_seq_length: Optional[int] = field(default=None)
#     overwrite_cache: bool = field(default=False)


# class CustomDataset(Dataset):
#     def __init__(self, training_args, data_args, model_args, train_tokenized_file):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args
#         raw = load_dataset("json", data_files=[train_tokenized_file], cache_dir=self.model_args.cache_dir)
#         self.data = raw["train"]
#         if self.data_args.max_train_samples is not None:
#             m = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(m))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         ex = self.data[i]
#         assert "input_ids" in ex and "labels" in ex
#         return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Logging setup
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
#     if is_distributed:
#         logger.warning(f"Process rank: {dist.get_rank()}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
#     else:
#         logger.warning(f"Running single-GPU. Device: {training_args.device}, n_gpu: {training_args.n_gpu}")

#     logger.info(f"Training parameters {training_args}")
#     set_seed(training_args.seed)

#     # --- Tokenizer ---
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

#     # Qwen2 + FlashAttention-2 requires LEFT padding for batched forward/eval
#     model_name_lower = model_args.model_name_or_path.lower()
#     if "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
#         if tokenizer.padding_side != "left":
#             logger.info("Forcing tokenizer.padding_side = 'left' for Qwen2 + FlashAttention-2.")
#         tokenizer.padding_side = "left"

#     # LLaMA-3 specific pad-token fallback (keep your original behavior)
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     # Ensure a pad token exists (fallback to EOS or create one)
#     if tokenizer.pad_token_id is None:
#         if tokenizer.eos_token_id is not None:
#             logger.info("Setting pad_token to eos_token for this tokenizer.")
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             logger.info("Adding a new <|pad|> token to tokenizer vocab.")
#             tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

#     # --- Model ---
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=("flash_attention_2" if model_args.use_flash_attn else "eager"),
#         trust_remote_code=True,
#     )

#     # Make sure model/generation config know the pad token id
#     if getattr(model.config, "pad_token_id", None) is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
#     if hasattr(model, "generation_config"):
#         if getattr(model.generation_config, "pad_token_id", None) is None:
#             model.generation_config.pad_token_id = tokenizer.pad_token_id

#     # Recommended with gradient checkpointing to avoid cache incompatibility
#     if getattr(training_args, "gradient_checkpointing", False):
#         try:
#             model.config.use_cache = False
#         except Exception:
#             pass

#     # If adding a pad token expanded the tokenizer, resize embeddings
#     embeddings = model.get_input_embeddings()
#     if is_distributed:
#         with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#             embedding_size = embeddings.weight.shape[0]
#     else:
#         embedding_size = embeddings.weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         logging.warning("len(tokenizer) > embedding_size; resizing token embeddings...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     # --- Datasets ---
#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file) if data_args.test_tokenized_file else None

#     # --- Trainer ---
#     if training_args.loss == "hybrid":
#         logger.info("Using HybridSFTTrainer (FY sparsemax + NS).")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
#         )
#     else:
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.52.4"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )

#     # --- Train ---
#     logger.info("*** Train ***")
#     checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)

#     # LLaMA-3 generation eos fix (keep your original behavior)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python
# # coding=utf-8
# import logging, os, sys
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# from dataclasses import dataclass, field
# from typing import Optional, Literal

# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# import transformers
# from datasets import load_dataset
# from torch.utils.data import Dataset

# from packaging import version
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )

# # Our custom hybrid/ablations trainer
# from hybrid_trainer import HybridSFTTrainer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(
#         default="gem",
#         metadata={
#             "help": "Which training loss to use.",
#             "choices": ["gem", "ce", "hybrid", "sparsemax", "ns_only"],
#         },
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(default="linear", metadata={"help": "Function h in GEM.", "choices": ["logsigmoid", "linear"]})
#     print_entropy: bool = field(default=False, metadata={"help": "Print entropy during training"})

#     # Hybrid / NS args
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for Negative Sampling loss."})
#     ns_type: str = field(default="top_k", metadata={"help": "Negative sampling type.", "choices": ["top_k", "bottom_p", "support_set"]})
#     ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
#     ns_bottom_p: float = field(default=0.9, metadata={"help": "BOTTOM-p by COUNT (fraction of vocab to suppress)."})
#     ns_temperature: float = field(default=1.0, metadata={"help": "Temperature applied to sparsemax path (like colleague)."})

#     # Evaluation cadence
#     evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no")
#     eval_steps: Optional[int] = field(default=None)


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(metadata={"help": "HF model path or identifier"})
#     cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir"})
#     use_flash_attn: bool = field(default=True, metadata={"help": "Use FlashAttention-2 when available"})


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(default=None, metadata={"help": "Path to tokenized train jsonl"})
#     test_tokenized_file: Optional[str] = field(default=None, metadata={"help": "Path to tokenized eval jsonl"})
#     max_train_samples: Optional[int] = field(default=None)
#     max_seq_length: Optional[int] = field(default=None)
#     overwrite_cache: bool = field(default=False)


# class CustomDataset(Dataset):
#     def __init__(self, training_args, data_args, model_args, train_tokenized_file):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args
#         raw = load_dataset("json", data_files=[train_tokenized_file], cache_dir=self.model_args.cache_dir)
#         self.data = raw["train"]
#         if self.data_args.max_train_samples is not None:
#             m = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(m))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         ex = self.data[i]
#         assert "input_ids" in ex and "labels" in ex
#         return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Logging setup
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
#     if is_distributed:
#         logger.warning(f"Process rank: {dist.get_rank()}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
#     else:
#         logger.warning(f"Running single-GPU. Device: {training_args.device}, n_gpu: {training_args.n_gpu}")

#     logger.info(f"Training parameters {training_args}")
#     set_seed(training_args.seed)

#     # --- Tokenizer ---
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

#     # Qwen2 + FlashAttention-2 requires LEFT padding for batched forward/eval
#     model_name_lower = model_args.model_name_or_path.lower()
#     if "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
#         if tokenizer.padding_side != "left":
#             logger.info("Forcing tokenizer.padding_side = 'left' for Qwen2 + FlashAttention-2.")
#         tokenizer.padding_side = "left"

#     # LLaMA-3 specific pad-token fallback
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     # Ensure a pad token exists (fallback to EOS or create one)
#     if tokenizer.pad_token_id is None:
#         if tokenizer.eos_token_id is not None:
#             logger.info("Setting pad_token to eos_token for this tokenizer.")
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             logger.info("Adding a new <|pad|> token to tokenizer vocab.")
#             tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

#     # --- Model ---
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=("flash_attention_2" if model_args.use_flash_attn else "eager"),
#         trust_remote_code=True,
#     )

#     # Make sure model/generation config know the pad token id
#     if getattr(model.config, "pad_token_id", None) is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
#     if hasattr(model, "generation_config"):
#         if getattr(model.generation_config, "pad_token_id", None) is None:
#             model.generation_config.pad_token_id = tokenizer.pad_token_id

#     # Recommended with gradient checkpointing to avoid cache incompatibility
#     if getattr(training_args, "gradient_checkpointing", False):
#         try:
#             model.config.use_cache = False
#         except Exception:
#             pass

#     # If adding a pad token expanded the tokenizer, resize embeddings
#     embeddings = model.get_input_embeddings()
#     if is_distributed:
#         with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#             embedding_size = embeddings.weight.shape[0]
#     else:
#         embedding_size = embeddings.weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         logging.warning("len(tokenizer) > embedding_size; resizing token embeddings...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     # --- Datasets ---
#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file) if data_args.test_tokenized_file else None

#     # --- Trainer ---
#     if training_args.loss in {"hybrid", "sparsemax", "ns_only"}:
#         logger.info(f"Using HybridSFTTrainer for loss='{training_args.loss}'.")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
#         )
#     else:
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
#         if version.parse(transformers.__version__) >= version.parse("4.52.4"):
#             from sft_trainer_v2 import SFTTrainer
#         else:
#             from sft_trainer import SFTTrainer
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
#             preprocess_logits_for_metrics=None,
#             compute_metrics=None,
#         )

#     # --- Train ---
#     logger.info("*** Train ***")
#     checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)

#     # LLaMA-3 generation eos fix (keep your original behavior)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python
# # coding=utf-8
# import logging, os, sys, math
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# from dataclasses import dataclass, field
# from typing import Optional, Literal

# import datasets
# import torch
# import torch.distributed as dist
# import deepspeed
# import transformers
# from datasets import load_dataset
# from torch.utils.data import Dataset

# from packaging import version
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     HfArgumentParser,
#     DataCollatorForSeq2Seq,
#     set_seed,
# )

# # Our custom hybrid/ablations trainer
# from hybrid_trainer import HybridSFTTrainer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
#     loss: str = field(
#         default="gem",
#         metadata={
#             "help": "Which training loss to use.",
#             "choices": ["gem", "ce", "hybrid", "sparsemax", "ns_only"],
#         },
#     )
#     gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
#     gem_h: str = field(default="linear", metadata={"help": "Function h in GEM.", "choices": ["logsigmoid", "linear"]})
#     print_entropy: bool = field(default=False, metadata={"help": "Print entropy during training"})

#     # Hybrid / NS args
#     ns_alpha: float = field(default=0.5, metadata={"help": "Weight for Negative Sampling loss."})
#     ns_type: str = field(default="top_k", metadata={"help": "Negative sampling type.", "choices": ["top_k", "bottom_p", "support_set"]})
#     ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
#     ns_bottom_p: float = field(default=0.9, metadata={"help": "BOTTOM-p by COUNT (fraction of vocab to suppress)."})
#     ns_temperature: float = field(default=1.0, metadata={"help": "Temperature applied to sparsemax path (like colleague)."})

#     # Evaluation cadence
#     evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no")
#     eval_steps: Optional[int] = field(default=None)

#     # --- NEFT (our wrapper; set >0 to enable) ---
#     neft_alpha: float = field(default=0.0, metadata={"help": "NEFT noise scale alpha; 0 disables NEFT"})

#     neft_alpha: float = field(default=0.0, metadata={"help": "NEFT noise scale alpha; 0 disables NEFT"})
#     neft_impl: str = field(
#         default="input_embeds",
#         metadata={"help": "Which NEFT implementation to use.", "choices": ["input_embeds", "official"]},
#     )

# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(metadata={"help": "HF model path or identifier"})
#     cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir"})
#     use_flash_attn: bool = field(default=True, metadata={"help": "Use FlashAttention-2 when available"})


# @dataclass
# class DataArguments:
#     train_tokenized_file: str = field(default=None, metadata={"help": "Path to tokenized train jsonl"})
#     test_tokenized_file: Optional[str] = field(default=None, metadata={"help": "Path to tokenized eval jsonl"})
#     max_train_samples: Optional[int] = field(default=None)
#     max_seq_length: Optional[int] = field(default=None)
#     overwrite_cache: bool = field(default=False)


# class CustomDataset(Dataset):
#     def __init__(self, training_args, data_args, model_args, train_tokenized_file):
#         self.training_args = training_args
#         self.data_args = data_args
#         self.model_args = model_args
#         raw = load_dataset("json", data_files=[train_tokenized_file], cache_dir=self.model_args.cache_dir)
#         self.data = raw["train"]
#         if self.data_args.max_train_samples is not None:
#             m = min(len(self.data), self.data_args.max_train_samples)
#             self.data = self.data.select(range(m))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         ex = self.data[i]
#         assert "input_ids" in ex and "labels" in ex
#         return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


# # --- NEFT: add uniform noise to embeddings, scaled by alpha/sqrt(L*d) ---
# def _add_neft_noise(inputs, model, alpha: float):
#     """
#     NEFT: add iid Uniform[-1,1] noise to input embeddings, scaled by alpha/sqrt(L*d).
#     L = per-sample non-pad length (from attention_mask), d = embedding dim.
#     """
#     if alpha <= 0.0 or "input_ids" not in inputs:
#         return inputs  # either disabled or already using inputs_embeds

#     input_ids = inputs["input_ids"]
#     attn = inputs.get("attention_mask", None)
#     embed = model.get_input_embeddings()(input_ids)  # [B,T,d]
#     B, T, d = embed.shape

#     if attn is not None:
#         lengths = attn.sum(dim=1).clamp(min=1).to(embed.dtype)  # [B]
#     else:
#         lengths = torch.full((B,), T, dtype=embed.dtype, device=embed.device)

#     scales = alpha / torch.sqrt(lengths * d)  # [B]
#     scales = scales.view(B, 1, 1)

#     # sample noise in fp32 for numerical stability, then cast to embed dtype
#     noise = torch.empty_like(embed, dtype=torch.float32).uniform_(-1.0, 1.0)
#     noise = (noise.to(embed.dtype) * scales)
#     if attn is not None:
#         noise = noise * attn.unsqueeze(-1).to(embed.dtype)  # zero noise on pads

#     noisy_embed = embed + noise

#     out = dict(inputs)  # shallow copy
#     out["inputs_embeds"] = noisy_embed
#     out.pop("input_ids")
#     return out


# def build_ce_trainer_with_optional_neft(SFTTrainerCls, model, training_args, train_dataset, test_dataset, tokenizer):
#     class NeftSFTTrainer(SFTTrainerCls):
#         def compute_loss(
#             self,
#             model,
#             inputs,
#             return_outputs: bool = False,
#             num_items_in_batch=None,   # <-- accept HF's kwarg (4.52+)
#         ):
#             if model.training and getattr(self.args, "neft_alpha", 0.0) > 0.0:
#                 inputs = _add_neft_noise(inputs, model, float(self.args.neft_alpha))
#             # forward all args to base class (compat with HF 4.52+)
#             return super().compute_loss(
#                 model,
#                 inputs,
#                 return_outputs=return_outputs,
#                 num_items_in_batch=num_items_in_batch,
#             )

#     trainer_cls = NeftSFTTrainer if getattr(training_args, "neft_alpha", 0.0) > 0.0 else SFTTrainerCls

#     return trainer_cls(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         tokenizer=tokenizer,
#         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
#         preprocess_logits_for_metrics=None,
#         compute_metrics=None,
#     )


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Logging setup
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     if training_args.should_log:
#         transformers.utils.logging.set_verbosity_info()
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
#     if is_distributed:
#         logger.warning(f"Process rank: {dist.get_rank()}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
#     else:
#         logger.warning(f"Running single-GPU. Device: {training_args.device}, n_gpu: {training_args.n_gpu}")

#     logger.info(f"Training parameters {training_args}")
#     set_seed(training_args.seed)

#     # --- Tokenizer ---
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

#     # Qwen2 + FlashAttention-2 requires LEFT padding for batched forward/eval
#     model_name_lower = model_args.model_name_or_path.lower()
#     if "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
#         if tokenizer.padding_side != "left":
#             logger.info("Forcing tokenizer.padding_side = 'left' for Qwen2 + FlashAttention-2.")
#         tokenizer.padding_side = "left"

#     # LLaMA-3 specific pad-token fallback
#     if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
#         tokenizer.pad_token_id = len(tokenizer) - 1
#         tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

#     # Ensure a pad token exists (fallback to EOS or create one)
#     if tokenizer.pad_token_id is None:
#         if tokenizer.eos_token_id is not None:
#             logger.info("Setting pad_token to eos_token for this tokenizer.")
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             logger.info("Adding a new <|pad|> token to tokenizer vocab.")
#             tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

#     # --- Model ---
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype="auto",
#         attn_implementation=("flash_attention_2" if model_args.use_flash_attn else "eager"),
#         trust_remote_code=True,
#     )
#     # If NEFT is requested with the official patch, apply it now.
#     if training_args.neft_alpha > 0 and training_args.neft_impl == "official":
#         from neft_official import NEFTune_safe
#         model = NEFTune_safe(model, noise_alpha=training_args.neft_alpha)
#     # Make sure model/generation config know the pad token id
#     if getattr(model.config, "pad_token_id", None) is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
#     if hasattr(model, "generation_config"):
#         if getattr(model.generation_config, "pad_token_id", None) is None:
#             model.generation_config.pad_token_id = tokenizer.pad_token_id

#     # Recommended with gradient checkpointing to avoid cache incompatibility
#     if getattr(training_args, "gradient_checkpointing", False):
#         try:
#             model.config.use_cache = False
#         except Exception:
#             pass

#     # If adding a pad token expanded the tokenizer, resize embeddings
#     embeddings = model.get_input_embeddings()
#     if is_distributed:
#         with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
#             embedding_size = embeddings.weight.shape[0]
#     else:
#         embedding_size = embeddings.weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         logging.warning("len(tokenizer) > embedding_size; resizing token embeddings...")
#         model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

#     # --- Datasets ---
#     train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
#     test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file) if data_args.test_tokenized_file else None

#     # --- Trainer ---
#     if training_args.loss in {"hybrid", "sparsemax", "ns_only"}:
#         logger.info(f"Using HybridSFTTrainer for loss='{training_args.loss}'.")
#         trainer = HybridSFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=test_dataset,
#             tokenizer=tokenizer,
#             data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
#         )
#     else:
#         logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss (NEFT alpha={training_args.neft_alpha}).")
#         if version.parse(transformers.__version__) >= version.parse("4.52.4"):
#             from sft_trainer_v2 import SFTTrainer as SFTTrainerCls
#         else:
#             from sft_trainer import SFTTrainer as SFTTrainerCls

#         trainer = build_ce_trainer_with_optional_neft(
#             SFTTrainerCls=SFTTrainerCls,
#             model=model,
#             training_args=training_args,
#             train_dataset=train_dataset,
#             test_dataset=test_dataset,
#             tokenizer=tokenizer,
#         )

#     # --- Train ---
#     logger.info("*** Train ***")
#     checkpoint = training_args.resume_from_checkpoint
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)

#     # LLaMA-3 generation eos fix (keep your original behavior)
#     if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
#         model.generation_config.eos_token_id = [128001, 128009]
#     trainer.save_model()

#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# coding=utf-8
import logging, os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from dataclasses import dataclass, field
from typing import Optional, Literal

import datasets
import torch
import torch.distributed as dist
import deepspeed
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
)

# Our custom hybrid/ablations trainer
from hybrid_trainer import HybridSFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Args
# =========================
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})

    loss: str = field(
        default="gem",
        metadata={
            "help": "Which training loss to use.",
            "choices": ["gem", "ce", "hybrid", "sparsemax", "ns_only"],
        },
    )
    gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
    gem_h: str = field(default="linear", metadata={"help": "Function h in GEM.", "choices": ["logsigmoid", "linear"]})
    print_entropy: bool = field(default=False, metadata={"help": "Print entropy during training"})

    # Hybrid / NS args
    ns_alpha: float = field(default=0.5, metadata={"help": "Weight for Negative Sampling loss."})
    ns_type: str = field(default="top_k", metadata={"help": "Negative sampling type.", "choices": ["top_k", "bottom_p", "support_set"]})
    ns_top_k: int = field(default=10, metadata={"help": "K for top-k negative sampling."})
    ns_bottom_p: float = field(default=0.9, metadata={"help": "BOTTOM-p by COUNT (fraction of vocab to suppress)."})
    ns_temperature: float = field(default=1.0, metadata={"help": "Temperature applied to sparsemax path (like colleague)."})

    # Evaluation cadence
    evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no")
    eval_steps: Optional[int] = field(default=None)

    # --- NEFT ---
    neft_alpha: float = field(default=0.0, metadata={"help": "NEFT noise scale alpha; 0 disables NEFT"})
    neft_impl: str = field(
        default="input_embeds",
        metadata={"help": "Which NEFT implementation to use.", "choices": ["input_embeds", "official"]},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model path or identifier"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir"})
    use_flash_attn: bool = field(default=True, metadata={"help": "Use FlashAttention-2 when available"})


@dataclass
class DataArguments:
    train_tokenized_file: str = field(default=None, metadata={"help": "Path to tokenized train jsonl"})
    test_tokenized_file: Optional[str] = field(default=None, metadata={"help": "Path to tokenized eval jsonl"})
    max_train_samples: Optional[int] = field(default=None)
    max_seq_length: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)


# =========================
# Dataset
# =========================
class CustomDataset(Dataset):
    def __init__(self, training_args, data_args, model_args, train_tokenized_file):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        raw = load_dataset("json", data_files=[train_tokenized_file], cache_dir=self.model_args.cache_dir)
        self.data = raw["train"]
        if self.data_args.max_train_samples is not None:
            m = min(len(self.data), self.data_args.max_train_samples)
            self.data = self.data.select(range(m))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        assert "input_ids" in ex and "labels" in ex
        return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


# =========================
# NEFT (input_embeds variant)
# =========================
def _add_neft_noise(inputs, model, alpha: float):
    """
    NEFT: add iid Uniform[-1,1] noise to input embeddings, scaled by alpha/sqrt(L*d).
    L = per-sample non-pad length (from attention_mask), d = embedding dim.
    """
    if alpha <= 0.0 or "input_ids" not in inputs:
        return inputs  # either disabled or already using inputs_embeds

    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask", None)
    embed = model.get_input_embeddings()(input_ids)  # [B,T,d]
    B, T, d = embed.shape

    if attn is not None:
        lengths = attn.sum(dim=1).clamp(min=1).to(embed.dtype)  # [B]
    else:
        lengths = torch.full((B,), T, dtype=embed.dtype, device=embed.device)

    scales = alpha / torch.sqrt(lengths * d)  # [B]
    scales = scales.view(B, 1, 1)

    # sample noise in fp32 for numerical stability, then cast to embed dtype
    noise = torch.empty_like(embed, dtype=torch.float32).uniform_(-1.0, 1.0)
    noise = (noise.to(embed.dtype) * scales)
    if attn is not None:
        noise = noise * attn.unsqueeze(-1).to(embed.dtype)  # zero noise on pads

    noisy_embed = embed + noise

    out = dict(inputs)  # shallow copy
    out["inputs_embeds"] = noisy_embed
    out.pop("input_ids")
    return out


def build_ce_trainer_with_optional_neft(SFTTrainerCls, model, training_args, train_dataset, test_dataset, tokenizer):
    """
    - If neft_impl == "official": use the plain SFT trainer (embedding is already monkey-patched).
    - If neft_impl == "input_embeds": wrap compute_loss to inject noise via inputs_embeds.
    """
    if getattr(training_args, "neft_alpha", 0.0) > 0.0 and getattr(training_args, "neft_impl", "input_embeds") == "official":
        # Plain SFT; official NEFT patch already applied to embedding forward
        return SFTTrainerCls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
            preprocess_logits_for_metrics=None,
            compute_metrics=None,
        )

    class NeftSFTTrainer(SFTTrainerCls):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,   # HF >= 4.52 compatibility
        ):
            if (
                model.training
                and getattr(self.args, "neft_alpha", 0.0) > 0.0
                and getattr(self.args, "neft_impl", "input_embeds") == "input_embeds"
            ):
                inputs = _add_neft_noise(inputs, model, float(self.args.neft_alpha))
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    trainer_cls = NeftSFTTrainer if getattr(training_args, "neft_alpha", 0.0) > 0.0 else SFTTrainerCls
    return trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )


# =========================
# Main
# =========================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    is_distributed = "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1
    if is_distributed:
        logger.warning(f"Process rank: {dist.get_rank()}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
    else:
        logger.warning(f"Running single-GPU. Device: {training_args.device}, n_gpu: {training_args.n_gpu}")

    logger.info(f"Training parameters {training_args}")
    set_seed(training_args.seed)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # Qwen2 + FlashAttention-2 requires LEFT padding for batched forward/eval
    model_name_lower = model_args.model_name_or_path.lower()
    if "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
        if tokenizer.padding_side != "left":
            logger.info("Forcing tokenizer.padding_side = 'left' for Qwen2 + FlashAttention-2.")
        tokenizer.padding_side = "left"

    # LLaMA-3 specific pad-token fallback
    if "llama-3" in tokenizer.name_or_path.lower() and tokenizer.pad_token is None:
        tokenizer.pad_token_id = len(tokenizer) - 1
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

    # Ensure a pad token exists (fallback to EOS or create one)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.info("Setting pad_token to eos_token for this tokenizer.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.info("Adding a new <|pad|> token to tokenizer vocab.")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=("flash_attention_2" if model_args.use_flash_attn else "eager"),
        trust_remote_code=True,
    )

    # Apply "official" NEFT patch (monkey-patch embeddings) if requested
    if training_args.neft_alpha > 0 and training_args.neft_impl == "official":
        from neft_official import NEFTune_safe
        model = NEFTune_safe(model, noise_alpha=training_args.neft_alpha)

    # Make sure model/generation config know the pad token id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Recommended with gradient checkpointing to avoid cache incompatibility
    if getattr(training_args, "gradient_checkpointing", False):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # If adding a pad token expanded the tokenizer, resize embeddings
    embeddings = model.get_input_embeddings()
    if is_distributed:
        with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
            embedding_size = embeddings.weight.shape[0]
    else:
        embedding_size = embeddings.weight.shape[0]
    if len(tokenizer) > embedding_size:
        logging.warning("len(tokenizer) > embedding_size; resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # --- Datasets ---
    train_dataset = CustomDataset(training_args, data_args, model_args, data_args.train_tokenized_file)
    test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file) if data_args.test_tokenized_file else None

    # --- Trainer ---
    if training_args.loss in {"hybrid", "sparsemax", "ns_only"}:
        logger.info(f"Using HybridSFTTrainer for loss='{training_args.loss}'.")
        trainer = HybridSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
        )
    else:
        logger.info(
            "Using default SFTTrainer for '%s' loss (NEFT alpha=%s, impl=%s).",
            training_args.loss, training_args.neft_alpha, training_args.neft_impl
        )
        if version.parse(transformers.__version__) >= version.parse("4.52.4"):
            from sft_trainer_v2 import SFTTrainer as SFTTrainerCls
        else:
            from sft_trainer import SFTTrainer as SFTTrainerCls

        trainer = build_ce_trainer_with_optional_neft(
            SFTTrainerCls=SFTTrainerCls,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
        )

    # --- Train ---
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # LLaMA-3 generation eos fix (keep your original behavior)
    if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
        model.generation_config.eos_token_id = [128001, 128009]
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()