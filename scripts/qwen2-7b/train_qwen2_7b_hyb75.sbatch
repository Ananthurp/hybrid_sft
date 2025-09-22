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
        logger.info(f"Using default SFTTrainer for '{training_args.loss}' loss.")
        if version.parse(transformers.__version__) >= version.parse("4.52.4"):
            from sft_trainer_v2 import SFTTrainer
        else:
            from sft_trainer import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8),
            preprocess_logits_for_metrics=None,
            compute_metrics=None,
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