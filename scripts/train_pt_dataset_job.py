# /// script
# dependencies = [
#   "accelerate>=1.0.0",
#   "bitsandbytes>=0.43.0",
#   "datasets>=3.0.0",
#   "huggingface-hub>=0.30.0",
#   "peft>=0.13.0",
#   "torchvision>=0.20.0",
#   "transformers>=4.56.0",
#   "trl>=0.12.0",
#   "pyyaml>=6.0.2",
#   "wandb>=0.18.0",
# ]
# ///

"""Hugging Face Jobs entry point for PT dataset training.

This script is the remote training layer of the project architecture: it loads a
published pruning-transition dataset from the Hub, trains a LoRA adapter via TRL
on prompt/completion rows derived from canonical PT rows, logs metrics to W&B, and
pushes the adapter model to the Hub. It is model-agnostic — BASE_MODEL, dataset
repo, and all hyperparameters come from the training config YAML or env overrides.
It runs as a uv PEP 723 script on Hugging Face Jobs with HF_TOKEN and WANDB_API_KEY
supplied as secrets; the training config path is set via TRAINING_CONFIG env var.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import whoami
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


TRAINING_CONFIG = Path(os.environ.get("TRAINING_CONFIG", "configs/train/training_gemma4_gsm8k_100.yaml"))


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _value(config: dict[str, Any], env_name: str, config_name: str, default: str) -> str:
    if env_name in os.environ:
        return os.environ[env_name]
    if config_name in config:
        return str(config[config_name])
    return default


def _int_value(config: dict[str, Any], env_name: str, config_name: str, default: int) -> int:
    return int(_value(config, env_name, config_name, str(default)))


def _float_value(config: dict[str, Any], env_name: str, config_name: str, default: float) -> float:
    return float(_value(config, env_name, config_name, str(default)))


def _report_to(config: dict[str, Any]) -> list[str]:
    if "REPORT_TO" in os.environ:
        return [item.strip() for item in os.environ["REPORT_TO"].split(",") if item.strip()]
    value = config.get("report_to", ["wandb"])
    if isinstance(value, str):
        return [value]
    return list(value)


config = _load_config(TRAINING_CONFIG)

BASE_MODEL = _value(config, "BASE_MODEL", "base_model", "google/gemma-4-E2B-it")
MAX_STEPS = _int_value(config, "MAX_STEPS", "max_steps", 200)
LEARNING_RATE = _float_value(config, "LEARNING_RATE", "learning_rate", 1e-4)
PER_DEVICE_TRAIN_BATCH_SIZE = _int_value(
    config, "PER_DEVICE_TRAIN_BATCH_SIZE", "per_device_train_batch_size", 1
)
GRADIENT_ACCUMULATION_STEPS = _int_value(
    config, "GRADIENT_ACCUMULATION_STEPS", "gradient_accumulation_steps", 4
)
MAX_LENGTH = _int_value(config, "MAX_LENGTH", "max_length", 1024)
WANDB_PROJECT = _value(config, "WANDB_PROJECT", "project", "reasoning-pruning")
RUN_NAME = _value(config, "RUN_NAME", "run_name", "pt-dataset-training")
REPORT_TO = _report_to(config)
OUTPUT_DIR = _value(config, "OUTPUT_DIR", "output_dir", "pt-trained-adapter")
token = os.environ.get("HF_TOKEN")
namespace = whoami(token=token)["name"] if token else "local"
HUB_MODEL_ID = _value(
    config,
    "HUB_MODEL_ID",
    "hub_model_id",
    f"{namespace}/pt-trained-adapter",
)
HUB_DATASET_ID = _value(
    config,
    "HUB_DATASET_ID",
    "hub_dataset_id",
    f"{namespace}/reasoning-pruning-pt-dataset",
)
# config_name for the training subset (published with config_name="training", split="train")
HUB_DATASET_CONFIG = _value(config, "HUB_DATASET_CONFIG", "dataset_config", "training")
HUB_DATASET_REVISION = os.environ.get("HUB_DATASET_REVISION")


dataset = load_dataset(
    HUB_DATASET_ID,
    HUB_DATASET_CONFIG,
    split="train",
    revision=HUB_DATASET_REVISION,
    token=token,
)

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_RUN_GROUP"] = os.environ.get("ROUND_ID", "reasoning-pruning")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
    ),
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        hub_model_id=HUB_MODEL_ID,
        push_to_hub=bool(token),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_steps=MAX_STEPS,
        bf16=True,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        report_to=REPORT_TO,
        run_name=RUN_NAME,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "quantization_config": quantization_config,
        },
    ),
)

trainer.train()
if token:
    trainer.push_to_hub(commit_message=f"Train PT-dataset adapter: {RUN_NAME}")
