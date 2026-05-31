"""Load training workflow configuration for reasoning-pruning.

This module owns the config contract for the training side of the project
architecture. It validates the base model, published PT dataset, output Hub
model repo, W&B reporting, and small training hyperparameters consumed by HF
Jobs training scripts. It is intentionally separate from dataset-builder config
so dataset creation and training can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    hub_model_id: str
    hub_dataset_id: str
    max_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int | None
    report_to: list[str]
    project: str
    run_name: str


def load_training_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    _require(raw, "base_model")
    _require(raw, "hub_model_id")
    _require(raw, "hub_dataset_id")

    return TrainingConfig(
        base_model=str(raw["base_model"]),
        hub_model_id=str(raw["hub_model_id"]),
        hub_dataset_id=str(raw["hub_dataset_id"]),
        max_steps=int(raw.get("max_steps", 3)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        max_length=raw.get("max_length"),
        report_to=_as_report_to(raw.get("report_to", ["wandb"])),
        project=str(raw.get("project", "reasoning-pruning")),
        run_name=str(raw.get("run_name", "gemma4-e2b-it-smoke")),
    )


def _require(raw: dict[str, Any], key: str) -> None:
    if not raw.get(key):
        raise ValueError(f"missing required training config field: {key}")


def _as_report_to(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("report_to must be a string or list of strings")
