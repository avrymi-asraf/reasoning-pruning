"""Create model-version lineage records for accepted checkpoints.

This module is the lightweight model-registry layer in the project architecture:
it turns an accepted training checkpoint into a documented model-card payload
that records parent model, PT dataset version, training config, W&B run,
evaluation metrics, and acceptance reason. It runs locally under uv before a
future Hub push or inside a Hugging Face Jobs promotion step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelPromotionRecord:
    model_id: str
    parent_model_id: str
    pt_dataset_id: str
    pt_dataset_revision: str
    training_config: str
    wandb_run: str
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    acceptance_reason: str = ""


def build_model_card(record: ModelPromotionRecord) -> str:
    metric_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(record.evaluation_results.items())
    )
    if not metric_lines:
        metric_lines = "- no evaluation metrics recorded"

    return (
        f"# {record.model_id}\n\n"
        "## Reasoning-Pruning Lineage\n\n"
        f"- parent_model_id: {record.parent_model_id}\n"
        f"- pt_dataset_id: {record.pt_dataset_id}\n"
        f"- pt_dataset_revision: {record.pt_dataset_revision}\n"
        f"- training_config: {record.training_config}\n"
        f"- wandb_run: {record.wandb_run}\n\n"
        "## Evaluation Results\n\n"
        f"{metric_lines}\n\n"
        "## Acceptance Reason\n\n"
        f"{record.acceptance_reason}\n"
    )
