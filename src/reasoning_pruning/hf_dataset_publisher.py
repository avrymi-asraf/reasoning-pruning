"""Publish canonical pruning-transition rows as Hugging Face Datasets.

This module is the Hub persistence layer in the project architecture: it keeps
canonical `input_x -> target_y` rows as the source dataset form and adds a
derived prompt/completion view for TRL training. It is safe to import during
local uv tests because Hugging Face dependencies are loaded only when publishing.
"""

from __future__ import annotations

from typing import Any

from reasoning_pruning.data import transition_row_to_prompt_completion


def prepare_hf_dataset_payload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "canonical": rows,
        "training": [transition_row_to_prompt_completion(row) for row in rows],
    }


def push_pt_dataset_to_hub(
    *,
    rows: list[dict[str, Any]],
    hub_dataset_id: str,
    token: str | None,
    private: bool = False,
) -> str:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Publishing requires the optional 'datasets' package. Run via uv with "
            "the dataset-creation dependencies or submit the workflow to HF Jobs."
        ) from exc

    payload = prepare_hf_dataset_payload(rows)
    # canonical and training have different schemas, so push as separate configs
    # (sub-datasets) within the same repo — the standard HF pattern for this.
    Dataset.from_list(payload["canonical"]).push_to_hub(
        hub_dataset_id, config_name="canonical", split="train", token=token, private=private
    )
    Dataset.from_list(payload["training"]).push_to_hub(
        hub_dataset_id, config_name="training", split="train", token=token, private=private
    )
    return hub_dataset_id
