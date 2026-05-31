"""Tests for model promotion lineage records.

These tests protect the model-versioning part of the HF-centered workflow:
accepted checkpoints must carry parent model, PT dataset, training, W&B, and
evaluation evidence. They run locally under uv/pytest without pushing to Hub.
"""

from reasoning_pruning.model_registry import ModelPromotionRecord, build_model_card


def test_builds_model_card_with_training_and_acceptance_lineage():
    record = ModelPromotionRecord(
        model_id="avreymi/gemma-4-E2B-it-reasoning-pruning-r1",
        parent_model_id="google/gemma-4-E2B-it",
        pt_dataset_id="avreymi/reasoning-pruning-transitions-r1",
        pt_dataset_revision="abc123",
        training_config="configs/train/training_gemma4_smoke.yaml",
        wandb_run="https://wandb.ai/example/run",
        evaluation_results={"transition_accuracy": 0.75},
        acceptance_reason="Improved transition accuracy without regression.",
    )

    card = build_model_card(record)

    assert "parent_model_id: google/gemma-4-E2B-it" in card
    assert "pt_dataset_revision: abc123" in card
    assert "transition_accuracy: 0.75" in card
    assert "Improved transition accuracy without regression." in card
