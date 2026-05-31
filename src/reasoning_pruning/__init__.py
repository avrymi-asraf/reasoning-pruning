"""Reasoning-pruning transition training utilities.

This package groups the local helpers that support the project architecture:
dataset-builder config, training config, reasoning-unit splitting, pruning
decisions, automatic PT dataset construction, and training-format conversion.
The package is used by uv-driven local tests and should remain aligned with the
Hugging Face Jobs dataset/training flows documented in AGENTS.md.
"""

__all__ = [
    "config",
    "data",
    "dataset_builder_config",
    "hf_dataset_publisher",
    "model_registry",
    "model_clients",
    "pruning_decision",
    "pt_dataset_builder",
    "question_source",
    "reasoning_units",
    "trace_generation",
    "training_config",
    "ui_or_cli",
]
