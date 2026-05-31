"""Compatibility imports for workflow configuration.

The project now keeps dataset-builder and training config contracts in separate
modules so the two workflow halves can evolve independently. This module remains
as a thin compatibility layer for older local imports, while new code should use
`dataset_builder_config.py` or `training_config.py` directly. It runs only in
uv-managed local tooling and tests.
"""

from __future__ import annotations

from pathlib import Path

from reasoning_pruning.dataset_builder_config import (
    DatasetBuilderConfig,
    load_dataset_builder_config,
)
from reasoning_pruning.training_config import TrainingConfig, load_training_config


DatasetCreationConfig = DatasetBuilderConfig


def load_dataset_build_config(path: Path) -> DatasetBuilderConfig:
    return load_dataset_builder_config(path)
