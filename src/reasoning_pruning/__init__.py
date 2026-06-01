"""Reasoning-pruning transition training utilities.

The package now keeps data creation small: `data_creation` owns the PT dataset
algorithm, `clients` owns model providers, and `cli` owns local command-line
wiring. Training configuration and model lineage helpers remain separate because
they are downstream of dataset creation. The package is used by uv-driven local
checks and Hugging Face Jobs.
"""

__all__ = [
    "cli",
    "clients",
    "data_creation",
    "model_registry",
    "training_config",
]
