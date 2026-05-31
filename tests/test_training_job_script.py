"""Tests for the Hugging Face Jobs training entry point.

These tests ensure the remote training path consumes a published PT dataset
instead of creating hand-written smoke rows inside the trainer. They run locally
under uv/pytest as a static guard before the script is submitted to HF Jobs.
"""

from pathlib import Path


def test_training_job_loads_hf_dataset_instead_of_inline_smoke_examples():
    script = Path("scripts/train_pt_dataset_job.py").read_text()

    assert "load_dataset" in script
    assert "Dataset.from_list" not in script
    assert "def build_example" not in script


def test_training_job_uses_training_config_not_dataset_builder_config():
    script = Path("scripts/train_pt_dataset_job.py").read_text()

    assert "TRAINING_CONFIG" in script
    assert "configs/train/training_gemma4_gsm8k_100.yaml" in script
    assert "dataset_builder_smoke.yaml" not in script


def test_training_job_uses_config_name_for_dataset_loading():
    script = Path("scripts/train_pt_dataset_job.py").read_text()

    # Must load via config_name (second positional arg), not split="training"
    assert "HUB_DATASET_CONFIG" in script
    assert 'split="train"' in script
    # Must NOT use the old incorrect split-name approach
    assert 'split=HUB_DATASET_SPLIT' not in script
