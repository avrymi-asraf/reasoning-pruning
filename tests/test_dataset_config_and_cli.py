"""Tests for config-first PT dataset creation interfaces.

These tests protect the user-facing workflow layer in AGENTS.md: dataset
creation must be discoverable from uv-driven CLI commands and meaningful
experiment behavior must come from structured config. They run locally without
network access while keeping the Hugging Face Dataset publishing boundary
explicit.
"""

from pathlib import Path
import os
import subprocess
import sys

from reasoning_pruning.dataset_builder_config import load_dataset_builder_config
from reasoning_pruning.hf_dataset_publisher import prepare_hf_dataset_payload
from reasoning_pruning import ui_or_cli
from reasoning_pruning.ui_or_cli import build_parser, load_env_file


def test_loads_dataset_builder_config_from_yaml():
    config = load_dataset_builder_config(Path("configs/data/dataset_builder_gsm8k_100_gemma4.yaml"))

    assert config.round_id == "gsm8k-gemma4-100-r1"
    assert config.hub_dataset_id == "avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1"
    assert config.source_type == "hf_dataset"
    assert config.source_dataset == "openai/gsm8k"
    assert config.source_subset == "main"
    assert config.source_split == "train"
    assert config.source_question_field == "question"
    assert config.code_version == "local-dev"
    assert config.generator["model_id"] == "google/gemma-4-E2B-it"
    assert config.decision["model_id"] == "gemini-3.1-flash-lite"
    assert config.max_pruning_depth == 1
    assert config.unit_split_strategy == "numbered_or_lines"


def test_prepares_canonical_and_training_payloads_for_hf_dataset():
    rows = [
        {
            "id": "pt-1",
            "question": "Q?",
            "input_x": "Question:\nQ?\n\nUseful reasoning prefix:\nA.",
            "target_y": "C.",
            "pruning_depth": 0,
            "metadata": {"removed_span": "B."},
        }
    ]

    payload = prepare_hf_dataset_payload(rows)

    assert payload["canonical"][0]["input_x"].endswith("A.")
    assert payload["training"][0]["prompt"].endswith("Continue with the next useful reasoning step:")
    assert payload["training"][0]["completion"] == "C."


def test_cli_exposes_dataset_creation_and_inspection_commands():
    parser = build_parser()

    create_args = parser.parse_args(
        ["build-dataset", "--config", "configs/data/dataset_builder_gsm8k_100_gemma4.yaml", "--dry-run"]
    )
    inspect_args = parser.parse_args(["inspect-dataset", "--config", "configs/data/dataset_builder_gsm8k_100_gemma4.yaml"])

    assert create_args.command == "build-dataset"
    assert create_args.dry_run is True
    assert inspect_args.command == "inspect-dataset"


def test_cli_script_runs_dry_run_without_packaging_install():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reasoning_pruning_cli.py",
            "build-dataset",
            "--config",
            "configs/data/dataset_builder_gsm8k_100_gemma4.yaml",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"canonical": []' in result.stdout


def test_load_env_file_sets_missing_values_without_overwriting_existing(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("export HF_TOKEN=from-file\nGEMINI_API_KEY=from-file\nEXISTING=from-file\n")
    monkeypatch.setenv("EXISTING", "already-set")

    load_env_file(path)

    assert os.environ["HF_TOKEN"] == "from-file"
    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["EXISTING"] == "already-set"


def test_cli_reports_runtime_errors_without_traceback(monkeypatch, capsys):
    def fail_create(args):
        raise RuntimeError("Gemini API error 400: API key expired.")

    monkeypatch.setattr(ui_or_cli, "_create_pt_dataset", fail_create)

    code = ui_or_cli.main(
        ["build-dataset", "--config", "configs/data/dataset_builder_gsm8k_100_gemma4.yaml", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "Gemini API error 400" in captured.err
