"""Command-line interface for reasoning-pruning workflows.

This module is the discoverable user interface layer in the project
architecture: it exposes dataset-builder commands separately from training and
lineage commands. It runs locally through uv, reads dataset-builder config only
for dataset commands, and keeps HF publishing explicit so serious dataset
creation can later be submitted through HF Jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from reasoning_pruning.dataset_builder_config import load_dataset_builder_config
from reasoning_pruning.hf_dataset_publisher import prepare_hf_dataset_payload, push_pt_dataset_to_hub
from reasoning_pruning.model_clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.pt_dataset_builder import DatasetBuildConfig, build_pt_dataset
from reasoning_pruning.question_source import QuestionSourceConfig, load_questions_from_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasoning-pruning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("build-dataset")
    create.add_argument("--config", required=True)
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--limit", type=int, default=None)

    inspect = subparsers.add_parser("inspect-dataset")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--limit", type=int, default=5)

    subparsers.add_parser("launch-training-job")
    subparsers.add_parser("evaluate-checkpoint")
    subparsers.add_parser("promote-checkpoint")
    subparsers.add_parser("inspect-lineage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(Path(".env"))
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-dataset":
            return _create_pt_dataset(args)
        if args.command == "inspect-dataset":
            return _inspect_examples(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    raise SystemExit(f"{args.command} is declared but not implemented yet")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _create_pt_dataset(args: argparse.Namespace) -> int:
    config = load_dataset_builder_config(Path(args.config))
    rows = _build_rows_from_config(config)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        print(json.dumps(prepare_hf_dataset_payload(rows), indent=2))
        return 0

    token = os.environ.get("HF_TOKEN")
    push_pt_dataset_to_hub(
        rows=rows,
        hub_dataset_id=config.hub_dataset_id,
        token=token,
        private=config.private,
    )
    print(config.hub_dataset_id)
    return 0


def _inspect_examples(args: argparse.Namespace) -> int:
    config = load_dataset_builder_config(Path(args.config))
    rows = _build_rows_from_config(config)[: args.limit]
    print(json.dumps(rows, indent=2))
    return 0


def _build_rows_from_config(config) -> list[dict]:
    if config.source_type == "local_file" and config.source_questions_path is None:
        return []
    if config.source_type == "local_file" and not Path(config.source_questions_path).exists():
        return []
    questions = load_questions_from_source(
        QuestionSourceConfig(
            source_type=config.source_type,
            source_dataset=config.source_dataset,
            source_dataset_revision=config.source_dataset_revision,
            source_questions_path=config.source_questions_path,
            source_subset=config.source_subset,
            source_split=config.source_split,
            source_question_field=config.source_question_field,
            source_limit=config.source_limit,
        ),
        hf_token=os.environ.get("HF_TOKEN"),
    )
    generator = create_generator_from_config(config.generator, config.generation)
    decision_model = create_decision_model_from_config(config.decision, config.pruning)
    return build_pt_dataset(
        questions=questions,
        generator=generator,
        decision_model=decision_model,
        config=DatasetBuildConfig(
            round_id=config.round_id,
            max_pruning_depth=config.max_pruning_depth,
            max_examples_per_question=config.max_examples_per_question,
            unit_split_strategy=config.unit_split_strategy,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
