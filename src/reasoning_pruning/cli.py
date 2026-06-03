"""Command line interface for reasoning-pruning data creation.

This module is the local user-facing wrapper around the simple data-creation
flow: load config, load questions, create model clients, build rows, and either
inspect or publish them. It keeps training and lineage placeholders out of the
core dataset algorithm. It runs through `uv run python scripts/reasoning_pruning_cli.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from reasoning_pruning.clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.data_creation import (
    DataCreationConfig,
    build_pt_dataset,
    load_data_creation_config,
    load_questions,
    prepare_hf_dataset_payload,
    push_pt_dataset_to_hub,
)


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
        key = key.strip().removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _create_pt_dataset(args: argparse.Namespace) -> int:
    config = load_data_creation_config(Path(args.config))
    rows = _build_rows_from_config(config)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        print(json.dumps(prepare_hf_dataset_payload(rows), indent=2))
        return 0

    push_pt_dataset_to_hub(
        rows=rows,
        hub_dataset_id=config.hub_dataset_id,
        token=os.environ.get("HF_TOKEN"),
        private=config.private,
    )
    print(config.hub_dataset_id)
    return 0


def _inspect_examples(args: argparse.Namespace) -> int:
    config = load_data_creation_config(Path(args.config))
    print(json.dumps(_build_rows_from_config(config)[: args.limit], indent=2))
    return 0


def _build_rows_from_config(config: DataCreationConfig) -> list[dict]:
    if config.source_type == "local_file" and config.source_questions_path is None:
        return []
    if config.source_type == "local_file" and not Path(config.source_questions_path).exists():
        return []

    questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))
    generator = create_generator_from_config(config.generator, config.generation, max_units_per_batch=config.max_units_per_batch)
    decision_model = create_decision_model_from_config(config.decision, config.pruning)
    return build_pt_dataset(
        questions=questions,
        generator=generator,
        decision_model=decision_model,
        config=config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
