"""Run a qualitative reasoning-pruning inspection from the command line.

This script is the local entry point for printing each stage of one pruning-data
example from question selection through the next-depth context. It delegates the
actual loop to `reasoning_pruning.qualitative_inspection` so notebook and CLI
inspection stay uniform. It is intended for cheap local or API-backed inspection
runs, not for publishing production datasets.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_pruning.cli import load_env_file
from reasoning_pruning.clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.data_creation import load_data_creation_config, load_questions
from reasoning_pruning.qualitative_inspection import run_qualitative_pruning_inspection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qualitative-pruning-inspection")
    parser.add_argument("--config", default="configs/data/qualitative_inspection_gemma4_api.yaml")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--question", default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--prompts-dir", default="prompts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(Path(".env"))
    args = build_parser().parse_args(argv)
    config = load_data_creation_config(Path(args.config))
    if args.max_depth is not None:
        config = replace(config, max_pruning_depth=args.max_depth, max_examples_per_question=args.max_depth)
    if args.max_retries is not None:
        config = replace(config, max_retries_per_depth=args.max_retries)

    question = args.question or _select_question(config, args.question_index)
    generator = create_generator_from_config(
        config.generator,
        config.generation,
        max_units_per_batch=config.max_units_per_batch,
    )
    decision_model = create_decision_model_from_config(config.decision, config.pruning, prompts_dir=args.prompts_dir)
    run_qualitative_pruning_inspection(
        question=question,
        generator=generator,
        decision_model=decision_model,
        config=config,
    )
    return 0


def _select_question(config, question_index: int) -> str:
    questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))
    if not questions:
        raise RuntimeError("No questions were loaded for qualitative inspection.")
    if question_index < 0 or question_index >= len(questions):
        raise RuntimeError(f"question index {question_index} is outside 0..{len(questions) - 1}")
    return questions[question_index]


if __name__ == "__main__":
    raise SystemExit(main())
