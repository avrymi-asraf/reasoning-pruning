"""Generate and cache reasoning traces from G for local D-prompt experimentation.

This script is the first half of the D-prompt playground. It runs the generator
model G on a set of questions and saves the raw reasoning traces to a local JSONL
file. The resulting file is the "trace database" that replay_decisions.py consumes
repeatedly without re-running G. Intended for local execution (Gemini as G) or HF
Jobs (Gemma-4 as G); the generator provider is determined by the config YAML.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoning_pruning.dataset_builder_config import load_dataset_builder_config
from reasoning_pruning.model_clients import create_generator_from_config
from reasoning_pruning.question_source import QuestionSourceConfig, load_questions_from_source
from reasoning_pruning.reasoning_units import split_reasoning_units
from reasoning_pruning.ui_or_cli import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate G traces and save to JSONL.")
    parser.add_argument("--config", required=True, help="Path to dataset builder YAML config.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to process.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    config = load_dataset_builder_config(Path(args.config))
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
    if args.limit:
        questions = questions[: args.limit]

    generator = create_generator_from_config(config.generator, config.generation)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    print(f"Generating traces for {total} questions → {output_path}")

    with output_path.open("w") as f:
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{total}] {question[:80]}...")
            context = f"Question:\n{question}"
            trace = generator.generate_reasoning(question=question, context=context)
            units = split_reasoning_units(trace.text, strategy=config.unit_split_strategy)
            row = {"question": question, "trace": trace.text, "units": units}
            f.write(json.dumps(row) + "\n")

    print(f"Done. {total} traces saved to {output_path}")


if __name__ == "__main__":
    main()
