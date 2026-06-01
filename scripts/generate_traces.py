"""Generate and cache reasoning traces from G for D-prompt experiments.

This script is the first half of the local prompt playground: it loads the same
data-creation config as the real dataset builder, runs only the generator G, and
writes raw traces plus split units to JSONL. The cached file lets
`replay_decisions.py` iterate on D prompts without re-running G. It is intended
for local uv execution with a lightweight generator provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoning_pruning.cli import load_env_file
from reasoning_pruning.clients import create_generator_from_config
from reasoning_pruning.data_creation import load_data_creation_config, load_questions, split_reasoning_units


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate G traces and save to JSONL.")
    parser.add_argument("--config", required=True, help="Path to data-creation YAML config.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to process.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    config = load_data_creation_config(Path(args.config))
    questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))
    if args.limit:
        questions = questions[: args.limit]

    generator = create_generator_from_config(config.generator, config.generation)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating traces for {len(questions)} questions → {output_path}")
    with output_path.open("w") as f:
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:80]}...")
            context = f"Question:\n{question}"
            trace = generator.generate_reasoning(question=question, context=context)
            units = split_reasoning_units(trace.text, strategy=config.unit_split_strategy)
            f.write(json.dumps({"question": question, "trace": trace.text, "units": units}) + "\n")

    print(f"Done. {len(questions)} traces saved to {output_path}")


if __name__ == "__main__":
    main()
