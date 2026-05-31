"""Replay D (decision model) on cached G traces with any prompt file.

This script is the D-prompt playground. Load a JSONL trace file produced by
generate_traces.py, point it at any .txt prompt, and see exactly what D decides
for each trace — without re-running G. Swap prompts/conservative-skip-v1.txt
for a copy with edits and re-run this script to compare behavior. Runs locally;
requires GEMINI_API_KEY in .env.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoning_pruning.model_clients import (
    gemini_generate_text,
    parse_json_pruning_decision,
)
from reasoning_pruning.ui_or_cli import load_env_file

DIVIDER = "─" * 60


def _format_prompt(template: str, prompt_version: str, trace: dict) -> str:
    numbered = "\n".join(f"{i}: {u}" for i, u in enumerate(trace["units"]))
    return template.format(
        prompt_version=prompt_version,
        question=trace["question"],
        context=f"Question:\n{trace['question']}",
        reasoning_units=numbered,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D on cached traces with a prompt file.")
    parser.add_argument("--traces", required=True, help="JSONL file from generate_traces.py.")
    parser.add_argument("--prompt", required=True, help="Path to a .txt decision prompt template.")
    parser.add_argument("--limit", type=int, default=None, help="Max traces to evaluate.")
    parser.add_argument("--model", default="gemini-2.0-flash-lite", help="Gemini model for D.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    traces = [json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()]
    if args.limit:
        traces = traces[: args.limit]

    prompt_version = Path(args.prompt).stem
    template = Path(args.prompt).read_text()

    total = len(traces)
    print(f"Loaded {total} traces. Running D with: {args.prompt}")
    print(DIVIDER)

    removal_count = 0

    for i, trace in enumerate(traces, 1):
        question = trace["question"]
        units = trace["units"]

        print(f"[{i}/{total}] Q: {question[:100]}")
        print("Units:")
        for idx, unit in enumerate(units):
            print(f"  {idx}: {unit}")

        prompt = _format_prompt(template, prompt_version, trace)
        raw = gemini_generate_text(
            model=args.model,
            prompt=prompt,
            generation_config={"responseMimeType": "application/json", "temperature": 0.0},
            api_key_env="GEMINI_API_KEY",
            transport=None,
        )
        decision = parse_json_pruning_decision(raw)

        if decision.has_removal:
            removal_count += 1
            removed_text = " | ".join(units[decision.removed_start_index : decision.removed_end_index + 1])
            print(f"Decision: has_removal=True  span={decision.removed_start_index}-{decision.removed_end_index}")
            print(f'  Removed: "{removed_text}"')
        else:
            print("Decision: has_removal=False")

        print(f"  Reason: {decision.reason}")
        print(f"  Can continue: {decision.can_continue_after_skip}")
        print(DIVIDER)

    print(f"Summary: {removal_count}/{total} had removals")


if __name__ == "__main__":
    main()
