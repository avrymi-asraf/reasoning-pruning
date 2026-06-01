"""Replay D on cached G traces with any prompt file.

This script is the second half of the local prompt playground. It reads JSONL
traces from `generate_traces.py`, formats a candidate decision prompt, and calls
Gemini D to inspect safe-skip decisions without regenerating traces. It runs
locally through uv and requires `GEMINI_API_KEY` in the environment or `.env`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoning_pruning.cli import load_env_file
from reasoning_pruning.clients import gemini_generate_text, parse_json_pruning_decision

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
    print(f"Loaded {len(traces)} traces. Running D with: {args.prompt}")
    print(DIVIDER)

    removal_count = 0
    for i, trace in enumerate(traces, 1):
        units = trace["units"]
        print(f"[{i}/{len(traces)}] Q: {trace['question'][:100]}")
        print("Units:")
        for idx, unit in enumerate(units):
            print(f"  {idx}: {unit}")

        raw = gemini_generate_text(
            model=args.model,
            prompt=_format_prompt(template, prompt_version, trace),
            generation_config={"responseMimeType": "application/json", "temperature": 0.0},
            api_key_env="GEMINI_API_KEY",
            transport=None,
        )
        decision = parse_json_pruning_decision(raw)

        if decision.has_removal and decision.removed_start_index is not None and decision.removed_end_index is not None:
            removal_count += 1
            removed = " | ".join(units[decision.removed_start_index : decision.removed_end_index + 1])
            print(f"Decision: has_removal=True  span={decision.removed_start_index}-{decision.removed_end_index}")
            print(f'  Removed: "{removed}"')
        else:
            print("Decision: has_removal=False")

        print(f"  Reason: {decision.reason}")
        print(f"  Can continue: {decision.can_continue_after_skip}")
        print(DIVIDER)

    print(f"Summary: {removal_count}/{len(traces)} had removals")


if __name__ == "__main__":
    main()
