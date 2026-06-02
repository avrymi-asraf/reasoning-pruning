"""Replay D on reasoning traces with any prompt file.

The second half of the local prompt playground. Loads traces from a local JSONL
file (generate_traces.py output) or directly from `avreymi/reasoning-traces-gemma4-100`
on the Hub. Calls Gemini D, then prints the decision and the actual PT row that
would be emitted (input_x → target_y). Swap prompt files and re-run to compare
versions without regenerating traces. Runs locally via uv and requires
`GEMINI_API_KEY` (and `HF_TOKEN` for Hub loading) in the environment or `.env`.
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
HUB_REPO = "avreymi/reasoning-traces-gemma4-100"


def load_traces_from_hub(config: str, limit: int | None) -> list[dict]:
    import os
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(HUB_REPO, name=config, split="train", token=token)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"question": r["question"], "units": r["units"], "source": r.get("source_dataset", config)} for r in ds]


def _format_prompt(template: str, prompt_version: str, trace: dict) -> str:
    numbered = "\n".join(f"{i}: {u}" for i, u in enumerate(trace["units"]))
    return template.format(
        prompt_version=prompt_version,
        question=trace["question"],
        context=f"Question:\n{trace['question']}",
        reasoning_units=numbered,
    )


def _build_pt_row(trace: dict, decision) -> dict | None:
    """Return the PT row if the decision is valid, else None."""
    if not (
        decision.has_removal
        and decision.can_continue_after_skip
        and decision.removed_start_index is not None
        and decision.removed_end_index is not None
    ):
        return None
    units = trace["units"]
    end = decision.removed_end_index
    if end + 1 >= len(units):
        return None
    useful_prefix = units[: decision.removed_start_index]
    target_y = units[end + 1]
    input_x = f"Question:\n{trace['question']}"
    if useful_prefix:
        input_x += "\n" + "\n".join(useful_prefix)
    return {
        "input_x": input_x,
        "target_y": target_y,
        "removed_span": units[decision.removed_start_index : end + 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D on traces and preview PT rows.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", help="Local JSONL file from generate_traces.py.")
    source.add_argument("--hub", nargs="?", const="all", metavar="CONFIG",
                        help="Load from HF Hub. Optionally specify config name (default: all).")
    parser.add_argument("--prompt", required=True, help="Path to a .txt decision prompt template.")
    parser.add_argument("--limit", type=int, default=None, help="Max traces to evaluate.")
    parser.add_argument("--model", default="gemini-2.0-flash-lite", help="Gemini model for D.")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    if args.hub is not None:
        config = args.hub
        print(f"Loading traces from Hub: {HUB_REPO} (config={config})")
        traces = load_traces_from_hub(config, args.limit)
    else:
        traces = [json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()]
        if args.limit:
            traces = traces[: args.limit]

    prompt_version = Path(args.prompt).stem
    template = Path(args.prompt).read_text()
    print(f"Loaded {len(traces)} traces. Running D with: {args.prompt}")
    print(DIVIDER)

    removal_count = 0
    valid_row_count = 0
    for i, trace in enumerate(traces, 1):
        units = trace["units"]
        source_label = trace.get("source", "")
        print(f"[{i}/{len(traces)}] {f'({source_label}) ' if source_label else ''}Q: {trace['question'][:100]}")
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

        row = _build_pt_row(trace, decision)
        if row:
            valid_row_count += 1
            print("PT row:")
            print(f"  input_x : {row['input_x']!r}")
            print(f"  target_y: {row['target_y']!r}")
            print(f"  removed : {row['removed_span']}")
        else:
            print("PT row: [none — invalid decision or no following unit]")

        print(DIVIDER)

    print(f"Summary: {removal_count}/{len(traces)} had removals, {valid_row_count}/{len(traces)} valid PT rows")


if __name__ == "__main__":
    main()
