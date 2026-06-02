# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=4.0.0",
#   "huggingface-hub>=0.30.0",
#   "torchvision>=0.20.0",
#   "transformers>=4.47.0",
# ]
# ///

"""Generate 100 Gemma-4 reasoning traces across 4 datasets.

Runs on Hugging Face Jobs (GPU). Generates 25 traces each from GSM8K,
MATH competition problems, AQUA-RAT algebra problems, and BBH logical-deduction
puzzles. Output is pushed to HUB_OUTPUT as a single HF dataset with columns:
source_dataset, question, trace, units. No decision model (D) is involved —
this is raw G output only.
"""

from __future__ import annotations

import os
import re

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = os.environ.get("GENERATOR_MODEL", "google/gemma-4-E2B-it")
HUB_OUTPUT = os.environ.get("HUB_OUTPUT", "avreymi/reasoning-traces-gemma4-100")
N_PER_SOURCE = int(os.environ.get("N_PER_SOURCE", "25"))
# Comma-separated source names to skip (already pushed in a previous run).
_SKIP = set(s.strip() for s in os.environ.get("SKIP_SOURCES", "").split(",") if s.strip())

# 25 questions × 4 sources = 100 traces
SOURCES = [
    {
        "name": "gsm8k",
        "id": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "field": "question",
    },
    {
        "name": "math",
        "id": "lighteval/MATH-Hard",
        "subset": None,
        "split": "train",
        "field": "problem",
    },
    {
        "name": "aqua_rat",
        "id": "deepmind/aqua_rat",
        "subset": None,
        "split": "train",
        "field": "question",
    },
    {
        "name": "bbh_logical_deduction",
        "id": "lukaemon/bbh",
        "subset": "logical_deduction_seven_objects",
        "split": "test",
        "field": "input",
    },
]

_NUMBERED_PREFIX = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s-]+)")


def _load_questions(src: dict, n: int, token: str | None) -> list[str]:
    kw: dict = {"token": token}
    ds = (
        load_dataset(src["id"], src["subset"], split=src["split"], **kw)
        if src["subset"]
        else load_dataset(src["id"], split=src["split"], **kw)
    )
    return [ds[i][src["field"]] for i in range(min(n, len(ds)))]


def _split_units(text: str) -> list[str]:
    lines = [_NUMBERED_PREFIX.sub("", line).strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        return lines
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]|[^.!?]+$", text) if s.strip()]


def _make_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    content = (
        f"Question:\n{question}\n\n"
        "Continue the reasoning. Write each step as a concrete computation, deduction, "
        "or fact. Do not write goal statements or intent — compute directly."
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def main() -> None:
    token = os.environ.get("HF_TOKEN")

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        token=token,
    )

    all_rows: list[dict] = []
    for src in SOURCES:
        if src["name"] in _SKIP:
            print(f"\n=== {src['name']} — skipped (SKIP_SOURCES) ===")
            continue
        print(f"\n=== {src['name']} — loading {N_PER_SOURCE} questions ===")
        questions = _load_questions(src, N_PER_SOURCE, token)
        batch: list[dict] = []
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question[:80]}...")
            prompt = _make_prompt(tokenizer, question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                )
            generated = output_ids[0][inputs["input_ids"].shape[-1] :]
            trace = tokenizer.decode(generated, skip_special_tokens=True).strip()
            units = _split_units(trace)
            batch.append(
                {
                    "source_dataset": src["name"],
                    "question": question,
                    "trace": trace,
                    "units": units,
                }
            )

        # Push each source immediately so a later timeout doesn't lose completed work.
        # Each source gets its own config_name; "all" is pushed at the end.
        print(f"  Pushing {len(batch)} {src['name']} traces...")
        Dataset.from_list(batch).push_to_hub(
            HUB_OUTPUT, config_name=src["name"], split="train", token=token, private=True
        )
        all_rows.extend(batch)

    # Build the combined 'all' config by loading all per-source configs from Hub
    # (handles the case where some sources were skipped via SKIP_SOURCES).
    combined: list[dict] = []
    for src in SOURCES:
        try:
            src_ds = load_dataset(HUB_OUTPUT, config_name=src["name"], split="train", token=token)
            combined.extend(src_ds.to_list())
            print(f"  Loaded {len(src_ds)} {src['name']} rows from Hub")
        except Exception as exc:
            print(f"  Warning: could not load {src['name']} from Hub ({exc}), using in-memory rows")
            combined.extend(r for r in all_rows if r["source_dataset"] == src["name"])

    print(f"\nTotal: {len(combined)} traces across {len(SOURCES)} datasets")
    Dataset.from_list(combined).push_to_hub(
        HUB_OUTPUT, config_name="all", split="train", token=token, private=True
    )
    print(f"Pushed combined 'all' config to {HUB_OUTPUT}")


if __name__ == "__main__":
    main()
