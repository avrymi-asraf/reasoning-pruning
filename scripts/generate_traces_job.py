# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=4.0.0",
#   "huggingface-hub>=0.30.0",
#   "torchvision>=0.20.0",
#   "transformers>=4.47.0",
#   "reasoning-pruning-codex @ git+https://github.com/avrymi-asraf/reasoning-pruning.git",
# ]
# ///

"""Generate raw Gemma-4 reasoning traces from reasoning-spectrum-qa.

Runs on Hugging Face Jobs (GPU). Pulls N questions from
`avreymi/reasoning-spectrum-qa` (the single PT source), formats each full prompt
with the shared `format_spectrum_question` (context + question + choices, never
the answer), and pushes traces to HUB_OUTPUT with columns:
source_dataset, reasoning_family, question, trace, units. No decision model (D)
is involved — this is raw G output only.
"""

from __future__ import annotations

import os
import re

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from reasoning_pruning.data_creation import format_spectrum_question

MODEL_ID = os.environ.get("GENERATOR_MODEL", "avreymi/gemma-4-E2B-it-reasoning-pruning")
HUB_OUTPUT = os.environ.get("HUB_OUTPUT", "avreymi/reasoning-traces-spectrum-gemma4")
SOURCE_DATASET = "avreymi/reasoning-spectrum-qa"
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "100"))
# Re-push the full accumulated dataset every PUSH_EVERY rows so a job timeout
# (spectrum traces can run long) never loses already-generated work.
PUSH_EVERY = int(os.environ.get("PUSH_EVERY", "25"))

_NUMBERED_PREFIX = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s-]+)")


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

    print(f"Loading {N_QUESTIONS} questions from {SOURCE_DATASET}")
    source = load_dataset(SOURCE_DATASET, split="data", token=token)
    rows: list[dict] = []
    for i in range(min(N_QUESTIONS, len(source))):
        row = source[i]
        question = format_spectrum_question(row)
        print(f"  [{i + 1}/{N_QUESTIONS}] ({row['reasoning_family']}) {question[:80]}...")
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
        rows.append(
            {
                "source_dataset": SOURCE_DATASET,
                "reasoning_family": row["reasoning_family"],
                "question": question,
                "trace": trace,
                "units": _split_units(trace),
            }
        )
        if len(rows) % PUSH_EVERY == 0:
            print(f"  Checkpoint push: {len(rows)} traces -> {HUB_OUTPUT}")
            Dataset.from_list(rows).push_to_hub(HUB_OUTPUT, split="train", token=token, private=True)

    print(f"\nTotal: {len(rows)} traces")
    Dataset.from_list(rows).push_to_hub(HUB_OUTPUT, split="train", token=token, private=True)
    print(f"Pushed traces to {HUB_OUTPUT}")


if __name__ == "__main__":
    main()
