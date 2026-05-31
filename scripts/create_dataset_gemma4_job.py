# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=4.0.0",
#   "huggingface-hub>=0.30.0",
#   "torchvision>=0.20.0",
#   "transformers>=4.47.0",
# ]
# ///

"""Hugging Face Jobs entry point for PT dataset creation with Gemma-4-E2B-it as G.

This script is the remote dataset-creation layer of the project architecture: it
loads Gemma-4-E2B-it as the generator G (self-distillation — the same model that
will be trained), uses Gemini Flash Lite as the conservative decision model D, runs
the automatic pruning-transition loop over GSM8K questions, and pushes the canonical
and training configs to a private Hub dataset. It runs as a uv PEP 723 script on
Hugging Face Jobs with HF_TOKEN and GEMINI_API_KEY supplied as secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any
from urllib import error, request

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────

ROUND_ID = os.environ.get("ROUND_ID", "gsm8k-gemma4-100-r1")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "google/gemma-4-E2B-it")
DECISION_MODEL = os.environ.get("DECISION_MODEL", "gemini-3.1-flash-lite")
PROMPT_VERSION = "conservative-skip-v1"
SOURCE_DATASET = "openai/gsm8k"
SOURCE_SUBSET = "main"
SOURCE_SPLIT = "train"
SOURCE_QUESTION_FIELD = "question"
SOURCE_LIMIT = int(os.environ.get("SOURCE_LIMIT", "100"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.7"))
MAX_PRUNING_DEPTH = int(os.environ.get("MAX_PRUNING_DEPTH", "1"))
MAX_EXAMPLES_PER_QUESTION = int(os.environ.get("MAX_EXAMPLES_PER_QUESTION", "1"))
HUB_DATASET_ID = os.environ.get("HUB_DATASET_ID", "avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1")
PRIVATE = os.environ.get("PRIVATE", "true").lower() != "false"

HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ── Reasoning unit splitting ───────────────────────────────────────────────────

_NUMBERED_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s-]+)")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]|[^.!?]+$")


def _clean_unit(text: str) -> str:
    return _NUMBERED_PREFIX_RE.sub("", text).strip()


def split_reasoning_units(text: str) -> list[str]:
    units = [_clean_unit(line) for line in text.splitlines()]
    units = [u for u in units if u]
    if len(units) > 1:
        return units
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


# ── Decision prompt ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Decision prompt version: {prompt_version}

You are a conservative pruning decision model. Your job: find the first reasoning unit that is pure filler and can be removed without any loss of correctness.

STRICT REMOVAL CONDITIONS — all must hold:
1. The unit contains NO computation, NO numeric value, NO logical deduction, and NO new fact. It is pure filler (e.g. a numbering artifact, a commentary, a restatement of the problem, or a statement of intent).
2. The unit at index removed_end_index+1 — which becomes the training target — contains ACTUAL reasoning: a numeric computation, a derived fact, or a logical deduction. It must NOT be another goal/intent statement ('Determine X', 'We need to find Y', 'Calculate Z', 'Convert A to B').
3. Removing the span leaves the reasoning coherent.

REMOVABLE examples: '1.' (bare numbering), 'Let me think.' (filler), 'This follows the standard approach.' (commentary).
NOT REMOVABLE: 'Convert 50 minutes to hours.' (goal statement — not a computation), 'Determine the rate per minute.' (intent, not math), '$12 × 50/60 = $10.' (actual computation — keep it), '50/60 = 5/6 hours.' (actual math — keep it).

If the next unit after the candidate removal is itself a goal/intent statement, set has_removal=false.

Question:
{question}

Current context:
{context}

Reasoning units:
{reasoning_units}

Return only JSON: has_removal (bool), removed_start_index (int), removed_end_index (int), reason (string), can_continue_after_skip (bool).
Set can_continue_after_skip=true only when the unit at removed_end_index+1 contains actual math, logic, or a concrete fact — never a goal or intent statement.
"""


def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE


def _decision_prompt(*, question: str, context: str, reasoning_units: list[str]) -> str:
    template = _load_prompt_template()
    numbered = "\n".join(f"{i}: {u}" for i, u in enumerate(reasoning_units))
    return template.format(
        prompt_version=PROMPT_VERSION,
        question=question,
        context=context,
        reasoning_units=numbered,
    )


# ── Gemini REST decision model ─────────────────────────────────────────────────

def _gemini_call(*, prompt: str, model: str, max_output_tokens: int, temperature: float) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini error {exc.code}: {detail[:300]}") from exc
    parts = (raw.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _ask_decision_model(
    *, question: str, context: str, units: list[str]
) -> dict[str, Any]:
    prompt = _decision_prompt(question=question, context=context, reasoning_units=units)
    text = _gemini_call(prompt=prompt, model=DECISION_MODEL, max_output_tokens=256, temperature=0.0)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"has_removal": False}
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return {"has_removal": False}


# ── Generator (Gemma-4-E2B-it) ─────────────────────────────────────────────────

def _load_generator():
    print(f"Loading {GENERATOR_MODEL} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    print("Generator loaded.")
    return tokenizer, model


def _generate_reasoning(*, tokenizer, model, context: str) -> str:
    user_content = (
        f"{context}\n\n"
        "Continue the reasoning. Write each step as a concrete computation, deduction, "
        "or fact. Do not write goal statements or intent — compute directly."
    )
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_content

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            do_sample=True,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── PT row builder ─────────────────────────────────────────────────────────────

def _stable_id(question: str, depth: int, start: int, end: int) -> str:
    payload = {"round_id": ROUND_ID, "question": question, "depth": depth,
               "removed_start_index": start, "removed_end_index": end}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"pt-{digest[:16]}"


def _build_pt_row(
    *,
    question: str,
    all_units: list[str],
    removed_start: int,
    removed_end: int,
    depth: int,
    decision_reason: str,
    question_index: int,
    original_trace: str,
) -> dict[str, Any]:
    next_index = removed_end + 1
    prefix_units = all_units[:removed_start]
    removed_units = all_units[removed_start: removed_end + 1]
    target_y = all_units[next_index]
    prefix = "\n".join(prefix_units) if prefix_units else "(empty)"
    input_x = f"Question:\n{question}\n\nUseful reasoning prefix:\n{prefix}"
    return {
        "id": _stable_id(question, depth, removed_start, removed_end),
        "question": question,
        "original_trace": original_trace,
        "input_x": input_x,
        "target_y": target_y,
        "pruning_depth": depth,
        "metadata": {
            "source_model": GENERATOR_MODEL,
            "source_model_revision": None,
            "round_id": ROUND_ID,
            "source_dataset": SOURCE_DATASET,
            "source_dataset_revision": "main",
            "decision_model": DECISION_MODEL,
            "decision_config": {
                "conservative": True,
                "require_following_step": True,
                "max_output_tokens": 256,
                "temperature": 0.0,
                "prompt_version": PROMPT_VERSION,
            },
            "removed_span": "\n".join(removed_units),
            "removed_start_index": removed_start,
            "removed_end_index": removed_end,
            "generation_config": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": GENERATION_TEMPERATURE,
                "do_sample": True,
            },
            "code_version": "hf-job",
            "decision_reason": decision_reason,
            "source_question_index": question_index,
        },
    }


def _row_to_prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["prompt"] = f"{row['input_x']}\n\nContinue with the next useful reasoning step:"
    out["completion"] = row["target_y"]
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

def _format_context(question: str, accepted_units: list[str]) -> str:
    base = f"Question:\n{question}"
    if not accepted_units:
        return base
    return f"{base}\n\nReasoning so far:\n" + "\n".join(accepted_units)


def build_pt_dataset(*, questions: list[str], tokenizer, model) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q_idx, question in enumerate(questions):
        print(f"[{q_idx + 1}/{len(questions)}] {question[:80]}")
        accepted: list[str] = []
        for depth in range(MAX_PRUNING_DEPTH):
            if len([r for r in rows if r["question"] == question]) >= MAX_EXAMPLES_PER_QUESTION:
                break
            context = _format_context(question, accepted)
            trace = _generate_reasoning(tokenizer=tokenizer, model=model, context=context)
            units = split_reasoning_units(trace)
            if len(units) < 2:
                break
            decision = _ask_decision_model(question=question, context=context, units=units)
            if not decision.get("has_removal"):
                break
            start = decision.get("removed_start_index")
            end = decision.get("removed_end_index")
            can_continue = decision.get("can_continue_after_skip", False)
            if not can_continue or start is None or end is None:
                break
            if start < 0 or end < start or end + 1 >= len(units):
                break
            abs_start = len(accepted) + start
            abs_end = len(accepted) + end
            all_units = accepted + units
            row = _build_pt_row(
                question=question,
                all_units=all_units,
                removed_start=abs_start,
                removed_end=abs_end,
                depth=depth,
                decision_reason=str(decision.get("reason", "")),
                question_index=q_idx,
                original_trace=trace,
            )
            rows.append(row)
            print(f"  depth={depth} removed='{all_units[abs_start:abs_end+1]}' target='{row['target_y'][:60]}'")
            accepted.extend(units[:start])
            accepted.append(units[end + 1])
    print(f"Built {len(rows)} PT rows from {len(questions)} questions.")
    return rows


# ── Push to Hub ────────────────────────────────────────────────────────────────

def push_to_hub(rows: list[dict[str, Any]]) -> None:
    training_rows = [_row_to_prompt_completion(r) for r in rows]
    Dataset.from_list(rows).push_to_hub(
        HUB_DATASET_ID, config_name="canonical", split="train", token=HF_TOKEN, private=PRIVATE
    )
    Dataset.from_list(training_rows).push_to_hub(
        HUB_DATASET_ID, config_name="training", split="train", token=HF_TOKEN, private=PRIVATE
    )
    print(f"Pushed {len(rows)} rows to {HUB_DATASET_ID}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {SOURCE_LIMIT} questions from {SOURCE_DATASET}...")
    ds = load_dataset(SOURCE_DATASET, SOURCE_SUBSET, split=SOURCE_SPLIT, token=HF_TOKEN)
    questions = [str(row[SOURCE_QUESTION_FIELD]) for row in ds.select(range(SOURCE_LIMIT))]

    tokenizer, model = _load_generator()
    rows = build_pt_dataset(questions=questions, tokenizer=tokenizer, model=model)

    if rows:
        push_to_hub(rows)
    else:
        print("No PT rows generated — nothing to push.")
