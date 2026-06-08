"""Wiring tests for the G/D model boundaries.

Fast test: the Gemini decision client turns a model response into a
`PruningDecision` the loop can use (connectivity, via a stub transport — no live
call, no assertions about request URL/schema internals). Slow test: the real
Gemini pipeline, gated behind env vars, is the only place real model output is
exercised in the suite. The richer "does the output make sense" check lives in
the pipeline inspection run, not here.
"""

import json
import os

import pytest

from reasoning_pruning.clients import (
    GeminiDecisionModel,
    create_decision_model_from_config,
    create_generator_from_config,
)
from reasoning_pruning.data_creation import build_pt_dataset, transition_row_to_prompt_completion

from fakes import make_config


def test_gemini_decision_response_parses_into_a_usable_decision(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def _transport(url, body):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "has_removal": True,
                                        "removed_start_index": 0,
                                        "removed_end_index": 0,
                                        "reason": "The first unit is filler.",
                                        "can_continue_after_skip": True,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

    decision_model = GeminiDecisionModel(
        decision_model="gemini-test",
        decision_config={"temperature": 0.0, "max_output_tokens": 128},
        transport=_transport,
    )

    decision = decision_model.find_first_removable_span(
        question="What is 2 + 3?",
        context="Question:\nWhat is 2 + 3?",
        reasoning_units=["We need to solve it.", "2 + 3 = 5.", "The answer is 5."],
    )

    assert decision.valid_for(["We need to solve it.", "2 + 3 = 5.", "The answer is 5."])
    assert decision.removed_start_index == 0 and decision.removed_end_index == 0


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_pipeline_generates_reasoning_and_produces_rows():
    generator = create_generator_from_config(
        {"provider": "gemini", "model_id": "gemini-2.0-flash-lite"},
        {"maxOutputTokens": 512, "temperature": 0.7},
    )
    decision_model = create_decision_model_from_config(
        {"provider": "gemini-json", "model_id": "gemini-2.0-flash-lite", "prompt_version": "conservative-skip-v1"},
        {"temperature": 0.0},
    )

    rows = build_pt_dataset(
        questions=["If x + 2 = 5, what is x?"],
        generator=generator,
        decision_model=decision_model,
        config=make_config("live-test"),
    )
    for row in rows:
        assert row["input_x"] and row["target_y"]
        training = transition_row_to_prompt_completion(row)
        assert training["prompt"] and training["completion"]
