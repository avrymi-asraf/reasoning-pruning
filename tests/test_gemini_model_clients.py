"""Tests for Gemini-backed generator and pruning decision clients.

These tests protect the live-model boundary used by the dataset-builder flow:
Gemini can generate reasoning traces and, as the D model, return conservative
JSON pruning decisions. They run locally with fake HTTP transports so no API key
or network is required during verification.
"""

from reasoning_pruning.model_clients import (
    GeminiJsonDecisionModel,
    GeminiReasoningGenerator,
    create_decision_model_from_config,
    create_generator_from_config,
)


def test_gemini_decision_model_parses_json_response(monkeypatch):
    requests = []

    def fake_transport(url, body):
        requests.append((url, body))
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"has_removal": true, "removed_start_index": 1, '
                                    '"removed_end_index": 1, "reason": "repeated", '
                                    '"can_continue_after_skip": true}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = GeminiJsonDecisionModel(
        decision_model="gemini-3.1-flash-lite",
        decision_config={"temperature": 0.0},
        transport=fake_transport,
    )

    decision = model.find_first_removable_span(
        question="Q?",
        context="Question:\nQ?",
        reasoning_units=["A", "B", "C"],
    )

    assert decision.has_removal is True
    assert decision.removed_start_index == 1
    assert decision.can_continue_after_skip is True
    assert "gemini-3.1-flash-lite:generateContent" in requests[0][0]
    assert requests[0][1]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_generator_returns_trace_text(monkeypatch):
    def fake_transport(url, body):
        return {"candidates": [{"content": {"parts": [{"text": "1. Start\n2. Continue"}]}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    generator = GeminiReasoningGenerator(
        source_model="gemini-3.5-flash",
        generation_config={"temperature": 0.2},
        transport=fake_transport,
    )

    trace = generator.generate_reasoning(question="Q?", context="Question:\nQ?")

    assert trace.text == "1. Start\n2. Continue"
    assert trace.generation_config["temperature"] == 0.2


def test_factories_create_gemini_generator_and_decision_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = create_generator_from_config(
        {"provider": "gemini", "model_id": "gemini-3.5-flash"}, {"max_output_tokens": 128}
    )
    decision = create_decision_model_from_config(
        {
            "provider": "gemini-json",
            "model_id": "gemini-3.1-flash-lite",
            "prompt_version": "conservative-skip-v1",
        },
        {"temperature": 0.0},
    )

    assert generator.source_model == "gemini-3.5-flash"
    assert decision.decision_model == "gemini-3.1-flash-lite"


def test_gemini_transport_errors_are_actionable(monkeypatch):
    def fake_transport(url, body):
        raise RuntimeError("Gemini API error 400: API key expired.")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = GeminiJsonDecisionModel(
        decision_model="gemini-3.1-flash-lite",
        transport=fake_transport,
    )

    try:
        model.find_first_removable_span(
            question="Q?",
            context="Question:\nQ?",
            reasoning_units=["A", "B"],
        )
    except RuntimeError as exc:
        assert "API key expired" in str(exc)
    else:
        raise AssertionError("expected Gemini transport failure")
