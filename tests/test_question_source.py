"""Tests for spectrum question assembly.

The one contract that matters here: the prompt body shown to G must contain the
context and choices a row needs to be answerable, but must NEVER contain the
answer fields — leaking them corrupts every derived training row.
"""

from reasoning_pruning.data_creation import format_spectrum_question


def test_spectrum_question_includes_context_and_choices_but_never_the_answer():
    row = {
        "input_mode": "question_with_context_and_choices",
        "question": "What regulates body processes?",
        "context": "The hypothalamus produces hormones that regulate body processes.",
        "choices": [{"label": "A", "text": "pancreas"}, {"label": "B", "text": "hypothalamus"}],
        "gold_answer": "hypothalamus",
        "gold_answer_label": "B",
        "reference_solution": "It is the hypothalamus because ...",
        "supporting_facts": "hypothalamus -> hormones",
    }
    text = format_spectrum_question(row)

    assert "The hypothalamus produces hormones" in text
    assert "(A) pancreas" in text and "(B) hypothalamus" in text
    assert row["reference_solution"] not in text
    assert row["supporting_facts"] not in text
