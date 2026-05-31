"""Tests for generated-reasoning unit splitting.

These tests protect the first transformation in the automatic PT dataset flow:
model text must be converted into stable reasoning units before pruning
decisions are made. They run locally under uv/pytest and keep the splitting
strategy simple enough for the HF Jobs data-creation path to reuse.
"""

from reasoning_pruning.reasoning_units import split_reasoning_units


def test_splits_numbered_lines_into_clean_reasoning_units():
    text = "1. Add the known values.\n2. This is a routine arithmetic setup.\n3. Return 5."

    units = split_reasoning_units(text, strategy="numbered_or_lines")

    assert units == [
        "Add the known values.",
        "This is a routine arithmetic setup.",
        "Return 5.",
    ]


def test_splits_sentences_when_line_strategy_has_one_block():
    text = "Add the known values. This is a routine arithmetic setup. Return 5."

    units = split_reasoning_units(text, strategy="sentences")

    assert units == [
        "Add the known values.",
        "This is a routine arithmetic setup.",
        "Return 5.",
    ]
