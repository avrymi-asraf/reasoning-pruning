"""Split generated reasoning traces into pruning units.

This module is the reasoning-unit layer in the project architecture: it turns
raw text emitted by the generator model into small ordered units for the
decision model. It is used by local uv tools and can run unchanged inside a
Hugging Face Jobs dataset-creation script before rows are published to Hub.
"""

from __future__ import annotations

import re


_NUMBERED_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s-]+)")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]|[^.!?]+$")


def split_reasoning_units(text: str, *, strategy: str = "numbered_or_lines") -> list[str]:
    if strategy == "numbered_or_lines":
        units = [_clean_unit(line) for line in text.splitlines()]
        units = [unit for unit in units if unit]
        if len(units) > 1:
            return units
        return _split_sentences(text)
    if strategy == "lines":
        return [unit for unit in (_clean_unit(line) for line in text.splitlines()) if unit]
    if strategy == "sentences":
        return _split_sentences(text)
    raise ValueError(f"unknown reasoning unit split strategy: {strategy}")


def _split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _clean_unit(text: str) -> str:
    return _NUMBERED_PREFIX_RE.sub("", text).strip()
