"""Interfaces for generator-model reasoning traces.

This module defines the trace-generation boundary in the project architecture:
the PT dataset builder asks the current generator model version to continue
reasoning from a question/context and receives generated text plus config
metadata. Implementations may run locally for smoke tests or inside Hugging Face
Jobs for real dataset creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedTrace:
    text: str
    generation_config: dict[str, Any] = field(default_factory=dict)


class ReasoningGenerator(Protocol):
    source_model: str
    source_model_revision: str | None

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        """Generate reasoning text from the current question and pruned context."""
