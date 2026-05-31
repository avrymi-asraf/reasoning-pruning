"""Represent and validate pruning decisions.

This module is the narrow decision layer in the project architecture: a
decision model returns whether the first safely removable reasoning span exists
and where it is. The automatic dataset builder consumes this object locally or
inside Hugging Face Jobs without asking the decision model to rewrite answers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PruningDecision:
    has_removal: bool
    removed_start_index: int | None
    removed_end_index: int | None
    reason: str
    can_continue_after_skip: bool

    def valid_span_for(self, unit_count: int) -> bool:
        if not self.has_removal or not self.can_continue_after_skip:
            return False
        if self.removed_start_index is None or self.removed_end_index is None:
            return False
        if self.removed_start_index < 0 or self.removed_end_index < self.removed_start_index:
            return False
        if self.removed_end_index + 1 >= unit_count:
            return False
        return self.removed_end_index < unit_count
