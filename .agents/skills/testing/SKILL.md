---
name: testing
description: Testing philosophy for this project — behavior tests only, no implementation dictation, heavy tests opt-in.
---

# Testing Philosophy

## The Rule

Tests check that the code **runs correctly and produces the right outcomes**. They do not dictate *how* the code achieves those outcomes.

## What to test (3–4 tests per feature)

Test the main behavioral contract of the system:

- Does the pipeline produce rows with `input_x` and `target_y`?
- Can those rows be converted to training format?
- Does the pipeline stop gracefully when no removal is found?
- Does a live LLM call return a non-empty response?

## What NOT to test

**Implementation details that don't matter to the caller:**

- Exact field names in `metadata` (the caller only needs `input_x`/`target_y`)
- Internal context string formats (`context_before_generation` exact value)
- Unit-splitting internals (which strategy produced which list)
- CLI parser argument structure
- Script source code contents (`assert "load_dataset" in script.read_text()` — never do this)
- Config YAML field values (those are config, not behavior)
- Exact URL formats in HTTP calls
- Internal attribute names on model clients

## Heavy tests (model loading) — always opt-in

Any test that loads a transformer model or calls a live API **must** be guarded:

```python
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_pipeline(): ...
```

Default `uv run pytest` must complete in seconds on any machine, including those without a GPU or API key. If it crashes or hangs, the test suite has a guard missing.

## Fake models for fast tests

Use simple inline fakes that satisfy the generator/decision-model protocols:

```python
class _FakeGenerator:
    source_model = "fake-g"
    source_model_revision = "v0"
    def generate_reasoning(self, *, question, context):
        return GeneratedTrace(text="Step 1. Filler step. Step 3 result.", generation_config={})

class _FakeDecisionModel:
    decision_model = "fake-d"
    def find_first_removable_span(self, *, question, context, reasoning_units):
        return PruningDecision(has_removal=True, removed_start_index=1, removed_end_index=1,
                               reason="filler", can_continue_after_skip=True)
```

The fake trace must have at least 3 units so the decision to skip index 1 leaves a valid target at index 2.

## Current test file

`tests/test_pipeline.py` — three tests:

1. `test_pipeline_produces_valid_training_rows` — fast, fake models
2. `test_pipeline_returns_empty_when_no_removal_found` — fast, fake models
3. `test_live_gemini_pipeline_generates_reasoning_and_produces_rows` — skipped by default
