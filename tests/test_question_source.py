"""Tests for loading source questions into the PT dataset loop.

These tests protect the start of the automatic data-generation code flow:
source questions must enter the generator as questions, not as prebuilt
prompt/completion rows. They run locally under uv/pytest and keep the source
dataset boundary explicit for Hugging Face dataset-creation jobs.
"""

from pathlib import Path

from reasoning_pruning.question_source import QuestionSourceConfig, load_questions, load_questions_from_source


def test_loads_questions_from_plain_text_file(tmp_path: Path):
    path = tmp_path / "questions.txt"
    path.write_text("What is 2 + 3?\n\nWhat comes after 8?\n")

    assert load_questions(path) == ["What is 2 + 3?", "What comes after 8?"]


def test_loads_questions_from_jsonl_question_field(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question": "What is 2 + 3?"}\n{"question": "What comes after 8?"}\n')

    assert load_questions(path) == ["What is 2 + 3?", "What comes after 8?"]


def test_loads_questions_from_jsonl_configured_question_field(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"prompt": "What is 2 + 3?"}\n{"prompt": "What comes after 8?"}\n')

    assert load_questions(path, question_field="prompt") == [
        "What is 2 + 3?",
        "What comes after 8?",
    ]


def test_loads_questions_from_hf_dataset_with_configured_field(monkeypatch):
    calls = {}

    def fake_loader(path, *, split, revision, token):
        calls.update({"path": path, "split": split, "revision": revision, "token": token})
        return [{"prompt": "Question A"}, {"prompt": "Question B"}, {"prompt": ""}]

    config = QuestionSourceConfig(
        source_type="hf_dataset",
        source_dataset="org/questions",
        source_dataset_revision="abc123",
        source_questions_path=None,
        source_subset=None,
        source_split="validation",
        source_question_field="prompt",
        source_limit=1,
    )

    questions = load_questions_from_source(config, hf_loader=fake_loader, hf_token="secret")

    assert questions == ["Question A"]
    assert calls == {
        "path": "org/questions",
        "split": "validation",
        "revision": "abc123",
        "token": "secret",
    }


def test_loads_questions_from_hf_dataset_subset_when_configured():
    calls = {}

    def fake_loader(path, name=None, *, split, revision, token):
        calls.update(
            {"path": path, "name": name, "split": split, "revision": revision, "token": token}
        )
        return [{"question": "Question A"}]

    config = QuestionSourceConfig(
        source_type="hf_dataset",
        source_dataset="openai/gsm8k",
        source_dataset_revision="main",
        source_questions_path=None,
        source_subset="main",
        source_split="train",
        source_question_field="question",
        source_limit=200,
    )

    assert load_questions_from_source(config, hf_loader=fake_loader) == ["Question A"]
    assert calls["name"] == "main"
