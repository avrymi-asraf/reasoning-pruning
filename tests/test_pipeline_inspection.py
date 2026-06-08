"""Smoke test for the pipeline inspector.

The inspector runs the production `build_rows_for_question` loop with a printing
observer, so this only confirms it stays wired: it runs end to end and returns
the same rows the production loop would. The actual value of the inspector is
reading its printed output against a live model, which a unit test cannot judge.
"""

from reasoning_pruning.pipeline_inspection import run_pipeline_inspection

from fakes import FakeDecisionModel, FakeGenerator, NoRemovalDecisionModel, make_config


def test_inspection_runs_the_production_loop_and_returns_rows(capsys):
    rows = run_pipeline_inspection(
        question="What is 2 + 3?",
        generator=FakeGenerator(),
        decision_model=FakeDecisionModel(),
        config=make_config(),
    )

    assert len(rows) == 1
    assert rows[0]["input_x"] and rows[0]["target_y"]
    assert capsys.readouterr().out, "inspector should print its stages"


def test_inspection_renders_the_reject_and_stop_paths_without_crashing(capsys):
    rows = run_pipeline_inspection(
        question="What is 2 + 3?",
        generator=FakeGenerator(),
        decision_model=NoRemovalDecisionModel(),
        config=make_config(),
    )

    output = capsys.readouterr().out
    assert rows == []
    assert "Attempt rejected" in output  # observer.rejected path
    assert "Stopping after depth" in output  # observer.stopped path
