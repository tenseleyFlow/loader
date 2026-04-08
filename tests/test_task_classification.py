"""Tests for runtime-owned task classification helpers."""

from __future__ import annotations

from loader.runtime.task_classification import (
    estimate_complexity,
    get_token_budget,
    is_conversational,
)


def test_is_conversational_detects_small_talk() -> None:
    assert is_conversational("hi there") is True
    assert is_conversational("thanks") is True
    assert is_conversational("what can you do?") is True


def test_is_conversational_rejects_actionable_tasks() -> None:
    assert is_conversational("create a new README for this repo") is False
    assert is_conversational("run the tests and fix failures") is False


def test_estimate_complexity_and_token_budget_cover_common_paths() -> None:
    assert estimate_complexity("hi") == "trivial"
    assert estimate_complexity("show me the config file") == "simple"
    assert estimate_complexity(
        "implement a website and then refactor the database layer"
    ) == "complex"
    assert get_token_budget("simple") == (512, 4096)
