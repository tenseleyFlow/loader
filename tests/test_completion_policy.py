"""Tests for completion-policy helpers."""

from loader.runtime.completion_policy import CompletionPolicy
from loader.runtime.task_completion import (
    detect_premature_completion,
    get_continuation_prompt,
)


def test_completion_policy_finalize_response_text_keeps_original_response() -> None:
    response = CompletionPolicy.finalize_response_text(
        content="Inspected the file successfully.",
        actions_taken=["read: README.md"],
    )

    assert response == "Inspected the file successfully."


def test_detect_premature_completion_respects_explicit_done_without_actions() -> None:
    assert detect_premature_completion(
        "Explain how Loader works.",
        "Done.",
        [],
    ) is False


def test_get_continuation_prompt_surfaces_missing_verification_steps() -> None:
    prompt = get_continuation_prompt(
        "Create the script and test that it works.",
        ["write: script.py"],
        "The script has been created.",
    )

    assert "Run the tests" in prompt or "verify it works" in prompt
