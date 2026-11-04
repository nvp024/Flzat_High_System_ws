"""Step-2 pure policy tests for stop/cancel and natural Action B answers."""

import pytest

from flzat_cognitive_pipeline.interaction import (
    assess_user_answer,
    is_stop_command,
    validate_safe_identifier,
)


@pytest.mark.parametrize(
    "text",
    [
        "stop",
        "Stop now!",
        "cancel the task",
        "emergency stop.",
        "dừng lại",
    ],
)
def test_stop_guard_accepts_normal_variants_before_busy(text):
    assert is_stop_command(text)


@pytest.mark.parametrize(
    "text",
    ["get the book", "do not stop near the door", "can you stop by the table"],
)
def test_stop_guard_does_not_match_ordinary_requests(text):
    assert not is_stop_command(text)


def test_confirm_negative_with_correction_preserves_full_answer():
    result = assess_user_answer("No, use the blue bottle on the table instead.")
    assert result.disposition == "ANSWERED"
    assert result.raw_answer == "No, use the blue bottle on the table instead."


@pytest.mark.parametrize("text", ["I refuse to answer", "skip", "tôi từ chối"])
def test_explicit_refusal_is_declined(text):
    assert assess_user_answer(text).disposition == "DECLINED"


@pytest.mark.parametrize("text", ["I don't know", "not sure", "huh", "không biết"])
def test_unclear_answer_is_not_declined_or_timed_out(text):
    assert assess_user_answer(text).disposition == "UNCLEAR"


def test_contract_identifiers_reject_path_characters():
    assert validate_safe_identifier("task-0001-c01-i01", "interaction_id")
    with pytest.raises(ValueError):
        validate_safe_identifier("../../task", "interaction_id")
