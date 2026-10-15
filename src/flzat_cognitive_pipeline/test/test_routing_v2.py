import json

import pytest

from flzat_cognitive_pipeline.domain import (
    ModelDecision,
    OutcomeKind,
    ProcessedOutcome,
)
from flzat_cognitive_pipeline.routing import (
    LowStatusAdapter,
    ResponseRouter,
    build_low_cancel,
)


def decision(question=None, speech="hello"):
    return ModelDecision(
        "answer",
        "conversation",
        "user",
        (),
        "response_only",
        (),
        question,
        speech,
    )


def test_response_router_never_publishes_rejected_or_clarification():
    router = ResponseRouter("safe fallback")
    rejected = router.route(ProcessedOutcome(OutcomeKind.REJECTED, error="bad"))
    clarify = router.route(
        ProcessedOutcome(
            OutcomeKind.NEEDS_CLARIFICATION,
            decision("Which one?", ""),
        )
    )
    assert rejected.publish_command is False
    assert rejected.speech == "safe fallback"
    assert clarify.publish_command is False
    assert clarify.speech == "Which one?"


def test_low_status_is_strict_and_correlated():
    raw = json.dumps(
        {
            "schema_version": "flzat.low_status.v2",
            "run_id": "run-20261014T000000Z",
            "command_id": "task-0001-c01",
            "state": "running",
            "error": "",
        }
    )
    status = LowStatusAdapter().parse(
        raw, "run-20261014T000000Z", "task-0001-c01"
    )
    assert status.state == "running"
    with pytest.raises(ValueError, match="unknown or stale"):
        LowStatusAdapter().parse(
            raw, "run-20261014T000000Z", "task-0001-c02"
        )


def test_low_cancel_is_versioned_canonical_and_correlated():
    raw = build_low_cancel(
        "run-20261014T000000Z",
        "task-0001-c01",
        "task-0001",
        "task-0001-r01",
    )

    assert raw == (
        '{"command_id":"task-0001-c01","reason":"user_cancelled",'
        '"request_id":"task-0001-r01","run_id":"run-20261014T000000Z",'
        '"schema_version":"flzat.low_cancel.v2","task_id":"task-0001"}'
    )
    with pytest.raises(ValueError, match="nonempty"):
        build_low_cancel("", "task-0001-c01", "task-0001", "task-0001-r01")
