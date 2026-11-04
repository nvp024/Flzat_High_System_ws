import json

import pytest

from flzat_cognitive_pipeline.domain import ContextSnapshot, OutcomeKind
from flzat_cognitive_pipeline.output_validator import (
    CognitiveOutputValidator,
    RepairPolicy,
)


def snapshot(request_id="request-1"):
    return ContextSnapshot(request_id, "demo", "map", ())


def validate(value, request_id="request-1"):
    raw = value if isinstance(value, str) else json.dumps(value)
    return CognitiveOutputValidator().validate(
        raw,
        snapshot(request_id),
        expected_request_id=request_id,
    )


def test_response_only_goal_and_clarification_are_distinct():
    reply = validate({"speech": "Hello.", "goal": None, "interaction": None})
    goal = validate(
        {
            "speech": "I will pass that goal to the robot.",
            "goal": {
                "description": "Move near the user",
                "constraints": ["Approach safely"],
                "selected_context_refs": [],
                "missing_information": [],
            },
            "interaction": None,
        }
    )
    ask = validate(
        {
            "speech": "",
            "goal": None,
            "interaction": {"kind": "ASK", "question": "Which bottle?"},
        }
    )

    assert reply.kind == OutcomeKind.RESPONSE_ONLY
    assert goal.kind == OutcomeKind.READY_FOR_LOW
    assert ask.kind == OutcomeKind.NEEDS_CLARIFICATION


def test_json_fence_is_normalized_but_schema_is_not_repaired():
    fenced = validate(
        "```json\n"
        '{"speech":"Hello.","goal":null,"interaction":null}'
        "\n```"
    )
    malformed = validate('{"speech":"Hello."}')

    assert fenced.valid
    assert not malformed.valid
    assert "schema" in malformed.error


def test_stale_snapshot_is_rejected():
    validator = CognitiveOutputValidator()
    result = validator.validate(
        '{"speech":"Hello.","goal":null,"interaction":null}',
        snapshot("old-request"),
        expected_request_id="new-request",
    )

    assert not result.valid
    assert "stale" in result.error


@pytest.mark.parametrize("maximum", [-1, 3])
def test_repair_policy_has_bounded_attempts(maximum):
    with pytest.raises(ValueError):
        RepairPolicy(maximum)


def test_repair_policy_stops_after_configured_limit():
    policy = RepairPolicy(2)
    assert policy.next_repair_number(1) == 1
    assert policy.next_repair_number(2) == 2
    assert policy.next_repair_number(3) is None
