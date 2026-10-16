import json

import pytest

from flzat_cognitive_pipeline.domain import (
    ContextItem,
    ContextSnapshot,
    OutcomeKind,
    TrustedPosition,
)
from flzat_cognitive_pipeline.output_validator import (
    ModeOutputValidator,
    RepairPolicy,
)


def snapshot():
    return ContextSnapshot(
        request_id="task-0001-r01",
        environment_id="demo",
        map_id="map",
        items=(
            ContextItem(
                context_ref="memory:book-1",
                kind="environment_object",
                object_id="book-1",
                label="book",
                description="A book on a table.",
                relationships=("on:table-1",),
                scene="living_room",
                retrieval_score=0.9,
                position=TrustedPosition("map", 1.0, 2.0, 0.5),
            ),
        ),
    )


def validate(mode, value):
    return ModeOutputValidator(["navigate", "fetch_object"]).validate(
        mode,
        json.dumps(value),
        snapshot(),
        expected_request_id="task-0001-r01",
    )


def test_visual_qa_schema_is_strict():
    result = validate(
        "visual_qa",
        {
            "reply": "There is a book on the table.",
            "observed_objects": ["book", "table"],
            "confidence": 0.91,
        },
    )
    assert result.valid
    assert result.kind == OutcomeKind.RESPONSE_ONLY
    assert result.value["observed_objects"] == ["book", "table"]


def test_general_qa_cannot_request_robot_action():
    accepted = validate(
        "general_qa",
        {
            "reply": "Beijing is the capital of China.",
            "speak": True,
            "requires_robot_action": False,
        },
    )
    rejected = validate(
        "general_qa",
        {
            "reply": "I will navigate there.",
            "speak": True,
            "requires_robot_action": True,
        },
    )
    assert accepted.valid
    assert not rejected.valid
    assert "cannot require robot action" in rejected.error


def test_social_skill_must_exist_in_trusted_registry():
    accepted = validate(
        "social_dialogue",
        {"reply": "Would you like me to help?", "speak": True, "suggested_skill": "navigate"},
    )
    rejected = validate(
        "social_dialogue",
        {"reply": "I can fly.", "speak": True, "suggested_skill": "fly"},
    )
    assert accepted.valid
    assert not rejected.valid
    assert "trusted registry" in rejected.error


def test_clarification_is_one_question_and_never_a_command():
    accepted = validate("clarification", {"reply": "Which book do you mean?"})
    rejected = validate("clarification", {"reply": "I will get the book."})
    assert accepted.kind == OutcomeKind.NEEDS_CLARIFICATION
    assert accepted.valid
    assert not rejected.valid


def test_cross_mode_schema_is_rejected():
    visual = {
        "reply": "A book.",
        "observed_objects": ["book"],
        "confidence": 0.8,
    }
    assert not validate("general_qa", visual).valid
    assert not validate("social_dialogue", visual).valid
    assert not validate("clarification", visual).valid


def test_task_validator_rejects_model_generated_coordinates():
    value = {
        "intent": "fetch_object",
        "target": {"type": "object", "name": "book"},
        "selected_context_refs": ["memory:book-1"],
        "execution_strategy": "direct",
        "missing_information": [],
        "clarification_question": None,
        "speech": "I will fetch the book.",
        "coordinates": {"x": 99.0, "y": 99.0},
    }
    result = validate("task_command", value)
    assert not result.valid
    assert "fields" in result.error


def test_task_validator_produces_original_trusted_outcome():
    value = {
        "intent": "fetch_object",
        "target": {"type": "object", "name": "book"},
        "selected_context_refs": ["memory:book-1"],
        "execution_strategy": "direct",
        "missing_information": [],
        "clarification_question": None,
        "speech": "I will fetch the book.",
    }
    result = validate("task_command", value)
    assert result.valid
    assert result.kind == OutcomeKind.READY_FOR_LOW
    assert result.task_outcome is not None


def test_json_fence_is_normalized_but_nested_fence_is_rejected():
    validator = ModeOutputValidator([])
    raw = '```json\n{"reply":"Beijing.","speak":true,"requires_robot_action":false}\n```'
    result = validator.validate(
        "general_qa", raw, snapshot(), expected_request_id="task-0001-r01"
    )
    nested = validator.validate(
        "general_qa",
        "```json\n```\n{}\n```",
        snapshot(),
        expected_request_id="task-0001-r01",
    )
    assert result.valid
    assert not nested.valid


def test_repair_policy_stops_after_two_repairs():
    policy = RepairPolicy(2)
    assert policy.next_repair_number(1) == 1
    assert policy.next_repair_number(2) == 2
    assert policy.next_repair_number(3) is None
    with pytest.raises(ValueError):
        RepairPolicy(3)

