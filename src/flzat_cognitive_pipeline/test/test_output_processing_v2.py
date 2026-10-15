import json

from flzat_cognitive_pipeline.domain import (
    ContextItem,
    ContextSnapshot,
    OutcomeKind,
    TrustedPosition,
)
from flzat_cognitive_pipeline.output_processing import StrictOutputProcessor


def snapshot(position=True):
    return ContextSnapshot(
        request_id="req-1",
        environment_id="demo",
        map_id="map",
        items=(
            ContextItem(
                context_ref="memory:cup-1",
                kind="environment_object",
                object_id="cup-1",
                label="water_cup",
                description="A cup on a table.",
                relationships=("on:table-1",),
                scene="kitchen",
                retrieval_score=0.9,
                position=TrustedPosition("map", 5.2, 3.4, 0.8) if position else None,
            ),
        ),
    )


def response(**overrides):
    value = {
        "intent": "fetch_object",
        "target": {"type": "object", "name": "water_cup"},
        "selected_context_refs": ["memory:cup-1"],
        "execution_strategy": "direct",
        "missing_information": [],
        "clarification_question": None,
        "speech": "I will ask the robot to fetch the cup.",
    }
    value.update(overrides)
    return json.dumps(value)


def process(raw, value=None, request_id="req-1"):
    return StrictOutputProcessor().process(
        raw, value or snapshot(), expected_request_id=request_id
    )


def test_valid_direct_and_search_decisions_are_ready():
    direct = process(response())
    search = process(
        response(
            selected_context_refs=[],
            execution_strategy="search",
        ),
        snapshot(position=False),
    )
    assert direct.kind == OutcomeKind.READY_FOR_LOW
    assert search.kind == OutcomeKind.READY_FOR_LOW


def test_clarification_and_response_only_are_classified():
    clarify = process(
        response(
            selected_context_refs=[],
            execution_strategy="clarify",
            missing_information=["which cup"],
            clarification_question="Which cup do you mean?",
        )
    )
    answer = process(
        response(
            intent="answer",
            target={"type": "conversation", "name": "user"},
            selected_context_refs=[],
            execution_strategy="response_only",
            speech="The kitchen is on your left.",
        )
    )
    assert clarify.kind == OutcomeKind.NEEDS_CLARIFICATION
    assert answer.kind == OutcomeKind.RESPONSE_ONLY


def test_malformed_extra_field_and_unknown_intent_are_rejected():
    assert process("not-json").kind == OutcomeKind.REJECTED
    extra = json.loads(response())
    extra["coordinates"] = {"x": 99}
    assert process(json.dumps(extra)).kind == OutcomeKind.REJECTED
    assert process(response(intent="fly_robot")).kind == OutcomeKind.REJECTED


def test_single_outer_json_fence_is_normalized():
    result = process("```json\n" + response() + "\n```")
    assert result.kind == OutcomeKind.READY_FOR_LOW


def test_incomplete_or_nested_fences_are_rejected():
    incomplete = process("```json\n" + response())
    nested = process("```json\n```\n" + response() + "\n```")
    assert incomplete.kind == OutcomeKind.REJECTED
    assert nested.kind == OutcomeKind.REJECTED


def test_unknown_and_stale_context_are_rejected():
    unknown = process(response(selected_context_refs=["memory:other"]))
    stale = process(response(), request_id="req-other")
    assert unknown.kind == OutcomeKind.REJECTED
    assert "unknown context_ref" in unknown.error
    assert stale.kind == OutcomeKind.REJECTED
    assert "stale" in stale.error


def test_direct_requires_trusted_position_and_no_unresolved_fields():
    no_position = process(response(), snapshot(position=False))
    unresolved = process(response(missing_information=["location"]))
    assert no_position.kind == OutcomeKind.REJECTED
    assert "trusted position" in no_position.error
    assert unresolved.kind == OutcomeKind.REJECTED
