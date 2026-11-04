import json
from pathlib import Path

from flzat_cognitive_pipeline.command import HighLevelHandoffAssembler
from flzat_cognitive_pipeline.domain import (
    CognitiveDecision,
    CognitiveGoal,
    ContextItem,
    ContextSnapshot,
    OutcomeKind,
    PipelineState,
    ProcessedOutcome,
    TaskRecord,
    TrustedPosition,
)
from flzat_cognitive_pipeline.output_processing import CognitiveOutputProcessor
from flzat_cognitive_pipeline.prompt_composer import PromptComposer


ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return ContextSnapshot(
        request_id="task-0001-r01",
        environment_id="hotel",
        map_id="hotel-map",
        items=(
            ContextItem(
                context_ref="memory:bottle-1",
                kind="environment_object",
                object_id="bottle-1",
                label="blue bottle",
                description="A blue bottle previously observed on the table.",
                relationships=("on:table-1",),
                scene="kitchen",
                retrieval_score=0.93,
                position=TrustedPosition("map", 2.0, 3.0, 0.8),
                attributes=(("color", "blue"),),
                first_seen_ros_ns=100,
                last_seen_ros_ns=200,
                localization_quality=0.8,
                confidence=0.9,
            ),
            ContextItem(
                context_ref="memory:table-1",
                kind="environment_object",
                object_id="table-1",
                label="table",
                description="A table in the kitchen.",
                relationships=("inside:kitchen",),
                scene="kitchen",
                retrieval_score=0.88,
                position=None,
            ),
        ),
    )


def response(*, speech="", goal=None, interaction=None):
    return json.dumps(
        {"speech": speech, "goal": goal, "interaction": interaction}
    )


def process(raw):
    return CognitiveOutputProcessor().process(
        raw,
        snapshot(),
        expected_request_id="task-0001-r01",
    )


def test_conversation_action_and_combined_decisions_are_not_route_blocked():
    conversation = process(
        response(speech="I am here with you. What would you like to talk about?")
    )
    action = process(
        response(
            speech="I will come closer so we can talk.",
            goal={
                "description": "Move near the user and remain available to talk",
                "constraints": ["Approach the user safely"],
                "selected_context_refs": [],
                "missing_information": [],
            },
        )
    )

    assert conversation.kind == OutcomeKind.RESPONSE_ONLY
    assert action.kind == OutcomeKind.READY_FOR_LOW
    assert action.decision.speech


def test_detailed_command_keeps_order_constraints_and_multiple_entities():
    outcome = process(
        response(
            speech="I understand the sequence.",
            goal={
                "description": (
                    "Go to the table, pick up the blue bottle, then remain "
                    "near the table so the user can continue talking"
                ),
                "constraints": [
                    "Approach the table before handling the bottle",
                    "Use the blue bottle on the table",
                    "Remain available for conversation afterward",
                ],
                "selected_context_refs": [
                    "memory:table-1",
                    "memory:bottle-1",
                ],
                "missing_information": [],
            },
        )
    )

    assert outcome.kind == OutcomeKind.READY_FOR_LOW
    assert "then remain" in outcome.decision.goal.description
    assert outcome.decision.goal.constraints == (
        "Approach the table before handling the bottle",
        "Use the blue bottle on the table",
        "Remain available for conversation afterward",
    )
    assert outcome.decision.goal.selected_context_refs == (
        "memory:table-1",
        "memory:bottle-1",
    )


def test_clarification_has_no_goal_and_unknown_refs_are_rejected():
    clarify = process(
        response(
            interaction={"kind": "ASK", "question": "Which bottle do you mean?"}
        )
    )
    unknown = process(
        response(
            goal={
                "description": "Bring the bottle",
                "constraints": [],
                "selected_context_refs": ["memory:unknown"],
                "missing_information": [],
            }
        )
    )
    both = process(
        response(
            goal={
                "description": "Bring a bottle",
                "constraints": [],
                "selected_context_refs": [],
                "missing_information": [],
            },
            interaction={"kind": "ASK", "question": "Which bottle?"},
        )
    )

    assert clarify.kind == OutcomeKind.NEEDS_CLARIFICATION
    assert unknown.kind == OutcomeKind.REJECTED
    assert both.kind == OutcomeKind.REJECTED


def test_prompt_contains_one_unified_runtime_context_without_router_mode():
    memory = snapshot()
    prompt = PromptComposer(str(ROOT / "prompts")).compose(
        transcript="Bring the blue bottle on the table and stay to talk.",
        context_items=[item.prompt_value() for item in memory.items],
        context_refs=memory.refs,
        dialogue_history=[
            {"mode": "cognitive", "user": "Hello", "assistant": "Hello."}
        ],
        mode="cognitive",
        trusted_context={"high_low_boundary": {"low_plans": True}},
        context_availability={
            "memory": {"status": "available", "reason": "2 candidates"},
            "vision": {"status": "unavailable", "reason": "no camera"},
        },
        interaction_context={
            "owner": "HIGH",
            "original_request": "Bring it here.",
            "turns": [
                {"question": "Which object?", "answer": "The blue bottle."}
            ],
        },
    )

    runtime = json.loads(prompt.user_prompt.split("RUNTIME_INPUT_JSON\n", 1)[1])
    assert prompt.output_schema == "CognitiveDecision.v1"
    assert "router_mode" not in runtime
    assert runtime["context_availability"]["vision"]["status"] == "unavailable"
    assert runtime["interaction_context"]["owner"] == "HIGH"
    assert len(runtime["rag_context"]) == 2


def test_handoff_keeps_every_entity_and_never_borrows_a_position():
    task = TaskRecord(
        run_id="run-20261103T000000Z",
        task_id="task-0001",
        request_id="task-0001-r01",
        original_transcript="Bring the blue bottle on the table.",
        latest_transcript="Bring the blue bottle on the table.",
        mode="cognitive",
        state=PipelineState.OUTPUT_VALIDATING,
    )
    decision = CognitiveDecision(
        speech="I will pass that goal to the robot.",
        goal=CognitiveGoal(
            description="Bring the blue bottle that is on the table to the user",
            constraints=("Use the blue bottle",),
            selected_context_refs=("memory:bottle-1", "memory:table-1"),
            missing_information=(),
        ),
        interaction=None,
    )
    outcome = ProcessedOutcome(OutcomeKind.READY_FOR_LOW, decision)

    assembler = HighLevelHandoffAssembler(
        clock=lambda: "2026-11-03T00:00:00+00:00"
    )
    handoff_id, payload = assembler.assemble(task, outcome, snapshot())
    value = json.loads(payload)

    assert value["schema_version"] == "flzat.high_handoff.v2"
    assert set(value) == {
        "schema_version", "handoff_id", "user_request", "goal", "context", "entities"
    }
    assert value["goal"]["description"].startswith("Bring the blue bottle")
    assert value["goal"]["constraints"] == ["Use the blue bottle"]
    assert value["goal"]["missing_information"] == []
    assert "resolved" not in value["user_request"]
    assert value["context"] == {
        "environment_id": "hotel", "map_id": "hotel-map",
        "source": "environment_memory", "current_positions_confirmed": False,
    }
    assert [item["id"] for item in value["entities"]] == [
        "bottle-1",
        "table-1",
    ]
    bottle, table = value["entities"]
    assert bottle["position"]["frame_id"] == "map"
    assert bottle["position"]["x"] == 2.0
    assert bottle["attributes"] == {"color": "blue"}
    assert bottle["relationships"] == ["on:table-1"]
    assert "source" not in bottle
    assert "context_ref" not in bottle
    assert "attributes" not in table
    assert "position" not in table
    audit = assembler.audit_record(task, handoff_id, snapshot())
    assert audit["snapshot"]["items"][0]["last_seen_ros_ns"] == 200
    assert audit["snapshot"]["items"][0]["context_ref"] == "memory:bottle-1"
    assert audit["created_at"] == "2026-11-03T00:00:00+00:00"


def test_compact_handoff_keeps_clarification_and_missing_details_without_entities():
    task = TaskRecord(
        run_id="run-20261104T000000Z", task_id="task-0001",
        request_id="task-0001-r01", original_transcript="Bring me that.",
        latest_transcript="The blue bottle, without moving the red one.",
        clarification_history=[
            ("Which object?", "The blue bottle, without moving the red one.")
        ],
    )
    outcome = ProcessedOutcome(
        OutcomeKind.READY_FOR_LOW,
        CognitiveDecision(
            speech="I'll look for your blue bottle.",
            goal=CognitiveGoal(
                "Bring the blue bottle to the user", ("Do not move the red bottle",),
                (), ("Bottle location",),
            ),
            interaction=None,
        ),
    )
    _, payload = HighLevelHandoffAssembler().assemble(task, outcome, snapshot())
    value = json.loads(payload)
    assert value["entities"] == []
    assert value["goal"]["missing_information"] == ["Bottle location"]
    assert value["user_request"]["original"] == "Bring me that."
    assert value["user_request"]["clarifications"] == [{
        "question": "Which object?",
        "answer": "The blue bottle, without moving the red one.",
    }]


def test_handoff_rejects_stale_snapshot():
    task = TaskRecord(
        run_id="run-20261103T000000Z",
        task_id="task-0001",
        request_id="other-request",
        original_transcript="Bring the bottle.",
        latest_transcript="Bring the bottle.",
    )
    outcome = ProcessedOutcome(
        OutcomeKind.READY_FOR_LOW,
        CognitiveDecision(
            speech="",
            goal=CognitiveGoal("Bring the bottle", (), (), ()),
            interaction=None,
        ),
    )

    try:
        HighLevelHandoffAssembler().assemble(task, outcome, snapshot())
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale snapshot was accepted")
