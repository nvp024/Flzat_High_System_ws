import json

import pytest

from flzat_cognitive_pipeline.artifacts import (
    ArtifactConflictError,
    CommandArtifactWriter,
)
from flzat_cognitive_pipeline.command import TrustedCommandAssembler, canonical_json
from flzat_cognitive_pipeline.domain import (
    ContextItem,
    ContextSnapshot,
    ModelDecision,
    OutcomeKind,
    PipelineState,
    ProcessedOutcome,
    TaskRecord,
    TrustedPosition,
)


def task():
    return TaskRecord(
        run_id="run-20261014T000000Z",
        task_id="task-0001",
        request_id="task-0001-r01",
        original_transcript="get the cup",
        latest_transcript="get the cup",
        state=PipelineState.OUTPUT_VALIDATING,
    )


def context(position=True):
    return ContextSnapshot(
        "task-0001-r01",
        "demo",
        "map",
        (
            ContextItem(
                "memory:cup-1",
                "environment_object",
                "cup-1",
                "water_cup",
                "A water cup.",
                ("on:table-1",),
                "kitchen",
                0.9,
                TrustedPosition("map", 5.2, 3.4, 0.8) if position else None,
            ),
        ),
    )


def outcome(strategy="direct", refs=("memory:cup-1",)):
    return ProcessedOutcome(
        OutcomeKind.READY_FOR_LOW,
        ModelDecision(
            intent="fetch_object",
            target_type="object",
            target_name="water_cup",
            selected_context_refs=refs,
            execution_strategy=strategy,
            missing_information=(),
            clarification_question=None,
            speech="I will ask the robot to fetch it.",
        ),
    )


def assembler():
    return TrustedCommandAssembler(
        clock=lambda: "2026-10-14T00:00:00+00:00",
    )


def test_direct_command_uses_only_trusted_snapshot_coordinates():
    command_id, payload = assembler().assemble(task(), outcome(), context())
    value = json.loads(payload)

    assert command_id == "task-0001-c01"
    assert value["run_id"] == "run-20261014T000000Z"
    assert value["trusted_context"]["position"] == {
        "frame_id": "map",
        "x": 5.2,
        "y": 3.4,
        "z": 0.8,
    }
    assert value["target"]["memory_id"] == "cup-1"
    assert payload == canonical_json(value)


def test_search_command_has_semantic_hints_but_no_fake_position():
    _, payload = assembler().assemble(
        task(), outcome("search", ("memory:cup-1",)), context(position=False)
    )
    value = json.loads(payload)
    assert "position" not in value["trusted_context"]
    assert value["semantic_hints"] == ["scene:kitchen", "on:table-1"]


def test_direct_uses_matching_target_position_not_first_selected_reference():
    table = ContextItem(
        "memory:table-1",
        "environment_object",
        "table-1",
        "table",
        "A table.",
        (),
        "kitchen",
        0.95,
        TrustedPosition("map", 1.0, 2.0, 0.0),
    )
    cup = context().items[0]
    mixed = ContextSnapshot("task-0001-r01", "demo", "map", (table, cup))
    selected = outcome("direct", ("memory:table-1", "memory:cup-1"))

    _, payload = assembler().assemble(task(), selected, mixed)

    assert json.loads(payload)["trusted_context"]["position"] == {
        "frame_id": "map",
        "x": 5.2,
        "y": 3.4,
        "z": 0.8,
    }


def test_artifact_json_jsonl_and_topic_payload_are_byte_identical(tmp_path):
    command_id, payload = assembler().assemble(task(), outcome(), context())
    writer = CommandArtifactWriter(tmp_path)

    path = writer.write(
        "run-20261014T000000Z", "task-0001", command_id, payload
    )
    repeated = writer.write(
        "run-20261014T000000Z", "task-0001", command_id, payload
    )

    assert repeated == path
    assert path.read_text(encoding="utf-8") == payload
    audit = tmp_path / "run-20261014T000000Z" / "commands.jsonl"
    assert audit.read_text(encoding="utf-8") == payload + "\n"
    assert not list(path.parent.glob("*.tmp"))


def test_duplicate_command_with_other_bytes_is_rejected(tmp_path):
    command_id, payload = assembler().assemble(task(), outcome(), context())
    writer = CommandArtifactWriter(tmp_path)
    writer.write("run-20261014T000000Z", "task-0001", command_id, payload)
    changed = payload.replace("water_cup", "coffee_mug")

    with pytest.raises(ArtifactConflictError):
        writer.write(
            "run-20261014T000000Z", "task-0001", command_id, changed
        )
