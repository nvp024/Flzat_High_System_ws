"""Verify the compact producer/consumer contract without running a ROS graph."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from flzat_cognitive_pipeline.artifacts import HandoffArtifactWriter
from flzat_cognitive_pipeline.command import HighLevelHandoffAssembler
from flzat_cognitive_pipeline.domain import (
    CognitiveDecision, CognitiveGoal, ContextSnapshot, OutcomeKind,
    ProcessedOutcome, TaskRecord,
)
from flzat_high_bringup.low_system_mock import LowSystemMock


@pytest.fixture
def handoff_request(tmp_path):
    task = TaskRecord(
        run_id="run-20261104T000000Z", task_id="task-0001",
        request_id="task-0001-r01", original_transcript="Find the blue bottle",
        latest_transcript="Find the blue bottle",
    )
    snapshot = ContextSnapshot(task.request_id, "demo_1", "demo_1_test", ())
    outcome = ProcessedOutcome(
        OutcomeKind.READY_FOR_LOW,
        CognitiveDecision(
            speech="I'll look for your bottle.",
            goal=CognitiveGoal("Find the blue bottle", (), (), ("Bottle location",)),
            interaction=None,
        ),
    )
    handoff_id, payload = HighLevelHandoffAssembler().assemble(task, outcome, snapshot)
    path = HandoffArtifactWriter(tmp_path).write(
        task.run_id, task.task_id, handoff_id, payload
    )
    return SimpleNamespace(
        run_id=task.run_id, task_id=task.task_id,
        handoff_id=handoff_id, json_path=str(path),
    )


def test_low_reads_compact_handoff_from_real_assembler_and_writer(handoff_request):
    value = LowSystemMock._read_handoff(handoff_request)
    assert value["schema_version"] == "flzat.high_handoff.v2"
    assert value["goal"]["constraints"] == []
    assert value["goal"]["missing_information"] == ["Bottle location"]
    assert value["entities"] == []


@pytest.mark.parametrize("field,value", [
    ("run_id", "run-other"), ("task_id", "task-0002"),
    ("handoff_id", "task-0002-c01"),
])
def test_low_rejects_file_from_a_different_goal(handoff_request, field, value):
    setattr(handoff_request, field, value)
    with pytest.raises(ValueError, match="path does not match"):
        LowSystemMock._read_handoff(handoff_request)


@pytest.mark.parametrize("mutation", ["schema", "identity", "constraints", "missing"])
def test_low_rejects_invalid_compact_payload(handoff_request, mutation):
    path = Path(handoff_request.json_path)
    value = json.loads(path.read_text())
    if mutation == "schema":
        value["schema_version"] = "flzat.high_handoff.v1"
    elif mutation == "identity":
        value["handoff_id"] = "task-0002-c01"
    elif mutation == "constraints":
        value["goal"]["constraints"] = "not an array"
    else:
        del value["goal"]["missing_information"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        LowSystemMock._read_handoff(handoff_request)


def test_watcher_finds_compact_payload_but_not_request_audit(handoff_request):
    path = Path(handoff_request.json_path)
    audit = path.parent / "audit.json"
    audit.write_text('{"schema_version":"flzat.handoff_audit.v1"}', encoding="utf-8")
    tool = Path(__file__).resolve().parents[3] / "tools/pretty_handoff_watcher.py"
    functions = runpy.run_path(str(tool))
    assert functions["handoff_files"](path.parent.parent.parent) == [path]
