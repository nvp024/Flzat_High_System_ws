import json

import pytest

from flzat_cognitive_pipeline.artifacts import (
    ArtifactConflictError,
    HandoffArtifactWriter,
    RequestArtifactWriter,
)
from flzat_cognitive_pipeline.command import canonical_json


def handoff_payload() -> str:
    return canonical_json(
        {
            "schema_version": "flzat.high_handoff.v2",
            "handoff_id": "task-0001-c01",
            "goal": {"description": "Bring the cup"},
        }
    )


def test_handoff_json_and_audit_payload_are_byte_identical(tmp_path):
    payload = handoff_payload()
    writer = HandoffArtifactWriter(tmp_path)

    path = writer.write(
        "run-20261014T000000Z", "task-0001", "task-0001-c01", payload
    )
    repeated = writer.write(
        "run-20261014T000000Z", "task-0001", "task-0001-c01", payload
    )

    assert repeated == path
    assert path.read_text(encoding="utf-8") == payload
    audit = tmp_path / "run-20261014T000000Z" / "handoffs.jsonl"
    assert audit.read_text(encoding="utf-8") == payload + "\n"
    assert not list(path.parent.glob("*.tmp"))


def test_duplicate_handoff_with_other_bytes_is_rejected(tmp_path):
    payload = handoff_payload()
    writer = HandoffArtifactWriter(tmp_path)
    writer.write("run-20261014T000000Z", "task-0001", "task-0001-c01", payload)

    with pytest.raises(ArtifactConflictError):
        writer.write(
            "run-20261014T000000Z",
            "task-0001",
            "task-0001-c01",
            payload.replace("Bring the cup", "Bring the book"),
        )


@pytest.mark.parametrize("field,value", [
    ("schema_version", "flzat.high_handoff.v1"),
    ("handoff_id", "task-0002-c01"),
])
def test_handoff_writer_rejects_wrong_schema_or_id(tmp_path, field, value):
    payload = json.loads(handoff_payload())
    payload[field] = value
    with pytest.raises(ValueError, match="schema or handoff ID"):
        HandoffArtifactWriter(tmp_path).write(
            "run-20261014T000000Z", "task-0001", "task-0001-c01",
            canonical_json(payload),
        )
    assert not list(tmp_path.rglob("*.json"))


def test_request_artifacts_are_atomic_correlated_and_immutable(tmp_path):
    writer = RequestArtifactWriter(tmp_path)
    value = {
        "schema_version": "flzat.transcript.v1",
        "task_id": "task-0001",
        "request_id": "task-0001-r01",
        "transcript": "get the book",
    }

    path = writer.write_json(
        "run-20261014T000000Z",
        "task-0001",
        "task-0001-r01",
        "transcript.json",
        value,
    )
    repeated = writer.write_json(
        "run-20261014T000000Z",
        "task-0001",
        "task-0001-r01",
        "transcript.json",
        value,
    )

    assert repeated == path
    assert path.parent.name == "task-0001-r01"
    assert json.loads(path.read_text(encoding="utf-8")) == value
    assert not list(path.parent.glob("*.tmp"))

    with pytest.raises(ArtifactConflictError):
        writer.write_json(
            "run-20261014T000000Z",
            "task-0001",
            "task-0001-r01",
            "transcript.json",
            {**value, "transcript": "different"},
        )


def test_request_artifact_rejects_path_traversal(tmp_path):
    writer = RequestArtifactWriter(tmp_path)
    with pytest.raises(ValueError, match="filename"):
        writer.write_text(
            "run-20261014T000000Z",
            "task-0001",
            "task-0001-r01",
            "../secret.txt",
            "no",
        )
