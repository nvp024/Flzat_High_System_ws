import json

import pytest

from flzat_memory.storage.memory_record import (
    EMBEDDING_MODEL,
    IncomingMemoryObservation,
    MapPosition,
    MemoryRecord,
    RecordValidationError,
    RobotPose,
    embedding_text,
    new_record,
)


def observation(**overrides):
    values = {
        "environment_id": "hotel_demo",
        "map_id": "map-session",
        "observation_id": "observation-1",
        "detector_class": "bottle",
        "label": "water_bottle",
        "description": "A blue water bottle on the counter.",
        "attributes": (("color", "blue"),),
        "relationships": ("on the counter",),
        "scene": "hotel_lobby",
        "map_position": MapPosition("map", 5.4, 3.2, 0.9),
        "robot_pose": RobotPose(4.8, 2.7, 0.0, 0.6),
        "observed_utc": "2026-08-27T10:00:00+00:00",
        "observed_ros_ns": 123_000_000_000,
        "detector_confidence": 0.94,
        "semantic_confidence": 0.92,
        "localization_quality": 0.90,
    }
    values.update(overrides)
    return IncomingMemoryObservation(**values)


def test_canonical_record_round_trip_and_confidence():
    record = new_record(observation(), "object-1", "images/observation_1.jpg")

    recovered = MemoryRecord.from_json(record.to_json())

    assert recovered == record
    assert recovered.schema_version == "environment_memory.v1"
    assert recovered.embedding_model == EMBEDDING_MODEL
    assert recovered.confidence == pytest.approx(0.90)
    assert recovered.seen_count == 1


def test_embedding_text_contains_only_planned_semantics():
    text = embedding_text(observation())

    assert text == (
        "water_bottle. A blue water bottle on the counter. "
        "Scene: hotel_lobby. Attributes: color: blue. "
        "Relationships: on the counter."
    )
    assert "5.4" not in text
    assert "object_id" not in text
    assert "confidence" not in text


def test_rejects_non_map_position_and_invalid_confidence():
    with pytest.raises(RecordValidationError, match="map_position frame"):
        new_record(
            observation(map_position=MapPosition("odom", 1.0, 2.0, 3.0)),
            "object-1",
            "images/a.jpg",
        )
    with pytest.raises(RecordValidationError, match="semantic_confidence"):
        new_record(
            observation(semantic_confidence=1.2),
            "object-1",
            "images/a.jpg",
        )


def test_stored_json_has_structured_numeric_metadata():
    payload = json.loads(
        new_record(observation(), "object-1", "images/a.jpg").to_json()
    )

    assert payload["map_position"] == {
        "frame_id": "map",
        "x": 5.4,
        "y": 3.2,
        "z": 0.9,
    }
    assert isinstance(payload["first_seen_ros_ns"], int)
    assert payload["attributes"] == {"color": "blue"}
