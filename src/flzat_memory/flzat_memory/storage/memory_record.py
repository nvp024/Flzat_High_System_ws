"""Define the canonical Version 1 object-memory schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "environment_memory.v1"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_DIMENSION = 384


class RecordValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MapPosition:
    frame_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class IncomingMemoryObservation:
    environment_id: str
    map_id: str
    observation_id: str
    detector_class: str
    label: str
    description: str
    attributes: tuple[tuple[str, str], ...]
    relationships: tuple[str, ...]
    scene: str
    map_position: MapPosition
    robot_pose: RobotPose
    observed_utc: str
    observed_ros_ns: int
    detector_confidence: float
    semantic_confidence: float
    localization_quality: float

    @property
    def confidence(self) -> float:
        return min(
            self.detector_confidence,
            self.semantic_confidence,
            self.localization_quality,
        )


@dataclass(frozen=True)
class MemoryRecord:
    schema_version: str
    object_id: str
    environment_id: str
    map_id: str
    detector_class: str
    label: str
    description: str
    attributes: tuple[tuple[str, str], ...]
    relationships: tuple[str, ...]
    scene: str
    map_position: MapPosition
    robot_pose: RobotPose
    first_seen_utc: str
    last_seen_utc: str
    first_seen_ros_ns: int
    last_seen_ros_ns: int
    seen_count: int
    detector_confidence: float
    semantic_confidence: float
    localization_quality: float
    confidence: float
    image_ref: str
    embedding_model: str

    def to_json(self) -> str:
        payload = asdict(self)
        payload["attributes"] = dict(self.attributes)
        payload["relationships"] = list(self.relationships)
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_json(cls, value: str) -> "MemoryRecord":
        try:
            payload = json.loads(value)
            payload["map_position"] = MapPosition(**payload["map_position"])
            payload["robot_pose"] = RobotPose(**payload["robot_pose"])
            payload["attributes"] = tuple(payload["attributes"].items())
            payload["relationships"] = tuple(payload["relationships"])
            record = cls(**payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecordValidationError(f"invalid stored record: {exc}") from exc
        validate_record(record)
        return record


def validate_incoming(value: IncomingMemoryObservation) -> None:
    _identifier(value.environment_id, "environment_id")
    _identifier(value.map_id, "map_id")
    _identifier(value.observation_id, "observation_id")
    _semantic_identifier(value.detector_class, "detector_class")
    _semantic_identifier(value.label, "label")
    _semantic_identifier(value.scene, "scene")
    _short_text(value.description, "description", 240)
    if value.map_position.frame_id != "map":
        raise RecordValidationError("map_position frame must be 'map'")
    if value.observed_ros_ns <= 0:
        raise RecordValidationError("observed ROS timestamp must be positive")
    _utc_timestamp(value.observed_utc, "observed_utc")
    if len(value.attributes) > 8 or len(value.relationships) > 5:
        raise RecordValidationError("semantic collection limits exceeded")
    seen_keys = set()
    for key, item in value.attributes:
        normalized = _semantic_identifier(key, "attribute key")
        if normalized in seen_keys:
            raise RecordValidationError("attribute keys must be unique")
        seen_keys.add(normalized)
        _short_text(item, f"attribute {key}", 120)
    for relationship in value.relationships:
        _short_text(relationship, "relationship", 120)
    _finite(value.map_position.x, "map x")
    _finite(value.map_position.y, "map y")
    _finite(value.map_position.z, "map z")
    _finite(value.robot_pose.x, "robot x")
    _finite(value.robot_pose.y, "robot y")
    _finite(value.robot_pose.z, "robot z")
    _finite(value.robot_pose.yaw, "robot yaw")
    for name, confidence in (
        ("detector_confidence", value.detector_confidence),
        ("semantic_confidence", value.semantic_confidence),
        ("localization_quality", value.localization_quality),
    ):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RecordValidationError(f"{name} must be finite and in [0, 1]")


def validate_record(value: MemoryRecord) -> None:
    if value.schema_version != SCHEMA_VERSION:
        raise RecordValidationError("unsupported record schema")
    if value.embedding_model != EMBEDDING_MODEL:
        raise RecordValidationError("unexpected embedding model")
    if value.seen_count < 1:
        raise RecordValidationError("seen_count must be positive")
    if value.first_seen_ros_ns > value.last_seen_ros_ns:
        raise RecordValidationError("first_seen must not follow last_seen")
    first_seen_utc = _utc_timestamp(value.first_seen_utc, "first_seen_utc")
    last_seen_utc = _utc_timestamp(value.last_seen_utc, "last_seen_utc")
    if first_seen_utc > last_seen_utc:
        raise RecordValidationError("first_seen UTC must not follow last_seen UTC")
    incoming = IncomingMemoryObservation(
        environment_id=value.environment_id,
        map_id=value.map_id,
        observation_id=value.object_id,
        detector_class=value.detector_class,
        label=value.label,
        description=value.description,
        attributes=value.attributes,
        relationships=value.relationships,
        scene=value.scene,
        map_position=value.map_position,
        robot_pose=value.robot_pose,
        observed_utc=value.last_seen_utc,
        observed_ros_ns=value.last_seen_ros_ns,
        detector_confidence=value.detector_confidence,
        semantic_confidence=value.semantic_confidence,
        localization_quality=value.localization_quality,
    )
    validate_incoming(incoming)
    if not value.object_id or not value.image_ref:
        raise RecordValidationError("object_id and image_ref are required")
    expected = min(
        value.detector_confidence,
        value.semantic_confidence,
        value.localization_quality,
    )
    if not math.isclose(value.confidence, expected, abs_tol=1e-6):
        raise RecordValidationError("overall confidence is inconsistent")


def new_record(
    observation: IncomingMemoryObservation, object_id: str, image_ref: str
) -> MemoryRecord:
    validate_incoming(observation)
    record = MemoryRecord(
        schema_version=SCHEMA_VERSION,
        object_id=object_id,
        environment_id=observation.environment_id,
        map_id=observation.map_id,
        detector_class=observation.detector_class,
        label=observation.label,
        description=observation.description,
        attributes=observation.attributes,
        relationships=observation.relationships,
        scene=observation.scene,
        map_position=observation.map_position,
        robot_pose=observation.robot_pose,
        first_seen_utc=observation.observed_utc,
        last_seen_utc=observation.observed_utc,
        first_seen_ros_ns=observation.observed_ros_ns,
        last_seen_ros_ns=observation.observed_ros_ns,
        seen_count=1,
        detector_confidence=observation.detector_confidence,
        semantic_confidence=observation.semantic_confidence,
        localization_quality=observation.localization_quality,
        confidence=observation.confidence,
        image_ref=image_ref,
        embedding_model=EMBEDDING_MODEL,
    )
    validate_record(record)
    return record


def embedding_text(value: MemoryRecord | IncomingMemoryObservation) -> str:
    attributes = ", ".join(f"{key}: {item}" for key, item in value.attributes)
    relationships = ", ".join(value.relationships)
    description = value.description.strip()
    description_suffix = "" if description.endswith((".", "!", "?")) else "."
    return (
        f"{value.label}. {description}{description_suffix} Scene: {value.scene}. "
        f"Attributes: {attributes}. Relationships: {relationships}."
    )


def with_confidences(record: MemoryRecord) -> MemoryRecord:
    return replace(
        record,
        confidence=min(
            record.detector_confidence,
            record.semantic_confidence,
            record.localization_quality,
        ),
    )


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise RecordValidationError(f"{name} must be a nonempty safe identifier")
    return value


def _short_text(value: str, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise RecordValidationError(f"{name} must contain 1 to {maximum} characters")


def _semantic_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_]+", value):
        raise RecordValidationError(f"{name} must be lowercase snake_case")
    return value


def _finite(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise RecordValidationError(f"{name} must be finite")


def _utc_timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RecordValidationError(f"{name} must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0.0:
        raise RecordValidationError(f"{name} must use UTC")
    return parsed
