"""Pure domain models for the FLZAT High-Level Cognitive Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineState(str, Enum):
    NEW = "NEW"
    CONTEXT_RETRIEVAL = "CONTEXT_RETRIEVAL"
    PROMPT_READY = "PROMPT_READY"
    VLM_QUEUED = "VLM_QUEUED"
    VLM_ACTIVE = "VLM_ACTIVE"
    OUTPUT_VALIDATING = "OUTPUT_VALIDATING"
    WAITING_CLARIFICATION = "WAITING_CLARIFICATION"
    RESPONSE_ONLY = "RESPONSE_ONLY"
    REJECTED = "REJECTED"
    HANDOFF_READY = "HANDOFF_READY"
    WRITING_ARTIFACT = "WRITING_ARTIFACT"
    HANDOFF_PENDING = "HANDOFF_PENDING"
    HANDOFF_ACCEPTED = "HANDOFF_ACCEPTED"
    LOW_RUNNING = "LOW_RUNNING"
    REFUSED = "REFUSED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


TERMINAL_STATES = {
    PipelineState.RESPONSE_ONLY,
    PipelineState.REJECTED,
    PipelineState.REFUSED,
    PipelineState.SUCCEEDED,
    PipelineState.FAILED,
    PipelineState.CANCELLED,
    PipelineState.TIMED_OUT,
}


class OutcomeKind(str, Enum):
    READY_FOR_LOW = "READY_FOR_LOW"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    RESPONSE_ONLY = "RESPONSE_ONLY"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class TrustedPosition:
    frame_id: str
    x: float
    y: float
    z: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class MemoryCandidate:
    object_id: str
    environment_id: str
    map_id: str
    label: str
    description: str
    relationships: tuple[str, ...]
    scene: str
    score: float
    position: TrustedPosition | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    first_seen_ros_ns: int = 0
    last_seen_ros_ns: int = 0
    localization_quality: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class ContextItem:
    context_ref: str
    kind: str
    object_id: str
    label: str
    description: str
    relationships: tuple[str, ...]
    scene: str
    retrieval_score: float
    position: TrustedPosition | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    first_seen_ros_ns: int = 0
    last_seen_ros_ns: int = 0
    localization_quality: float = 0.0
    confidence: float = 0.0

    def prompt_value(self) -> dict[str, Any]:
        """Return model-visible semantics without trusted coordinates."""
        return {
            "context_ref": self.context_ref,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "attributes": {key: value for key, value in self.attributes},
            "relationships": list(self.relationships),
            "scene": self.scene,
            "has_position": self.position is not None,
            "first_seen_ros_ns": self.first_seen_ros_ns,
            "last_seen_ros_ns": self.last_seen_ros_ns,
            "localization_quality": round(self.localization_quality, 6),
            "confidence": round(self.confidence, 6),
            "source": "environment_memory",
            "retrieval_score": round(self.retrieval_score, 6),
        }


@dataclass(frozen=True)
class ContextSnapshot:
    request_id: str
    environment_id: str
    map_id: str
    items: tuple[ContextItem, ...]
    truncated: bool = False

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(item.context_ref for item in self.items)

    def resolve(self, context_ref: str) -> ContextItem:
        for item in self.items:
            if item.context_ref == context_ref:
                return item
        raise KeyError(f"unknown context_ref: {context_ref}")


@dataclass(frozen=True)
class CognitiveGoal:
    """Natural-language objective produced by High, not an execution plan."""

    description: str
    constraints: tuple[str, ...]
    selected_context_refs: tuple[str, ...]
    missing_information: tuple[str, ...]


@dataclass(frozen=True)
class CognitiveInteraction:
    """One High-owned request for clarification."""

    kind: str
    question: str


@dataclass(frozen=True)
class CognitiveDecision:
    """Unified companion decision: speech, an optional goal, or clarification."""

    speech: str
    goal: CognitiveGoal | None
    interaction: CognitiveInteraction | None


@dataclass(frozen=True)
class ProcessedOutcome:
    kind: OutcomeKind
    decision: CognitiveDecision | None = None
    error: str = ""


@dataclass(frozen=True)
class TransitionEvent:
    state: PipelineState
    timestamp: str
    reason: str


@dataclass
class TaskRecord:
    run_id: str
    task_id: str
    request_id: str
    original_transcript: str
    latest_transcript: str
    mode: str = "cognitive"
    state: PipelineState = PipelineState.NEW
    clarification_history: list[tuple[str, str]] = field(default_factory=list)
    pending_question: str = ""
    handoff_id: str = ""
    transitions: list[TransitionEvent] = field(default_factory=list)

    @property
    def query_text(self) -> str:
        parts = [self.original_transcript]
        for question, answer in self.clarification_history:
            parts.append(f"Question: {question}")
            parts.append(f"Answer: {answer}")
        return "\n".join(parts)
