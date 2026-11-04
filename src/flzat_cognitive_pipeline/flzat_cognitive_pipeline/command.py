"""Assemble High-to-Low handoffs exclusively from validated trusted context."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from typing import Any, Callable

from flzat_cognitive_pipeline.domain import (
    CognitiveDecision,
    ContextItem,
    ContextSnapshot,
    OutcomeKind,
    ProcessedOutcome,
    TaskRecord,
)


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _handoff_id(task: TaskRecord) -> str:
    return f"{task.task_id}-c01"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HighLevelHandoffAssembler:
    """Build a grounded High-to-Low artifact without planning Low execution."""

    def __init__(
        self,
        *,
        handoff_id_factory: Callable[[TaskRecord], str] = _handoff_id,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._handoff_id_factory = handoff_id_factory
        self._clock = clock

    def assemble(
        self,
        task: TaskRecord,
        outcome: ProcessedOutcome,
        snapshot: ContextSnapshot,
    ) -> tuple[str, str]:
        if outcome.kind != OutcomeKind.READY_FOR_LOW:
            raise ValueError("only READY_FOR_LOW outcomes can become handoffs")
        decision = outcome.decision
        if not isinstance(decision, CognitiveDecision) or decision.goal is None:
            raise ValueError("handoff requires a validated cognitive goal")
        if snapshot.request_id != task.request_id:
            raise ValueError("cannot assemble from a stale context snapshot")
        handoff_id = self._handoff_id_factory(task).strip()
        if not handoff_id:
            raise ValueError("handoff ID factory returned an empty value")
        selected = [
            snapshot.resolve(context_ref)
            for context_ref in decision.goal.selected_context_refs
        ]
        payload = {
            "schema_version": "flzat.high_handoff.v2",
            "handoff_id": handoff_id,
            "user_request": {
                "original": task.original_transcript,
                "clarifications": [
                    {"question": question, "answer": answer}
                    for question, answer in task.clarification_history
                ],
            },
            "goal": {
                "description": decision.goal.description,
                "constraints": list(decision.goal.constraints),
                "missing_information": list(decision.goal.missing_information),
            },
            "context": {
                "environment_id": snapshot.environment_id,
                "map_id": snapshot.map_id,
                "source": "environment_memory",
                "current_positions_confirmed": False,
            },
            "entities": [self._grounded_entity(item) for item in selected],
        }
        return handoff_id, canonical_json(payload)

    def audit_record(
        self, task: TaskRecord, handoff_id: str, snapshot: ContextSnapshot
    ) -> dict[str, Any]:
        """Retain full snapshot provenance locally, outside the Low payload."""
        if snapshot.request_id != task.request_id:
            raise ValueError("cannot audit a stale context snapshot")
        return {
            "schema_version": "flzat.handoff_audit.v1",
            "run_id": task.run_id,
            "task_id": task.task_id,
            "request_id": task.request_id,
            "handoff_id": handoff_id,
            "created_at": self._clock(),
            "snapshot": asdict(snapshot),
        }

    @staticmethod
    def _grounded_entity(item: ContextItem) -> dict[str, Any]:
        entity: dict[str, Any] = {
            "id": item.object_id,
            "label": item.label,
        }
        # Omit empty/exactly redundant fields, never guess which details matter.
        if item.description and item.description.strip().casefold() != item.label.strip().casefold():
            entity["description"] = item.description
        if item.attributes:
            entity["attributes"] = dict(item.attributes)
        if item.relationships:
            entity["relationships"] = list(item.relationships)
        if item.scene:
            entity["scene"] = item.scene
        if item.position is not None:
            entity["position"] = item.position.as_dict()
        return entity
