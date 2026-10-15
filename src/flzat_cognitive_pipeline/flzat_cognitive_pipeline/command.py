"""Assemble Low-System commands exclusively from validated trusted context."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Callable

from flzat_cognitive_pipeline.domain import (
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


def _command_id(task: TaskRecord) -> str:
    return f"{task.task_id}-c01"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrustedCommandAssembler:
    def __init__(
        self,
        *,
        command_id_factory: Callable[[TaskRecord], str] = _command_id,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._command_id_factory = command_id_factory
        self._clock = clock

    def assemble(
        self,
        task: TaskRecord,
        outcome: ProcessedOutcome,
        snapshot: ContextSnapshot,
    ) -> tuple[str, str]:
        if outcome.kind != OutcomeKind.READY_FOR_LOW or outcome.decision is None:
            raise ValueError("only READY_FOR_LOW outcomes can become commands")
        if snapshot.request_id != task.request_id:
            raise ValueError("cannot assemble from a stale context snapshot")
        decision = outcome.decision
        selected = [snapshot.resolve(ref) for ref in decision.selected_context_refs]
        target_item = self._select_target(selected, decision.target_name)
        command_id = self._command_id_factory(task)
        if not command_id.strip():
            raise ValueError("command ID factory returned an empty value")

        trusted_context: dict[str, Any] = {
            "environment_id": snapshot.environment_id,
            "map_id": snapshot.map_id,
        }
        if target_item is not None and target_item.scene:
            trusted_context["room"] = target_item.scene
        if decision.execution_strategy == "direct":
            positioned = (
                target_item
                if target_item is not None and target_item.position is not None
                else next(
                    (item for item in selected if item.position is not None), None
                )
            )
            if positioned is None:
                raise ValueError("direct command has no trusted position")
            trusted_context["position"] = positioned.position.as_dict()

        target: dict[str, Any] = {
            "type": decision.target_type,
            "name": decision.target_name,
        }
        if target_item is not None:
            target["memory_id"] = target_item.object_id
        hints = self._semantic_hints(selected)
        payload = {
            "schema_version": "flzat.high_command.v2",
            "run_id": task.run_id,
            "command_id": command_id,
            "task_id": task.task_id,
            "request_id": task.request_id,
            "created_at": self._clock(),
            "intent": decision.intent,
            "execution_mode": decision.execution_strategy,
            "target": target,
            "trusted_context": trusted_context,
            "semantic_hints": hints,
            "missing_information": [],
            "context_refs": list(decision.selected_context_refs),
            "status": "validated",
        }
        return command_id, canonical_json(payload)

    @staticmethod
    def _select_target(
        selected: list[ContextItem], target_name: str
    ) -> ContextItem | None:
        normalized = target_name.casefold()
        return next(
            (item for item in selected if item.label.casefold() == normalized),
            selected[0] if selected else None,
        )

    @staticmethod
    def _semantic_hints(selected: list[ContextItem]) -> list[str]:
        values: list[str] = []
        for item in selected:
            if item.scene:
                values.append(f"scene:{item.scene}")
            values.extend(item.relationships)
        return list(dict.fromkeys(values))
