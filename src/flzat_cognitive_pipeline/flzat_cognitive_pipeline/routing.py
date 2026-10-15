"""Deterministic response and Low-System status routing."""

from __future__ import annotations

from dataclasses import dataclass
import json

from flzat_cognitive_pipeline.domain import OutcomeKind, ProcessedOutcome


@dataclass(frozen=True)
class RouteDecision:
    publish_command: bool
    speech: str
    terminal: bool


class ResponseRouter:
    def __init__(self, rejection_speech: str = "I could not safely process that request."):
        self.rejection_speech = rejection_speech.strip()

    def route(self, outcome: ProcessedOutcome) -> RouteDecision:
        decision = outcome.decision
        if outcome.kind == OutcomeKind.READY_FOR_LOW:
            return RouteDecision(True, decision.speech if decision else "", False)
        if outcome.kind == OutcomeKind.NEEDS_CLARIFICATION:
            question = decision.clarification_question if decision else ""
            return RouteDecision(False, question or self.rejection_speech, False)
        if outcome.kind == OutcomeKind.RESPONSE_ONLY:
            return RouteDecision(False, decision.speech if decision else "", True)
        return RouteDecision(False, self.rejection_speech, True)


@dataclass(frozen=True)
class LowStatus:
    run_id: str
    command_id: str
    state: str
    error: str


class LowStatusAdapter:
    ALLOWED_STATES = {"accepted", "running", "succeeded", "failed", "cancelled"}
    KEYS = {"schema_version", "run_id", "command_id", "state", "error"}

    def parse(
        self,
        raw: str,
        expected_run_id: str,
        expected_command_id: str,
    ) -> LowStatus:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Low status is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != self.KEYS:
            raise ValueError("Low status fields do not match v2 schema")
        if value["schema_version"] != "flzat.low_status.v2":
            raise ValueError("unsupported Low status schema")
        run_id = value["run_id"]
        command_id = value["command_id"]
        state = value["state"]
        error = value["error"]
        if not isinstance(run_id, str) or run_id != expected_run_id:
            raise ValueError("Low status run_id is unknown or stale")
        if not isinstance(command_id, str) or command_id != expected_command_id:
            raise ValueError("Low status command_id is unknown or stale")
        if not isinstance(state, str) or state not in self.ALLOWED_STATES:
            raise ValueError("Low status state is unsupported")
        if not isinstance(error, str):
            raise ValueError("Low status error must be a string")
        return LowStatus(run_id, command_id, state, error)


def build_low_cancel(
    run_id: str,
    command_id: str,
    task_id: str,
    request_id: str,
    reason: str = "user_cancelled",
) -> str:
    values = (run_id, command_id, task_id, request_id, reason)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Low cancel identifiers and reason must be nonempty strings")
    return json.dumps(
        {
            "schema_version": "flzat.low_cancel.v2",
            "run_id": run_id,
            "command_id": command_id,
            "task_id": task_id,
            "request_id": request_id,
            "reason": reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
