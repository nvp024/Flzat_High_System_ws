"""Validation and bounded format repair for CognitiveDecision.v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flzat_cognitive_pipeline.domain import ContextSnapshot, OutcomeKind, ProcessedOutcome
from flzat_cognitive_pipeline.output_processing import CognitiveOutputProcessor


@dataclass(frozen=True)
class ValidatedCognitiveOutput:
    kind: OutcomeKind
    value: dict[str, Any] | None = None
    reply: str = ""
    task_outcome: ProcessedOutcome | None = None
    error: str = ""

    @property
    def valid(self) -> bool:
        return not self.error and self.value is not None

    def validation_record(self, attempt: int) -> dict[str, Any]:
        return {
            "schema_version": "flzat.validator_result.v1",
            "attempt": attempt,
            "mode": "cognitive",
            "valid": self.valid,
            "outcome": self.kind.value,
            "error": self.error,
        }


class CognitiveOutputValidator:
    """Validate the only model output contract used by the High pipeline."""

    def __init__(self) -> None:
        self._processor = CognitiveOutputProcessor()

    def validate(
        self,
        raw_response: str,
        snapshot: ContextSnapshot,
        *,
        expected_request_id: str,
    ) -> ValidatedCognitiveOutput:
        outcome = self._processor.process(
            raw_response,
            snapshot,
            expected_request_id=expected_request_id,
        )
        if outcome.kind == OutcomeKind.REJECTED:
            return ValidatedCognitiveOutput(outcome.kind, error=outcome.error)
        decision = outcome.decision
        if decision is None or not hasattr(decision, "goal"):
            return ValidatedCognitiveOutput(
                OutcomeKind.REJECTED,
                error="validated cognitive decision is missing",
            )
        goal = decision.goal
        interaction = decision.interaction
        value = {
            "speech": decision.speech,
            "goal": None
            if goal is None
            else {
                "description": goal.description,
                "constraints": list(goal.constraints),
                "selected_context_refs": list(goal.selected_context_refs),
                "missing_information": list(goal.missing_information),
            },
            "interaction": None
            if interaction is None
            else {"kind": interaction.kind, "question": interaction.question},
        }
        reply = interaction.question if interaction is not None else decision.speech
        return ValidatedCognitiveOutput(
            kind=outcome.kind,
            value=value,
            reply=reply,
            task_outcome=outcome,
        )


@dataclass(frozen=True)
class RepairPolicy:
    max_repair_attempts: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.max_repair_attempts <= 2:
            raise ValueError("max_repair_attempts must be between 0 and 2")

    def next_repair_number(self, attempts_sent: int) -> int | None:
        if attempts_sent < 1:
            raise ValueError("attempts_sent must include the original inference")
        repair_number = attempts_sent
        return repair_number if repair_number <= self.max_repair_attempts else None
