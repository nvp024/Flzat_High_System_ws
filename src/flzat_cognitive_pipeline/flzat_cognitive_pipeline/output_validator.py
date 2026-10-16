"""Strict mode-specific output validation for Qwen3-VL responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable

from flzat_cognitive_pipeline.domain import (
    ContextSnapshot,
    OutcomeKind,
    ProcessedOutcome,
)
from flzat_cognitive_pipeline.output_processing import StrictOutputProcessor
from flzat_cognitive_pipeline.profile_contracts import PROFILE_CONTRACTS


class ModeOutputError(ValueError):
    """Raised when a response violates its routed mode schema."""


@dataclass(frozen=True)
class ValidatedModeOutput:
    mode: str
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
            "mode": self.mode,
            "valid": self.valid,
            "outcome": self.kind.value,
            "error": self.error,
        }


class ModeOutputValidator:
    """Dispatch exact schemas without permitting cross-mode output."""

    def __init__(self, allowed_skills: Iterable[str]) -> None:
        self._task = StrictOutputProcessor()
        self._allowed_skills = frozenset(
            value.strip() for value in allowed_skills if value.strip()
        )

    def validate(
        self,
        mode: str,
        raw_response: str,
        snapshot: ContextSnapshot,
        *,
        expected_request_id: str,
    ) -> ValidatedModeOutput:
        if mode not in PROFILE_CONTRACTS:
            return self._invalid(mode, f"unsupported validator mode: {mode}")
        if mode == "task_command":
            return self._validate_task(raw_response, snapshot, expected_request_id)
        try:
            value = self._parse_object(raw_response)
            if mode == "visual_qa":
                return self._validate_visual(value)
            if mode == "general_qa":
                return self._validate_general(value)
            if mode == "social_dialogue":
                return self._validate_social(value)
            return self._validate_clarification(value)
        except ModeOutputError as exc:
            return self._invalid(mode, str(exc))

    def _validate_task(
        self,
        raw_response: str,
        snapshot: ContextSnapshot,
        expected_request_id: str,
    ) -> ValidatedModeOutput:
        outcome = self._task.process(
            raw_response,
            snapshot,
            expected_request_id=expected_request_id,
        )
        if outcome.kind == OutcomeKind.REJECTED:
            return self._invalid("task_command", outcome.error)
        decision = outcome.decision
        if decision is None:
            return self._invalid("task_command", "validated task decision is missing")
        value = {
            "intent": decision.intent,
            "target": {
                "type": decision.target_type,
                "name": decision.target_name,
            },
            "selected_context_refs": list(decision.selected_context_refs),
            "execution_strategy": decision.execution_strategy,
            "missing_information": list(decision.missing_information),
            "clarification_question": decision.clarification_question,
            "speech": decision.speech,
        }
        reply = (
            decision.clarification_question
            if outcome.kind == OutcomeKind.NEEDS_CLARIFICATION
            else decision.speech
        ) or ""
        return ValidatedModeOutput(
            mode="task_command",
            kind=outcome.kind,
            value=value,
            reply=reply,
            task_outcome=outcome,
        )

    def _validate_visual(self, value: dict[str, Any]) -> ValidatedModeOutput:
        self._exact_keys(value, {"reply", "observed_objects", "confidence"})
        reply = self._text(value["reply"], "reply", maximum=1000)
        objects = value["observed_objects"]
        if not isinstance(objects, list) or len(objects) > 30:
            raise ModeOutputError("observed_objects must be a list of at most 30 names")
        normalized = [
            self._text(item, "observed_objects item", maximum=128)
            for item in objects
        ]
        confidence = value["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ModeOutputError("confidence must be a finite number from 0 to 1")
        output = {
            "reply": reply,
            "observed_objects": normalized,
            "confidence": float(confidence),
        }
        return ValidatedModeOutput(
            "visual_qa", OutcomeKind.RESPONSE_ONLY, output, reply
        )

    def _validate_general(self, value: dict[str, Any]) -> ValidatedModeOutput:
        self._exact_keys(value, {"reply", "speak", "requires_robot_action"})
        reply = self._text(value["reply"], "reply", maximum=1000)
        if value["speak"] is not True:
            raise ModeOutputError("general_qa speak must be true")
        if value["requires_robot_action"] is not False:
            raise ModeOutputError("general_qa cannot require robot action")
        output = {
            "reply": reply,
            "speak": True,
            "requires_robot_action": False,
        }
        return ValidatedModeOutput(
            "general_qa", OutcomeKind.RESPONSE_ONLY, output, reply
        )

    def _validate_social(self, value: dict[str, Any]) -> ValidatedModeOutput:
        self._exact_keys(value, {"reply", "speak", "suggested_skill"})
        reply = self._text(value["reply"], "reply", maximum=1000)
        if value["speak"] is not True:
            raise ModeOutputError("social_dialogue speak must be true")
        skill = value["suggested_skill"]
        if skill is not None:
            skill = self._text(skill, "suggested_skill", maximum=128)
            if skill not in self._allowed_skills:
                raise ModeOutputError("suggested_skill is not in the trusted registry")
        output = {"reply": reply, "speak": True, "suggested_skill": skill}
        return ValidatedModeOutput(
            "social_dialogue", OutcomeKind.RESPONSE_ONLY, output, reply
        )

    def _validate_clarification(
        self, value: dict[str, Any]
    ) -> ValidatedModeOutput:
        self._exact_keys(value, {"reply"})
        reply = self._text(value["reply"], "reply", maximum=240)
        if "\n" in reply or not reply.endswith("?"):
            raise ModeOutputError("clarification reply must be one concise question")
        return ValidatedModeOutput(
            "clarification",
            OutcomeKind.NEEDS_CLARIFICATION,
            {"reply": reply},
            reply,
        )

    @classmethod
    def _parse_object(cls, raw_response: str) -> dict[str, Any]:
        text = cls._strip_single_json_fence(raw_response)
        if not text:
            raise ModeOutputError("model response is empty")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModeOutputError(f"response is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ModeOutputError("response must be a JSON object")
        return value

    @staticmethod
    def _strip_single_json_fence(raw_response: str) -> str:
        text = raw_response.strip()
        if not text.startswith("```") and not text.endswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ModeOutputError("response contains an incomplete code fence")
        if lines[0].strip().casefold() not in {"```", "```json"}:
            raise ModeOutputError("response code fence must contain JSON")
        inner = "\n".join(lines[1:-1]).strip()
        if "```" in inner:
            raise ModeOutputError("response contains nested code fences")
        return inner

    @staticmethod
    def _exact_keys(value: dict[str, Any], keys: set[str]) -> None:
        if set(value) != keys:
            raise ModeOutputError("response fields do not match the mode schema")

    @staticmethod
    def _text(value: Any, name: str, *, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ModeOutputError(f"{name} must be a nonempty string")
        result = value.strip()
        if len(result) > maximum:
            raise ModeOutputError(f"{name} exceeds {maximum} characters")
        return result

    @staticmethod
    def _invalid(mode: str, error: str) -> ValidatedModeOutput:
        return ValidatedModeOutput(
            mode=mode,
            kind=OutcomeKind.REJECTED,
            error=error.strip() or "output validation failed",
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
        return (
            repair_number
            if repair_number <= self.max_repair_attempts
            else None
        )

