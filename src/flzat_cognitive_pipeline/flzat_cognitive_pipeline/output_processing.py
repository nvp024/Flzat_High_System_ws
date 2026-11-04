"""Strict processing for the unified High-level cognitive decision."""

from __future__ import annotations

import json
from typing import Any

from flzat_cognitive_pipeline.domain import (
    CognitiveDecision,
    CognitiveGoal,
    CognitiveInteraction,
    ContextSnapshot,
    OutcomeKind,
    ProcessedOutcome,
)


class CognitiveDecisionError(ValueError):
    """Raised when model output violates CognitiveDecision.v1."""


_COGNITIVE_KEYS = {"speech", "goal", "interaction"}
_GOAL_KEYS = {
    "description",
    "constraints",
    "selected_context_refs",
    "missing_information",
}
_INTERACTION_KEYS = {"kind", "question"}


class CognitiveOutputProcessor:
    """Validate one decision without repairing or inventing trusted data."""

    def process(
        self,
        raw_response: str,
        snapshot: ContextSnapshot,
        *,
        expected_request_id: str,
    ) -> ProcessedOutcome:
        try:
            value = self._parse_object(raw_response)
            if set(value) != _COGNITIVE_KEYS:
                raise CognitiveDecisionError(
                    "CognitiveDecision fields do not match v1 schema"
                )
            speech = self._text(value["speech"], "speech", allow_empty=True)
            goal = self._parse_goal(value["goal"])
            interaction = self._parse_interaction(value["interaction"])
            if goal is None and interaction is None and not speech:
                raise CognitiveDecisionError(
                    "decision must contain speech, a goal, or an interaction"
                )
            if goal is not None and interaction is not None:
                raise CognitiveDecisionError(
                    "a clarification interaction cannot dispatch a goal"
                )
            self._validate_context(goal, snapshot, expected_request_id)
            decision = CognitiveDecision(speech, goal, interaction)
            if interaction is not None:
                return ProcessedOutcome(OutcomeKind.NEEDS_CLARIFICATION, decision)
            if goal is not None:
                return ProcessedOutcome(OutcomeKind.READY_FOR_LOW, decision)
            return ProcessedOutcome(OutcomeKind.RESPONSE_ONLY, decision)
        except CognitiveDecisionError as exc:
            return ProcessedOutcome(OutcomeKind.REJECTED, error=str(exc))

    def _parse_goal(self, value: Any) -> CognitiveGoal | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != _GOAL_KEYS:
            raise CognitiveDecisionError("goal fields do not match v1 schema")
        return CognitiveGoal(
            description=self._text(value["description"], "goal.description"),
            constraints=self._text_list(
                value["constraints"], "goal.constraints", maximum=16
            ),
            selected_context_refs=self._text_list(
                value["selected_context_refs"],
                "goal.selected_context_refs",
                maximum=10,
            ),
            missing_information=self._text_list(
                value["missing_information"],
                "goal.missing_information",
                maximum=16,
            ),
        )

    def _parse_interaction(self, value: Any) -> CognitiveInteraction | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != _INTERACTION_KEYS:
            raise CognitiveDecisionError("interaction fields do not match v1 schema")
        kind = self._text(value["kind"], "interaction.kind")
        if kind != "ASK":
            raise CognitiveDecisionError("High interaction kind must be ASK")
        question = self._text(value["question"], "interaction.question")
        if "\n" in question or not question.endswith("?"):
            raise CognitiveDecisionError(
                "interaction.question must be one concise question"
            )
        return CognitiveInteraction(kind=kind, question=question)

    @classmethod
    def _parse_object(cls, raw_response: str) -> dict[str, Any]:
        text = cls.strip_single_json_fence(raw_response)
        if not text:
            raise CognitiveDecisionError("model response is empty")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CognitiveDecisionError(
                f"response is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise CognitiveDecisionError("response must be a JSON object")
        return value

    @staticmethod
    def strip_single_json_fence(raw_response: str) -> str:
        """Normalize Qwen's common outer JSON fence without repairing content."""

        text = raw_response.strip()
        if not text.startswith("```") and not text.endswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise CognitiveDecisionError("response contains an incomplete code fence")
        if lines[0].strip().casefold() not in {"```", "```json"}:
            raise CognitiveDecisionError("response code fence must contain JSON")
        inner = "\n".join(lines[1:-1]).strip()
        if "```" in inner:
            raise CognitiveDecisionError("response contains nested code fences")
        return inner

    @staticmethod
    def _validate_context(
        goal: CognitiveGoal | None,
        snapshot: ContextSnapshot,
        expected_request_id: str,
    ) -> None:
        if snapshot.request_id != expected_request_id:
            raise CognitiveDecisionError("context snapshot is stale or cross-request")
        if goal is None:
            return
        refs = goal.selected_context_refs
        if len(set(refs)) != len(refs):
            raise CognitiveDecisionError(
                "goal.selected_context_refs contains duplicates"
            )
        available = set(snapshot.refs)
        unknown = [ref for ref in refs if ref not in available]
        if unknown:
            raise CognitiveDecisionError(f"unknown context_ref: {unknown[0]}")

    @staticmethod
    def _text(value: Any, name: str, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise CognitiveDecisionError(f"{name} must be a string")
        result = value.strip()
        if not allow_empty and not result:
            raise CognitiveDecisionError(f"{name} cannot be empty")
        if len(result) > 2000:
            raise CognitiveDecisionError(f"{name} exceeds 2000 characters")
        return result

    @classmethod
    def _text_list(
        cls,
        value: Any,
        name: str,
        *,
        maximum: int,
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or len(value) > maximum:
            raise CognitiveDecisionError(
                f"{name} must be a list of at most {maximum} strings"
            )
        return tuple(cls._text(item, name) for item in value)
