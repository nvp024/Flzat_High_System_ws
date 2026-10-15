"""Strict ModelDecision parsing; no repair, guessing or permissive fallback."""

from __future__ import annotations

import json
from typing import Any

from flzat_cognitive_pipeline.domain import (
    ContextSnapshot,
    ModelDecision,
    OutcomeKind,
    ProcessedOutcome,
)


class ModelDecisionError(ValueError):
    pass


INTENT_ALLOW_LIST = {
    "fetch_object",
    "navigate",
    "find_object",
    "manipulate_object",
    "answer",
    "cancel",
}
ACTION_INTENTS = {
    "fetch_object",
    "navigate",
    "find_object",
    "manipulate_object",
}
STRATEGIES = {"direct", "search", "clarify", "response_only"}
_TOP_KEYS = {
    "intent",
    "target",
    "selected_context_refs",
    "execution_strategy",
    "missing_information",
    "clarification_question",
    "speech",
}
_TARGET_KEYS = {"type", "name"}


class StrictOutputProcessor:
    def process(
        self,
        raw_response: str,
        snapshot: ContextSnapshot,
        *,
        expected_request_id: str,
    ) -> ProcessedOutcome:
        try:
            decision = self._parse(raw_response)
            self._validate_context(decision, snapshot, expected_request_id)
            return self._classify(decision, snapshot)
        except ModelDecisionError as exc:
            return ProcessedOutcome(OutcomeKind.REJECTED, error=str(exc))

    def _parse(self, raw_response: str) -> ModelDecision:
        text = self._strip_single_json_fence(raw_response)
        if not text:
            raise ModelDecisionError("model response is empty")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelDecisionError(f"response is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict) or set(value) != _TOP_KEYS:
            raise ModelDecisionError("ModelDecision fields do not match v1 schema")
        target = value["target"]
        if not isinstance(target, dict) or set(target) != _TARGET_KEYS:
            raise ModelDecisionError("target fields do not match v1 schema")

        intent = self._text(value["intent"], "intent")
        if intent not in INTENT_ALLOW_LIST:
            raise ModelDecisionError(f"intent is not allowed: {intent}")
        target_type = self._text(target["type"], "target.type")
        target_name = self._text(target["name"], "target.name")
        strategy = self._text(value["execution_strategy"], "execution_strategy")
        if strategy not in STRATEGIES:
            raise ModelDecisionError(f"unsupported execution_strategy: {strategy}")
        refs = self._text_list(value["selected_context_refs"], "selected_context_refs", 5)
        if len(set(refs)) != len(refs):
            raise ModelDecisionError("selected_context_refs contains duplicates")
        missing = self._text_list(value["missing_information"], "missing_information", 8)
        question = value["clarification_question"]
        if question is not None:
            question = self._text(question, "clarification_question")
        speech = self._text(value["speech"], "speech", allow_empty=True)
        return ModelDecision(
            intent=intent,
            target_type=target_type,
            target_name=target_name,
            selected_context_refs=refs,
            execution_strategy=strategy,
            missing_information=missing,
            clarification_question=question,
            speech=speech,
        )

    @staticmethod
    def _strip_single_json_fence(raw_response: str) -> str:
        """Normalize Qwen's common outer JSON fence without repairing content."""

        text = raw_response.strip()
        if not text.startswith("```") and not text.endswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ModelDecisionError("response contains an incomplete code fence")
        opener = lines[0].strip().casefold()
        if opener not in {"```json", "```"}:
            raise ModelDecisionError("response code fence must contain JSON")
        inner = "\n".join(lines[1:-1]).strip()
        if "```" in inner:
            raise ModelDecisionError("response contains nested code fences")
        return inner

    @staticmethod
    def _validate_context(
        decision: ModelDecision,
        snapshot: ContextSnapshot,
        expected_request_id: str,
    ) -> None:
        if snapshot.request_id != expected_request_id:
            raise ModelDecisionError("context snapshot is stale or cross-request")
        available = set(snapshot.refs)
        unknown = [ref for ref in decision.selected_context_refs if ref not in available]
        if unknown:
            raise ModelDecisionError(f"unknown context_ref: {unknown[0]}")

    @staticmethod
    def _classify(
        decision: ModelDecision,
        snapshot: ContextSnapshot,
    ) -> ProcessedOutcome:
        if decision.execution_strategy == "clarify":
            if not decision.clarification_question:
                raise ModelDecisionError("clarify requires clarification_question")
            if not decision.missing_information:
                raise ModelDecisionError("clarify requires missing_information")
            return ProcessedOutcome(OutcomeKind.NEEDS_CLARIFICATION, decision)

        if decision.execution_strategy == "response_only" or decision.intent in {
            "answer",
            "cancel",
        }:
            if not decision.speech:
                raise ModelDecisionError("response-only decision requires speech")
            return ProcessedOutcome(OutcomeKind.RESPONSE_ONLY, decision)

        if decision.intent not in ACTION_INTENTS:
            raise ModelDecisionError("intent cannot be dispatched to Low System")
        if decision.missing_information or decision.clarification_question is not None:
            raise ModelDecisionError("dispatchable decision contains unresolved information")

        if decision.execution_strategy == "direct":
            if not decision.selected_context_refs:
                raise ModelDecisionError("direct strategy requires a context_ref")
            if not any(
                snapshot.resolve(ref).position is not None
                for ref in decision.selected_context_refs
            ):
                raise ModelDecisionError("direct strategy requires trusted position")
        elif decision.execution_strategy != "search":
            raise ModelDecisionError("action intent requires direct or search strategy")
        return ProcessedOutcome(OutcomeKind.READY_FOR_LOW, decision)

    @staticmethod
    def _text(value: Any, name: str, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise ModelDecisionError(f"{name} must be a string")
        result = value.strip()
        if not allow_empty and not result:
            raise ModelDecisionError(f"{name} cannot be empty")
        if len(result) > 512:
            raise ModelDecisionError(f"{name} exceeds 512 characters")
        return result

    @classmethod
    def _text_list(
        cls,
        value: Any,
        name: str,
        maximum: int,
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or len(value) > maximum:
            raise ModelDecisionError(f"{name} must be a list of at most {maximum}")
        return tuple(cls._text(item, name) for item in value)
