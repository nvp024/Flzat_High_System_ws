"""Trusted, bounded context providers selected by ContextPolicy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from flzat_cognitive_pipeline.context_policy import ContextRequirements
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import ContextSnapshot, TaskRecord


class TrustedYamlProvider:
    def __init__(self, path: str | Path, root_key: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.root_key = root_key
        try:
            value = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"cannot load trusted context {self.path}: {exc}") from exc
        if not isinstance(value, dict) or root_key not in value:
            raise ValueError(f"trusted context requires root key '{root_key}'")
        self._value = self._bounded_value(value[root_key])

    @classmethod
    def _bounded_value(cls, value: Any, depth: int = 0) -> Any:
        if depth > 5:
            raise ValueError("trusted context nesting exceeds five levels")
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value.strip()[:1000]
        if isinstance(value, list):
            if len(value) > 50:
                raise ValueError("trusted context list exceeds 50 entries")
            return [cls._bounded_value(item, depth + 1) for item in value]
        if isinstance(value, dict):
            if len(value) > 50:
                raise ValueError("trusted context mapping exceeds 50 entries")
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("trusted context keys must be nonempty strings")
                result[key.strip()[:128]] = cls._bounded_value(item, depth + 1)
            return result
        raise ValueError(f"unsupported trusted context value: {type(value).__name__}")

    def get(self) -> Any:
        return deepcopy(self._value)


@dataclass(frozen=True)
class AdaptiveContext:
    mode: str
    context_items: list[dict[str, Any]]
    context_refs: tuple[str, ...]
    dialogue_history: list[dict[str, str]]
    trusted_context: dict[str, Any]
    include_rag_context: bool


class AdaptiveContextBuilder:
    def __init__(
        self,
        *,
        skill_registry: TrustedYamlProvider,
        safety_policy: TrustedYamlProvider,
        user_profile: TrustedYamlProvider,
        conversation_store: ConversationStore,
    ) -> None:
        self._skills = skill_registry
        self._safety = safety_policy
        self._profile = user_profile
        self._conversation = conversation_store

    def build(
        self,
        policy: ContextRequirements,
        task: TaskRecord,
        snapshot: ContextSnapshot,
    ) -> AdaptiveContext:
        trusted: dict[str, Any] = {}
        if policy.mode in {"task_command", "social_dialogue"}:
            trusted["skill_registry"] = self._skills.get()
        if policy.mode == "task_command":
            trusted["safety_policy"] = self._safety.get()
        if policy.needs_user_profile:
            trusted["user_profile"] = self._profile.get()

        dialogue: list[dict[str, str]] = []
        if policy.needs_conversation_memory and policy.mode in {
            "social_dialogue",
            "clarification",
        }:
            dialogue.extend(self._conversation.recent())
        dialogue.extend(
            {"question": question, "answer": answer}
            for question, answer in task.clarification_history
        )
        return AdaptiveContext(
            mode=policy.mode,
            context_items=[item.prompt_value() for item in snapshot.items],
            context_refs=snapshot.refs,
            dialogue_history=dialogue,
            trusted_context=trusted,
            include_rag_context=policy.needs_task_memory,
        )
