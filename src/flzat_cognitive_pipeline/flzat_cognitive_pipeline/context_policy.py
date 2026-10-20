"""Authoritative route-to-context policy for the cognitive manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


ROUTE_MODES = (
    "task_command",
    "visual_qa",
    "general_qa",
    "social_dialogue",
    "clarification",
)
CONTEXT_FLAGS = (
    "needs_vision",
    "needs_task_memory",
    "needs_conversation_memory",
    "needs_user_profile",
    "needs_external_knowledge",
)


class ContextPolicyError(ValueError):
    """Raised when a routed request cannot be authorized safely."""


@dataclass(frozen=True)
class ContextRequirements:
    mode: str
    needs_vision: bool
    needs_task_memory: bool
    needs_conversation_memory: bool
    needs_user_profile: bool
    needs_external_knowledge: bool
    corrected_flags: tuple[str, ...] = ()

    @property
    def allows_low_command(self) -> bool:
        return self.mode == "task_command"

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "needs_vision": self.needs_vision,
            "needs_task_memory": self.needs_task_memory,
            "needs_conversation_memory": self.needs_conversation_memory,
            "needs_user_profile": self.needs_user_profile,
            "needs_external_knowledge": self.needs_external_knowledge,
            "corrected_flags": list(self.corrected_flags),
        }


class ContextPolicy:
    """Canonicalize untrusted router flags using trusted per-mode rules."""

    def __init__(self, rules: Mapping[str, Mapping[str, bool]]) -> None:
        if set(rules) != set(ROUTE_MODES):
            raise ContextPolicyError(
                "context policy must define exactly: " + ", ".join(ROUTE_MODES)
            )
        canonical: dict[str, dict[str, bool]] = {}
        for mode in ROUTE_MODES:
            values = rules[mode]
            if not isinstance(values, Mapping):
                raise ContextPolicyError(f"context policy for {mode} must be a mapping")
            if any(name not in values for name in CONTEXT_FLAGS):
                raise ContextPolicyError(
                    f"context policy for {mode} is missing context flags"
                )
            normalized: dict[str, bool] = {}
            for name in CONTEXT_FLAGS:
                value = values[name]
                if not isinstance(value, bool):
                    raise ContextPolicyError(
                        f"context policy {mode}.{name} must be boolean"
                    )
                normalized[name] = value
            if normalized["needs_external_knowledge"]:
                raise ContextPolicyError(
                    "external knowledge cannot be enabled without a provider"
                )
            canonical[mode] = normalized
        self._canonical = canonical

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContextPolicy":
        config_path = Path(path).expanduser().resolve()
        try:
            value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ContextPolicyError(
                f"cannot load context policy {config_path}: {exc}"
            ) from exc
        routes = value.get("routes") if isinstance(value, dict) else None
        if not isinstance(routes, dict):
            raise ContextPolicyError("context policy YAML requires a routes mapping")
        rules = {
            mode: routes[mode]
            for mode in ROUTE_MODES
            if mode in routes
        }
        return cls(rules)

    def resolve(
        self,
        mode: str,
        proposed_flags: Mapping[str, bool],
    ) -> ContextRequirements:
        routed_mode = mode.strip()
        effective_mode = routed_mode
        if effective_mode not in ROUTE_MODES:
            raise ContextPolicyError(f"unsupported route mode: {effective_mode}")
        if any(name not in proposed_flags for name in CONTEXT_FLAGS):
            raise ContextPolicyError("routed command is missing context flags")
        if any(not isinstance(proposed_flags[name], bool) for name in CONTEXT_FLAGS):
            raise ContextPolicyError("routed context flags must be boolean")
        if proposed_flags["needs_external_knowledge"]:
            raise ContextPolicyError(
                "external knowledge was requested but no provider is configured"
            )

        canonical = dict(self._canonical[effective_mode])
        corrected = tuple(
            name
            for name in CONTEXT_FLAGS
            if proposed_flags[name] != canonical[name]
        )
        return ContextRequirements(
            mode=effective_mode,
            corrected_flags=corrected,
            **canonical,
        )

    def requirements_for(self, mode: str) -> ContextRequirements:
        if mode not in ROUTE_MODES:
            raise ContextPolicyError(f"unsupported route mode: {mode}")
        values = dict(self._canonical[mode])
        return ContextRequirements(mode=mode, **values)
