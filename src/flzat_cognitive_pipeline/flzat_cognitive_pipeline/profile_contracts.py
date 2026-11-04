"""Trusted mode-to-prompt/output contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileContract:
    mode: str
    output_schema: str
    validator: str


PROFILE_CONTRACTS: dict[str, ProfileContract] = {
    "cognitive": ProfileContract(
        "cognitive", "CognitiveDecision.v1", "cognitive_decision_v1"
    ),
}
