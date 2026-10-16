"""Trusted mode-to-prompt/output contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileContract:
    mode: str
    output_schema: str
    validator: str


PROFILE_CONTRACTS: dict[str, ProfileContract] = {
    "task_command": ProfileContract(
        "task_command", "ModelDecision.v1", "task_command_v1"
    ),
    "visual_qa": ProfileContract("visual_qa", "VisualQA.v1", "visual_qa_v1"),
    "general_qa": ProfileContract("general_qa", "GeneralQA.v1", "general_qa_v1"),
    "social_dialogue": ProfileContract(
        "social_dialogue", "SocialDialogue.v1", "social_dialogue_v1"
    ),
    "clarification": ProfileContract(
        "clarification", "Clarification.v1", "clarification_v1"
    ),
}

