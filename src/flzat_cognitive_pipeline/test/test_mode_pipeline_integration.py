import json
from pathlib import Path

import pytest

from flzat_cognitive_pipeline.context_policy import ContextPolicy
from flzat_cognitive_pipeline.domain import (
    ContextItem,
    ContextSnapshot,
    OutcomeKind,
    TrustedPosition,
)
from flzat_cognitive_pipeline.output_validator import ModeOutputValidator
from flzat_cognitive_pipeline.prompt_composer import PromptComposer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PROMPTS = ROOT / "prompts"


def snapshot():
    return ContextSnapshot(
        request_id="task-0001-r01",
        environment_id="demo_1",
        map_id="demo_1_test",
        items=(
            ContextItem(
                context_ref="memory:book-1",
                kind="environment_object",
                object_id="book-1",
                label="book",
                description="A book on a table.",
                relationships=("on:table-1",),
                scene="living_room",
                retrieval_score=0.9,
                position=TrustedPosition("map", 1.0, 2.0, 0.5),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("mode", "transcript", "response", "expected_kind"),
    (
        (
            "task_command",
            "get the book",
            {
                "intent": "fetch_object",
                "target": {"type": "object", "name": "book"},
                "selected_context_refs": ["memory:book-1"],
                "execution_strategy": "direct",
                "missing_information": [],
                "clarification_question": None,
                "speech": "I will fetch the book.",
            },
            OutcomeKind.READY_FOR_LOW,
        ),
        (
            "visual_qa",
            "what is on the table",
            {
                "reply": "There is a book on the table.",
                "observed_objects": ["book", "table"],
                "confidence": 0.9,
            },
            OutcomeKind.RESPONSE_ONLY,
        ),
        (
            "general_qa",
            "what is the capital of China",
            {
                "reply": "Beijing is the capital of China.",
                "speak": True,
                "requires_robot_action": False,
            },
            OutcomeKind.RESPONSE_ONLY,
        ),
        (
            "social_dialogue",
            "I feel tired",
            {
                "reply": "Would you like to rest?",
                "speak": True,
                "suggested_skill": None,
            },
            OutcomeKind.RESPONSE_ONLY,
        ),
        (
            "clarification",
            "get that for me",
            {"reply": "Which object do you want me to get?"},
            OutcomeKind.NEEDS_CLARIFICATION,
        ),
    ),
)
def test_route_policy_prompt_and_validator_are_aligned(
    mode, transcript, response, expected_kind
):
    policy = ContextPolicy.from_yaml(CONFIG / "intent_routes.yaml").requirements_for(
        mode
    )
    memory = snapshot()
    context_items = (
        [item.prompt_value() for item in memory.items]
        if policy.needs_task_memory
        else []
    )
    context_refs = memory.refs if policy.needs_task_memory else ()
    dialogue = (
        [{"user": "previous request", "assistant": "previous reply", "mode": mode}]
        if policy.needs_conversation_memory
        else []
    )
    trusted = None
    if mode == "task_command":
        trusted = {
            "skill_registry": [{"name": "fetch_object"}],
            "safety_policy": ["Do not invent coordinates."],
        }
    elif mode == "social_dialogue":
        trusted = {
            "skill_registry": [{"name": "fetch_object"}],
            "user_profile": {"preferred_language": "en"},
        }

    prompt = PromptComposer(str(PROMPTS)).compose(
        transcript=transcript,
        context_items=context_items,
        context_refs=context_refs,
        dialogue_history=dialogue,
        mode=mode,
        trusted_context=trusted,
        include_rag_context=policy.needs_task_memory,
    )
    validated = ModeOutputValidator(["fetch_object"]).validate(
        mode,
        json.dumps(response),
        memory,
        expected_request_id=memory.request_id,
    )

    assert prompt.mode == mode
    assert prompt.validator.endswith("_v1")
    assert validated.valid
    assert validated.kind == expected_kind
    assert policy.allows_low_command is (mode == "task_command")
