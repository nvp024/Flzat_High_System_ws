from pathlib import Path

import pytest

from flzat_cognitive_pipeline.context_policy import (
    CONTEXT_FLAGS,
    ContextPolicy,
    ContextPolicyError,
)
from flzat_cognitive_pipeline.context_providers import (
    AdaptiveContextBuilder,
    TrustedYamlProvider,
)
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import ContextSnapshot, TaskRecord


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def context_policy():
    return ContextPolicy.from_yaml(CONFIG / "intent_routes.yaml")


def flags(**overrides):
    value = {name: False for name in CONTEXT_FLAGS}
    value.update(overrides)
    return value


def test_policy_canonicalizes_spoofed_router_flags():
    policy = context_policy().resolve("task_command", flags())

    assert policy.needs_vision is True
    assert policy.needs_task_memory is True
    assert set(policy.corrected_flags) == {"needs_vision", "needs_task_memory"}
    assert policy.allows_low_command is True


def test_policy_selects_only_required_mode_context():
    policy = context_policy()

    visual = policy.resolve("visual_qa", flags(needs_vision=True))
    general = policy.resolve("general_qa", flags())
    social = policy.resolve(
        "social_dialogue",
        flags(needs_conversation_memory=True, needs_user_profile=True),
    )
    clarify = policy.resolve(
        "clarification",
        flags(needs_conversation_memory=True),
    )

    assert visual.needs_vision and not visual.needs_task_memory
    assert not general.needs_vision and not general.needs_task_memory
    assert not social.needs_conversation_memory and social.needs_user_profile
    assert "needs_conversation_memory" in social.corrected_flags
    assert clarify.needs_conversation_memory and not clarify.needs_vision
    assert not any(
        item.allows_low_command for item in (visual, general, social, clarify)
    )


def test_external_knowledge_is_rejected_without_provider():
    with pytest.raises(ContextPolicyError, match="no provider"):
        context_policy().resolve(
            "general_qa",
            flags(needs_external_knowledge=True),
        )


def test_resolved_general_question_does_not_enable_task_context():
    policy = context_policy().resolve(
        "general_qa",
        flags(),
    )

    assert policy.mode == "general_qa"
    assert policy.needs_task_memory is False
    assert policy.needs_vision is False


def test_adaptive_context_keeps_conversation_separate_from_object_memory():
    conversation = ConversationStore(max_turns=3)
    conversation.append("social_dialogue", "I feel tired", "Please rest.")
    builder = AdaptiveContextBuilder(
        skill_registry=TrustedYamlProvider(CONFIG / "skill_registry.yaml", "skills"),
        safety_policy=TrustedYamlProvider(
            CONFIG / "safety_policy.yaml", "safety_policy"
        ),
        user_profile=TrustedYamlProvider(
            CONFIG / "user_profile.yaml", "user_profile"
        ),
        conversation_store=conversation,
    )
    task = TaskRecord("run-1", "task-0001", "task-0001-r01", "hello", "hello")
    snapshot = ContextSnapshot("task-0001-r01", "demo", "map", ())
    policy = context_policy().requirements_for("social_dialogue")

    context = builder.build(policy, task, snapshot)

    assert context.context_items == []
    assert context.context_refs == ()
    assert context.include_rag_context is False
    assert context.dialogue_history == []
    assert context.trusted_context["user_profile"]["preferred_language"] == "en"
    assert context.trusted_context["skill_registry"][0]["name"] == "navigate"


def test_task_clarification_uses_only_bounded_task_history():
    conversation = ConversationStore(max_turns=3)
    conversation.append("task_command", "get that", "Which object?")
    builder = AdaptiveContextBuilder(
        skill_registry=TrustedYamlProvider(CONFIG / "skill_registry.yaml", "skills"),
        safety_policy=TrustedYamlProvider(
            CONFIG / "safety_policy.yaml", "safety_policy"
        ),
        user_profile=TrustedYamlProvider(
            CONFIG / "user_profile.yaml", "user_profile"
        ),
        conversation_store=conversation,
    )
    task = TaskRecord("run-1", "task-0001", "task-0001-r02", "get that", "book")
    task.clarification_history.append(("Which object?", "book"))
    snapshot = ContextSnapshot("task-0001-r02", "demo", "map", ())
    policy = context_policy().requirements_for("task_command")

    context = builder.build(policy, task, snapshot)

    assert context.dialogue_history == [{"question": "Which object?", "answer": "book"}]


def test_context_policy_is_loaded_from_yaml(tmp_path):
    malformed = tmp_path / "routes.yaml"
    malformed.write_text("routes:\n  task_command: {}\n", encoding="utf-8")

    with pytest.raises(ContextPolicyError, match="define exactly"):
        ContextPolicy.from_yaml(malformed)
