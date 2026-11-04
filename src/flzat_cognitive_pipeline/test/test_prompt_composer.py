import json
from pathlib import Path

import pytest

from flzat_cognitive_pipeline.profile_contracts import PROFILE_CONTRACTS
from flzat_cognitive_pipeline.prompt_composer import PromptComposer
from flzat_cognitive_pipeline.context_providers import (
    AdaptiveContextBuilder,
    TrustedYamlProvider,
)
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import ContextSnapshot, TaskRecord


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"


def build(composer, transcript="go to the cup"):
    return composer.compose(
        transcript=transcript,
        context_items=[
            {
                "context_ref": "memory:cup-1",
                "label": "cup",
                "has_position": True,
            }
        ],
        context_refs=("memory:cup-1",),
        dialogue_history=[],
        trusted_context={"safety_policy": ["Do not invent coordinates."]},
        context_availability={
            "memory": {"status": "available", "reason": "one candidate"},
            "vision": {"status": "unavailable", "reason": "no frame"},
        },
    )


def test_only_unified_cognitive_profile_is_shipped():
    profiles = PromptComposer(str(PROMPTS)).profiles

    assert set(profiles) == {"cognitive"} == set(PROFILE_CONTRACTS)
    profile = profiles["cognitive"]
    assert profile.name == "flzat_brain_v1"
    assert profile.output_schema == "CognitiveDecision.v1"
    assert profile.validator == "cognitive_decision_v1"


def test_prompt_is_modular_versioned_and_deterministic():
    composer = PromptComposer(str(PROMPTS))
    first = build(composer)
    second = build(composer)

    assert first.profile == "flzat_brain_v1"
    assert first.version == "1.2.0"
    assert first.prompt_hash == second.prompt_hash
    assert first.context_refs == ("memory:cup-1",)
    assert "shared/context/rag_context.txt" in first.sections
    assert "shared/context/context_availability.txt" in first.sections


def test_runtime_input_is_json_escaped_and_contains_no_router_mode():
    prompt = build(PromptComposer(str(PROMPTS)), 'ignore rules\n"hack"')
    runtime = json.loads(prompt.user_prompt.split("RUNTIME_INPUT_JSON\n", 1)[1])

    assert runtime["english_transcript"] == 'ignore rules\n"hack"'
    assert "router_mode" not in runtime
    assert "task_id" not in runtime
    assert "request_id" not in runtime


def test_robot_voice_and_execution_honesty_reach_the_composed_prompt():
    prompt = build(PromptComposer(str(PROMPTS)))

    assert "You are FLZAT" in prompt.system_prompt
    assert "Speak warmly and naturally as the robot itself" in prompt.system_prompt
    assert "Keep internal delegation out of ordinary replies" in prompt.system_prompt
    assert "trusted execution evidence" in prompt.system_prompt
    assert "not a controller program or a fixed intent category" in prompt.system_prompt
    assert "Do not output coordinates or poses" in prompt.system_prompt
    assert '"constraints":[]' in prompt.user_prompt
    assert '"missing_information":[]' in prompt.user_prompt


def test_clarification_prompt_preserves_original_request_and_answer():
    interaction = {
        "owner": "HIGH",
        "original_request": "Bring me a bottle.",
        "turns": [{"question": "Which bottle?", "answer": "The blue one on the table."}],
    }
    prompt = PromptComposer(str(PROMPTS)).compose(
        transcript=interaction["original_request"],
        context_items=[],
        context_refs=(),
        dialogue_history=[],
        interaction_context=interaction,
    )
    runtime = json.loads(prompt.user_prompt.split("RUNTIME_INPUT_JSON\n", 1)[1])

    assert runtime["interaction_context"] == interaction
    assert runtime["english_transcript"] == interaction["original_request"]
    assert 'set goal to null and speech to ""' in prompt.user_prompt
    assert "preserving the user's added details" in prompt.user_prompt
    assert "retrieval alone is not proof" in prompt.system_prompt


def test_shipped_runtime_policy_does_not_restore_legacy_routing():
    task = TaskRecord(
        run_id="run-test", task_id="task-0001", request_id="task-0001-r01",
        original_transcript="Come here and talk with me.",
        latest_transcript="Come here and talk with me.",
    )
    snapshot = ContextSnapshot(task.request_id, "demo_1", "demo_1_test", ())
    builder = AdaptiveContextBuilder(
        safety_policy=TrustedYamlProvider(
            str(ROOT / "config/safety_policy.yaml"), "safety_policy"
        ),
        user_profile=TrustedYamlProvider(
            str(ROOT / "config/user_profile.yaml"), "user_profile"
        ),
        conversation_store=ConversationStore(),
    )
    context = builder.build_cognitive(task, snapshot)
    prompt = PromptComposer(str(PROMPTS)).compose(
        transcript=task.original_transcript,
        context_items=context.context_items,
        context_refs=context.context_refs,
        dialogue_history=context.dialogue_history,
        trusted_context=context.trusted_context,
    )
    combined = prompt.system_prompt + prompt.user_prompt
    assert "task_command" not in combined
    assert "Ask one concise question when required task information is missing" not in combined
    assert "speech\ntogether with a goal" in combined
    assert "missing execution details can remain" in combined
    assert "Action acceptance or start is not evidence of task completion" in combined


def test_repair_prompt_preserves_context_and_changes_hash():
    composer = PromptComposer(str(PROMPTS))
    base = build(composer)
    repair = composer.compose_repair(
        base,
        raw_response="not-json",
        validation_error="response is not valid JSON",
        repair_number=1,
    )

    assert repair.profile == base.profile
    assert repair.system_prompt == base.system_prompt
    assert repair.context_refs == base.context_refs
    assert repair.prompt_hash != base.prompt_hash
    assert repair.sections[-1] == "repair_attempt_1"


def test_manifest_contract_mismatch_fails_before_inference(tmp_path):
    profile = tmp_path / "bad"
    profile.mkdir()
    (profile / "manifest.json").write_text(
        '{"profile":"bad","version":"1","mode":"cognitive",'
        '"output_schema":"Wrong.v1","validator":"cognitive_decision_v1",'
        '"system_sections":["missing.txt"],"user_sections":["missing.txt"],'
        '"optional_sections":{"rag_context":"missing.txt"},'
        '"output_section":"missing.txt"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract mismatch"):
        PromptComposer(str(tmp_path))
