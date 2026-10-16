from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent


def test_routed_command_interface_is_owned_by_high_interfaces():
    interface = (
        SRC_ROOT
        / "flzat_high_interfaces"
        / "msg"
        / "RoutedCommand.msg"
    ).read_text(encoding="utf-8")

    for field in (
        "router_version",
        "utterance",
        "mode",
        "confidence",
        "needs_vision",
        "needs_task_memory",
        "needs_conversation_memory",
        "needs_user_profile",
        "needs_external_knowledge",
        "route_names",
        "route_scores",
        "latency_ms",
    ):
        assert field in interface


def test_cognitive_manager_consumes_only_routed_user_input():
    source = (PACKAGE_ROOT / "flzat_cognitive_pipeline/cognitive_manager.py").read_text(
        encoding="utf-8"
    )

    assert "RoutedCommand" in source
    assert '"/audio_events"' not in source
    assert '"/flzat/high_system/text_command"' not in source
    assert "SpeechToText" not in source
    assert "ContextPolicy" in source
    assert "runtime.policy.allows_low_command" in source


def test_manager_selects_memory_and_frame_from_authoritative_policy():
    source = (PACKAGE_ROOT / "flzat_cognitive_pipeline/cognitive_manager.py").read_text(
        encoding="utf-8"
    )

    assert "if policy.needs_task_memory:" in source
    assert "if runtime.policy.needs_vision:" in source
    assert "frame_skipped" in source


def test_safety_guard_uses_manager_cancel_adapter_to_preserve_low_contract():
    router_source = (
        PACKAGE_ROOT / "flzat_cognitive_pipeline/intent_router.py"
    ).read_text(encoding="utf-8")
    manager_source = (
        PACKAGE_ROOT / "flzat_cognitive_pipeline/cognitive_manager.py"
    ).read_text(encoding="utf-8")

    assert 'String, "/flzat/high_system/cancel", qos' in router_source
    assert "build_low_cancel(" in manager_source
    assert '"low_cancel_topic", "/flzat/low_system/cancel"' in manager_source
