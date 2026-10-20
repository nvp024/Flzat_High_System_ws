import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_frame_adapter_is_registered():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "static_frame_server" in source


def test_fake_vlm_default_is_exact_json():
    source = (ROOT / "flzat_high_bringup/fake_servers.py").read_text(
        encoding="utf-8"
    )
    marker = '"response_text",\n'
    assert marker in source


def test_fake_vlm_transport_allows_text_only_requests():
    source = (ROOT / "flzat_high_bringup/fake_servers.py").read_text(
        encoding="utf-8"
    )
    goal_source = source.split("def _goal(request)", 1)[1].split(
        "def _execute", 1
    )[0]
    assert "request.image.data" not in goal_source


def test_low_status_example_is_json_serializable():
    value = {
        "schema_version": "flzat.low_status.v2",
        "run_id": "run-20261014T000000Z",
        "command_id": "task-0001-c01",
        "state": "accepted",
        "error": "",
    }
    assert json.loads(json.dumps(value))["command_id"] == "task-0001-c01"


def test_low_mock_correlates_commands_by_run_and_command_id():
    source = (ROOT / "flzat_high_bringup/low_system_mock.py").read_text(
        encoding="utf-8"
    )
    assert "command_key = (run_id, command_id)" in source
    assert "self._commands[command_key]" in source


def test_fake_routing_sequence_does_not_consume_main_responses():
    import threading
    from types import SimpleNamespace
    from flzat_high_bringup.fake_servers import FakeVlmServer

    values = {
        "routing_response_sequence": ["routing-1", "routing-2"],
        "response_sequence": ["main", "repair"],
    }
    fake = SimpleNamespace(
        _response_lock=threading.Lock(),
        _response_index=0,
        _routing_response_index=0,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )
    assert FakeVlmServer._next_routing_response(fake) == "routing-1"
    assert FakeVlmServer._next_response(fake) == "main"
    assert FakeVlmServer._next_routing_response(fake) == "routing-2"
    assert FakeVlmServer._next_response(fake) == "repair"
