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
