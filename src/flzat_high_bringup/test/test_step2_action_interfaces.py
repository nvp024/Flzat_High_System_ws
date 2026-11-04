"""Generated Action A/B contract shape and launch wiring."""

from pathlib import Path

from flzat_high_interfaces.action import AskUser, HandleIntent


ROOT = Path(__file__).resolve().parents[1]


def test_action_a_contract_fields_are_path_only():
    goal = HandleIntent.Goal()
    assert goal.get_fields_and_field_types() == {
        "run_id": "string",
        "task_id": "string",
        "handoff_id": "string",
        "json_path": "string",
    }
    assert HandleIntent.Result().get_fields_and_field_types() == {
        "outcome": "string",
        "detail": "string",
    }
    assert HandleIntent.Feedback().get_fields_and_field_types() == {
        "stage": "string",
        "detail": "string",
    }


def test_action_b_contract_keeps_natural_answer_and_error_separate():
    goal_fields = AskUser.Goal().get_fields_and_field_types()
    assert tuple(goal_fields) == (
        "run_id",
        "task_id",
        "handoff_id",
        "interaction_id",
        "kind",
        "content_hint",
        "options",
        "timeout_s",
    )
    assert AskUser.Result().get_fields_and_field_types() == {
        "outcome": "string",
        "answer": "string",
        "error_message": "string",
    }


def test_launch_uses_action_mock_not_old_topic_transport():
    launch = (ROOT / "launch/high_system.launch.py").read_text(encoding="utf-8")
    mock = (ROOT / "flzat_high_bringup/low_system_mock.py").read_text(
        encoding="utf-8"
    )
    manager = (
        ROOT.parent
        / "flzat_cognitive_pipeline/flzat_cognitive_pipeline/cognitive_manager.py"
    ).read_text(encoding="utf-8")
    assert '"low_result_outcome"' in launch
    assert "ActionServer(" in mock and "HandleIntent" in mock
    assert "ActionClient(" in mock and "AskUser" in mock
    assert "self._command_pub.publish" not in manager
    assert "self._low_cancel_pub.publish" not in manager
