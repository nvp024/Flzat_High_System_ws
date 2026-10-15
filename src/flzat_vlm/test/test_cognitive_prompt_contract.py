from pathlib import Path


SRC_ROOT = Path(__file__).parents[2]


def test_run_vlm_contract_accepts_versioned_precomposed_prompt():
    action = (
        SRC_ROOT / "flzat_high_interfaces/action/RunVlm.action"
    ).read_text(encoding="utf-8")
    node = (SRC_ROOT / "flzat_vlm/flzat_vlm/vlm_node.py").read_text(
        encoding="utf-8"
    )

    for field in (
        "sensor_msgs/CompressedImage image",
        "string transcript",
        "string request_id",
        "string system_prompt",
        "string user_prompt",
        "string prompt_version",
        "string prompt_hash",
    ):
        assert field in action
    assert "MultimodalEvent" not in action
    assert "expected_hash = hashlib.sha256" in node
    assert "system_prompt = request.system_prompt" in node


def test_vlm_node_has_no_second_prompt_or_grounding_path():
    node = (SRC_ROOT / "flzat_vlm/flzat_vlm/vlm_node.py").read_text(
        encoding="utf-8"
    )
    assert "from flzat_vlm.prompting" not in node
    assert "self._prompt_builder" not in node
    assert "GroundObjects" not in node
    assert "grounding" not in node.lower()
