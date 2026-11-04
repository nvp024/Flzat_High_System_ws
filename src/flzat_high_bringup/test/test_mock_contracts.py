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


def test_low_mock_correlates_handoffs_and_reads_path_before_result():
    source = (ROOT / "flzat_high_bringup/low_system_mock.py").read_text(
        encoding="utf-8"
    )
    assert "key = (run_id, handoff_id)" in source
    assert "duplicate handoff_id" in source
    assert "Path(str(request.json_path)).resolve(strict=True)" in source
    assert '"flzat.high_handoff.v2"' in source
