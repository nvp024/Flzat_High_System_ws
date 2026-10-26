from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_launch_can_switch_real_and_fake_boundaries():
    source = (ROOT / "launch/high_system.launch.py").read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("use_fake_speech"' in source
    assert 'DeclareLaunchArgument("use_fake_vlm"' in source
    assert 'DeclareLaunchArgument("use_low_mock"' in source
    assert '"run_camera_node",' in source
    assert '"run_vad_node",' in source
    assert '"run_stt_node",' in source
    assert 'DeclareLaunchArgument("whisper_model_path", default_value="")' in source
    assert 'DeclareLaunchArgument("whisper_beam_size", default_value="5")' in source
    assert '"run_tts_node",' in source
    assert 'NotSubstitution(LaunchConfiguration("use_fake_speech"))' in source
    assert '"frame_source",' in source
    assert 'EqualsSubstitution(LaunchConfiguration("frame_source"), "camera")' in source
    assert 'EqualsSubstitution(LaunchConfiguration("frame_source"), "static")' in source
    assert 'executable="static_frame_server"' in source
    assert 'executable="cognitive_manager"' in source
    assert 'executable="intent_router"' in source
    assert 'executable="speech_ingress"' in source
    assert 'executable="memory_query_node"' in source
    assert 'DeclareLaunchArgument("intent_routes_path"' in source
    assert '"config_file": LaunchConfiguration("intent_routes_path")' in source
    assert '"context_policy_path": LaunchConfiguration("intent_routes_path")' in source
    assert 'DeclareLaunchArgument("skill_registry_path"' in source
    assert 'DeclareLaunchArgument("safety_policy_path"' in source
    assert 'DeclareLaunchArgument("user_profile_path"' in source
    assert 'DeclareLaunchArgument("max_repair_attempts"' in source
    assert "UnlessCondition" in source
    assert 'DeclareLaunchArgument(\n            "environment_id",' in source
    assert 'DeclareLaunchArgument(\n            "map_id",' in source
    assert 'DeclareLaunchArgument("map_id", default_value=' not in source
    assert (
        'DeclareLaunchArgument("embedding_local_files_only", default_value="false")'
        in source
    )
    assert "do_image_splitting" not in source


def test_local_io_launch_owns_only_hardware_boundaries():
    source = (ROOT / "launch/local_io.launch.py").read_text(encoding="utf-8")

    assert 'executable="camera_node"' not in source
    assert '"/launch/flzat_vision_context.launch.py"' in source
    assert 'executable="vad_node"' in source
    assert 'executable="tts_node"' in source
    assert '"piper_model_path": LaunchConfiguration("piper_model_path")' in source
    assert 'executable="stt_node"' not in source
    assert 'executable="speech_ingress"' not in source
    assert 'DeclareLaunchArgument("mic_device"' in source
    assert 'LaunchConfiguration("mic_device"), value_type=str' in source
    assert '"debug_fps",' in source
    assert 'DeclareLaunchArgument("piper_model_path", default_value="")' in source


def test_low_mock_supports_lifecycle_and_cancel_contracts():
    source = (ROOT / "flzat_high_bringup/low_system_mock.py").read_text(
        encoding="utf-8"
    )

    assert "status_sequence" in source
    assert '"running"' in source
    assert '"succeeded"' in source
    assert '"failed"' in source
    assert '"cancelled"' in source
    assert '"flzat.low_cancel.v2"' in source
