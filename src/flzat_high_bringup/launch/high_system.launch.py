from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    AndSubstitution,
    EqualsSubstitution,
    LaunchConfiguration,
    NotSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from flzat_speech.launch_config import speech_arguments, speech_parameters


DEFAULT_FAKE_RESPONSE = (
    '{"intent":"fetch_object","target":{"type":"object",'
    '"name":"water_cup"},"selected_context_refs":[],'
    '"execution_strategy":"search","missing_information":[],'
    '"clarification_question":null,"speech":'
    '"I will ask the robot to search for the water cup."}'
)


def generate_launch_description():
    config = (
        get_package_share_directory("flzat_high_bringup")
        + "/config/high_system.yaml"
    )
    intent_routes = (
        get_package_share_directory("flzat_cognitive_pipeline")
        + "/config/intent_routes.yaml"
    )
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("flzat_vision_context"),
            "/launch/flzat_vision_context.launch.py",
        ]),
        condition=IfCondition(
            AndSubstitution(
                EqualsSubstitution(LaunchConfiguration("frame_source"), "camera"),
                LaunchConfiguration("run_camera_node"),
            )
        ),
    )
    real_vlm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("flzat_vlm"),
            "/launch/vlm_server.launch.py",
        ]),
        condition=UnlessCondition(LaunchConfiguration("use_fake_vlm")),
        launch_arguments={
            "backend": LaunchConfiguration("vlm_backend"),
            "model_id": LaunchConfiguration("model_id"),
            "model_revision": LaunchConfiguration("model_revision"),
            "device": LaunchConfiguration("device"),
            "dtype": LaunchConfiguration("dtype"),
            "quantization": LaunchConfiguration("quantization"),
            "max_new_tokens": LaunchConfiguration("max_new_tokens"),
            "trust_remote_code": LaunchConfiguration("trust_remote_code"),
            "local_files_only": LaunchConfiguration("local_files_only"),
            "qwen_min_image_pixels": LaunchConfiguration(
                "qwen_min_image_pixels"
            ),
            "qwen_max_image_pixels": LaunchConfiguration(
                "qwen_max_image_pixels"
            ),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("storage_root", default_value="~/.local/share/flzat/environment_memory"),
        DeclareLaunchArgument(
            "environment_id",
            description="Required environment directory containing the RAG database",
        ),
        DeclareLaunchArgument(
            "map_id",
            description="Required map identifier recorded in the memory manifest",
        ),
        DeclareLaunchArgument("embedding_backend", default_value="sentence_transformer"),
        DeclareLaunchArgument("embedding_local_files_only", default_value="false"),
        DeclareLaunchArgument("output_root", default_value="~/.local/share/flzat/high_system/commands"),
        DeclareLaunchArgument("enable_text_command", default_value="false"),
        DeclareLaunchArgument("use_fake_speech", default_value="false"),
        DeclareLaunchArgument("use_fake_vlm", default_value="false"),
        DeclareLaunchArgument("use_low_mock", default_value="false"),
        DeclareLaunchArgument(
            "run_camera_node",
            default_value="true",
            description="Start the hardware camera provider in this process",
        ),
        DeclareLaunchArgument(
            "run_vad_node",
            default_value="true",
            description="Start microphone capture and VAD in this process",
        ),
        DeclareLaunchArgument(
            "run_stt_node",
            default_value="true",
            description="Start the real speech-to-text action server",
        ),
        *speech_arguments(),
        DeclareLaunchArgument(
            "run_tts_node",
            default_value="true",
            description="Start the real text-to-speech action server",
        ),
        DeclareLaunchArgument(
            "frame_source",
            default_value="camera",
            choices=["camera", "static"],
            description="Frame provider used by the cognitive pipeline",
        ),
        DeclareLaunchArgument("static_image_path", default_value=""),
        DeclareLaunchArgument("request_timeout_s", default_value="60.0"),
        DeclareLaunchArgument("max_repair_attempts", default_value="2"),
        DeclareLaunchArgument("health_period_s", default_value="2.0"),
        DeclareLaunchArgument("intent_routes_path", default_value=intent_routes),
        DeclareLaunchArgument("skill_registry_path", default_value=""),
        DeclareLaunchArgument("safety_policy_path", default_value=""),
        DeclareLaunchArgument("user_profile_path", default_value=""),
        DeclareLaunchArgument("vlm_backend", default_value="qwen3_vl"),
        DeclareLaunchArgument("model_id", default_value="Qwen/Qwen3-VL-2B-Instruct"),
        DeclareLaunchArgument("model_revision", default_value="main"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("dtype", default_value="auto"),
        DeclareLaunchArgument("quantization", default_value="none"),
        DeclareLaunchArgument("max_new_tokens", default_value="256"),
        DeclareLaunchArgument("trust_remote_code", default_value="false"),
        DeclareLaunchArgument("local_files_only", default_value="false"),
        DeclareLaunchArgument(
            "qwen_min_image_pixels", default_value=str(256 * 256)
        ),
        DeclareLaunchArgument(
            "qwen_max_image_pixels", default_value=str(768 * 768)
        ),
        DeclareLaunchArgument("fake_transcript", default_value="go to the water cup"),
        DeclareLaunchArgument("fake_vlm_response", default_value=DEFAULT_FAKE_RESPONSE),
        DeclareLaunchArgument("fake_vlm_second_response", default_value=""),
        DeclareLaunchArgument(
            "fake_routing_response",
            default_value='{"mode":null,"relation_to_pending":"new_request",'
            '"needs_clarification":true,'
            '"clarification_question":"Could you clarify your request?"}',
        ),
        DeclareLaunchArgument("fake_vlm_delay_s", default_value="0.0"),
        DeclareLaunchArgument("fake_vlm_ignore_cancel", default_value="false"),
        DeclareLaunchArgument("low_transition_delay_s", default_value="0.1"),
        DeclareLaunchArgument("low_terminal_state", default_value="succeeded"),
        DeclareLaunchArgument(
            "low_failure_error", default_value="configured Low System failure"
        ),
        vision,
        Node(
            package="flzat_high_bringup",
            executable="static_frame_server",
            name="static_frame_server",
            output="screen",
            condition=IfCondition(
                EqualsSubstitution(LaunchConfiguration("frame_source"), "static")
            ),
            parameters=[{"image_path": LaunchConfiguration("static_image_path")}],
        ),
        Node(
            package="flzat_memory",
            executable="memory_query_node",
            name="memory_query_node",
            output="screen",
            parameters=[config, {
                "storage_root": LaunchConfiguration("storage_root"),
                "environment_id": LaunchConfiguration("environment_id"),
                "map_id": LaunchConfiguration("map_id"),
                "embedding_backend": LaunchConfiguration("embedding_backend"),
                "embedding_local_files_only": ParameterValue(
                    LaunchConfiguration("embedding_local_files_only"), value_type=bool
                ),
            }],
        ),
        Node(
            package="flzat_speech",
            executable="vad_node",
            name="vad_node",
            output="screen",
            parameters=[speech_parameters("vad")],
            condition=IfCondition(
                AndSubstitution(
                    NotSubstitution(LaunchConfiguration("use_fake_speech")),
                    LaunchConfiguration("run_vad_node"),
                )
            ),
        ),
        Node(
            package="flzat_speech",
            executable="stt_node",
            name="stt_node",
            output="screen",
            condition=IfCondition(
                AndSubstitution(
                    NotSubstitution(LaunchConfiguration("use_fake_speech")),
                    LaunchConfiguration("run_stt_node"),
                )
            ),
            parameters=[speech_parameters("stt")],
        ),
        Node(
            package="flzat_speech",
            executable="tts_node",
            name="tts_node",
            output="screen",
            condition=IfCondition(
                AndSubstitution(
                    NotSubstitution(LaunchConfiguration("use_fake_speech")),
                    LaunchConfiguration("run_tts_node"),
                )
            ),
            parameters=[speech_parameters("tts")],
        ),
        Node(
            package="flzat_high_bringup",
            executable="fake_stt_server",
            name="fake_stt_server",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_fake_speech")),
            parameters=[{"transcript": LaunchConfiguration("fake_transcript")}],
        ),
        Node(
            package="flzat_high_bringup",
            executable="fake_tts_server",
            name="fake_tts_server",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_fake_speech")),
        ),
        Node(
            package="flzat_speech",
            executable="speech_ingress",
            name="speech_ingress",
            output="screen",
            parameters=[speech_parameters("ingress")],
        ),
        Node(
            package="flzat_cognitive_pipeline",
            executable="intent_router",
            name="intent_router",
            output="screen",
            parameters=[config, {
                "config_file": LaunchConfiguration("intent_routes_path"),
                "embedding_backend": LaunchConfiguration("embedding_backend"),
                "embedding_device": "cpu",
                "embedding_local_files_only": ParameterValue(
                    LaunchConfiguration("embedding_local_files_only"), value_type=bool
                ),
                "enable_text_command": ParameterValue(
                    LaunchConfiguration("enable_text_command"), value_type=bool
                ),
            }],
        ),
        real_vlm,
        Node(
            package="flzat_high_bringup",
            executable="fake_vlm_server",
            name="fake_vlm_server",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_fake_vlm")),
            parameters=[{
                "response_text": ParameterValue(
                    LaunchConfiguration("fake_vlm_response"), value_type=str
                ),
                "delay_s": ParameterValue(
                    LaunchConfiguration("fake_vlm_delay_s"), value_type=float
                ),
                "ignore_cancel": ParameterValue(
                    LaunchConfiguration("fake_vlm_ignore_cancel"), value_type=bool
                ),
                "second_response_text": ParameterValue(
                    LaunchConfiguration("fake_vlm_second_response"), value_type=str
                ),
                "routing_response_text": ParameterValue(
                    LaunchConfiguration("fake_routing_response"), value_type=str
                ),
            }],
        ),
        Node(
            package="flzat_cognitive_pipeline",
            executable="cognitive_manager",
            name="cognitive_manager",
            output="screen",
            parameters=[config, {
                "tts_timeout_s": ParameterValue(
                    LaunchConfiguration("tts_timeout_s"), value_type=float
                ),
                "environment_id": LaunchConfiguration("environment_id"),
                "map_id": LaunchConfiguration("map_id"),
                "output_root": LaunchConfiguration("output_root"),
                "context_policy_path": LaunchConfiguration("intent_routes_path"),
                "skill_registry_path": LaunchConfiguration("skill_registry_path"),
                "safety_policy_path": LaunchConfiguration("safety_policy_path"),
                "user_profile_path": LaunchConfiguration("user_profile_path"),
                "request_timeout_s": ParameterValue(
                    LaunchConfiguration("request_timeout_s"), value_type=float
                ),
                "max_repair_attempts": ParameterValue(
                    LaunchConfiguration("max_repair_attempts"), value_type=int
                ),
                "health_period_s": ParameterValue(
                    LaunchConfiguration("health_period_s"), value_type=float
                ),
            }],
        ),
        Node(
            package="flzat_high_bringup",
            executable="low_system_mock",
            name="low_system_mock",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_low_mock")),
            parameters=[{
                "status_sequence": ["accepted"],
                "terminal_state": LaunchConfiguration("low_terminal_state"),
                "failure_error": LaunchConfiguration("low_failure_error"),
                "transition_delay_s": ParameterValue(
                    LaunchConfiguration("low_transition_delay_s"), value_type=float
                ),
            }],
        ),
    ])
