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
    '{"speech":"I will ask the robot to find the water cup.",'
    '"goal":{"description":"Find the water cup requested by the user",'
    '"constraints":[],"selected_context_refs":[],'
    '"missing_information":["current water cup location"]},'
    '"interaction":null}'
)


def generate_launch_description():
    log_arguments = ["--ros-args", "--log-level", LaunchConfiguration("log_level")]
    config = (
        get_package_share_directory("flzat_high_bringup")
        + "/config/high_system.yaml"
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
        launch_arguments={"log_level": LaunchConfiguration("log_level")}.items(),
    )
    real_vlm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("flzat_vlm"),
            "/launch/vlm_server.launch.py",
        ]),
        condition=UnlessCondition(LaunchConfiguration("use_fake_vlm")),
        launch_arguments={
            "log_level": LaunchConfiguration("log_level"),
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
        DeclareLaunchArgument(
            "log_level", default_value="info",
            choices=["debug", "info", "warn", "error", "fatal"],
            description="ROS node log verbosity; info is concise, debug includes details",
        ),
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
        DeclareLaunchArgument("output_root", default_value="~/.local/share/flzat/high_system/handoffs"),
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
        DeclareLaunchArgument("fake_vlm_delay_s", default_value="0.0"),
        DeclareLaunchArgument("fake_vlm_ignore_cancel", default_value="false"),
        DeclareLaunchArgument("low_transition_delay_s", default_value="0.1"),
        DeclareLaunchArgument("low_result_outcome", default_value="STARTED_PLAN"),
        DeclareLaunchArgument(
            "low_failure_error", default_value="configured Low System failure"
        ),
        DeclareLaunchArgument("low_ask_user_kind", default_value=""),
        DeclareLaunchArgument(
            "low_ask_user_content_hint",
            default_value="Which object do you mean?",
        ),
        DeclareLaunchArgument("low_ask_user_timeout_s", default_value="30.0"),
        DeclareLaunchArgument("low_ask_user_retry_interval_s", default_value="1.0"),
        DeclareLaunchArgument("low_ask_user_max_attempts", default_value="10"),
        vision,
        Node(
            package="flzat_high_bringup",
            executable="static_frame_server",
            arguments=log_arguments,
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
            arguments=log_arguments,
            name="knowledge_server",
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
            arguments=log_arguments,
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
            arguments=log_arguments,
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
            arguments=log_arguments,
            name="speech_output",
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
            arguments=log_arguments,
            name="fake_stt_server",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_fake_speech")),
            parameters=[{"transcript": LaunchConfiguration("fake_transcript")}],
        ),
        Node(
            package="flzat_high_bringup",
            executable="fake_tts_server",
            arguments=log_arguments,
            name="fake_tts_server",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_fake_speech")),
        ),
        Node(
            package="flzat_speech",
            executable="speech_ingress",
            arguments=log_arguments,
            name="speech_input",
            output="screen",
            parameters=[speech_parameters("ingress")],
        ),
        real_vlm,
        Node(
            package="flzat_high_bringup",
            executable="fake_vlm_server",
            arguments=log_arguments,
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
            }],
        ),
        Node(
            package="flzat_cognitive_pipeline",
            executable="cognitive_manager",
            arguments=log_arguments,
            name="vlm_brain",
            output="screen",
            parameters=[config, {
                "tts_timeout_s": ParameterValue(
                    LaunchConfiguration("tts_timeout_s"), value_type=float
                ),
                "environment_id": LaunchConfiguration("environment_id"),
                "map_id": LaunchConfiguration("map_id"),
                "output_root": LaunchConfiguration("output_root"),
                "enable_text_command": ParameterValue(
                    LaunchConfiguration("enable_text_command"), value_type=bool
                ),
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
            arguments=log_arguments,
            name="low_system_mock",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_low_mock")),
            parameters=[{
                "result_outcome": LaunchConfiguration("low_result_outcome"),
                "failure_error": LaunchConfiguration("low_failure_error"),
                "ask_user_kind": LaunchConfiguration("low_ask_user_kind"),
                "ask_user_content_hint": LaunchConfiguration(
                    "low_ask_user_content_hint"
                ),
                "ask_user_timeout_s": ParameterValue(
                    LaunchConfiguration("low_ask_user_timeout_s"), value_type=float
                ),
                "ask_user_retry_interval_s": ParameterValue(
                    LaunchConfiguration("low_ask_user_retry_interval_s"),
                    value_type=float,
                ),
                "ask_user_max_attempts": ParameterValue(
                    LaunchConfiguration("low_ask_user_max_attempts"), value_type=int
                ),
                "transition_delay_s": ParameterValue(
                    LaunchConfiguration("low_transition_delay_s"), value_type=float
                ),
            }],
        ),
    ])
