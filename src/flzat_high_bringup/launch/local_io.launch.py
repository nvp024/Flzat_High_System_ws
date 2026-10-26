"""Hardware I/O boundary for a split local/cloud deployment."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("flzat_vision_context"),
            "/launch/flzat_vision_context.launch.py",
        ]),
        condition=IfCondition(LaunchConfiguration("run_camera_node")),
        launch_arguments={
            "camera_index": LaunchConfiguration("camera_index"),
            "camera_width": LaunchConfiguration("camera_width"),
            "camera_height": LaunchConfiguration("camera_height"),
            "camera_fps": LaunchConfiguration("camera_fps"),
            "debug_fps": LaunchConfiguration("debug_fps"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("run_camera_node", default_value="true"),
        DeclareLaunchArgument("run_vad_node", default_value="true"),
        DeclareLaunchArgument("run_tts_node", default_value="true"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="15.0"),
        DeclareLaunchArgument(
            "debug_fps",
            default_value="0.0",
            description="Disable debug image topics by default over a WAN link",
        ),
        DeclareLaunchArgument("mic_device", default_value="sysdefault"),
        DeclareLaunchArgument("vad_silence_ms", default_value="500"),
        DeclareLaunchArgument("tts_volume", default_value="1.0"),
        DeclareLaunchArgument("piper_model_path", default_value=""),
        DeclareLaunchArgument("piper_config_path", default_value=""),
        DeclareLaunchArgument("piper_length_scale", default_value="1.0"),
        DeclareLaunchArgument("piper_use_cuda", default_value="false"),
        DeclareLaunchArgument("tts_output_sample_rate", default_value="48000"),
        DeclareLaunchArgument("tts_sentence_pause_ms", default_value="200"),
        DeclareLaunchArgument("tts_max_chunk_characters", default_value="240"),
        vision,
        Node(
            package="flzat_speech",
            executable="vad_node",
            name="vad_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("run_vad_node")),
            parameters=[{
                "mic_device": ParameterValue(
                    LaunchConfiguration("mic_device"), value_type=str
                ),
                "silence_ms": ParameterValue(
                    LaunchConfiguration("vad_silence_ms"), value_type=int
                ),
            }],
        ),
        Node(
            package="flzat_speech",
            executable="tts_node",
            name="tts_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("run_tts_node")),
            parameters=[{
                "tts_volume": ParameterValue(
                    LaunchConfiguration("tts_volume"), value_type=float
                ),
                "piper_model_path": LaunchConfiguration("piper_model_path"),
                "piper_config_path": LaunchConfiguration("piper_config_path"),
                "piper_length_scale": ParameterValue(
                    LaunchConfiguration("piper_length_scale"), value_type=float
                ),
                "piper_use_cuda": ParameterValue(
                    LaunchConfiguration("piper_use_cuda"), value_type=bool
                ),
                "output_sample_rate": ParameterValue(
                    LaunchConfiguration("tts_output_sample_rate"), value_type=int
                ),
                "sentence_pause_ms": ParameterValue(
                    LaunchConfiguration("tts_sentence_pause_ms"), value_type=int
                ),
                "max_chunk_characters": ParameterValue(
                    LaunchConfiguration("tts_max_chunk_characters"), value_type=int
                ),
            }],
        ),
    ])
