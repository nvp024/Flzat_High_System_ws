"""Hardware I/O boundary for a split local/cloud deployment."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from flzat_speech.launch_config import speech_arguments, speech_parameters


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
            "camera_buffer_duration_s": LaunchConfiguration("camera_buffer_duration_s"),
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
        DeclareLaunchArgument("camera_buffer_duration_s", default_value="35.0"),
        DeclareLaunchArgument(
            "debug_fps",
            default_value="0.0",
            description="Disable debug image topics by default over a WAN link",
        ),
        *speech_arguments(("vad", "tts")),
        vision,
        Node(
            package="flzat_speech",
            executable="vad_node",
            name="vad_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("run_vad_node")),
            parameters=[speech_parameters("vad")],
        ),
        Node(
            package="flzat_speech",
            executable="tts_node",
            name="tts_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("run_tts_node")),
            parameters=[speech_parameters("tts")],
        ),
    ])
