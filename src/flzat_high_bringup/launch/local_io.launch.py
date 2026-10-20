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
        DeclareLaunchArgument("tts_rate", default_value="175"),
        DeclareLaunchArgument("tts_volume", default_value="1.0"),
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
                "tts_rate": ParameterValue(
                    LaunchConfiguration("tts_rate"), value_type=int
                ),
                "tts_volume": ParameterValue(
                    LaunchConfiguration("tts_volume"), value_type=float
                ),
            }],
        ),
    ])
