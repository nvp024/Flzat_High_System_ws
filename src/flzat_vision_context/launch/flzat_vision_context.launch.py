from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="15.0"),
        DeclareLaunchArgument("debug_fps", default_value="5.0"),
        DeclareLaunchArgument("camera_buffer_duration_s", default_value="35.0"),
        Node(
            package="flzat_vision_context",
            executable="camera_node",
            name="camera_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "buffer_duration_s": ParameterValue(
                    LaunchConfiguration("camera_buffer_duration_s"), value_type=float
                ),
                "camera_index": ParameterValue(
                    LaunchConfiguration("camera_index"), value_type=int
                ),
                "camera_width": ParameterValue(
                    LaunchConfiguration("camera_width"), value_type=int
                ),
                "camera_height": ParameterValue(
                    LaunchConfiguration("camera_height"), value_type=int
                ),
                "camera_fps": ParameterValue(
                    LaunchConfiguration("camera_fps"), value_type=float
                ),
                "debug_fps": ParameterValue(
                    LaunchConfiguration("debug_fps"), value_type=float
                ),
            }],
        ),
    ])
