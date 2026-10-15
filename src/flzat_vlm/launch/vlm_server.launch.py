from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="qwen3_vl"),
        DeclareLaunchArgument(
            "model_id",
            default_value="Qwen/Qwen3-VL-2B-Instruct",
        ),
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
        Node(
            package="flzat_vlm",
            executable="vlm_node",
            name="vlm_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "backend": ParameterValue(
                    LaunchConfiguration("backend"), value_type=str
                ),
                "model_id": ParameterValue(
                    LaunchConfiguration("model_id"), value_type=str
                ),
                "model_revision": ParameterValue(
                    LaunchConfiguration("model_revision"), value_type=str
                ),
                "device": ParameterValue(
                    LaunchConfiguration("device"), value_type=str
                ),
                "dtype": ParameterValue(
                    LaunchConfiguration("dtype"), value_type=str
                ),
                "quantization": ParameterValue(
                    LaunchConfiguration("quantization"), value_type=str
                ),
                "max_new_tokens": ParameterValue(
                    LaunchConfiguration("max_new_tokens"), value_type=int
                ),
                "trust_remote_code": ParameterValue(
                    LaunchConfiguration("trust_remote_code"), value_type=bool
                ),
                "local_files_only": ParameterValue(
                    LaunchConfiguration("local_files_only"), value_type=bool
                ),
                "qwen_min_image_pixels": ParameterValue(
                    LaunchConfiguration("qwen_min_image_pixels"), value_type=int
                ),
                "qwen_max_image_pixels": ParameterValue(
                    LaunchConfiguration("qwen_max_image_pixels"), value_type=int
                ),
            }],
        ),
    ])
