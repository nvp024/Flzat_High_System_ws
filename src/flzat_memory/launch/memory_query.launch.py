from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "environment_id",
            description="Required environment directory containing the RAG database",
        ),
        DeclareLaunchArgument(
            "map_id",
            description="Required map identifier recorded in the memory manifest",
        ),
        DeclareLaunchArgument("storage_root", default_value=""),
        DeclareLaunchArgument("embedding_device", default_value="cpu"),
        DeclareLaunchArgument(
            "embedding_backend", default_value="sentence_transformer"
        ),
        DeclareLaunchArgument(
            "embedding_local_files_only", default_value="false"
        ),
        Node(
            package="flzat_memory",
            executable="memory_query_node",
            name="memory_query_node",
            output="screen",
            parameters=[{
                "environment_id": LaunchConfiguration("environment_id"),
                "map_id": LaunchConfiguration("map_id"),
                "storage_root": LaunchConfiguration("storage_root"),
                "embedding_device": LaunchConfiguration("embedding_device"),
                "embedding_backend": LaunchConfiguration("embedding_backend"),
                "embedding_local_files_only": ParameterValue(
                    LaunchConfiguration("embedding_local_files_only"),
                    value_type=bool,
                ),
            }],
        ),
    ])
