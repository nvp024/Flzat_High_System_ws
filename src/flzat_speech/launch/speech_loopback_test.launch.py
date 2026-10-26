"""Standalone hardware diagnostic: microphone VAD -> STT -> TTS echo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "vad_silence_ms",
            default_value="500",
            description="Silence required to finish an utterance.",
        ),
        DeclareLaunchArgument(
            "whisper_language",
            default_value="en",
            description="Whisper language code; empty enables auto detection.",
        ),
        DeclareLaunchArgument(
            "whisper_model_path",
            default_value="",
            description="Local Whisper checkpoint; empty uses models/base.en.pt.",
        ),
        DeclareLaunchArgument("whisper_beam_size", default_value="5"),
        DeclareLaunchArgument("whisper_initial_prompt", default_value=""),
        DeclareLaunchArgument(
            "whisper_max_no_speech_probability", default_value="0.65"
        ),
        DeclareLaunchArgument("whisper_min_avg_logprob", default_value="-1.0"),
        Node(
            package="flzat_speech",
            executable="vad_node",
            name="vad_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "silence_ms": ParameterValue(
                    LaunchConfiguration("vad_silence_ms"), value_type=int
                ),
            }],
        ),
        Node(
            package="flzat_speech",
            executable="stt_node",
            name="stt_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "language": ParameterValue(
                    LaunchConfiguration("whisper_language"), value_type=str
                ),
                "whisper_model_path": LaunchConfiguration("whisper_model_path"),
                "beam_size": ParameterValue(
                    LaunchConfiguration("whisper_beam_size"), value_type=int
                ),
                "initial_prompt": LaunchConfiguration("whisper_initial_prompt"),
                "max_no_speech_probability": ParameterValue(
                    LaunchConfiguration("whisper_max_no_speech_probability"),
                    value_type=float,
                ),
                "min_avg_logprob": ParameterValue(
                    LaunchConfiguration("whisper_min_avg_logprob"), value_type=float
                ),
            }],
        ),
        Node(
            package="flzat_speech",
            executable="tts_node",
            name="tts_node",
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="flzat_speech",
            executable="audio_loopback_node",
            name="audio_loopback_node",
            output="screen",
            emulate_tty=True,
        ),
    ])
