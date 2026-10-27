"""Reusable VAD, STT and TTS servers without an audio dispatcher."""

from launch import LaunchDescription
from launch_ros.actions import Node

from flzat_speech.launch_config import speech_arguments, speech_parameters


def generate_launch_description():
    return LaunchDescription([
        *speech_arguments(("vad", "stt", "tts")),
        Node(
            package="flzat_speech", executable="vad_node", name="vad_node",
            output="screen", emulate_tty=True,
            parameters=[speech_parameters("vad")],
        ),
        Node(
            package="flzat_speech", executable="stt_node", name="stt_node",
            output="screen", emulate_tty=True,
            parameters=[speech_parameters("stt")],
        ),
        Node(
            package="flzat_speech", executable="tts_node", name="tts_node",
            output="screen", emulate_tty=True,
            parameters=[speech_parameters("tts")],
        ),
    ])
