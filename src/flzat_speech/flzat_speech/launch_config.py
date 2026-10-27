"""Shared speech launch arguments; keep test and production settings identical."""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


# (launch argument, ROS node parameter, default). Model files remain local.
SPEECH_OPTIONS = {
    "vad": (
        ("mic_device", "mic_device", "sysdefault"),
        ("vad_silence_ms", "silence_ms", 500),
        ("tts_resume_delay_s", "tts_resume_delay_s", 0.8),
        ("speech_max_duration_s", "max_segment_duration_s", 30.0),
    ),
    "stt": (
        ("whisper_model_path", "whisper_model_path", ""),
        ("whisper_language", "language", "en"),
        ("whisper_beam_size", "beam_size", 5),
        ("whisper_initial_prompt", "initial_prompt", ""),
        ("whisper_max_no_speech_probability", "max_no_speech_probability", 0.65),
        ("whisper_min_avg_logprob", "min_avg_logprob", -1.0),
        ("whisper_max_decode_tokens", "max_decode_tokens", 64),
        ("speech_max_duration_s", "max_audio_duration_s", 30.0),
    ),
    "ingress": (
        ("pending_audio_max_age_s", "pending_audio_max_age_s", 2.0),
        ("stt_timeout_s", "stt_timeout_s", 30.0),
        ("tts_timeout_s", "tts_timeout_s", 65.0),
    ),
    "tts": (
        ("tts_volume", "tts_volume", 1.0),
        ("piper_model_path", "piper_model_path", ""),
        ("piper_config_path", "piper_config_path", ""),
        ("piper_length_scale", "piper_length_scale", 1.0),
        ("piper_use_cuda", "piper_use_cuda", False),
        ("tts_output_sample_rate", "output_sample_rate", 48000),
        ("tts_sentence_pause_ms", "sentence_pause_ms", 200),
        ("tts_max_chunk_characters", "max_chunk_characters", 240),
        ("tts_speech_timeout_s", "speech_timeout_s", 60.0),
        ("tts_cancel_grace_s", "cancel_grace_s", 2.0),
    ),
}


def speech_arguments(groups=("vad", "stt", "ingress", "tts")):
    defaults = {
        name: default for group in groups
        for name, _, default in SPEECH_OPTIONS[group]
    }
    return [DeclareLaunchArgument(name, default_value=str(default))
            for name, default in defaults.items()]


def speech_parameters(group):
    return {
        parameter: ParameterValue(LaunchConfiguration(name), value_type=type(default))
        for name, parameter, default in SPEECH_OPTIONS[group]
    }
