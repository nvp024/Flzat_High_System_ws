from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from flzat_high_interfaces.action import SpeechToText
from flzat_high_interfaces.msg import SpeechAudio

EXPECTED_SAMPLE_RATE: int = 16_000
DEFAULT_WHISPER_MODEL: str = "base.en.pt"
DEFAULT_BEAM_SIZE: int = 5
DEFAULT_MAX_NO_SPEECH_PROBABILITY: float = 0.65
DEFAULT_MIN_AVG_LOGPROB: float = -1.0
MAX_COMPRESSION_RATIO: float = 2.4
MAX_WORDS_PER_SECOND: float = 5.0
MAX_DECODE_TOKENS: int = 64


def _default_whisper_model_path() -> str:
    """Find the conventional model path without requiring a sourced shell."""
    colcon_prefixes = os.environ.get("COLCON_PREFIX_PATH", "")
    for prefix_value in filter(None, colcon_prefixes.split(os.pathsep)):
        prefix = Path(prefix_value).expanduser()
        # Colcon may use <workspace>/install or
        # <workspace>/install/<package> as an environment prefix.
        for models_dir in (prefix / "models", prefix.parent / "models"):
            if models_dir.is_dir():
                return str(models_dir / DEFAULT_WHISPER_MODEL)
        if prefix.parent.name == "install":
            models_dir = prefix.parent.parent / "models"
            if models_dir.is_dir():
                return str(models_dir / DEFAULT_WHISPER_MODEL)

    # This also works with --symlink-install because resolve() reaches src/.
    models_dir = Path(__file__).resolve().parents[3] / "models"
    return str(models_dir / DEFAULT_WHISPER_MODEL)


def _build_transcribe_options(
    language: str | None,
    beam_size: int,
    initial_prompt: str,
) -> dict[str, object]:
    """Build deterministic decoding options for independent robot commands."""
    options: dict[str, object] = {
        "task": "transcribe",
        "fp16": False,
        "temperature": 0.0,
        "beam_size": beam_size,
        "condition_on_previous_text": False,
        "sample_len": MAX_DECODE_TOKENS,
        "verbose": None,
    }
    if language is not None:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    return options


def _extract_confident_transcript(
    result_data: dict,
    max_no_speech_probability: float,
    min_avg_logprob: float,
    audio_duration_s: float,
) -> str:
    """Keep only Whisper segments that contain sufficiently confident speech."""
    segments = result_data.get("segments")
    if not isinstance(segments, list):
        return str(result_data.get("text", "")).strip()

    accepted: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        no_speech_probability = float(segment.get("no_speech_prob", 0.0))
        average_log_probability = float(segment.get("avg_logprob", 0.0))
        compression_ratio = float(segment.get("compression_ratio", 0.0))
        if no_speech_probability > max_no_speech_probability:
            continue
        if average_log_probability < min_avg_logprob:
            continue
        if compression_ratio > MAX_COMPRESSION_RATIO:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            accepted.append(text)
    transcript = " ".join(accepted)
    if audio_duration_s > 0.0:
        words_per_second = len(transcript.split()) / audio_duration_s
        if words_per_second > MAX_WORDS_PER_SECOND:
            return ""
    return transcript


def _load_stt_model(model_path: str):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Whisper model not found at {path}. "
            "Set the 'whisper_model_path' ROS parameter to a local .pt model file."
        )
    import whisper
    model = whisper.load_model(str(path))
    return model


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("stt_node")

        self._cb_group = ReentrantCallbackGroup()
        self._goal_lock = threading.Lock()
        self._goal_active = False

        self.get_logger().info("Loading Whisper model …")
        self.declare_parameter("whisper_model_path", _default_whisper_model_path())
        self.declare_parameter("language", "en")
        self.declare_parameter("beam_size", DEFAULT_BEAM_SIZE)
        self.declare_parameter("initial_prompt", "")
        self.declare_parameter(
            "max_no_speech_probability",
            DEFAULT_MAX_NO_SPEECH_PROBABILITY,
        )
        self.declare_parameter("min_avg_logprob", DEFAULT_MIN_AVG_LOGPROB)
        self.declare_parameter("expected_sample_rate", EXPECTED_SAMPLE_RATE)
        self.declare_parameter("max_audio_duration_s", 30.0)
        model_path = (
            self.get_parameter("whisper_model_path")
            .get_parameter_value()
            .string_value.strip()
            or _default_whisper_model_path()
        )
        language = self.get_parameter("language").get_parameter_value().string_value.strip()
        self._language = language or None
        self._beam_size = self.get_parameter(
            "beam_size"
        ).get_parameter_value().integer_value
        self._initial_prompt = self.get_parameter(
            "initial_prompt"
        ).get_parameter_value().string_value.strip()
        self._max_no_speech_probability = self.get_parameter(
            "max_no_speech_probability"
        ).get_parameter_value().double_value
        self._min_avg_logprob = self.get_parameter(
            "min_avg_logprob"
        ).get_parameter_value().double_value
        self._expected_sample_rate = self.get_parameter(
            "expected_sample_rate"
        ).get_parameter_value().integer_value
        self._max_audio_duration_s = self.get_parameter(
            "max_audio_duration_s"
        ).get_parameter_value().double_value
        if self._expected_sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Whisper input must be {EXPECTED_SAMPLE_RATE} Hz; "
                "resampling is not implemented in this node."
            )
        if self._max_audio_duration_s <= 0.0:
            raise ValueError("Parameter 'max_audio_duration_s' must be greater than zero.")
        if self._beam_size < 1:
            raise ValueError("Parameter 'beam_size' must be at least 1.")
        if not 0.0 <= self._max_no_speech_probability <= 1.0:
            raise ValueError(
                "Parameter 'max_no_speech_probability' must be between 0.0 and 1.0."
            )
        self.get_logger().info(f"Whisper model path: {model_path}")
        self._stt_model = _load_stt_model(model_path)
        self.get_logger().info(
            f"Whisper model ready; language={self._language or 'auto'}; "
            f"beam_size={self._beam_size}; deterministic decoding enabled."
        )

        self._action_server = ActionServer(
            self,
            SpeechToText,
            "/stt_action",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self.get_logger().info("SttNode ready — /stt_action active.")

    def _goal_callback(self, goal_request) -> GoalResponse:
        packet = goal_request.audio_packet
        sample_count = len(packet.audio_data)
        if sample_count == 0:
            self.get_logger().warn("Rejected empty STT goal.")
            return GoalResponse.REJECT
        if packet.sample_rate != self._expected_sample_rate:
            self.get_logger().warn(
                f"Rejected STT goal with sample rate {packet.sample_rate}; "
                f"expected {self._expected_sample_rate} Hz."
            )
            return GoalResponse.REJECT
        max_samples = int(self._expected_sample_rate * self._max_audio_duration_s)
        if sample_count > max_samples:
            self.get_logger().warn(
                f"Rejected STT goal with {sample_count} samples; "
                f"limit is {max_samples} ({self._max_audio_duration_s:.1f} s)."
            )
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warn("Rejected STT goal because inference is already active.")
                return GoalResponse.REJECT
            self._goal_active = True

        self.get_logger().info(
            f"STT goal accepted: {sample_count} samples @ {packet.sample_rate} Hz"
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("STT cancel request received.")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        try:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("STT goal cancelled before inference started.")
                goal_handle.canceled()
                result = SpeechToText.Result()
                result.transcript = ""
                return result

            self.get_logger().info("Transcribing …")
            audio_packet: SpeechAudio = goal_handle.request.audio_packet
            audio_i16 = np.array(audio_packet.audio_data, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

            feedback_msg = SpeechToText.Feedback()
            feedback_msg.status = "inference_started"
            goal_handle.publish_feedback(feedback_msg)

            transcribe_options = _build_transcribe_options(
                self._language,
                self._beam_size,
                self._initial_prompt,
            )
            result_data = self._stt_model.transcribe(audio_f32, **transcribe_options)
            transcript = _extract_confident_transcript(
                result_data,
                self._max_no_speech_probability,
                self._min_avg_logprob,
                len(audio_f32) / self._expected_sample_rate,
            )
            if not transcript:
                self.get_logger().warn(
                    "Whisper result rejected: no sufficiently confident speech."
                )

            # Whisper cannot be safely interrupted mid-inference. The action is
            # nevertheless marked cancelled as soon as that inference returns.
            if goal_handle.is_cancel_requested:
                self.get_logger().info("STT goal cancelled after inference completed.")
                goal_handle.canceled()
                result = SpeechToText.Result()
                result.transcript = ""
                return result

            feedback_msg.status = "inference_complete"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"[STT] '{transcript}'")
            goal_handle.succeed()
            result = SpeechToText.Result()
            result.transcript = transcript
            return result
        except Exception as exc:
            self.get_logger().error(f"STT inference failed: {exc}")
            goal_handle.abort()
            result = SpeechToText.Result()
            result.transcript = ""
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False

    def destroy_node(self) -> None:
        if hasattr(self, "_action_server"):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = SttNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("SttNode shutting down …")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
