from __future__ import annotations

import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Optional

from flzat_high_interfaces.action import TextToSpeech
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool

TTS_VOLUME: float = 1.0
DEFAULT_TTS_ACTIVE_TOPIC: str = "/voice/tts_active"
DEFAULT_PIPER_VOICE: str = "en_US-lessac-medium"


def _default_piper_model_path() -> str:
    """Find the workspace-owned Piper model without assuming the current directory."""
    relative = Path("tts") / f"{DEFAULT_PIPER_VOICE}.onnx"
    colcon_prefixes = os.environ.get("COLCON_PREFIX_PATH", "")
    for prefix_value in filter(None, colcon_prefixes.split(os.pathsep)):
        prefix = Path(prefix_value).expanduser()
        for models_dir in (prefix / "models", prefix.parent / "models"):
            if models_dir.is_dir():
                return str(models_dir / relative)
        if prefix.parent.name == "install":
            models_dir = prefix.parent.parent / "models"
            if models_dir.is_dir():
                return str(models_dir / relative)
    return str(Path(__file__).resolve().parents[3] / "models" / relative)


def _split_speech_text(text: str, max_characters: int) -> tuple[str, ...]:
    """Normalize whitespace and bound synthesis chunks at natural boundaries."""
    value = " ".join(text.split())
    if not value:
        return ()
    sentences = re.split(r"(?<=[.!?])\s+", value)
    chunks: list[str] = []
    for sentence in sentences:
        remaining = sentence.strip()
        while len(remaining) > max_characters:
            boundary = max(
                remaining.rfind(marker, 0, max_characters + 1)
                for marker in (", ", "; ", ": ", " ")
            )
            if boundary <= 0:
                boundary = max_characters
            elif remaining[boundary] in ",;:":
                boundary += 1
            chunk = remaining[:boundary].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[boundary:].strip()
        if remaining:
            chunks.append(remaining)
    return tuple(chunks)


def _resample_pcm_int16(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample interleaved PCM to a device-friendly rate without extra dependencies."""
    if source_rate == target_rate or samples.shape[0] == 0:
        return samples
    target_frames = max(1, round(samples.shape[0] * target_rate / source_rate))
    source_positions = np.arange(samples.shape[0], dtype=np.float64)
    target_positions = np.linspace(
        0.0,
        float(samples.shape[0] - 1),
        target_frames,
    )
    channels = [
        np.interp(target_positions, source_positions, samples[:, index])
        for index in range(samples.shape[1])
    ]
    return np.clip(np.rint(np.column_stack(channels)), -32768, 32767).astype(
        np.int16
    )


class _PiperEngine:
    """Persistent Piper voice with cancellable, in-memory audio playback."""

    def __init__(
        self,
        stop_event: threading.Event,
        model_path: str,
        config_path: str,
        *,
        volume: float,
        length_scale: float,
        output_sample_rate: int,
        sentence_pause_ms: int,
        max_chunk_characters: int,
        use_cuda: bool,
    ) -> None:
        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(
                f"Piper model not found at {model}. Download {DEFAULT_PIPER_VOICE} "
                "or set the 'piper_model_path' ROS parameter."
            )
        config = Path(config_path).expanduser().resolve() if config_path else None
        if config is not None and not config.is_file():
            raise FileNotFoundError(f"Piper voice config not found at {config}")
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts is not installed; install the workspace requirements.txt"
            ) from exc

        self._stop_event = stop_event
        self._voice = PiperVoice.load(
            str(model),
            config_path=str(config) if config is not None else None,
            use_cuda=use_cuda,
        )
        self._synthesis_config = SynthesisConfig(
            volume=volume,
            length_scale=length_scale,
        )
        self._output_sample_rate = output_sample_rate
        self._sentence_pause_ms = sentence_pause_ms
        self._max_chunk_characters = max_chunk_characters
        self._stream = None
        self._stream_lock = threading.Lock()
        self.last_error: Optional[str] = None

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._stream_lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception as exc:
                self.last_error = f"Unable to stop Piper playback: {exc}"

    def speak(
        self,
        text: str,
        on_word_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        try:
            import sounddevice as sd
        except ImportError:
            self.last_error = "sounddevice is not installed"
            return False

        chunks = _split_speech_text(text, self._max_chunk_characters)
        if not chunks:
            self.last_error = "speech text is empty"
            return False

        stream = None
        audio_format: tuple[int, int, int] | None = None
        try:
            for text_index, speech_chunk in enumerate(chunks):
                if self._stop_event.is_set():
                    return False
                if on_word_callback is not None:
                    for word in speech_chunk.split():
                        on_word_callback(word)
                for audio_chunk in self._voice.synthesize(
                    speech_chunk,
                    syn_config=self._synthesis_config,
                ):
                    if self._stop_event.is_set():
                        return False
                    current_format = (
                        int(audio_chunk.sample_rate),
                        int(audio_chunk.sample_width),
                        int(audio_chunk.sample_channels),
                    )
                    if current_format[1] != 2:
                        raise RuntimeError("Piper must produce 16-bit PCM audio")
                    if stream is None:
                        audio_format = current_format
                        stream = sd.OutputStream(
                            samplerate=self._output_sample_rate,
                            channels=current_format[2],
                            dtype="int16",
                        )
                        with self._stream_lock:
                            self._stream = stream
                        stream.start()
                    elif current_format != audio_format:
                        raise RuntimeError("Piper changed audio format during synthesis")
                    samples = np.frombuffer(
                        audio_chunk.audio_int16_bytes, dtype=np.int16
                    ).reshape(-1, current_format[2])
                    stream.write(
                        _resample_pcm_int16(
                            samples,
                            current_format[0],
                            self._output_sample_rate,
                        )
                    )

                if (
                    stream is not None
                    and text_index + 1 < len(chunks)
                    and self._sentence_pause_ms > 0
                    and audio_format is not None
                ):
                    silence_frames = int(
                        self._output_sample_rate * self._sentence_pause_ms / 1000
                    )
                    stream.write(
                        np.zeros(
                            (silence_frames, audio_format[2]), dtype=np.int16
                        )
                    )
            if stream is None:
                raise RuntimeError("Piper produced no audio")
            stream.stop()
            return not self._stop_event.is_set()
        except Exception as exc:
            self.last_error = str(exc)
            return False
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            with self._stream_lock:
                self._stream = None


class TtsNode(Node):
    """A single-speaker TTS action server with explicit VAD muting state."""

    def __init__(self) -> None:
        super().__init__("tts_node")

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._goal_reserved = False
        self._active_token: Optional[object] = None
        self._active_engine: Optional[_PiperEngine] = None
        self._cb_group = ReentrantCallbackGroup()
        self._execution_done = threading.Event()
        self._execution_done.set()
        self._shutting_down = False
        self._speech_worker = None

        self.declare_parameter("tts_volume", TTS_VOLUME)
        self.declare_parameter("piper_model_path", "")
        self.declare_parameter("piper_config_path", "")
        self.declare_parameter("piper_length_scale", 1.0)
        self.declare_parameter("piper_use_cuda", False)
        self.declare_parameter("output_sample_rate", 48000)
        self.declare_parameter("sentence_pause_ms", 200)
        self.declare_parameter("max_chunk_characters", 240)
        self.declare_parameter("speech_timeout_s", 60.0)
        self.declare_parameter("cancel_grace_s", 2.0)
        self.declare_parameter("tts_active_topic", DEFAULT_TTS_ACTIVE_TOPIC)
        self._tts_volume = self.get_parameter("tts_volume").get_parameter_value().double_value
        piper_model_path = self.get_parameter(
            "piper_model_path"
        ).get_parameter_value().string_value.strip() or _default_piper_model_path()
        piper_config_path = self.get_parameter(
            "piper_config_path"
        ).get_parameter_value().string_value.strip()
        piper_length_scale = self.get_parameter(
            "piper_length_scale"
        ).get_parameter_value().double_value
        piper_use_cuda = self.get_parameter(
            "piper_use_cuda"
        ).get_parameter_value().bool_value
        output_sample_rate = self.get_parameter(
            "output_sample_rate"
        ).get_parameter_value().integer_value
        sentence_pause_ms = self.get_parameter(
            "sentence_pause_ms"
        ).get_parameter_value().integer_value
        max_chunk_characters = self.get_parameter(
            "max_chunk_characters"
        ).get_parameter_value().integer_value
        self._speech_timeout_s = self.get_parameter(
            "speech_timeout_s"
        ).get_parameter_value().double_value
        self._cancel_grace_s = self.get_parameter(
            "cancel_grace_s"
        ).get_parameter_value().double_value
        tts_active_topic = self.get_parameter(
            "tts_active_topic"
        ).get_parameter_value().string_value
        if not 0.0 <= self._tts_volume <= 1.0:
            raise ValueError("Parameter 'tts_volume' must be between 0.0 and 1.0.")
        if piper_length_scale <= 0.0:
            raise ValueError("Parameter 'piper_length_scale' must be greater than zero.")
        if output_sample_rate <= 0:
            raise ValueError("Parameter 'output_sample_rate' must be greater than zero.")
        if sentence_pause_ms < 0:
            raise ValueError("Parameter 'sentence_pause_ms' cannot be negative.")
        if max_chunk_characters < 40:
            raise ValueError("Parameter 'max_chunk_characters' must be at least 40.")
        if self._speech_timeout_s <= 0.0 or self._cancel_grace_s <= 0.0:
            raise ValueError("TTS timeout parameters must be greater than zero.")

        self.get_logger().info(f"Loading Piper voice from {piper_model_path} …")
        self._engine = _PiperEngine(
            self._stop_event,
            piper_model_path,
            piper_config_path,
            volume=self._tts_volume,
            length_scale=piper_length_scale,
            output_sample_rate=output_sample_rate,
            sentence_pause_ms=sentence_pause_ms,
            max_chunk_characters=max_chunk_characters,
            use_cuda=piper_use_cuda,
        )

        voice_state_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._tts_active_pub = self.create_publisher(Bool, tts_active_topic, voice_state_qos)

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            "/tts_action",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self._publish_tts_active(False)
        self.get_logger().info(
            f"TtsNode ready — Piper; /tts_action active; VAD state on {tts_active_topic}."
        )

    def _publish_tts_active(self, active: bool) -> None:
        self._tts_active_pub.publish(Bool(data=active))

    def _goal_callback(self, goal_request) -> GoalResponse:
        text = goal_request.text.strip()
        if not text:
            self.get_logger().warn("Rejected empty TTS goal.")
            return GoalResponse.REJECT

        with self._state_lock:
            if self._goal_reserved or self._shutting_down:
                self.get_logger().warn(
                    "Rejected TTS goal because another speech request is active."
                )
                return GoalResponse.REJECT
            self._goal_reserved = True

        self.get_logger().debug(f"TTS goal accepted: '{text[:60]}'")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("TTS cancel request — stopping engine.")
        self._stop_event.set()
        with self._state_lock:
            engine = self._active_engine
        if engine is not None:
            engine.request_stop()
        return CancelResponse.ACCEPT

    def _release_goal(self, token: object) -> None:
        """Release the single-speaker slot only if it belongs to this goal."""
        with self._state_lock:
            if self._active_token is not token:
                return
            self._active_engine = None
            self._active_token = None
            self._goal_reserved = False
            # Publish before another goal can claim the speaker slot.
            self._publish_tts_active(False)

    def _execute_callback(self, goal_handle):
        self._execution_done.clear()
        try:
            if self._shutting_down:
                goal_handle.abort()
                return TextToSpeech.Result(success=False)
            return self._execute_speech(goal_handle)
        finally:
            self._execution_done.set()

    def _execute_speech(self, goal_handle):
        text = goal_handle.request.text.strip()
        token = object()
        self.get_logger().info(f"[TTS] '{text}'")

        self._stop_event.clear()
        with self._state_lock:
            self._active_token = token
        self._publish_tts_active(True)

        if goal_handle.is_cancel_requested:
            self._stop_event.set()
            self._release_goal(token)
            goal_handle.canceled()
            result = TextToSpeech.Result()
            result.success = False
            return result

        feedback_msg = TextToSpeech.Feedback()

        def _on_word(word: str) -> None:
            feedback_msg.current_word = word
            goal_handle.publish_feedback(feedback_msg)

        success_container: list[bool] = [False]
        speech_done = threading.Event()
        timed_out = threading.Event()
        engine = self._engine

        def _speech_thread() -> None:
            try:
                success_container[0] = engine.speak(text, on_word_callback=_on_word)
            finally:
                speech_done.set()
                # If the action returned after an unresponsive engine, this
                # worker owns the eventual cleanup and VAD re-enable.
                if timed_out.is_set():
                    self._release_goal(token)

        with self._state_lock:
            self._active_engine = engine
        speech_thread = threading.Thread(target=_speech_thread, daemon=True)
        self._speech_worker = speech_thread
        speech_thread.start()

        deadline = time.monotonic() + self._speech_timeout_s
        while not speech_done.wait(timeout=0.1):
            if goal_handle.is_cancel_requested:
                engine.request_stop()
            if time.monotonic() >= deadline:
                timed_out.set()
                engine.request_stop()
                if not speech_done.wait(timeout=self._cancel_grace_s):
                    self.get_logger().error(
                        "TTS engine did not stop before the timeout grace period; "
                        "rejecting future TTS goals until its worker exits."
                    )
                break

        if timed_out.is_set() and not speech_done.is_set():
            goal_handle.abort()
            result = TextToSpeech.Result()
            result.success = False
            return result

        self._release_goal(token)

        result = TextToSpeech.Result()
        if goal_handle.is_cancel_requested:
            self.get_logger().info("TTS goal cancelled.")
            goal_handle.canceled()
            result.success = False
        elif timed_out.is_set():
            self.get_logger().error("TTS goal timed out.")
            goal_handle.abort()
            result.success = False
        elif not success_container[0]:
            self.get_logger().error(f"TTS engine failed: {engine.last_error or 'unknown error'}")
            goal_handle.abort()
            result.success = False
        else:
            self.get_logger().debug("TTS goal completed.")
            goal_handle.succeed()
            result.success = True
        return result

    def destroy_node(self) -> None:
        self._shutting_down = True
        self._stop_event.set()
        with self._state_lock:
            engine = self._active_engine
        if engine is not None:
            engine.request_stop()
        self._execution_done.wait()
        if self._speech_worker is not None:
            # A timed-out action may still own a synthesis worker. Do not destroy
            # publishers/action handles while that worker can use them.
            self._speech_worker.join()
        if hasattr(self, "_action_server"):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = None
    try:
        node = TtsNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None:
            node.get_logger().info("TtsNode shutting down …")
    finally:
        if node is not None:
            node._shutting_down = True
            node._engine.request_stop()
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
