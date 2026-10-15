from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BufferedFrame:
    """One JPEG frame and its ROS capture timestamp in nanoseconds."""

    stamp_ns: int
    jpeg_data: bytes
    frame_id: str


class FrameRingBuffer:
    """Thread-safe frame buffer bounded by both time and frame count."""

    def __init__(self, duration_s: float, max_frames: int) -> None:
        if duration_s <= 0.0:
            raise ValueError("duration_s must be greater than zero")
        if max_frames < 1:
            raise ValueError("max_frames must be at least one")
        self._duration_ns = int(duration_s * 1_000_000_000)
        self._frames: collections.deque[BufferedFrame] = collections.deque(
            maxlen=max_frames
        )
        self._lock = threading.Lock()

    def append(self, frame: BufferedFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            oldest_allowed_ns = frame.stamp_ns - self._duration_ns
            while self._frames and self._frames[0].stamp_ns < oldest_allowed_ns:
                self._frames.popleft()

    def nearest(self, stamp_ns: int, max_age_s: float) -> Optional[BufferedFrame]:
        if max_age_s < 0.0:
            return None
        max_age_ns = int(max_age_s * 1_000_000_000)
        with self._lock:
            if not self._frames:
                return None
            nearest_frame = min(
                self._frames,
                key=lambda frame: abs(frame.stamp_ns - stamp_ns),
            )
            if abs(nearest_frame.stamp_ns - stamp_ns) > max_age_ns:
                return None
            return nearest_frame

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
