"""Pure request timing and aggregate counters for the cognitive pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable


@dataclass
class RequestTrace:
    request_id: str
    task_id: str
    source: str
    started_ns: int
    last_mark_ns: int
    phase_ms: dict[str, float] = field(default_factory=dict)


class PipelineTelemetry:
    """Track one timing trace per request without depending on ROS."""

    def __init__(
        self,
        run_id: str,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id is required")
        self._run_id = run_id.strip()
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._traces: dict[str, RequestTrace] = {}
        self._completed = 0
        self._rejected = 0
        self._cancelled = 0
        self._timed_out = 0

    def begin(self, request_id: str, task_id: str, source: str) -> None:
        with self._lock:
            if not request_id.strip() or not task_id.strip() or not source.strip():
                raise ValueError("request_id, task_id and source are required")
            if request_id in self._traces:
                raise ValueError("request trace already exists")
            now = self._clock_ns()
            self._traces[request_id] = RequestTrace(
                request_id=request_id,
                task_id=task_id,
                source=source,
                started_ns=now,
                last_mark_ns=now,
            )

    def mark(self, request_id: str, phase: str) -> None:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace is None:
                return
            value = phase.strip()
            if not value:
                raise ValueError("phase cannot be empty")
            now = self._clock_ns()
            elapsed_ms = (now - trace.last_mark_ns) / 1_000_000.0
            trace.phase_ms[value] = round(max(0.0, elapsed_ms), 3)
            trace.last_mark_ns = now

    def finish(self, request_id: str, outcome: str) -> dict[str, object] | None:
        with self._lock:
            trace = self._traces.pop(request_id, None)
            if trace is None:
                return None
            value = outcome.strip().lower()
            if not value:
                raise ValueError("outcome cannot be empty")
            now = self._clock_ns()
            total_ms = (now - trace.started_ns) / 1_000_000.0
            if not math.isfinite(total_ms) or total_ms < 0.0:
                raise ValueError("telemetry clock produced invalid elapsed time")
            self._completed += 1
            if value == "rejected":
                self._rejected += 1
            elif value == "cancelled":
                self._cancelled += 1
            elif value == "timed_out":
                self._timed_out += 1
            return {
                "schema_version": "flzat.high_metrics.v2",
                "run_id": self._run_id,
                "request_id": trace.request_id,
                "task_id": trace.task_id,
                "source": trace.source,
                "outcome": value,
                "total_ms": round(total_ms, 3),
                "phase_ms": dict(trace.phase_ms),
            }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_traces": len(self._traces),
                "completed": self._completed,
                "rejected": self._rejected,
                "cancelled": self._cancelled,
                "timed_out": self._timed_out,
            }
