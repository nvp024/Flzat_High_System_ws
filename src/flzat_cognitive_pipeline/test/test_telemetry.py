from flzat_cognitive_pipeline.telemetry import PipelineTelemetry


class Clock:
    def __init__(self):
        self.value = 1_000_000_000

    def __call__(self):
        return self.value

    def advance_ms(self, value):
        self.value += int(value * 1_000_000)


def test_request_metrics_have_phase_total_and_correlation():
    clock = Clock()
    metrics = PipelineTelemetry("run-20261014T000000Z", clock)
    metrics.begin("task-0001-r01", "task-0001", "speech_audio")
    clock.advance_ms(12.5)
    metrics.mark("task-0001-r01", "rag")
    clock.advance_ms(20)

    result = metrics.finish("task-0001-r01", "dispatched")

    assert result == {
        "schema_version": "flzat.high_metrics.v2",
        "run_id": "run-20261014T000000Z",
        "request_id": "task-0001-r01",
        "task_id": "task-0001",
        "source": "speech_audio",
        "outcome": "dispatched",
        "total_ms": 32.5,
        "phase_ms": {"rag": 12.5},
    }
    assert metrics.snapshot()["completed"] == 1


def test_terminal_counters_and_late_marks_are_safe():
    clock = Clock()
    metrics = PipelineTelemetry("run-20261014T000000Z", clock)
    for number, outcome in enumerate(("rejected", "cancelled", "timed_out"), 1):
        task_id = f"task-{number:04d}"
        request_id = f"{task_id}-r01"
        metrics.begin(request_id, task_id, "text")
        metrics.finish(request_id, outcome)
        metrics.mark(request_id, "late")

    assert metrics.snapshot() == {
        "active_traces": 0,
        "completed": 3,
        "rejected": 1,
        "cancelled": 1,
        "timed_out": 1,
    }
