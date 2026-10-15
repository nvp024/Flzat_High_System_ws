# Test Report

Ngày consolidation: **2026-10-14**.

## Đã xác nhận local

- VAD/STT/TTS contracts và STT tiếng Anh.
- Chroma manifest/read-only retrieval và ROS QueryMemory.
- Context bounding, prompt modular/hash, clarification state.
- Qwen3-VL action contract, cancellation và shared inference broker.
- Strict output, trusted coordinate resolution, JSON/JSONL artifact.
- High ↔ Low command/status/cancel contract.
- Camera/frame buffer và static-image service adapter.
- Full mock/replay không cần model, microphone hoặc camera.

Kết quả sau consolidation trong Conda `py312`:

```text
Summary: 7 packages finished
Summary: 73 tests, 0 errors, 0 failures, 0 skipped
```

Canonical `high_system.launch.py` cũng đã chạy với `demo_1` (59 objects),
static frame, fake VLM và Low mock. Text command đi qua RAG/prompt/VLM/output,
publish command rồi nhận `accepted → running → succeeded`. Replay sau cleanup
trả về `passed: true`, `health_ready: true`, `metrics_seen: true`; một SIGINT
vào launch process cũng kết thúc sạch cả 7 node.

Mã correlation đã được rút gọn và phân cấp theo run:
`run-YYYYMMDDTHHMMSSZ`, `task-0001`, `task-0001-r01`,
`task-0001-c01`. Artifact được lưu theo
`<output_root>/<run_id>/<task_id>/<command_id>.json`; Low mock correlate bằng
cặp `(run_id, command_id)` để ID tuần tự không xung đột giữa các run.

Grounding/bounding-box thử nghiệm, Qwen2-VL, SmolVLM2 và prompt context không có
producer đã được xóa. Build/install/log cũ được xóa trước khi build lại để
không còn interface hoặc Python module stale trong overlay.

Lệnh regression:

```bash
source /opt/ros/jazzy/setup.bash
cd flzat_high_system_ws
colcon build --symlink-install
source install/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
colcon test --python-testing pytest
colcon test-result --verbose
```

## Chưa thể kết luận bằng local test

- chất lượng và latency Qwen3-VL trên GPU mục tiêu;
- semantic ranking bằng production sentence-transformer trên dataset thật;
- microphone/speaker thật;
- Low System/OpenArm end-to-end thật.

Lưu ý môi trường Conda `py312` tại thời điểm audit đang có package đã trôi khỏi
file chuẩn (`transformers 5.14.1`, `huggingface-hub 2.1.1`,
`sentence-transformers 6.0.0`), nên `pip check` báo không tương thích. Source đã
pin bộ dependency thống nhất trong `requirements.txt`; cần chạy lại
`pip install -r requirements.txt` trước lần test Qwen/RAG production. Mock,
build và contract test ở trên không phụ thuộc việc tải model.

Máy GPU/cloud chạy cùng source, launch và `requirements.txt`; không dùng một
pipeline cloud riêng.
