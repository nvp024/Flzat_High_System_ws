# Test Report

Ngày cập nhật: **2026-10-20**.

## Conditional Qwen routing — 20/10/2026

- Build riêng `flzat_cognitive_pipeline` và `flzat_high_bringup` thành công trong
  build/install tạm; dùng các ROS interfaces/dependencies đã build sẵn.
- **145 tests passed**, bao gồm test ROS topic/action thật với production Manager
  và FakeVlmServer. Không load Qwen, không cần GPU, RAG server, camera hoặc TTS.
- Đã kiểm tra: route chắc chắn chỉ gọi inference chính; route chưa chắc gọi
  classifier rồi main; "America" tiếp tục câu hỏi tổng thống với đầy đủ history;
  câu hỏi thủ đô mới tạo task mới; trả lời "the book" chỉ bật task RAG sau khi
  classifier xác định task_command.
- General-QA validator từ chối output find_object/ball; routing JSON sai, ID
  cũ, timeout, cancel và goal acceptance đến trễ không được dispatch command.
- Fake routing có response sequence độc lập với main/repair. Giới hạn số lượt
  clarification vẫn được kiểm tra, nhưng không chặn chuyển sang yêu cầu mới.
- Prompt/accuracy/latency của classifier Qwen thật vẫn cần test trên cloud.
  Test fake xác nhận luồng, không chứng minh model phân loại chính xác mọi câu.

Chạy lại từ workspace sau khi source ROS và `install/setup.bash`:

```bash
export PYTHONPATH="$PWD/src/flzat_cognitive_pipeline:$PWD/src/flzat_memory:$PWD/src/flzat_high_bringup:$PWD/src/flzat_vlm:$PWD/src/flzat_speech:$PWD/src/flzat_vision_context:$PYTHONPATH"
export ROS_DOMAIN_ID=183
export ROS_LOG_DIR=/tmp/flzat-routing-test-logs
export FLZAT_RUN_ROS_TESTS=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/flzat_cognitive_pipeline/test src/flzat_memory/test \
  src/flzat_high_bringup/test src/flzat_vlm/test \
  src/flzat_speech/test src/flzat_vision_context/test -q
```

Dùng ROS domain riêng chưa có node khác. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
tránh xung đột plugin launch_testing cũ với pytest trong môi trường local;
test tích hợp ROS ở trên tự tạo executor và không cần plugin đó.
Các kết quả bên dưới là lịch sử kiểm thử trước thay đổi routing.

## Đã xác nhận local

- VAD/STT/TTS contracts và STT tiếng Anh.
- Chroma manifest/read-only retrieval và ROS QueryMemory.
- Context bounding, prompt modular/hash, clarification state.
- Qwen3-VL action contract, cancellation và shared inference broker.
- Strict output, trusted coordinate resolution, JSON/JSONL artifact.
- High ↔ Low command/status/cancel contract.
- Camera/frame buffer và static-image service adapter.
- Full mock/replay không cần model, microphone hoặc camera.
- Phase 1 Intent Router: deterministic Safety Guard, cached route embeddings,
  `RoutedCommand`, STT ingress và Manager chỉ nhận routed input.
- Phase 2 Adaptive Context: authoritative YAML `ContextPolicy`, conditional
  RAG/frame lookup, trusted providers, bounded conversation context và
  text-only Qwen request.
- Phase 3: năm manifest-driven prompt profile, validator riêng theo mode,
  bounded output repair, request artifacts và telemetry theo từng attempt.

Kết quả sau consolidation trong Conda `py312`:

```text
Summary: 7 packages finished
121 tests passed
```

Canonical `high_system.launch.py` cũng đã chạy với `demo_1` (59 objects),
static frame, fake VLM và Low mock. Text command đi qua RAG/prompt/VLM/output,
publish command rồi nhận `accepted → running → succeeded`. Replay sau cleanup
trả về `passed: true`, `health_ready: true`, `metrics_seen: true`; một SIGINT
vào launch process cũng kết thúc sạch các node.

Acceptance sau khi thêm router trả về `actual_state: SUCCEEDED` và
`command_seen: true`. Câu `emergency stop` được router phân loại
`safety_stop` với `urgency: emergency` mà không tạo task/VLM request mới.
Router phát high-level cancel; CognitiveManager là adapter duy nhất được phép
tạo `flzat.low_cancel.v2`, nhờ đó không phá contract correlation hiện tại của
Low System.

Acceptance Giai đoạn 2 trên ROS 2 DDS thật với `demo_1`:

| Mode | Kết quả | RAG | Frame | High Command |
|---|---|---:|---:|---:|
| `task_command` | `SUCCEEDED` | Có | Có | Có |
| `general_qa` | `RESPONSE_ONLY` | Không | Không | Không |
| `visual_qa` | `RESPONSE_ONLY` | Không | Có | Không |
| `social_dialogue` | `RESPONSE_ONLY` | Không | Không | Không |
| `clarification` | `WAITING_CLARIFICATION` | Không | Không | Không |

Metrics xác nhận `general_qa`/`social_dialogue`/`clarification` có
`frame_skipped`, `visual_qa` có `frame_lookup`, và chỉ `task_command` có
`rag_context`.

Acceptance Giai đoạn 3 xác nhận cả năm mode dùng đúng validator: task publish
High Command; bốn mode còn lại không publish command. Một output đầu tiên sai
JSON đã được repair thành công ở attempt 2 bằng cùng request/context/frame.
Trường hợp luôn sai thực hiện đúng 1 inference gốc + 2 repair rồi `REJECTED`,
không tạo High Command. Request artifact chứa đầy đủ hai/ba raw output,
validator result, prompt hash và validated/failure record.

Acceptance deadline dùng inference delay `0.8 s` và request timeout `1.2 s`
đã chuyển request sang `TIMED_OUT` trong repair attempt 1, hủy VLM active và
không publish High Command. Như vậy deadline bao phủ toàn chuỗi repair.

Các acceptance trên dùng fake VLM để kiểm tra orchestration và contract; chất
lượng/latency output thực tế vẫn cần Qwen3-VL thật trên GPU/cloud.

Safety acceptance với fake VLM delay 10 giây cũng pass: `stop` được router xử
lý trong khoảng `0.048 ms`, request đang `VLM_ACTIVE` chuyển sang `CANCELLED`,
không có High Command nào được publish.

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
cd flzat_high_system_ws
conda activate py312
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q \
  src/flzat_cognitive_pipeline/test \
  src/flzat_vlm/test \
  src/flzat_high_bringup/test \
  src/flzat_speech/test \
  src/flzat_memory/test \
  src/flzat_vision_context/test
```

## Chưa thể kết luận bằng local test

- chất lượng và latency Qwen3-VL trên GPU mục tiêu;
- semantic ranking bằng production sentence-transformer trên dataset thật;
- latency `< 50 ms` của production sentence-transformer trên CPU mục tiêu;
- microphone/speaker thật;
- Low System/OpenArm end-to-end thật.

Lưu ý môi trường Conda `py312` tại thời điểm audit đang có package đã trôi khỏi
file chuẩn (`transformers 5.14.1`, `huggingface-hub 2.1.1`,
`sentence-transformers 6.0.0`), nên `pip check` báo không tương thích. Source đã
pin bộ dependency thống nhất trong `requirements.txt`; cần chạy lại
`pip install -r requirements.txt` trước lần test Qwen/RAG production. Mock,
build và contract test ở trên không phụ thuộc việc tải model.

Ngoài ra, `pytest 9.1.1` trong Conda không tương thích với plugin
`launch_testing` của ROS 2 Jazzy khi gọi `colcon test`. Báo cáo `121 passed` ở
trên dùng pytest trực tiếp và tắt plugin auto-load; đây là vấn đề test
environment, không phải lỗi source hoặc `colcon build`.

Máy GPU/cloud chạy cùng source, launch và `requirements.txt`; không dùng một
pipeline cloud riêng.
