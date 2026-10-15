# ROS 2 Interfaces

## Topics

| Topic | Type | Vai trò |
|---|---|---|
| `/audio_events` | `SpeechAudio` | VAD → Manager |
| `/flzat/high_system/text_command` | `std_msgs/String` | bypass VAD/STT khi test; mặc định tắt |
| `/flzat/high_system/command` | `std_msgs/String` | canonical `flzat.high_command.v2` |
| `/flzat/high_system/status` | `std_msgs/String` | trạng thái task |
| `/flzat/high_system/health` | `std_msgs/String` | readiness/dependencies |
| `/flzat/high_system/metrics` | `std_msgs/String` | latency/outcome |
| `/flzat/high_system/cancel` | `std_msgs/String` | yêu cầu hủy |
| `/flzat/low_system/status` | `std_msgs/String` | accepted/running/succeeded/failed/cancelled |
| `/flzat/low_system/cancel` | `std_msgs/String` | cancel đã correlate |

## Actions và services

| Name | Type |
|---|---|
| `/stt_action` | `SpeechToText` action |
| `/tts_action` | `TextToSpeech` action |
| `/vlm/run` | `RunVlm` action |
| `/environment_memory/query` | `QueryMemory` service |
| `/vision/get_frame_near` | `GetFrameNear` service |

`RunVlm` truyền trực tiếp đúng một `CompressedImage`, transcript, request ID
nội bộ, system/user prompt, prompt version và prompt hash. Không còn wrapper
`MultimodalEvent` hoặc mảng frame của PoC cũ. Backend reject metadata thiếu hoặc
hash sai. Kết quả lặp lại request ID để Manager bỏ stale response.

## RAG

Manager không mở trực tiếp SQLite. `MemoryQueryNode` kiểm tra manifest,
environment/map và mở Chroma read-only. `sentence_transformer` dùng production;
`deterministic_test` chỉ kiểm tra contract.

## Output

ModelDecision có đúng các field: `intent`, `target`,
`selected_context_refs`, `execution_strategy`, `missing_information`,
`clarification_question`, `speech`. Extra field, unknown ref, stale snapshot,
intent ngoài allow-list hoặc direct command không có trusted position đều bị
reject.

## Correlation IDs

Mỗi lần khởi động Manager tạo một `run_id` UTC dễ đọc, ví dụ
`run-20261014T153000Z`. Các ID bên trong run dùng số thứ tự:
`task-0001`, `task-0001-r01` và `task-0001-c01`. Clarification giữ nguyên
`task_id` nhưng tăng request thành `r02`, `r03`, ... . High command/status,
metrics và Low status/cancel đều mang `run_id` để không nhầm dữ liệu giữa hai
lần khởi động.
