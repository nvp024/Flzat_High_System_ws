# ROS 2 Interfaces

## Topics

| Topic | Type | Vai trò |
|---|---|---|
| `/audio_events` | `SpeechAudio` | VAD → SpeechIngress |
| `/flzat/high_system/stt_result` | `std_msgs/String` | SpeechIngress → IntentRouter |
| `/flzat/high_system/text_command` | `std_msgs/String` | test hook → IntentRouter; mặc định tắt |
| `/flzat/high_system/routed_command` | `RoutedCommand` | IntentRouter → CognitiveManager |
| `/flzat/high_system/command` | `std_msgs/String` | canonical `flzat.high_command.v2` |
| `/flzat/high_system/status` | `std_msgs/String` | trạng thái task |
| `/flzat/high_system/health` | `std_msgs/String` | readiness/dependencies |
| `/flzat/high_system/metrics` | `std_msgs/String` | latency/outcome |
| `/flzat/high_system/cancel` | `std_msgs/String` | Safety Guard/user → CognitiveManager |
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

`RunVlm` truyền một `CompressedImage`, transcript, request ID nội bộ,
system/user prompt, prompt version và prompt hash. `image.data` được phép rỗng
cho mode text-only; mode cần vision vẫn bắt buộc frame hợp lệ. Không còn wrapper
`MultimodalEvent` hoặc mảng frame của PoC cũ. Backend reject metadata thiếu hoặc
hash sai. Kết quả lặp lại request ID để Manager bỏ stale response.

`RoutedCommand` chứa router version, utterance, mode, confidence, language,
urgency, context flags, route names/scores và router latency. Manager kiểm tra
độ dài score arrays và giá trị hữu hạn trước khi nhận request.
Context flags trong message chỉ là đề xuất. Manager đọc trusted route rules từ
`intent_routes.yaml`, canonicalize lại bằng `ContextPolicy`, và reject yêu cầu
external knowledge vì provider đó chưa tồn tại.

Safety Guard không tự tạo payload Low cancel vì không sở hữu correlation state.
Nó phát high-level cancel; CognitiveManager chuyển thành
`flzat.low_cancel.v2` khi có command đang chạy. Khi hệ thống đang idle thì
không tạo payload Low giả.

## RAG

Manager không mở trực tiếp SQLite. `MemoryQueryNode` kiểm tra manifest,
environment/map và mở Chroma read-only. `sentence_transformer` dùng production;
`deterministic_test` chỉ kiểm tra contract.

Environment RAG chỉ được gọi cho `task_command`. `visual_qa`, `general_qa`,
`social_dialogue` và `clarification` không query Chroma trong Giai đoạn 2.
Health payload `flzat.high_health.v2` có thêm `mode_readiness` và `active_mode`;
camera/memory chỉ ảnh hưởng readiness của mode cần dependency tương ứng.

## Output theo mode

| Mode | Schema | Kết quả |
|---|---|---|
| `task_command` | `ModelDecision.v1` | trusted High Command hoặc clarification |
| `visual_qa` | `VisualQA.v1` | `reply`, `observed_objects`, `confidence` |
| `general_qa` | `GeneralQA.v1` | `reply`, `speak=true`, `requires_robot_action=false` |
| `social_dialogue` | `SocialDialogue.v1` | `reply`, `speak=true`, trusted/null `suggested_skill` |
| `clarification` | `Clarification.v1` | một câu hỏi ngắn trong `reply` |

`ModelDecision.v1` có đúng các field: `intent`, `target`,
`selected_context_refs`, `execution_strategy`, `missing_information`,
`clarification_question`, `speech`. Extra field, unknown ref, stale snapshot,
intent ngoài allow-list hoặc direct command không có trusted position đều bị
reject. Validator reject schema của mode khác; output QA/social/clarification
không có đường publish `/flzat/high_system/command`.

Output sai được repair tối đa hai lần bằng cùng Qwen/context/frame. Mỗi attempt
có raw response và validator result riêng trong request artifact. Khi hết giới
hạn hoặc hết deadline, request bị reject/timed out và không publish command.

## Correlation IDs

Mỗi lần khởi động Manager tạo một `run_id` UTC dễ đọc, ví dụ
`run-20261014T153000Z`. Các ID bên trong run dùng số thứ tự:
`task-0001`, `task-0001-r01` và `task-0001-c01`. Clarification giữ nguyên
`task_id` nhưng tăng request thành `r02`, `r03`, ... . High command/status,
metrics và Low status/cancel đều mang `run_id` để không nhầm dữ liệu giữa hai
lần khởi động.
