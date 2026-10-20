# Architecture

## Một runtime flow

```text
/audio_events → SpeechIngress → STT → /flzat/high_system/stt_result ─┐
text hook ────────────────────────────────────────────────────────────┤
                                                                     ▼
                                                        Safety Guard/Intent Router
                                                                     │
                                                            RoutedCommand
                                                                     │
                                                                     ▼
                                                            CognitiveManager
                                                                     │
                    ├─ uncertain/pending turn → text-only Qwen routing
                    │    ├─ resolved → mode + continue/new request
                    │    └─ missing information → relevant question → wait
                    ├─ ContextPolicy → canonical mode requirements
                    ├─ AdaptiveContextBuilder → trusted bounded providers
                    ├─ QueryMemory → read-only Chroma (conditional)
                    ├─ GetFrameNear → camera/static adapter (conditional)
                    ├─ profile PromptComposer → RunVlm → mode validator/repair
                    ├─ clarification/response → TTS
                    └─ trusted command → JSON/JSONL + ROS topic → Low System
```

Router dùng sentence-transformer, cache embedding của route examples khi khởi
động và chỉ encode user utterance cho mỗi request. Safety Guard chạy trước
semantic routing và phát high-level cancel mà không gọi RAG hoặc VLM.

Từ 20/10/2026, route `clarification` của embedding chỉ yêu cầu phân loại bổ sung.
Manager gọi Qwen bằng `prompts/routing/` trước RAG/frame; mọi lượt có task đang
chờ clarification cũng được kiểm tra `continue` hay `new_request`.
Không còn ép `clarification -> task_command`. Qwen trả mode đã xác định hoặc
null nếu chưa đủ thông tin, cùng câu hỏi làm rõ nếu cần.
Continuation giữ câu gốc và lịch sử question/answer trong prompt chuyên biệt.
New request kết thúc task đang chờ và tạo task mới với lịch sử riêng.
Routing output sai schema bị reject; không được chuyển sang task mặc định.

Manager không tin trực tiếp các context flag trong `RoutedCommand`.
`ContextPolicy` đọc cùng route rules từ YAML rồi canonicalize yêu cầu theo mode.
Do đó message giả mạo không thể làm task bỏ qua RAG/frame bắt buộc hoặc ép QA
đọc object memory.

Router và ContextPolicy nhận cùng một `intent_routes_path` từ launch, nên không
có hai bản cấu hình mode độc lập bị lệch nhau.

Adaptive context hiện đã hoạt động:

- `task_command`: environment RAG + frame + skill registry + safety policy;
- `visual_qa`: frame, không environment RAG;
- `general_qa`: text-only;
- `social_dialogue`: bounded conversation + read-only user profile;
- `clarification`: trạng thái chờ thông tin; nhánh routing chỉ nhận câu gốc,
  câu mới, mode đã biết, lịch sử làm rõ và câu hỏi đang chờ.

Conversation và user profile không được trộn vào Chroma object database. Skill
registry và safety policy là trusted YAML context, không phải semantic RAG.

Không còn `foundation_manager` hoặc `multimodal_manager`. VLM node không tự
chọn prompt hội thoại; nó chỉ nhận prompt đã được `PromptComposer` tạo và kiểm
tra SHA-256 trước inference.

PromptComposer chỉ ghép context được policy cho phép. Ảnh camera được gửi trực
tiếp qua `RunVlm`, không bị chuyển thành text context. Với mode text-only,
`CompressedImage.data` rỗng là hợp lệ và Qwen backend không chạy vision
preprocessing.

PromptComposer discover năm manifest và kiểm tra chuỗi contract
`router mode ↔ prompt profile ↔ output schema ↔ validator` trước inference:

- `task_command` → `ModelDecision.v1`;
- `visual_qa` → `VisualQA.v1`;
- `general_qa` → `GeneralQA.v1`;
- `social_dialogue` → `SocialDialogue.v1`;
- `clarification` → `Clarification.v1`.

Profile clarification cũ được giữ tương thích với validator, nhưng route
clarification từ embedding hiện đi qua `SemanticRoute.v1` trước. Prompt routing
riêng gồm ba file system/task/output, không tham gia danh sách profile inference
chính. General/visual QA cho phép lịch sử clarification của chính task.

Routing sử dụng cùng `/vlm/run`, với request ID có hậu tố `-routing`, ảnh rỗng,
và hash riêng. Main inference giữ request ID gốc. Cả hai dùng chung deadline và
cancellation; callback cũ không được tiếp tục task đã hủy. Fake VLM có fixture
routing riêng để không tiêu thụ response dành cho main inference/repair.

Nếu output sai schema, Manager dùng lại chính Qwen, frozen frame và context
snapshot để repair tối đa hai lần. Repair không query RAG/camera lại và vẫn
nằm trong deadline/cancellation của request. Chỉ output task hợp lệ mới được
đưa tới TrustedCommandAssembler.

## Trust boundary

- Qwen chọn intent, strategy và `context_ref`.
- Tọa độ không được lấy từ model output.
- `ContextRagManager` đóng băng kết quả RAG theo request nội bộ.
- `TrustedCommandAssembler` resolve tọa độ từ snapshot đáng tin cậy.
- Output chấp nhận JSON thuần hoặc đúng một outer fence ` ```json ... ``` `;
  nội dung bên trong vẫn phải đúng strict schema. Repair chỉ sửa format/schema,
  không được tạo thêm fact, target, coordinate, skill hoặc context reference.
- Artifact được tách theo `output_root/run_id/task_id/command_id.json`; audit
  JSONL nằm trong chính thư mục của run để ID tuần tự không đụng lần chạy cũ.
- Artifact request nằm ở `output_root/run_id/task_id/request_id/`, gồm router,
  context policy, RAG context, prompt metadata, từng VLM/validator attempt và
  validated output hoặc failure record.

## Test adapters

`static_frame_server` implement đúng `/vision/get_frame_near`, vì vậy ảnh file
đi qua cùng đường với camera thật. Fake STT/VLM/TTS/Low dùng cùng ROS contract
production. Chúng là adapter test, không phải pipeline thứ hai.

Qwen3-VL là backend production duy nhất phía sau `VlmNode`.

## Phạm vi

High System sở hữu voice, context/RAG, prompt, reasoning, clarification và
High-Level Command. SLAM, Nav2, depth/TF localization, manipulation và actuator
thuộc Low System.
