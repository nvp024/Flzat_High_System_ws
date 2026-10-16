# FLZAT Intent Router and Adaptive Context Plan

> Trạng thái 2026-10-15: Giai đoạn 1, 2 và phần source/local acceptance của
> Giai đoạn 3 đã triển khai. Full build, 121 unit/contract tests, ROS acceptance
> cho năm semantic mode và bounded repair đã pass. Cloud verification bằng
> Qwen3-VL thật cùng benchmark sentence-transformer production vẫn còn chờ.

## 1. Mục tiêu

Mở rộng FLZAT High System từ pipeline chủ yếu xử lý task tìm kiếm, định vị và
điều khiển robot thành pipeline có thể phân loại yêu cầu người dùng trước khi
gọi Qwen3-VL.

Flow mục tiêu:

```text
Voice → VAD → English STT ─┐
                           ├→ Safety Guard
Text command ──────────────┘
                              ↓
                         Intent Router
                              ↓
                    RoutedCommand + flags
                              ↓
                      CognitiveManager
                              ↓
                       Context Policy
                              ↓
              Mode-specific Context Builder
                              ↓
                 Mode-specific Prompt Profile
                              ↓
                         Qwen3-VL
                              ↓
                  Mode-specific Validator
                     ┌────────┴────────┐
                     ↓                 ↓
              High Command         TTS Reply
```

Chỉ `task_command` được phép tạo `/flzat/high_system/command` và gửi xuống Low
System. `safety_stop` không gọi RAG, camera hoặc Qwen3-VL.

## 2. Các quyết định đã chốt

- Qwen3-VL tiếp tục là production VLM duy nhất.
- Không thêm LLM riêng cho Intent Router.
- Router sử dụng sentence-transformer và cosine similarity.
- Route examples, threshold và context flags được cấu hình bằng YAML.
- Embedding của route examples được cache khi node khởi động.
- Pipeline `task_command` hiện tại được giữ làm nền tảng.
- Qwen không được tạo coordinates hoặc trusted High Command trực tiếp.
- `flzat.high_command.v2` và contract hiện tại với Low System không thay đổi.
- Mock mode hiện tại được giữ nguyên để regression test nhánh `task_command`.
- Các mode mới được kiểm tra bằng unit test, fixture và Qwen3-VL thật trên cloud.
- Không có silent fallback cho `environment_id` hoặc `map_id`.
- Skill do model đề xuất phải tồn tại trong trusted skill registry.

## 3. Router modes

| Mode | Mục đích | Camera | Task memory | Conversation | User profile | Kết quả |
|---|---|---:|---:|---:|---:|---|
| `safety_stop` | Dừng hoặc hủy | Không | Không | Không | Không | Cancel |
| `task_command` | Navigation, fetch, manipulation | Có | Có | Khi cần | Không | High Command |
| `visual_qa` | Hỏi về ảnh hiện tại | Có | Không mặc định | Không | Không | TTS reply |
| `general_qa` | Kiến thức chung | Không | Không | Không mặc định | Không | TTS reply |
| `social_dialogue` | Hội thoại xã hội | Không | Không | Có | Có | TTS reply |
| `clarification` | Yêu cầu mơ hồ | Không mặc định | Không | Có | Không | TTS question |

`needs_external_knowledge` được giữ cho khả năng mở rộng nhưng phải luôn là
`false` cho tới khi có một knowledge provider riêng.

---

# Giai đoạn 1 — Router, ROS contracts và STT ingress

## 1.1. ROS interface

Tạo:

```text
src/flzat_high_interfaces/msg/RoutedCommand.msg
```

Các trường dự kiến:

```text
string router_version
string utterance
string mode
float32 confidence
string language
string urgency
bool needs_vision
bool needs_task_memory
bool needs_conversation_memory
bool needs_user_profile
bool needs_external_knowledge
string[] route_names
float32[] route_scores
float32 latency_ms
```

Cập nhật `flzat_high_interfaces/CMakeLists.txt` và `package.xml`.

## 1.2. Router configuration

Tạo:

```text
src/flzat_cognitive_pipeline/config/intent_routes.yaml
```

File cấu hình chứa:

- embedding model ID và revision;
- examples của từng mode;
- minimum confidence;
- minimum Top-1/Top-2 margin;
- context flags mặc định;
- safety phrases và command patterns;
- urgency mapping;
- cache configuration.

Nếu score dưới threshold hoặc margin không đủ, router trả
`mode=clarification`.

## 1.3. Intent Router node

Tạo:

```text
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/intent_router.py
```

ROS interfaces:

```text
Sub:
  /flzat/high_system/text_command
  /flzat/high_system/stt_result

Pub:
  /flzat/high_system/routed_command
  /flzat/high_system/cancel
```

Xử lý mỗi request:

1. Validate và normalize transcript.
2. Chạy deterministic Safety Guard.
3. Nếu là safety command, phát high-level cancel và kết thúc.
4. Nếu bình thường, encode duy nhất user utterance.
5. Tính cosine similarity với cached example embeddings.
6. Áp dụng confidence threshold và margin.
7. Publish typed `RoutedCommand`.

Safety Guard không dùng substring tùy ý. Exact phrases và anchored patterns phải
được đọc từ YAML để hạn chế false-positive.

## 1.4. STT ingress

Hiện tại `CognitiveManager` trực tiếp nhận `/audio_events` và gọi STT action.
Để Router thực sự nằm sau STT, cần một speech ingress nhỏ:

```text
/audio_events → /stt_action → /flzat/high_system/stt_result
```

Sau thay đổi này, `CognitiveManager` không còn subscribe `/audio_events` và
không còn sở hữu STT ActionClient.

## 1.5. Testing giai đoạn 1

- Safety phrases luôn được xử lý trước semantic routing.
- Sáu mode có positive và negative test cases.
- Low confidence chuyển sang `clarification`.
- Top-1 và Top-2 quá gần chuyển sang `clarification`.
- YAML không hợp lệ làm node fail-fast.
- Route embeddings chỉ được encode một lần.
- `route_names` và `route_scores` luôn cùng độ dài.
- Text command và STT result đi qua cùng một classifier.
- Benchmark warm inference dưới 50 ms trên CPU mục tiêu.

## 1.6. Gate hoàn thành

- Router build và chạy độc lập.
- Có typed `RoutedCommand` trên ROS 2 topic.
- Safety stop không gọi RAG, camera hoặc VLM.
- Chưa thay đổi High Command contract.

---

# Giai đoạn 2 — Conditional CognitiveManager và Adaptive Context

**Trạng thái: hoàn tất source và local acceptance ngày 2026-10-15.**

Implementation dùng `intent_routes.yaml` làm nguồn cấu hình chung cho Router và
authoritative `ContextPolicy` qua cùng launch argument `intent_routes_path`,
tránh lặp/hard-code mode flags trong Manager.

## 2.1. Routed input

`CognitiveManager` chuyển sang subscribe:

```text
/flzat/high_system/routed_command
```

Manager tạo `TaskRecord` từ routed utterance và giữ nguyên format ID:

```text
run-YYYYMMDDTHHMMSSZ
task-XXXX
task-XXXX-rXX
task-XXXX-cXX
```

## 2.2. ContextPolicy

Router chỉ đề xuất context flags. `ContextPolicy` trong Manager là nguồn quyết
định authoritative và phải validate/canonicalize flags theo mode.

Điều này ngăn một routed message lỗi hoặc giả mạo làm `task_command` bỏ qua
grounding bắt buộc.

## 2.3. Context providers

Context Builder được tổ chức theo provider:

```text
EnvironmentMemoryProvider   # Chroma object/location memory
SkillRegistryProvider       # trusted YAML allow-list
SafetyPolicyProvider        # trusted static policy
ConversationMemoryProvider  # bounded recent turns
UserProfileProvider         # read-only profile
ExternalKnowledgeProvider   # chưa triển khai
```

Safety policy và skill registry là trusted context, không phải semantic RAG.
Conversation memory và user profile không được lưu chung vào object database.

## 2.4. Mode-specific context

### task_command

```text
camera + environment RAG + skill registry + safety policy
```

Tái sử dụng `ContextRagManager`, `Strict ModelDecision`,
`TrustedCommandAssembler` và command artifact hiện có.

Live robot pose/state không được đưa vào prompt ở phiên bản này. Low System là
nguồn trạng thái thực thi và tự định vị khi nhận High Command; High System
không tạo thêm một robot-state contract chưa được định nghĩa.

### visual_qa

```text
current camera frame
```

Phiên bản đầu không dùng environment RAG. Conditional visual memory có thể bổ
sung sau bằng trigger policy cấu hình.

### general_qa

```text
text-only, không environment RAG, không camera
```

### social_dialogue

```text
3–5 conversation turns + read-only user profile
```

### clarification

```text
short conversation/task history
```

Nếu TaskStore đang chờ clarification, utterance tiếp theo tiếp tục task hiện
tại bất kể router score, trừ khi Safety Guard phát hiện stop/cancel.

## 2.5. Text-only Qwen3-VL

Qwen backend hiện bắt buộc có ảnh. Cần mở rộng để các mode không cần vision có
thể gửi text-only request:

- giữ nguyên `RunVlm.action`;
- cho phép `image.data` rỗng khi `needs_vision=false`;
- Qwen backend tạo user message chỉ có text;
- không gọi frame service cho text-only mode;
- health readiness phải phụ thuộc mode thay vì luôn yêu cầu camera và memory.

## 2.6. Safety cancellation

Intent Router phát `/flzat/high_system/cancel`. `CognitiveManager` tiếp tục sở
hữu việc tạo payload `flzat.low_cancel.v2` vì chỉ Manager biết `run_id`,
`task_id`, `request_id` và `command_id`.

Không tạo Low cancel payload giả khi không có task active. Voice safety không
thay thế hardware emergency stop của Low System.

## 2.7. Testing giai đoạn 2

- Manager không tin context flags bị spoof.
- `task_command` gọi RAG và frame service.
- `visual_qa` gọi frame nhưng không gọi environment RAG.
- `general_qa` không gọi memory hoặc frame service.
- `social_dialogue` không gọi environment RAG.
- `clarification` sử dụng short conversation context.
- External knowledge flag bị reject khi chưa có provider.
- Safety stop trong lúc VLM active gửi cancellation ngay.
- Chỉ `task_command` có đường dẫn tới TrustedCommandAssembler.
- Existing `flzat.high_command.v2` tests tiếp tục pass.

## 2.8. Gate hoàn thành

- Manager chạy được theo route mode.
- Context provider chỉ được gọi khi policy yêu cầu.
- Nhánh task hiện tại vẫn tạo đúng High Command.
- General/social/clarification chạy text-only.

---

# Giai đoạn 3 — Prompt profiles, validators, artifacts và kiểm thử tổng hợp

**Trạng thái: hoàn tất source, unit test, build và local ROS acceptance ngày
2026-10-15. Cloud test Qwen3-VL thật chưa thực hiện trong đợt này.**

## 3.1. Prompt profiles

Xây dựng năm profile:

```text
flzat_cognitive_v1      # task_command
flzat_visual_qa_v1      # visual_qa
flzat_general_qa_v1     # general_qa
flzat_social_v1         # social_dialogue
flzat_clarify_v1        # clarification
```

`safety_stop` không có prompt profile.

Mỗi profile gồm manifest và các section nhỏ:

```text
manifest.json
system/
policies/
tasks/
context/
output/
```

Các policy dùng chung được đặt trong `prompts/shared/`. Không copy cùng một
policy sang cả năm profile.

## 3.2. Manifest-driven PromptComposer

`PromptComposer` không còn hard-code một profile và danh sách file cố định.
Composer phải kiểm tra đồng bộ:

```text
router mode
↔ manifest mode
↔ output schema
↔ validator
```

Nếu không khớp, request bị reject trước khi gọi VLM.

## 3.3. Validators

### TaskCommandValidator

Validate `ModelDecision.v1`, sau đó trusted code mới tạo
`flzat.high_command.v2`. Qwen không được tạo coordinates.

### VisualQAValidator

Output:

```json
{
  "reply": "There are books on the table.",
  "observed_objects": [],
  "confidence": 0.8
}
```

### GeneralQAValidator

Output:

```json
{
  "reply": "Beijing is the capital of China.",
  "speak": true,
  "requires_robot_action": false
}
```

`requires_robot_action` bắt buộc là `false`.

### SocialDialogueValidator

Output:

```json
{
  "reply": "You sound tired. Would you like to rest?",
  "speak": true,
  "suggested_skill": null
}
```

Nếu `suggested_skill` khác `null`, giá trị phải tồn tại trong trusted skill
registry và không được tự động dispatch xuống Low System.

### ClarificationValidator

Output:

```json
{
  "reply": "Which object do you want me to get?"
}
```

Chỉ chấp nhận một câu hỏi ngắn và không chứa robot action.

## 3.4. Output repair

Khi VLM output sai schema:

```text
original inference
→ validation
→ repair attempt 1
→ validation
→ repair attempt 2
→ validation
→ safe fallback
```

Repair phải:

- dùng cùng Qwen3-VL;
- giữ nguyên context snapshot;
- không query RAG lại;
- chỉ sửa format/schema;
- tuân theo cancellation và request deadline;
- không đoán coordinates hoặc target;
- tuyệt đối không publish task JSON sai xuống Low System.

## 3.5. Request artifacts

Mở rộng artifact theo request:

```text
output_root/
└── run-.../
    └── task-0001/
        ├── task-0001-c01.json
        └── task-0001-r01/
            ├── transcript.json
            ├── router.json
            ├── context_policy.json
            ├── rag_context.json
            ├── prompt_metadata.json
            ├── vlm_attempt_01.txt
            ├── validator_attempt_01.json
            └── validated_output.json
```

Manager ghi router output sau khi đã tạo task/request ID. Artifact phải atomic,
không overwrite giữa các run. Production cần cấu hình retention/redaction cho
raw transcript, conversation history và user profile.

## 3.6. Telemetry

Đo riêng:

```text
STT
router
RAG/context
frame lookup
prompt composition
VLM inference
repair attempts
validator
artifact writing
total request
```

## 3.7. Testing giai đoạn 3

- Năm manifest load đúng profile và output schema.
- Prompt composition deterministic và prompt hash ổn định.
- Mỗi validator chấp nhận đúng schema và reject cross-mode schema.
- QA/social/clarification không thể publish High Command.
- Skill ngoài registry bị reject.
- Model-generated coordinates bị reject.
- Repair dừng sau tối đa hai lần.
- Timeout bao phủ toàn bộ repair sequence.
- Artifacts không overwrite và correlation đúng request.
- Existing task command regression tests tiếp tục pass.
- `colcon build --symlink-install` thành công.
- Toàn bộ unit và integration tests pass.

## 3.8. Cloud tests với Qwen3-VL thật

Test tối thiểu:

```text
task_command:    "get the book on the table"
visual_qa:       "what is on the table?"
general_qa:      "what is the capital of China?"
social_dialogue: "I feel tired"
clarification:   "get that for me"
```

Theo dõi:

```text
/flzat/high_system/routed_command
/vlm/raw_response
/flzat/high_system/command
/flzat/high_system/status
/flzat/high_system/metrics
```

## 3.9. Gate hoàn thành

- Năm mode dùng đúng prompt và validator.
- Task command tạo High Command tương thích với Low System.
- Các mode còn lại chỉ tạo TTS reply.
- Cloud test Qwen3-VL xác nhận output thực tế.
- README, `docs/ARCHITECTURE.md`, `docs/INTERFACES.md` và test report được cập
  nhật sau khi implementation đã ổn định.

---

# 4. Danh sách file dự kiến

## File mới

```text
src/flzat_high_interfaces/msg/RoutedCommand.msg
src/flzat_cognitive_pipeline/config/intent_routes.yaml
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/intent_router.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/context_policy.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/conversation_store.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/output_validator.py
src/flzat_speech/flzat_speech/speech_ingress.py
src/flzat_cognitive_pipeline/prompts/shared/...
src/flzat_cognitive_pipeline/prompts/flzat_visual_qa_v1/...
src/flzat_cognitive_pipeline/prompts/flzat_general_qa_v1/...
src/flzat_cognitive_pipeline/prompts/flzat_social_v1/...
src/flzat_cognitive_pipeline/prompts/flzat_clarify_v1/...
```

## File cần sửa

```text
src/flzat_high_interfaces/CMakeLists.txt
src/flzat_high_interfaces/package.xml
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/cognitive_manager.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/prompt_composer.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/domain.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/task_store.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/artifacts.py
src/flzat_cognitive_pipeline/flzat_cognitive_pipeline/telemetry.py
src/flzat_cognitive_pipeline/setup.py
src/flzat_cognitive_pipeline/package.xml
src/flzat_vlm/flzat_vlm/vlm_node.py
src/flzat_vlm/flzat_vlm/backends/qwen3_vl.py
src/flzat_high_bringup/config/high_system.yaml
src/flzat_high_bringup/launch/high_system.launch.py
README.md
docs/ARCHITECTURE.md
docs/INTERFACES.md
docs/TEST_REPORT.md
```

# 5. Ngoài phạm vi hiện tại

- Không sửa Low System.
- Không thay đổi `flzat.high_command.v2`.
- Không thêm LLM router.
- Không thêm YOLO hoặc perception model khác.
- Không huấn luyện model mới.
- Không xây external knowledge base trong giai đoạn này.
- Không mở rộng mock VLM sang tất cả mode.
- Không để Qwen tạo trusted coordinates.
- Không thêm runtime dependency tới các workspace cũ.

# 6. Thứ tự triển khai

```text
1. Router + ROS contracts + STT ingress
2. Conditional Manager + ContextPolicy + text-only Qwen
3. Prompt profiles + validators + artifacts + tests + cloud verification
```

Không bắt đầu giai đoạn sau khi gate của giai đoạn trước chưa đạt.
