# FLZAT High System — kế hoạch refactor pipeline nhận thức

Ngày cập nhật: **04/11/2026**.

Trạng thái: **Cả ba bước đã triển khai và kiểm thử với Low mock.** Action A/B
đã có interface ROS 2 và được nối vào luồng High. Đây vẫn là contract kiểm thử
với mock, chưa phải xác nhận tương thích Low production.

Cập nhật JSON bàn giao (04/11/2026): đầu ra hiện tại dùng bản gọn
`flzat.high_handoff.v2`, giữ `goal.description`, `constraints` và
`missing_information`, yêu cầu gốc và các lượt clarification. Entity chỉ mang
thông tin ngữ nghĩa/vị trí cần thiết; metadata đầy đủ nằm trong
`handoff_audit.json` tại High. Các nhắc tới v1 bên dưới là mốc triển khai trước
đó; contract hiện hành được mô tả trong `HIGH_LOW_ACTION_CONTRACT.md`.

## Mục tiêu và phạm vi

Hoàn thiện pipeline High System trong workspace hiện tại, tái sử dụng các thành
phần đã có, không xây một hệ thống song song:

```text
Voice/Text
  → High thu thập context
  → VLM hiểu yêu cầu và hình thành mục tiêu high-level
  → Kiểm tra quyết định, bổ sung grounded entities
  → Ghi JSON hoàn tất
  → Action A gửi đường dẫn cho Low

Low cần tương tác → Action B
  ├─ ASK/CONFIRM → High hỏi → người dùng trả lời → High hiểu → Result B
  └─ INFORM/REFUSE → High phát thông báo → Result B
```

Chỉ triển khai **High và Low mock phục vụ kiểm thử**. Không triển khai Low LLM,
planner, BT, FAST-SKILL runtime hoặc controller.

### Các quy ước đã chốt

- High truyền mục tiêu và context; Low tự phân loại, phân rã và thực thi.
- Không ép VLM High chọn `fetch_object`, `navigation`, `manipulation`,
  FAST-SKILL/PLANNED-TASK hoặc `direct/search`.
- Lời đáp và mục tiêu hành động có thể cùng tồn tại trong một quyết định.
- JSON bảo toàn yêu cầu, ràng buộc, nhiều thực thể, quan hệ và nguồn dữ liệu.
  Vị trí phải gắn đúng thực thể và frame, lấy từ snapshot đã kiểm tra.
- A gửi đường dẫn file, không gửi toàn bộ JSON. High và Low dùng thư mục chung
  có cùng absolute path; file phải ghi hoàn tất trước Goal A.
- `STARTED_FAST/STARTED_PLAN` chỉ biểu thị Low đã khởi động thực thi.
- B giữ các kind `ASK | CONFIRM | REFUSE | INFORM`.
- Chỉ một phiên hỏi–đáp tại một thời điểm, không có hàng chờ B.
- Không thay thế nhiệm vụ hoặc tự hủy/tạo lại inference do input mới.
- Giữ voice half-duplex; text/cancel vẫn có đường tiếp nhận riêng.
- Chưa thêm topic/service theo dõi hoặc hủy nhiệm vụ Low sau khi A kết thúc.

### Hiện trạng làm cơ sở refactor

Giữ VAD/STT/TTS, camera/ring buffer/static image, truy vấn Chroma, Qwen backend,
PromptComposer, snapshot, atomic artifact writer, telemetry và kiểm tra JSON.

Thay tầng điều phối đang phụ thuộc router theo loại câu, schema intent kỹ thuật,
command topic cũ và cách gộp inference/hội thoại/thực thi thành một trạng thái.
Loại bỏ fallback gắn tọa độ của thực thể khác khi target thiếu vị trí.

Mốc test trước refactor trong phiên khảo sát: **210 passed, 1 skipped**. Đây là
test code hiện tại, không chứng minh A/B mới hoặc chất lượng prompt/Qwen mới.

## Bước 1 — Thống nhất nhận thức, context và JSON

**Trạng thái: Hoàn thành ngày 03/11/2026.** Luồng mặc định đã nhận trực tiếp
STT/text, dùng profile nhận thức `flzat_brain_v1`, snapshot RAG và frame tùy
khả dụng, rồi kiểm tra `CognitiveDecision.v1` và ghi
`flzat.high_handoff.v1`. Tại mốc Bước 1, router và command topic cũ còn được giữ
tạm để chuyển đổi; chúng đã được loại bỏ hoàn toàn ở Bước 3. Cơ chế hỏi lại do
High sở hữu đã được nối vào context; owner `LOW_B` được bổ sung ở Bước 2.

Kết quả xác minh Step 1: **197 passed, 1 skipped** cho toàn bộ `src`, bài ROS 2
tích hợp tùy chọn **1 passed**, build hai package cognitive/bringup thành công,
và smoke test launch đã ghi được handoff JSON từ text + RAG + static frame.

### Công việc

1. Bỏ embedding router và lượt Qwen phân loại khỏi luồng mặc định. Tách và giữ
   safety guard stop/cancel độc lập, không phụ thuộc router hoặc inference.
2. Dùng một bộ prompt nhận thức modular: vai trò companion, ranh giới High–Low,
   context, tương tác đang chờ, grounding và output contract.
3. Tổng hợp transcript, lịch sử giới hạn, ảnh nếu có, ứng viên RAG liên quan và
   trạng thái High thực sự biết. Không dùng nhãn general QA/visual QA/task để
   chặn quyền suy luận hoặc tạo mục tiêu của VLM.
4. Đánh dấu ảnh/RAG không khả dụng hoặc dữ liệu quá hạn; không làm mọi yêu cầu
   thất bại chỉ vì thiếu một nguồn context. Chưa thêm vòng ghi/cập nhật Chroma.
5. Giữ thuộc tính, quan hệ, thời điểm quan sát và nguồn từ bản ghi. Phân biệt
   lời người dùng, quan sát ảnh và bộ nhớ cũ; không biến quan hệ văn bản thành
   quan hệ hình học đã xác thực.
6. Quyết định VLM gồm lời đáp, mục tiêu ngôn ngữ tự nhiên tùy chọn, context refs,
   ràng buộc, thông tin chưa rõ và yêu cầu tương tác khi cần. Không tạo planner
   hoặc lựa chọn skill ở High.
7. JSON bàn giao lưu version, ID, nguyên văn yêu cầu, phần bổ sung đã được hiểu,
   mục tiêu, ràng buộc, thông tin thiếu và danh sách grounded entities.
   Mỗi thực thể giữ riêng ID, nguồn, quan hệ, vị trí/frame nếu có.
8. Giải context refs từ đúng snapshot; không chấp nhận tọa độ do VLM tạo và
   không fallback sang thực thể đầu tiên. Giữ kiểm tra schema và sửa định dạng
   có giới hạn.

Thiếu chi tiết thực thi không tự động chặn bàn giao. Nếu chưa hiểu chính ý định
người dùng, High hỏi làm rõ trước khi gửi A.

### Phiên hỏi–đáp dùng chung, phân biệt nguồn

Thiết kế phiên giữ `owner = HIGH | LOW_B`, ID yêu cầu, câu hỏi, lịch sử hỏi–đáp,
deadline và số lần hỏi lại. Bước 1 triển khai nhánh `HIGH`; Bước 2 bổ sung
`LOW_B`, `interaction_id` và Goal handle khi Action B tồn tại.

| Nguồn câu hỏi | Khi hiểu câu trả lời |
|---|---|
| HIGH | Bổ sung vào yêu cầu ban đầu, tiếp tục suy luận; có thể tạo mục tiêu và gửi A |
| LOW_B | Trả Result cho đúng B; không tự tạo A mới |

Không đổi nguồn phiên chỉ vì câu trả lời chứa động từ hành động. Chưa nhận diện
đổi chủ đề/câu hỏi ngoài lề thành yêu cầu mới trong phiên hỏi–đáp.

### Testing và tiêu chí hoàn thành

- Trò chuyện thuần túy không tạo hành động không có căn cứ.
- “Come here and sit with me so we can talk” có thể vừa đáp lời vừa tạo mục tiêu;
  không dùng nhãn “trò chuyện” để phủ quyết hành động.
- Command dài giữ đủ mục tiêu, thứ tự/ràng buộc người dùng nêu, không bị rút
  thành chỉ tên object và tọa độ.
- Nhiều thực thể giữ đúng nguồn, ID, vị trí và frame. Thiếu tọa độ vẫn biểu đạt
  được context ngữ nghĩa; không bịa dữ liệu để lấp chỗ trống.
- JSON lỗi, context ref giả, snapshot cũ và vị trí không hợp lệ bị chặn.
- Câu trả lời nguồn HIGH tiếp tục đúng yêu cầu. Quy tắc nguồn LOW_B không phát
  sinh A được kiểm thử ở Bước 2 sau khi Action B được triển khai.

**Hoàn thành khi:** prompt, context, validator và JSON thống nhất với ranh giới
High–Low mới; các test thuần logic trên chạy được không cần Qwen thật.

## Bước 2 — Action A/B và chính sách trạng thái tối giản

**Trạng thái: Hoàn thành ngày 04/11/2026 với Low mock.** Luồng bàn giao topic
cũ trong Manager/Low mock đã được thay bằng Action A gửi absolute path tới file
`flzat.high_handoff.v1`; JSON không còn giữ `command_id` dành cho adapter cũ.
Action B dùng chung speech/text ingress nhưng giữ owner LOW_B, nên câu trả lời
chỉ hoàn tất đúng Goal B và không tạo A mới.

Đã bổ sung BUSY/READY quan sát được, một reservation B duy nhất, retry hữu hạn
ở Low mock, phân biệt ASK/CONFIRM với INFORM/REFUSE, timeout/decline/unresolved,
và đường stop/cancel đi trước BUSY. Cancel A đang hoạt động chờ Result A thay
vì tự khẳng định Low đã dừng; sau `STARTED_*` hệ thống chỉ báo rõ chưa có contract
hủy Low execution trong phase này.

### Action A: `/llm/handle_intent`

High là client; Low mock là server.

- Goal đã triển khai: `run_id`, `task_id`, `handoff_id`, `json_path`.
- Feedback: `stage = deciding | validating | planning | waiting_for_user`, kèm
  thông tin mô tả.
- Result: `STARTED_FAST | STARTED_PLAN | REFUSED | FAILED | CANCELED`, kèm mô tả.

Ghi file immutable bằng atomic writer rồi mới gửi Goal. Chưa tự xóa artifact
vì High chưa biết Low dùng xong lúc nào. Không tự gửi lại A khi kết quả không
rõ hoặc sau restart; mock kiểm tra trùng `handoff_id`.

TTS đi kèm A có lifecycle riêng. A hoàn tất không đồng nghĩa TTS đã rảnh.
`waiting_for_user` chỉ cập nhật trạng thái; câu hỏi cụ thể phải đến qua B.

Quy ước mock đã triển khai: `STARTED_*`/`REFUSED` đi cùng ROS `SUCCEEDED`; `FAILED`
đi cùng `ABORTED`; `CANCELED` đi cùng ROS `CANCELED` sau xử lý cancel.
Goal rejection, mất kết nối và hết thời gian chờ là trạng thái giao tiếp riêng,
không tự gán thành outcome Low chưa trả. Mapping này cần đối chiếu với Low thật.

Sau `STARTED_*`, đóng lượt bàn giao High và lưu “Low đã báo bắt đầu”; không báo
nhiệm vụ thành công. High nhận tương tác mới khi tài nguyên đã rảnh.

### Action B: `/brain/ask_user`

High là server; Low mock là client.

- Goal đã triển khai: các ID liên kết, `interaction_id`, `kind`, `content_hint`,
  `options`, `timeout`.
- Feedback: `speaking | listening`.
- Result: `outcome`, `answer`, `error_message`.

**ASK/CONFIRM:** chấp nhận Goal → phát câu hỏi → chờ TTS và cooldown chống vọng
→ `listening` → nhận voice/text → hiểu câu trả lời trong context phiên → trả
Result hoặc hỏi lại có giới hạn.

Câu tự nhiên như “the bottle on the table” dùng VLM khi cần; không bắt buộc
yes/no hoặc khớp option. Nội dung/câu trả lời rõ ràng có thể xử lý trực tiếp,
không bắt buộc gọi VLM cho mọi B.

**INFORM/REFUSE:** phát thông báo → chờ TTS thành công → `ANSWERED` với answer
rỗng. Không phát `listening`, không chờ STT. `REFUSE` là Low nhờ High truyền đạt
lời từ chối, không phải người dùng từ chối trả lời.

### Phủ định, không hiểu và kết thúc phiên

**Đề xuất diễn giải cần thống nhất với Low:**

- Phủ định CONFIRM vẫn là `ANSWERED`. “No, use the blue bottle instead” giữ cả
  phủ định và thông tin bổ sung, không chỉ trả “no”. Đây vẫn là câu trả lời B;
  High không tự sửa nhiệm vụ hoặc gửi A mới.
- `DECLINED` dành cho người dùng chủ động không muốn trả lời/tham gia tương tác.
- Lưu câu trả lời nguyên văn trong artifact. Bản chuẩn hóa không thêm sự kiện
  hoặc quyết định bước thực thi thay Low.

Giới hạn một câu hỏi ban đầu và tối đa **hai lần hỏi lại**, trong cùng deadline.
Timeout B tính từ lúc chấp nhận Goal, gồm TTS và inference; không reset khi hỏi
lại. Phiên HIGH dùng deadline cấu hình, mặc định 60 giây.

- Hiểu câu trả lời: hoàn tất theo đúng nguồn phiên.
- Deadline hết: B trả `TIMEOUT`; HIGH đóng phiên timeout, không gửi A.
- Hết số lần hỏi lại trước deadline nhưng vẫn không hiểu: HIGH đóng với trạng
  thái nội bộ `UNRESOLVED_ANSWER`, không gửi A. Với B, **đề xuất** ROS `ABORTED`,
  outcome rỗng, `error_message` có lý do `UNRESOLVED_ANSWER`.

Không hiểu không phải từ chối hoặc timeout. Outcome hiện tại thiếu nhãn riêng;
mock dùng mapping đề xuất, chưa coi là Low thật đã chấp thuận.

| Tình huống B | ROS terminal status | Outcome |
|---|---|---|
| Trả lời hợp lệ, gồm phủ định CONFIRM | SUCCEEDED | ANSWERED |
| Chủ động từ chối trả lời | SUCCEEDED | DECLINED |
| Hết deadline | SUCCEEDED | TIMEOUT |
| INFORM/REFUSE phát lời xong | SUCCEEDED | ANSWERED, answer rỗng |
| Lỗi kỹ thuật | ABORTED | Rỗng; có error_message |
| Hết số lần hỏi lại nhưng chưa hiểu — đề xuất | ABORTED | Rỗng; UNRESOLVED_ANSWER |
| Low gửi ROS cancel được chấp nhận | CANCELED | Rỗng |

Giữ `SUPERSEDED` trong contract nhưng chưa sử dụng. Goal bị từ chối tiếp nhận
không có Result B. INFORM/REFUSE hoàn tất chỉ xác nhận đã phát thông báo,
không khẳng định người dùng đã nghe hoặc đồng ý.

### Chính sách input và BUSY

Tách trạng thái lượt nhận thức, phiên hỏi–đáp và bàn giao A. Callback ROS không
chờ đồng bộ A/B/TTS. Một inference Qwen tại một thời điểm; không hàng chờ input.

| Trạng thái High | Voice/text thông thường | Goal B mới |
|---|---|---|
| Rảnh | Nhận yêu cầu | Nhận nếu hợp lệ |
| Đang inference hoặc TTS | Từ chối với BUSY | Từ chối tiếp nhận |
| Chờ A, không inference/TTS/phiên hỏi–đáp | Từ chối yêu cầu mới với BUSY | Nhận B |
| Nghe câu trả lời cho HIGH | Bổ sung cho yêu cầu gốc | Không kỳ vọng B trong luồng tuần tự; guard chồng phiên được hoãn, xem ghi chú dưới |
| Nghe câu trả lời cho LOW_B | Xử lý để trả B | Từ chối tiếp nhận |
| Xử lý câu trả lời của một phiên | Từ chối input mới với BUSY | Từ chối tiếp nhận |
| A đã trả STARTED, High đã rảnh | Nhận lượt độc lập | Nhận B có liên kết hợp lệ |

BUSY phải xuất hiện trên topic trạng thái High hiện có và log ngắn: loại sự kiện
từ chối, resource/lý do, ID input hoặc interaction, ID phiên đang bận và chỉ báo
không tiếp nhận/không xếp hàng. Không biến task hiện tại thành FAILED.

Khi giải phóng tài nguyên, công bố lại khả năng nhận yêu cầu người dùng/B.
Không phát lời BUSY đè lên câu hỏi. Tiêu chí BUSY áp dụng cho input đã tới High;
mic vẫn tạm ngừng trong lúc TTS theo cơ chế half-duplex hiện tại.

Giữ kiểm tra ID/generation để bỏ callback trễ sau timeout/hủy tường minh;
không dùng input mới làm lý do hủy và tái tạo inference.

### B đến khi TTS đi kèm A còn hoạt động

High từ chối Goal B vì bận; không ngắt TTS, không giữ Goal chờ và không trả
Result giả. Không giả định B chỉ đến lúc High rảnh.

**Đề xuất phía Low/mock cần thống nhất:**

- Goal hợp lệ chưa được chấp nhận có thể retry hữu hạn. Mock mặc định mỗi một
  giây, tối đa mười lần thử tổng cộng và không vượt ngân sách thời gian tổng.
- Giữ `interaction_id`; mỗi lần thử là một ROS Goal mới. Không retry một Goal
  đã được chấp nhận.
- Timeout gửi lại là thời gian còn lại, không reset ngân sách.
- Hết ngân sách: ghi không thiết lập được tương tác, không coi như đã hỏi.
- ROS rejection không có Result giải thích BUSY; không coi mọi rejection là
  BUSY. Mock biết kịch bản và kiểm tra event trạng thái High để xác nhận lý do.
  Chính sách retry của Low thật phải được thống nhất riêng.

Retry nằm ở client Low/mock, không phải hàng chờ phía High.

### Stop/cancel độc lập BUSY

Safety guard xử lý stop/cancel tường minh trước BUSY và không cần Qwen.

- Dừng inference/TTS, đóng phiên High theo phạm vi yêu cầu hủy.
- A còn hoạt động: gửi ROS cancel và chờ xác nhận, không tự tuyên bố Low dừng.
- Sau STARTED: báo rõ chưa có cơ chế hủy nhiệm vụ Low trong phạm vi này.
- Người dùng stop khi B đang hoạt động: **đề xuất** B kết thúc ROS `ABORTED`,
  outcome rỗng, lý do `INTERRUPTED_BY_USER`. Không dùng ROS `CANCELED` khi client
  Low chưa gửi cancel.

Giữ đầu vào cancel tường minh hiện có và stop từ text/STT; không hứa nhận được
voice stop khi mic đang bị mute bởi TTS. Chưa bổ sung voice barge-in.

### Testing và tiêu chí hoàn thành

- File hoàn tất trước khi mock đọc A; các ID và kết quả A được liên kết đúng.
- A gọi B trước Result và B đến sau STARTED đều được xử lý khi High sẵn sàng.
- ASK/CONFIRM chờ input; INFORM/REFUSE hoàn tất chỉ với TTS, không cần STT.
- Câu trả lời tự nhiên, phủ định kèm bổ sung và từ chối trả lời được phân biệt.
- Hỏi lại hết lượt trước deadline khác TIMEOUT và DECLINED.
- B đến trong lúc TTS đi kèm A: bị reject, có BUSY event; mock retry thành công
  sau khi rảnh hoặc kết thúc rõ ràng khi hết ngân sách.
- B thứ hai/input lúc bận không được xếp hàng hoặc thay thế request hiện tại.
- Stop/cancel không bị BUSY chặn; TTS/VLM lỗi, timeout và callback trễ không gây
  trả Result hai lần, nói muộn hoặc dispatch A muộn.

**Hoàn thành khi:** High và mock chạy được vòng A → B → trả lời tự nhiên → Result
B, đúng chính sách đơn phiên và quan sát được mọi trường hợp từ chối/lỗi.

### Kết quả kiểm thử Bước 2

- Build `flzat_high_interfaces` và toàn bộ dependency tới
  `flzat_high_bringup`: **7 package thành công**.
- Toàn bộ pytest workspace: **228 passed, 2 skipped**. Hai test ROS live được
  skip mặc định để không tự mở DDS trong unit suite.
- ROS integration A → B → natural answer → Result B → `STARTED_PLAN`:
  **1 passed** với ROS domain riêng.
- Integration đã ép TTS giữ BUSY: B bị reject có status event, Low mock retry
  cùng `interaction_id`, sau đó B được nhận và câu trả lời không tạo task/A mới.
- `python compileall` và `git diff --check`: đạt.
- Vấn đề ament cũ thu được 0 test đã được xử lý ở Bước 3; `colcon test` hiện
  tìm và chạy đúng các test của hai package Python.

## Bước 3 — Đồng bộ workspace và kiểm thử toàn luồng

**Trạng thái: Hoàn thành ngày 04/11/2026.** Workspace chỉ còn profile
`flzat_brain_v1`, output `CognitiveDecision.v1`, artifact
`flzat.high_handoff.v1` và Action A/B. Router embedding/Qwen phân loại, các
schema intent kỹ thuật, `RoutedCommand`, command topic/low-status transport cũ,
fake routing fixtures và prompt profile theo route đã được gỡ khỏi source và
package install.

### Công việc

1. Đồng bộ Manager, launch, config, schema, prompt, validators, fake VLM và Low
   mock. Giữ nguyên khả năng chạy voice, text, camera và static image.
2. Gỡ các nhánh router/intent/transport Low cũ khỏi luồng production sau khi
   thay test tương ứng; không chạy hai đường bàn giao Low song song.
3. Giữ log ngắn và artifact đầy đủ: input, snapshot, prompt, raw/validated
   output, JSON bàn giao, sự kiện A/B, BUSY và lỗi tương tác.
4. Cập nhật README và tài liệu contract, lệnh test full voice và text/static
   image. Tách quy ước đã chốt, hành vi đã kiểm thử với mock và đề xuất cần Low
   xác nhận.

### Testing và tiêu chí hoàn thành

- Text + static image + RAG → VLM → JSON → A; fake voice đi qua cùng pipeline.
- Trò chuyện thuần túy không có hành động vô căn cứ; yêu cầu tương tác có hành
  động vẫn tạo mục tiêu, không bị chặn bằng nhãn route.
- High hỏi làm rõ → bổ sung yêu cầu → A; Low hỏi B → trả lời → Result B, không A.
- Nhiều thực thể giữ đúng quan hệ, nguồn, vị trí và frame trong file bàn giao.
- Chạy đủ test BUSY, B/TTS collision, retry, phủ định, từ chối, không hiểu,
  timeout, lỗi dịch vụ và stop/cancel của bước 2.
- STARTED không thành COMPLETED; restart không tự gửi lại artifact. Trạng thái
  gián đoạn/không rõ không được giả thành nhiệm vụ đã thành công.
- Chạy lại unit suite và ROS integration với fake backend, thư mục tạm và ROS
  domain riêng. Ghi rõ test đã chạy, skipped và chưa thể kiểm chứng.
- Qwen thật được test bằng cùng launch/pipeline trên cloud hoặc máy đủ mạnh,
  không viết một pipeline cloud song song. Mock không thay thế test chất lượng
  model, mic/TTS thật hoặc tích hợp Low thật.

**Hoàn thành khi:** luồng tối giản chạy end-to-end với mock, regression các module
tái sử dụng đạt yêu cầu, tài liệu và interface mới đồng bộ, không còn đường cũ
cùng phát lệnh. Không tuyên bố đã tích hợp Low production chỉ từ kết quả mock.

### Kết quả kiểm thử Bước 3

- Build sạch dependency tới `flzat_high_bringup`: **7 package thành công**.
- Toàn workspace: **179 passed, 2 skipped**; hai test skip là ROS live opt-in.
- ROS live với domain riêng: **2 passed**, gồm text → cognitive VLM và vòng
  A → B → natural answer → `STARTED_PLAN`.
- `colcon test` đã hoạt động đúng: cognitive **83 passed**; bringup
  **13 passed, 2 skipped**.
- `compileall` và `git diff --check`: đạt.
- Chưa chạy Qwen thật, mic/camera thật hoặc Low production trong Bước 3; các
  mục này giữ nguyên là kiểm chứng sau, không được suy diễn từ kết quả mock.

## Các điểm cần thống nhất với phía Low

Các kind/outcome và hướng A/B đã chốt; tên trường chi tiết trong `.action` vẫn là
thiết kế dự kiến. Trước tích hợp thật cần xác nhận:

- Mapping ROS terminal status với outcome nghiệp vụ, gồm kết quả A.
- Phủ định CONFIRM là ANSWERED; DECLINED là từ chối trả lời; giữ nội dung bổ sung.
- ABORTED/outcome rỗng cho UNRESOLVED_ANSWER và INTERRUPTED_BY_USER.
- Chính sách retry Goal B chưa được chấp nhận, deadline và chống trùng ID.
- Đường dẫn shared storage, quyền đọc file và quy trình dọn dữ liệu sau này.

Không ngầm mở rộng B thành API yêu cầu perception hoặc dùng B làm giao thức
điều khiển nhiệm vụ. Không coi mock đã thay mặt Low production chấp thuận.

## Hoãn sang giai đoạn sau

### Ghi chú rà soát 04/11/2026 — chồng phiên HIGH clarification và Action B

Theo quyết định của người dùng, **chưa sửa logic chồng phiên này** và không coi
đây là blocker cho luồng kiểm thử tuần tự hiện tại:

```text
High làm rõ yêu cầu với người dùng → hiểu mục tiêu → gửi A
→ Low xử lý → nếu cần thông tin bổ sung thì gửi B → High trả Result B
```

Với cùng một nhiệm vụ chưa gửi A, Low chưa nhận được việc nên chưa có lý do gửi
B cho nhiệm vụ đó. Giả định kiểm thử hiện tại là không còn nhiệm vụ Low trước
đó có thể gửi B trong lúc High làm rõ yêu cầu mới.

Rủi ro vẫn được giữ để theo dõi: sau khi A của nhiệm vụ trước trả `STARTED_*`,
Low có thể còn thực thi và gửi B muộn, đúng lúc High đang hỏi làm rõ nhiệm vụ
mới. Code hiện chưa chặn B bằng trạng thái chờ clarification của HIGH; đây là
giới hạn đã ghi nhận, không phải đã được guard bảo đảm. Nếu gặp tình huống này
khi chạy thực tế, bổ sung kiểm tra phiên đang chờ và regression test trước khi
mở rộng luồng. Không thêm hàng chờ, điều phối nhiều nhiệm vụ hay thay đổi
contract A/B trong lần ghi chú này.

### Các tính năng khác được hoãn

- Hàng chờ nhiều B, ưu tiên và điều phối nhiều nhiệm vụ.
- Đổi chủ đề/câu hỏi ngoài lề trong phiên hỏi–đáp rồi quay lại phiên cũ.
- Thay thế hoặc điều chỉnh nhiệm vụ theo ý mới của người dùng.
- Tự hủy và tái tạo inference khi có input mới.
- Theo dõi, truy vấn, hủy nhiệm vụ Low sau khi A kết thúc.
- Voice barge-in, quan sát chủ động liên tục và vòng cập nhật memory.
- API để Low yêu cầu High phân tích cảnh ngoài phạm vi B.

## Theo dõi triển khai

- [x] Bước 1: nhận thức, context, JSON và test thuần logic.
- [x] Bước 2: A/B, phiên đơn, BUSY, stop/cancel và test mock.
- [x] Bước 3: đồng bộ, ROS end-to-end, regression và tài liệu.
- [ ] Kiểm chứng Qwen thật và ghi nhận kết quả riêng.
- [ ] Đối chiếu contract với người phát triển Low trước tích hợp production.
