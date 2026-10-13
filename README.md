# 🦜 Tiger-TTS Telegram bot

Bot nhận văn bản tiếng Việt, trả về voice message (Tiger-TTS v3 Turbo, ONNX/CPU, 48 kHz).
Chạy được trên máy chỉ có CPU, khoảng 2 GB RAM.

| File | Vai trò |
|---|---|
| `main.py` | Bot. Lần chạy đầu tự cài thư viện còn thiếu vào `./.deps` và tải model vào `./hf_cache` |
| `requirements.txt` | Thư viện tối giản (không gradio/librosa/torch) |
| `voices/*.npz` | Giọng custom đã mã hoá sẵn (tạo bằng `make_voice.py`) |
| `make_voice.py` | Clip 3–8 s → `voices/<tên>.npz` |
| `.env.example` | Danh sách biến môi trường |

Tài nguyên đo được: thư viện ~240 MB + model ~0.6 GB ≈ **0.85 GB disk**; RAM ~1.2 GB khi rảnh, +0.45 GB khi tạo giọng (`CHUNK_CHARS=128`).

## 1. Tạo bot Telegram
1. Mở Telegram → chat với **@BotFather** → `/newbot` → đặt tên → nhận **token** dạng `123456:ABC...`.
2. Giữ token bí mật (không commit lên GitHub — `.env` đã nằm trong `.gitignore`).

## 2. (Tuỳ chọn) Thêm giọng custom
```bash
cd ../Tiger-TTS
uv run python ../vieneu-telegram-bot/make_voice.py /đường/dẫn/clip.wav ten_giong
```
Tạo `voices/ten_giong.npz` (+ file preview `.wav` để nghe thử, không cần commit).
Bước này cần ~2.1 GB RAM — nên chạy trên máy cá nhân, không chạy trên server nhỏ.

## 3. Chạy bot
```bash
python3.12 -m venv .venv
.venv/bin/python main.py --selftest       # tạo selftest.ogg, không cần Telegram
cp .env.example .env                      # điền TELEGRAM_BOT_TOKEN
.venv/bin/python main.py
```
Lần chạy đầu mất vài phút (cài thư viện ~240 MB, tải model ~0.6 GB); những lần sau bỏ qua. Log khởi động:
```
⏳ Loading Tiger-TTS v3 Turbo (ONNX, fp32, threads=2)...
✅ Model ready in ...s
🎙️ Custom voices: ngoc_huyen_truyen
🤖 Bot polling (allowed users: everyone)
```
Mỗi command được ghi log dạng `✅ <user_id> /<command> success in <giây>`.

## 4. Cấu hình (`.env` hoặc biến môi trường)
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Bắt buộc**. Token từ BotFather |
| `ALLOWED_USER_IDS` | trống = mọi người | User ID được dùng bot, cách nhau dấu phẩy (lấy bằng `/id`) |
| `DEFAULT_VOICE` | `Ngọc Huyền` | Giọng mặc định |
| `MAX_CHARS` | `600` | Số ký tự tối đa mỗi tin nhắn |
| `MAX_FILE_CHARS` | `40000` | Số ký tự tối đa mỗi file `.txt` (~35 phút audio) |
| `CHUNK_CHARS` | `128` | Số ký tự tối đa mỗi lần model xử lý — quyết định RAM đỉnh |
| `VIENEU_THREADS` | `2` | Số luồng CPU cho model |
| `VIENEU_PRECISION` | `fp32` | `int8` chỉ khi CPU có VNNI |

## 5. Dùng và khoá bot
1. Trong Telegram mở bot → `/start` → gửi một câu → nhận voice message.
2. Gửi `/id` → đặt `ALLOWED_USER_IDS=<id>` (nhiều người: `id1,id2`) → khởi động lại bot.
   Nên làm ngay — bot công khai ai cũng dùng được và sẽ chiếm hết CPU.

Gửi **file `.txt`** (vd. một chương truyện) → bot đọc từng đoạn, cập nhật tiến trình mỗi 15 giây (% và thời gian còn lại), trả về 1 file MP3. Một chương ~14.000 ký tự ≈ 13 phút audio, ~7 MB; trên Mac M (2 threads) mất 5 phút, CPU server yếu hơn sẽ lâu hơn (xem RTF trong log). Giới hạn: `MAX_FILE_CHARS`.

Lệnh: `/voices` (danh sách giọng) · `/voice Mai Anh` (đổi giọng) · `/voice ngoc_huyen_truyen` (giọng custom).

`/test` — nghe thử một giọng: bot đọc một câu mẫu ngắn (~4 giây audio) bằng giọng đó, không đổi giọng đang dùng.
`/test` (danh sách giọng, có số) · `/test 3` hoặc `/test Mai Anh` · gõ một phần tên như `/test trân` cũng được; nếu khớp nhiều giọng, bot liệt kê để chọn.

## 6. Tuỳ chỉnh tốc độ và RAM
- Log khởi động in `🖥️ CPU: ... · VNNI: yes/no` (trên Linux). Nếu `yes`: `VIENEU_PRECISION=int8` → nhanh hơn ~1.6×, RAM ~0.9 GB. Nghe thử, nếu méo tiếng thì quay lại `fp32`.
- `MAX_CHARS` nhỏ hơn → mỗi lần chờ ít hơn. Bot xử lý **từng yêu cầu một** để không vượt RAM.
- `CHUNK_CHARS`: RAM đỉnh tăng theo nó. Đo với file 3 KB: 256 → +1.3 GB (bị kill trên máy 2 GB), 128 → +0.45 GB, 96 → +0.3 GB. Không ảnh hưởng tốc độ.
- `VIENEU_THREADS=1` không giảm RAM, chỉ chậm hơn (~35%).

## Xử lý sự cố
| Triệu chứng | Cách xử lý |
|---|---|
| Bot bị kill / tự restart khi tạo giọng (hết RAM) | Giảm `CHUNK_CHARS` (96); trên Linux thêm `MALLOC_ARENA_MAX=2`; `int8` nếu CPU có VNNI |
| `No space left on device` | Cần ≥ 1.5 GB disk trống cho thư viện + model |
| `Thiếu TELEGRAM_BOT_TOKEN` | Chưa đặt biến trong `.env` / môi trường |
| `Conflict: terminated by other getUpdates` | Bot đang chạy ở chỗ khác với cùng token → tắt bản kia, hoặc `/revoke` token ở BotFather |
| Tiếng méo, rè | Đang dùng `int8` trên CPU không có VNNI → `VIENEU_PRECISION=fp32` |
