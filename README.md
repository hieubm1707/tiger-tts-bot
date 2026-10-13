# 🦜 Tiger-TTS Telegram bot — deploy lên BotKeep

Bot nhận văn bản tiếng Việt, trả về voice message (Tiger-TTS v3 Turbo, ONNX/CPU, 48 kHz).
Được thiết kế cho gói **BotKeep Founder Free** (2 GB RAM · 1.5 vCore · 2 GB disk).

| File | Vai trò |
|---|---|
| `main.py` | Bot. Lần chạy đầu tự cài thư viện vào `./.deps` và tải model vào `./hf_cache` |
| `requirements.txt` | Thư viện tối giản (không gradio/librosa/torch) |
| `voices/*.npz` | Giọng custom đã mã hoá sẵn (tạo bằng `make_voice.py` trên Mac) |
| `make_voice.py` | Chạy **trên Mac**: clip 3–8 s → `voices/<tên>.npz` |
| `samples/*.ogg` | Audio thử giọng có sẵn cho lệnh `/test` (tạo bằng `main.py --make-samples` trên Mac) |
| `.env.example` | Danh sách biến môi trường |

Tài nguyên đo được: thư viện ~240 MB + model ~0.6 GB ≈ **0.85 GB disk**; RAM 1.3–1.75 GB khi tạo giọng.

## 1. Tạo bot Telegram
1. Mở Telegram → chat với **@BotFather** → `/newbot` → đặt tên → nhận **token** dạng `123456:ABC...`.
2. Giữ token bí mật (không commit lên GitHub).

## 2. (Tuỳ chọn) Thêm giọng custom — trên Mac
```bash
cd ../Tiger-TTS
uv run python ../vieneu-telegram-bot/make_voice.py /đường/dẫn/clip.wav ten_giong
```
Tạo `voices/ten_giong.npz` (+ file preview `.wav` để nghe thử, không cần upload).
**Không clone trên BotKeep** — bước này cần ~2.1 GB RAM.

## 3. Thử trên Mac (tuỳ chọn)
```bash
python3.12 -m venv /tmp/botvenv && /tmp/botvenv/bin/python main.py --selftest   # tạo selftest.ogg
/tmp/botvenv/bin/python main.py --make-samples   # samples/<giọng>.ogg cho /test (~1 phút, ~1 MB)
cp .env.example .env   # điền TELEGRAM_BOT_TOKEN
/tmp/botvenv/bin/python main.py
```

## 4. Đẩy source lên GitHub (repo private)
```bash
git status            # kiểm tra: KHÔNG có .env, .venv/, .deps/, hf_cache/
git push
```
`.gitignore` loại `.env` (token), thư viện, model, file tạm. Repo chỉ ~1.5 MB: code, `voices/*.npz`, `samples/*.ogg`.
Model (~0.6 GB) không nằm trong repo — bot tự tải khi khởi động lần đầu.

## 5. Tạo server trên BotKeep
1. Đăng ký / đăng nhập [botkeep.cloud](https://botkeep.cloud) (Founder Free: đăng ký trước **15/10, 17:00 giờ Ý**).
2. Tạo server mới → loại **Applications & bots** → runtime **Python 3.12** (hoặc 3.13).
3. **Tài nguyên: dồn gần hết pool cho server này** — RAM **2 GB** (tối thiểu ~1.8 GB), CPU **1.5 vCore**, storage **≥ 1.5 GB**.
4. Nguồn: **GitHub** → cấp quyền đọc repo private → chọn repo, nhánh `master`.
5. Start command: `python main.py`
6. **Environment** — thêm biến:
   - `TELEGRAM_BOT_TOKEN` = token từ BotFather (đánh dấu **secret/protected**)
   - `ALLOWED_USER_IDS` = user ID của bạn (lấy bằng `/id` ở bước 7; để trống lúc đầu)
   - Tuỳ chọn: `DEFAULT_VOICE`, `MAX_CHARS`, `MAX_FILE_CHARS`, `VIENEU_THREADS`, `VIENEU_PRECISION` (xem `.env.example`)
7. Bấm **Start** → mở **Console**.

## 6. Lần khởi động đầu (mất vài phút)
BotKeep tự cài `requirements.txt` (~240 MB) trước dòng `__BOTKEEP_PROCESS_START__`; bot chỉ cài thêm `vieneu`.
Console sẽ hiện lần lượt:
```
Successfully installed ... (BotKeep cài requirements.txt)
__BOTKEEP_PROCESS_START__
📦 ... pip install ... --no-deps vieneu==3.8.3   ← chỉ 4 MB
⏳ Loading Tiger-TTS v3 Turbo (ONNX, fp32, threads=2)...   ← tải model ~0.6 GB lần đầu
✅ Model ready in ...s
🎙️ Custom voices: ngoc_huyen_truyen
🤖 Bot polling (allowed users: everyone)
```
Những lần sau bỏ qua cài đặt và tải model.

## 7. Dùng và khoá bot
1. Trong Telegram mở bot → `/start` → gửi một câu → nhận voice message.
2. Gửi `/id` → copy user ID → BotKeep **Environment** → `ALLOWED_USER_IDS=<id>` (nhiều người: `id1,id2`) → **Restart**.
   Nên làm ngay — bot công khai ai cũng dùng được và sẽ chiếm hết CPU.

Gửi **file `.txt`** (vd. một chương truyện) → bot đọc từng đoạn, báo tiến độ, trả về 1 file MP3. Một chương ~14.000 ký tự ≈ 13 phút audio, ~7 MB; trên Mac M (2 threads) mất 5 phút, trên BotKeep lâu hơn (xem RTF trong Console). Giới hạn: `MAX_FILE_CHARS` (mặc định 40000).

Lệnh: `/voices` (danh sách giọng) · `/voice Mai Anh` (đổi giọng) · `/voice ngoc_huyen_truyen` (giọng custom).

`/test` — nghe audio thử giọng có sẵn trong `samples/` (gửi file sẵn, **không tốn CPU** của server):
`/test` (danh sách) · `/test 3` hoặc `/test Mai Anh` (một clip; gõ một phần tên như `/test đức` → mọi giọng khớp) · `/test all` (tất cả).
Có thể bỏ thêm file `.ogg` (OPUS → voice message) / `.mp3` / `.m4a` / `.wav` bất kỳ vào `samples/`. Xoá một clip rồi chạy lại `--make-samples` để tạo lại.

## 8. Tuỳ chỉnh tốc độ
- Console in dòng `🖥️ CPU: ... · VNNI: yes/no` khi khởi động. Nếu `yes`: `VIENEU_PRECISION=int8` → nhanh hơn ~1.6×, RAM ~0.9 GB. Nghe thử, nếu méo tiếng thì quay lại `fp32`.
- `MAX_CHARS` nhỏ hơn → mỗi lần chờ ít hơn. Bot xử lý **từng yêu cầu một** để không vượt 2 GB RAM.

## Xử lý sự cố
| Triệu chứng | Cách xử lý |
|---|---|
| Bot tự restart / `Killed` khi tạo giọng | Thiếu RAM → tăng RAM server lên 2 GB, dùng `int8` nếu CPU có VNNI, giảm `MAX_CHARS` |
| `pip install` lỗi mạng / quyền | BotKeep có thể chặn mạng ra ngoài hoặc tự cài `requirements.txt` theo cách khác → xem Console, hỏi support |
| `No space left on device` khi `pip install ... -r requirements.txt` | Thư mục `.deps` cũ còn sót lại từ lần lỗi → xoá `.deps` (File Manager) rồi Restart. Bot tự cài toàn bộ chỉ khi BotKeep chưa cài requirements |
| `No space left on device` khi tải model | Tăng storage server (cần ≥ 1.5 GB) |
| `Thiếu TELEGRAM_BOT_TOKEN` | Chưa thêm biến trong Environment |
| `Conflict: terminated by other getUpdates` | Bot đang chạy ở chỗ khác (ví dụ trên Mac) → tắt bản kia |
| Tiếng méo, rè | Đang dùng `int8` trên CPU không có VNNI → `VIENEU_PRECISION=fp32` |
