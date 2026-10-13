"""Tiger-TTS Telegram bot for BotKeep (Founder Free: 2 GB RAM, 1.5 vCore, 2 GB disk).

Start command: python main.py
Env (BotKeep → Environment): TELEGRAM_BOT_TOKEN (required), see .env.example for the rest.

First start installs the dependencies into ./.deps and downloads the v3 Turbo model
(~580 MB) into ./hf_cache, so it takes a few minutes. Later starts reuse both.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
sys.path.insert(0, str(DEPS))
# Keep the model cache inside the project (counts toward the 2 GB, survives restarts).
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# hub 2.x's shared blob store shards files by hash, so a .onnx and its external .data
# resolve to different dirs and onnxruntime>=1.31 rejects it ("path escapes model directory").
os.environ.setdefault("HF_HUB_DISABLE_SHARED_BLOBS", "1")
# BotKeep (Pterodactyl) mounts /tmp as a small tmpfs: pip unpacking there and the .txt → MP3
# temp files run out of space, so keep temp files on the server disk instead.
(ROOT / ".tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(ROOT / ".tmp")


def _pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
           "--target", str(DEPS), *args]
    print("📦", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_deps() -> None:
    """Install only what is missing. BotKeep already pip-installs requirements.txt (into
    ~/.local) before starting us, so there only `vieneu` goes in — with --no-deps, to skip
    gradio/librosa (~500 MB)."""
    import importlib
    try:
        import telegram, dotenv, onnxruntime, sea_g2p  # noqa: F401
    except ImportError:  # local runs / hosts that don't install requirements.txt themselves
        _pip("-r", str(ROOT / "requirements.txt"))
        importlib.invalidate_caches()
    try:
        import vieneu  # noqa: F401
    except ImportError:
        _pip("--no-deps", "vieneu==3.8.3")
        importlib.invalidate_caches()


ensure_deps()

import asyncio  # noqa: E402
import functools  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from telegram import Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.error import TelegramError  # noqa: E402
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters  # noqa: E402

load_dotenv(ROOT / ".env")  # BotKeep manages .env for application profiles

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
for _n in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("tiger-bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
THREADS = int(os.getenv("VIENEU_THREADS", "2"))
PRECISION = os.getenv("VIENEU_PRECISION", "fp32")  # "int8" only if the CPU has VNNI
MAX_CHARS = int(os.getenv("MAX_CHARS", "600"))
MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "40000"))  # .txt upload; ~35 min of audio
# Max characters per model chunk (vieneu packs whole sentences up to this). Peak RAM grows with
# it: a 3 KB file measured +1.3 GB at vieneu's default 256, +0.45 GB at 128, +0.3 GB at 96.
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "128"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Ngọc Huyền")
ALLOWED_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
VOICES_DIR = ROOT / "voices"
PREFS_FILE = ROOT / "user_voices.json"
SAMPLE_TEXT = "Xin chào, đây là đoạn thử giọng. Chúc bạn nghe truyện vui vẻ."  # for /test

tts = None
custom_voices: dict = {}   # name -> {"speaker_emb", "codes"} from voices/*.npz
user_voice: dict = {}      # chat id (str) -> voice name
synth_lock = asyncio.Lock()  # one synthesis at a time: two in parallel would blow the 2 GB RAM
file_jobs: set = set()      # chat ids with a .txt job running (one per chat)
PARA_GAP = 0.5               # seconds of silence between paragraphs of a .txt file
PROGRESS_EVERY = 15          # seconds between progress edits of a .txt job
PROGRESS_NOTE = f"Tiến trình sẽ được cập nhật mỗi {PROGRESS_EVERY}s, vui lòng đợi"


def load_model() -> None:
    global tts
    from vieneu import Vieneu
    try:  # int8 needs VNNI; report it so you can decide on VIENEU_PRECISION
        info = Path("/proc/cpuinfo").read_text()
        model = next((l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("model name")), "?")
        log.info("🖥️ CPU: %s · VNNI: %s", model, "yes" if "vnni" in info else "no")
    except OSError:
        pass
    t0 = time.time()
    log.info("⏳ Loading Tiger-TTS v3 Turbo (ONNX, %s, threads=%d)...", PRECISION, THREADS)
    tts = Vieneu(threads=THREADS, precision=PRECISION)
    log.info("✅ Model ready in %.1fs", time.time() - t0)
    for f in sorted(VOICES_DIR.glob("*.npz")):
        d = np.load(f)
        custom_voices[f.stem] = {"speaker_emb": d["speaker_emb"], "codes": d["codes"]}
    if custom_voices:
        log.info("🎙️ Custom voices: %s", ", ".join(custom_voices))


def load_prefs() -> None:
    try:
        user_voice.update(json.loads(PREFS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass


def save_prefs() -> None:
    try:
        PREFS_FILE.write_text(json.dumps(user_voice, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log.warning("Could not save prefs: %s", e)


def all_voice_names() -> list:
    return list(custom_voices) + [vid for _, vid in tts.list_preset_voices()]


def resolve_voice(name: str):
    """Return a value for `infer(voice=...)`, or None if unknown."""
    if name in custom_voices:
        return custom_voices[name]
    resolved = tts.resolve_voice_name(name)
    return resolved if resolved else None


def synthesize_ogg(text: str, voice) -> tuple:
    t0 = time.time()
    audio = tts.infer(text, voice=voice, max_chars=CHUNK_CHARS)
    buf = io.BytesIO()
    sf.write(buf, audio, tts.sample_rate, format="OGG", subtype="OPUS")
    buf.seek(0)
    dur = len(audio) / tts.sample_rate
    gen = time.time() - t0
    log.info("🔊 %d chars → %.1fs audio in %.1fs (RTF %.2f)", len(text), dur, gen, gen / max(dur, 1e-6))
    return buf, dur


def synthesize_append(out: sf.SoundFile, text: str, voice) -> float:
    """Synthesize one paragraph and append it (+ a pause) to an open file. Returns seconds written."""
    audio = tts.infer(text, voice=voice, max_chars=CHUNK_CHARS)
    out.write(audio)
    out.write(np.zeros(int(PARA_GAP * tts.sample_rate), dtype=np.float32))
    return len(audio) / tts.sample_rate + PARA_GAP


def fmt_left(seconds: float) -> str:
    return f"~{seconds:.0f} giây" if seconds < 60 else f"~{seconds / 60:.0f} phút"


async def set_status(status, text: str) -> None:
    """Progress edits are best-effort: a network hiccup must not kill a long .txt job."""
    try:
        await status.edit_text(text)
    except TelegramError as e:
        log.warning("Status update failed (job continues): %s", e)


def allowed(update: Update) -> bool:
    return not ALLOWED_IDS or (update.effective_user and update.effective_user.id in ALLOWED_IDS)


def logged(handler):
    """Log every command: `✅ <user_id> /<command> success in <s>`, or ❌ with the error."""
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user.id if update.effective_user else "?"
        msg = update.effective_message
        cmd = (msg.text or "").split(maxsplit=1)[0].split("@")[0] if msg else handler.__name__
        t0 = time.time()
        try:
            await handler(update, context)
        except Exception as e:
            log.error("❌ %s %s failed in %.2fs: %s", user, cmd, time.time() - t0, e)
            raise
        log.info("✅ %s %s success in %.2fs", user, cmd, time.time() - t0)
    return wrapper


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        await update.message.reply_text(f"⛔ Bạn chưa được cấp quyền. User ID: {update.effective_user.id}")
        return
    voice = user_voice.get(str(update.effective_chat.id), DEFAULT_VOICE)
    await update.message.reply_text(
        "🦜 Tiger-TTS bot\n\n"
        "• Gửi văn bản tiếng Việt → nhận voice message.\n"
        "• Gửi file .txt (vd. một chương truyện) → nhận file MP3.\n"
        "• /voices — danh sách giọng\n"
        "• /voice <tên> — đổi giọng, ví dụ: /voice Mai Anh\n"
        "• /test <tên> — nghe thử một giọng bằng câu mẫu ngắn\n"
        "• /id — xem user ID\n\n"
        f"Giọng hiện tại: {voice} · Tối đa {MAX_CHARS} ký tự/lần."
    )


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"User ID: {update.effective_user.id}")


async def cmd_voices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    lines = [f"🎙️ {n} (custom)" for n in custom_voices]
    lines += [label for label, _ in tts.list_preset_voices()]
    await update.message.reply_text("\n".join(lines) + "\n\nĐổi giọng: /voice <tên>")


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    name = " ".join(context.args).strip()
    if not name:
        cur = user_voice.get(str(update.effective_chat.id), DEFAULT_VOICE)
        await update.message.reply_text(f"Giọng hiện tại: {cur}\nDùng: /voice <tên> (xem /voices)")
        return
    if resolve_voice(name) is None:
        await update.message.reply_text(f"❌ Không có giọng '{name}'. Xem /voices")
        return
    user_voice[str(update.effective_chat.id)] = name
    save_prefs()
    await update.message.reply_text(f"✅ Đã chọn giọng: {name}")


def nfc(s: str) -> str:  # macOS/iOS keyboards can send NFD, voice names are NFC
    return unicodedata.normalize("NFC", s)


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/test lists the voices, /test <số|tên> reads SAMPLE_TEXT with that voice."""
    if not allowed(update):
        return
    names = all_voice_names()
    arg = " ".join(context.args).strip()
    if not arg:
        lines = [f"{i}. {n}" for i, n in enumerate(names, 1)]
        await update.message.reply_text("🎧 Nghe thử giọng:\n" + "\n".join(lines)
                                        + "\n\nDùng: /test <số hoặc tên>, ví dụ /test 3 hoặc /test Mai Anh")
        return
    if arg.isdigit() and 1 <= int(arg) <= len(names):
        picked = [names[int(arg) - 1]]
    else:
        key = nfc(arg).casefold()
        picked = ([n for n in names if n.casefold() == key]
                  or [n for n in names if key in n.casefold()])
    if not picked:
        await update.message.reply_text(f"❌ Không có giọng '{arg}'. Xem /test")
        return
    if len(picked) > 1:
        await update.message.reply_text(f"🔎 Có {len(picked)} giọng khớp '{arg}':\n" + "\n".join(picked)
                                        + "\n\nGõ tên đầy đủ, ví dụ /test " + picked[0])
        return
    name = picked[0]
    if synth_lock.locked():
        await update.message.reply_text("⏳ Đang xử lý yêu cầu khác, bạn chờ chút nhé...")
    async with synth_lock:
        await update.effective_chat.send_action(ChatAction.RECORD_VOICE)
        try:
            buf, dur = await asyncio.to_thread(synthesize_ogg, SAMPLE_TEXT, resolve_voice(name))
        except Exception as e:
            log.exception("Test synthesis failed")
            await update.message.reply_text(f"❌ Lỗi khi tạo giọng: {e}")
            return
    await update.message.reply_voice(voice=buf, duration=int(round(dur)), caption=f"🎧 {name}")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        await update.message.reply_text(f"⛔ Bạn chưa được cấp quyền. User ID: {update.effective_user.id}")
        return
    text = (update.message.text or "").strip()
    if not text:
        return
    if len(text) > MAX_CHARS:
        await update.message.reply_text(f"✂️ Văn bản dài {len(text)} ký tự, tối đa {MAX_CHARS}. Hãy chia nhỏ.")
        return
    name = user_voice.get(str(update.effective_chat.id), DEFAULT_VOICE)
    voice = resolve_voice(name) or resolve_voice(DEFAULT_VOICE)
    if synth_lock.locked():
        await update.message.reply_text("⏳ Đang xử lý yêu cầu khác, bạn chờ chút nhé...")
    async with synth_lock:
        await update.effective_chat.send_action(ChatAction.RECORD_VOICE)
        try:
            buf, dur = await asyncio.to_thread(synthesize_ogg, text, voice)
        except Exception as e:  # keep the bot alive on a bad input
            log.exception("Synthesis failed")
            await update.message.reply_text(f"❌ Lỗi khi tạo giọng: {e}")
            return
    await update.message.reply_voice(voice=buf, duration=int(round(dur)), caption=f"🎙️ {name}")


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A .txt file → one MP3. Paragraphs are synthesized one by one and streamed to disk,
    so RAM stays flat however long the file is; the lock is taken per paragraph so short
    text messages can still get through while a long file is running."""
    if not allowed(update):
        await update.message.reply_text(f"⛔ Bạn chưa được cấp quyền. User ID: {update.effective_user.id}")
        return
    doc = update.message.document
    chat_id = update.effective_chat.id
    if chat_id in file_jobs:
        await update.message.reply_text("⏳ Đang đọc một file khác cho bạn, đợi xong rồi gửi tiếp nhé.")
        return
    raw = bytes(await (await doc.get_file()).download_as_bytearray())
    text = raw.decode("utf-8-sig", errors="replace")
    paras = [l.strip() for l in text.splitlines() if l.strip()]
    total = sum(len(p) for p in paras)
    if not paras:
        await update.message.reply_text("❌ File rỗng.")
        return
    if total > MAX_FILE_CHARS:
        await update.message.reply_text(f"✂️ File dài {total} ký tự, tối đa {MAX_FILE_CHARS}. Hãy chia nhỏ.")
        return
    name = user_voice.get(str(chat_id), DEFAULT_VOICE)
    voice = resolve_voice(name) or resolve_voice(DEFAULT_VOICE)
    title = Path(doc.file_name or "audio.txt").stem
    status = await update.message.reply_text(f"📖 {title}: {len(paras)} đoạn, {total} ký tự. Bắt đầu đọc...\n"
                                             f"{PROGRESS_NOTE}")
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    file_jobs.add(chat_id)
    t0 = last_edit = time.time()
    dur = 0.0
    done = 0
    try:
        with sf.SoundFile(path, "w", samplerate=tts.sample_rate, channels=1,
                          format="MP3", subtype="MPEG_LAYER_III") as out:
            for i, p in enumerate(paras, 1):
                async with synth_lock:
                    dur += await asyncio.to_thread(synthesize_append, out, p, voice)
                done += len(p)
                if time.time() - last_edit > PROGRESS_EVERY and i < len(paras):
                    left = (time.time() - t0) / done * (total - done)
                    await set_status(status, f"📖 {title}: {i}/{len(paras)} đoạn ({done * 100 // total}%) "
                                             f"· còn {fmt_left(left)}\n{PROGRESS_NOTE}")
                    last_edit = time.time()
        gen = time.time() - t0
        log.info("📖 %s: %d chars → %.1fs audio in %.1fs (RTF %.2f)", title, total, dur, gen, gen / dur)
        await set_status(status, f"📤 {title}: xong {dur / 60:.1f} phút audio sau {gen / 60:.1f} phút, đang gửi...")
        with open(path, "rb") as f:
            await update.message.reply_audio(audio=f, filename=f"{title}.mp3", title=title, performer=name,
                                             duration=int(round(dur)), write_timeout=300)
        try:
            await status.delete()
        except TelegramError:
            pass  # the MP3 is already sent; a leftover status message is harmless
    except Exception as e:  # keep the bot alive on a bad input
        log.exception("File synthesis failed")
        await update.message.reply_text(f"❌ Lỗi khi đọc file: {e}")
    finally:
        file_jobs.discard(chat_id)
        os.unlink(path)


def selftest() -> None:
    """`python main.py --selftest`: synthesize one sentence to selftest.ogg, no Telegram."""
    load_model()
    voice = resolve_voice(DEFAULT_VOICE)
    buf, dur = synthesize_ogg("Xin chào, đây là bài kiểm tra của bot VieNeu.", voice)
    (ROOT / "selftest.ogg").write_bytes(buf.getvalue())
    print(f"✅ selftest.ogg ({dur:.1f}s) — voices: {', '.join(all_voice_names()[:8])}...")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not TOKEN:
        sys.exit("❌ Thiếu TELEGRAM_BOT_TOKEN (BotKeep → Environment).")
    load_model()
    load_prefs()
    if resolve_voice(DEFAULT_VOICE) is None:
        sys.exit(f"❌ DEFAULT_VOICE '{DEFAULT_VOICE}' không tồn tại. Có: {all_voice_names()}")
    app = Application.builder().token(TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], logged(cmd_start)))
    app.add_handler(CommandHandler("id", logged(cmd_id)))
    app.add_handler(CommandHandler("voices", logged(cmd_voices)))
    app.add_handler(CommandHandler("voice", logged(cmd_voice)))
    app.add_handler(CommandHandler("test", logged(cmd_test)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.TXT | filters.Document.FileExtension("txt"), on_document))
    log.info("🤖 Bot polling (allowed users: %s)", sorted(ALLOWED_IDS) or "everyone")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
