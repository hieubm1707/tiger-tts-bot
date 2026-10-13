"""Run on your Mac (NOT on BotKeep — cloning peaks at ~2.1 GB RAM).

Encodes a 3–8 s reference clip into voices/<name>.npz (~12 KB), which the bot loads
at start-up and offers as a voice. Use a clean, single-speaker clip.

    cd ../Tiger-TTS && uv run python ../vieneu-telegram-bot/make_voice.py my_clip.wav my_voice
"""
import sys
from pathlib import Path

import numpy as np
from vieneu import Vieneu

if len(sys.argv) != 3:
    sys.exit("Usage: python make_voice.py <clip.wav> <voice_name>")
clip, name = sys.argv[1], sys.argv[2]
out = Path(__file__).resolve().parent / "voices" / f"{name}.npz"
out.parent.mkdir(exist_ok=True)
tts = Vieneu()
emb, codes = tts.encode_reference(clip, denoise=True)
np.savez(out, speaker_emb=emb, codes=codes)
tts.save(tts.infer("Xin chào, đây là giọng vừa được tạo cho bot.", voice={"speaker_emb": emb, "codes": codes}),
         str(out.with_suffix(".preview.wav")))
print(f"✅ {out} (+ preview: {out.with_suffix('.preview.wav').name} — nghe thử, không cần upload)")
