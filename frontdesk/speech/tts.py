"""Local text-to-speech via Piper, falling back to espeak-ng."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd

VOICE_DIR = Path(__file__).resolve().parents[2] / "models"
VOICE_NAME = "en_US-lessac-medium"


class Speaker:
    def __init__(self, voice_path: Path | None = None):
        path = voice_path or VOICE_DIR / f"{VOICE_NAME}.onnx"
        self.voice = None
        if path.exists():
            from piper import PiperVoice

            self.voice = PiperVoice.load(str(path))

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.voice is None:
            self._espeak(text)
            return

        chunks = [
            np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
            for c in self.voice.synthesize(text)
        ]
        if not chunks:
            return
        sd.play(np.concatenate(chunks), self.voice.config.sample_rate)
        sd.wait()

    @staticmethod
    def _espeak(text: str) -> None:
        if shutil.which("espeak-ng"):
            subprocess.run(["espeak-ng", text], check=False)
        else:
            print(f"[frontdesk] {text}")
