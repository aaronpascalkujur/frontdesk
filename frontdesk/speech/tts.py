"""Local text-to-speech via Piper, falling back to espeak-ng."""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

VOICE_DIR = Path(__file__).resolve().parents[2] / "models"
VOICE_NAME = "en_US-lessac-medium"
# How often the talking animation is fed while Piper speaks, in seconds.
FRAME = 1 / 30
INT16_FULL_SCALE = 32768.0


class Speaker:
    """Speaks text aloud, reporting amplitude as it goes.

    `on_level` receives the loudness of the audio actually being played, so a
    frontend animates against real speech rather than a canned loop. It is sent
    a final 0.0 once the utterance ends.
    """

    def __init__(
        self,
        voice_path: Path | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        path = voice_path or VOICE_DIR / f"{VOICE_NAME}.onnx"
        self.on_level = on_level
        self.voice = None
        if path.exists():
            from piper import PiperVoice

            self.voice = PiperVoice.load(str(path))

    def _report(self, level: float) -> None:
        if self.on_level is None:
            return
        try:
            self.on_level(level)
        except Exception:
            self.on_level = None  # a frontend that throws is not asked again

    @staticmethod
    def _envelope(audio: np.ndarray, rate: int) -> np.ndarray:
        """Per-frame loudness of the whole utterance, normalised to roughly 0..1."""
        hop = max(1, int(rate * FRAME))
        usable = (audio.size // hop) * hop
        if usable == 0:
            return np.zeros(0, dtype=np.float64)
        blocks = audio[:usable].astype(np.float64).reshape(-1, hop)
        return np.sqrt(np.mean(np.square(blocks), axis=1)) / INT16_FULL_SCALE

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
        audio = np.concatenate(chunks)
        rate = self.voice.config.sample_rate

        sd.play(audio, rate)
        # Walk the envelope against the wall clock rather than sleeping a fixed
        # step, so the animation cannot drift away from what is being heard.
        started = time.monotonic()
        for i, level in enumerate(self._envelope(audio, rate)):
            delay = (started + i * FRAME) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._report(float(level))
        sd.wait()
        self._report(0.0)

    @staticmethod
    def _espeak(text: str) -> None:
        if shutil.which("espeak-ng"):
            subprocess.run(["espeak-ng", text], check=False)
        else:
            print(f"[frontdesk] {text}")
