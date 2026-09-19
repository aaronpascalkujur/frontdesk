"""Microphone capture and local speech-to-text with faster-whisper."""

import queue
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
MODEL_SIZE = "base.en"


class Listener:
    def __init__(self, model_size: str = MODEL_SIZE, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def record_until_enter(self) -> np.ndarray:
        """Push-to-talk: capture from the default mic until the user presses Enter."""
        frames: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            frames.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback
        ):
            input()

        chunks = []
        while not frames.empty():
            chunks.append(frames.get())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()
