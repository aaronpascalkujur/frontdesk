"""Microphone capture and local speech-to-text with faster-whisper."""

import queue
import sys
from typing import NamedTuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
MODEL_SIZE = "base.en"


class Heard(NamedTuple):
    """A transcript plus whisper's own read on how sure it was.

    `avg_logprob` approaches 0 when confident and falls below about -1 when the
    audio was unclear; `no_speech_prob` rises when it may have been silence.
    Both are journalled so bad transcriptions can be found after the fact.
    """

    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


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

    def transcribe(self, audio: np.ndarray) -> Heard:
        if audio.size == 0:
            return Heard("")
        # transcribe() streams segments lazily; materialise them so the text and
        # the confidence figures come from the same pass.
        segments = list(self.model.transcribe(audio, language="en", beam_size=1)[0])
        text = " ".join(s.text.strip() for s in segments).strip()
        if not segments:
            return Heard(text)
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        no_speech = max(s.no_speech_prob for s in segments)
        return Heard(text, round(avg_logprob, 3), round(no_speech, 3))
