"""Microphone capture and local speech-to-text with faster-whisper."""

import queue
import sys
from typing import Callable, NamedTuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
MODEL_SIZE = "base.en"
# 512 frames at 16kHz is a 32ms block, so level callbacks arrive ~31 times a
# second — enough to animate a waveform smoothly without costing anything.
BLOCK_SIZE = 512


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
    """Captures speech. Knows nothing about how the wait is signalled or drawn.

    `on_open` fires the instant the stream is live and `on_level` reports input
    amplitude while it stays open, so a frontend can show the microphone state
    truthfully instead of guessing at it.
    """

    def __init__(
        self,
        model_size: str = MODEL_SIZE,
        sample_rate: int = SAMPLE_RATE,
        on_level: Callable[[float], None] | None = None,
        on_open: Callable[[], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.on_level = on_level
        self.on_open = on_open
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._warm_up()

    def _warm_up(self) -> None:
        """Open and close the mic once at startup.

        The first open of a cold device costs tens of milliseconds that come out
        of whatever the user says first. Spend it now, before anyone is talking.
        """
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32"
            ):
                pass
        except Exception as e:  # no mic yet is not fatal; recording will say so
            print(f"[audio] could not warm up input device: {e}", file=sys.stderr)

    def _report(self, block: np.ndarray) -> None:
        if self.on_level is None:
            return
        try:
            rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        except (ValueError, FloatingPointError):
            return
        if rms != rms:  # NaN from a device glitch
            return
        try:
            self.on_level(rms)
        except Exception:
            self.on_level = None  # a frontend that throws is not asked again

    def record_until(self, gate) -> np.ndarray:
        """Push-to-talk: capture from the default mic until `gate.wait()` returns.

        The gate is whatever ends the turn — Enter on a terminal, a click in the
        UI. Recording stops either way, including when the gate is closed out
        from under us during shutdown.
        """
        frames: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            frames.put(indata.copy())
            self._report(indata)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        ):
            # Announced only once the device is actually live. Without this cue
            # there is a window where the user is already talking and nothing is
            # being recorded, which costs the opening words of the sentence.
            if self.on_open:
                self.on_open()
            gate.wait()

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
