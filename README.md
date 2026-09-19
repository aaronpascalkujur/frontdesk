# Frontdesk

A voice front-end for [Mystin Office](https://github.com/aaronpascalkujur/mystin-office), an HTTP server that runs a roster of AI agents.

You speak a task. Frontdesk transcribes it on your machine, hands it to the office, and reads the agent's answer back when it lands.

```
Press Enter to speak, then Enter again to send.
[heard in 480ms] "summarize the notes from last week"
[frontdesk] On it.

[researcher · 22.4s · saved to 2026-09-19-summary.md]
Last week's notes cover three threads...

[frontdesk] researcher is done. Last week's notes cover three threads...
```

## Why it's built this way

**Speech never leaves the machine.** Transcription is [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base.en`, int8 on CPU) and synthesis is Piper (`piper-tts`), both running locally. A task is only sent anywhere once the office decides what to do with it — there's no cloud speech API in the path.

**Dispatch is threaded.** `run_task` blocks until the agent finishes, which can take minutes. Frontdesk runs each dispatch on a worker thread, so you can speak a second task while the first agent is still working. A lock around `say()` keeps two finished agents from talking over each other.

**Long answers are truncated out loud, not on disk.** Anything past 400 characters is cut at a word boundary when spoken, and Frontdesk tells you the filename the office wrote the full text to.

## Requirements

- Python 3.10+ (the code uses `X | None` annotations)
- A working microphone and speakers — `sounddevice` talks to PortAudio, so on Debian/Ubuntu you may need `sudo apt install libportaudio2`
- A Mystin Office server running at `http://127.0.0.1:4521`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Whisper model downloads itself on first run. The Piper voice does not — put `en_US-lessac-medium.onnx` and its `.json` config in `models/`:

```bash
mkdir -p models
# download en_US-lessac-medium (.onnx + .onnx.json) from
# https://huggingface.co/rhasspy/piper-voices into models/
```

Without that file Frontdesk falls back to `espeak-ng`, and without `espeak-ng` it just prints what it would have said. Both fallbacks are fine for testing the rest of the loop.

## Run

Start Mystin Office first (`npm start` in that repo), then:

```bash
python -m frontdesk
```

Press Enter to start recording, Enter again to send. Say "quit", "exit", "stop", or "goodbye" to shut down — Frontdesk waits for any in-flight agents before exiting. Ctrl-C does the same.

## Layout

```
frontdesk/
  core.py          turn loop, threaded dispatch, spoken-result truncation
  backend/
    mystin.py      HTTP client for Mystin Office (/api/agents, /api/task)
  speech/
    stt.py         mic capture + faster-whisper transcription
    tts.py         Piper synthesis with an espeak-ng fallback
```

## Configuration

There's no config file yet. The knobs are module constants:

| Constant | File | Default |
|---|---|---|
| `DEFAULT_BASE_URL` | `backend/mystin.py` | `http://127.0.0.1:4521` |
| `MODEL_SIZE` | `speech/stt.py` | `base.en` |
| `SAMPLE_RATE` | `speech/stt.py` | `16000` |
| `VOICE_NAME` | `speech/tts.py` | `en_US-lessac-medium` |
| `SPOKEN_RESULT_LIMIT` | `core.py` | `400` |
