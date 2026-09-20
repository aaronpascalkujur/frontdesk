# Frontdesk

A voice front-end for [Mystin Office](https://github.com/aaronpascalkujur/mystin-office), an HTTP server that runs a roster of AI agents.

You speak a task. Frontdesk transcribes it on your machine, hands it to the office, and reads the agent's answer back when it lands.

It runs in a terminal, or in a browser window that animates the microphone and the voice while it works.

```
Press Enter when you are ready to speak.
[listening — speak now, then press Enter]
[heard in 210ms] "hello"
[frontdesk] Hello. What can I get started for you?

Press Enter when you are ready to speak.
[listening — speak now, then press Enter]
[heard in 480ms] "summarize the notes from last week"
[frontdesk] On it.

[researcher · 22.4s · saved to 2026-09-19-summary.md]
Last week's notes cover three threads...

[frontdesk] researcher is done. Last week's notes cover three threads...
```

## Why it's built this way

**Speech never leaves the machine.** Transcription is [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base.en`, int8 on CPU) and synthesis is Piper (`piper-tts`), both running locally. A task is only sent anywhere once the office decides what to do with it — there's no cloud speech API in the path.

**Small talk never reaches an agent.** Saying "hello" used to cost a full agent run — thirty seconds for a greeting. Now every turn is triaged first: a table of common openers answers instantly with no network call, and anything else goes to a fast `haiku` model that either replies conversationally or hands the turn back for dispatch. Triage is deliberately biased toward dispatch — if the model is unsure, unreachable, or slow, the task goes to the office, so a real request is never swallowed by the chat layer.

**It gets faster at what you say often.** A phrase the model had to think about once is promoted into an instant reply, so the second time costs a dict lookup instead of a round trip — measured on real turns, 4.9s down to 0ms. Only short phrases are kept, learned entries can never shadow a built-in, and everything lands in `state/learned_replies.json` as plain text you can read and prune by hand.

**It refuses to learn from a bad transcript.** A turn whisper was unsure about still gets answered, but is never written to the cache. Without that gate a misheard phrase gets promoted once and then parroted back forever — which is exactly what happened with a clipped "how was your day" that reached the cache as "you".

**Every turn is journalled.** `state/journal.jsonl` records what was heard, how confident whisper was, which tier answered, and what the agent sent back. Nothing reads it yet; it exists so that triage and transcription can eventually be judged against what actually happened instead of guessed at.

**Dispatch is threaded.** `run_task` blocks until the agent finishes, which can take minutes. Frontdesk runs each dispatch on a worker thread, so you can speak a second task while the first agent is still working. A lock around `say()` keeps two finished agents from talking over each other.

**Long answers are truncated out loud, not on disk.** Anything past 400 characters is cut at a word boundary when spoken, and Frontdesk tells you the filename the office wrote the full text to.

**The loop does not know what it is drawn on.** It used to describe itself with `print()`, which only a terminal can read. Now it publishes events — state changes, transcripts, amplitude — and a frontend subscribes. The terminal frontend prints them; the browser frontend animates them. Adding the UI changed no turn logic, and the console path still runs with no server at all.

**The animation is driven by real audio, not a loop.** The microphone reports its actual input level while the stream is open, and Piper reports the loudness of the samples it is playing. The orb is following the sound you are making or hearing. The same honesty applies to the states: "listening" is published from inside the open audio stream, so the UI cannot claim the mic is live a moment before it is — the bug that once ate the first word of a sentence.

**The browser is presentation only.** The microphone, whisper and Piper all stay in Python, so there is no browser audio permission in the path and nothing about the speech-stays-local guarantee changes. The page talks to a standard-library HTTP server over Server-Sent Events, which is why the UI adds no dependency to `requirements.txt`.

## Requirements

- Python 3.10+ (the code uses `X | None` annotations)
- A working microphone and speakers — `sounddevice` talks to PortAudio, so on Debian/Ubuntu you may need `sudo apt install libportaudio2`
- A Mystin Office server running at `http://127.0.0.1:4521`
- A browser, only for `--ui`. The interface needs no extra packages — it is served by the standard library
- The `claude` CLI on your `PATH` and logged in — Mystin Office already needs it, and Frontdesk shells out to it for small-talk triage. Without it, triage is skipped and every turn goes to an agent

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

Start Mystin Office first (`npm start` in that repo), then pick a frontend.

**In a browser:**

```bash
python -m frontdesk --ui
```

This opens `http://127.0.0.1:4520`. Press space or click the orb to open the microphone, then press space again to send. The orb tracks your voice while you speak and the agent's while it answers; a dispatched task gets a card with a running clock until the answer lands. The terminal keeps a plain transcript alongside it.

`--no-browser` skips opening a window, `--port` moves the server, and `--host 0.0.0.0` exposes it beyond this machine — which also hands anyone on your network a button that turns your microphone on, so only do that deliberately.

**In a terminal:**

```bash
python -m frontdesk
```

Press Enter, wait for `[listening]`, then speak and press Enter again to send. Don't start talking before the cue — the mic is not open until it appears, and a clipped opening word wrecks the transcription ("how was your day" becomes "you"). The browser UI shows the same moment as the orb turning cyan.

Either way, say "quit", "exit", "stop", or "goodbye" to shut down — Frontdesk waits for any in-flight agents before exiting. Ctrl-C does the same.

## Layout

```
frontdesk/
  core.py          turn loop, threaded dispatch, spoken-result truncation
  chat.py          small-talk triage: table, learned cache, then fast model
  events.py        event bus, push-to-talk gates, state names
  console.py       terminal frontend — prints the events
  journal.py       append-only turn log + atomic JSON writes
  backend/
    mystin.py      HTTP client for Mystin Office (/api/agents, /api/task)
  speech/
    stt.py         mic capture + faster-whisper transcription
    tts.py         Piper synthesis with an espeak-ng fallback
  ui/              browser frontend — animates the same events
    server.py      stdlib HTTP + Server-Sent Events
    launch.py      wires the loop to the browser
    static/        index.html, style.css, app.js (canvas orb)
tests/
  test_frontdesk.py
state/             gitignored, created on first run
  journal.jsonl        one record per turn
  learned_replies.json phrases promoted out of the model path
```

`core.py` imports neither frontend. It emits; they subscribe.

Run the tests with:

```bash
python -m unittest discover tests
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
| `TRIAGE_MODEL` | `chat.py` | `haiku` |
| `SMALL_TALK` | `chat.py` | table of instant replies |
| `MAX_LEARNED_ENTRIES` | `chat.py` | `500` |
| `MAX_LEARNED_WORDS` | `chat.py` | `8` |
| `MIN_CACHE_CONFIDENCE` | `chat.py` | `-0.75` |
| `STATE_DIR` | `journal.py` | `state/` beside the package |
| `DEFAULT_PORT` | `ui/server.py` | `4520` |
| `HEARTBEAT` | `ui/server.py` | `15.0` seconds |
| `BLOCK_SIZE` | `speech/stt.py` | `512` (~31 level updates/sec) |

To forget everything learned so far, delete `state/learned_replies.json`. To
forget one bad reply, open it and remove that entry — the file is a flat map of
phrase to answer.
