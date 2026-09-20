"""Tell small talk apart from real work.

A greeting should not cost a thirty-second agent run, so obvious openers get an
instant canned reply and everything else goes to a fast model that either
answers conversationally or hands the turn back for dispatch.
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from .journal import STATE_DIR, atomic_write_json

TRIAGE_MODEL = "haiku"
TRIAGE_TIMEOUT = 20.0
TASK_SENTINEL = "TASK"

LEARNED_PATH = STATE_DIR / "learned_replies.json"
# A cached answer only pays off for phrases short enough to be said again
# verbatim. Long utterances are near-unique, so caching them just grows the file.
MAX_LEARNED_WORDS = 8
MAX_LEARNED_CHARS = 60
MAX_LEARNED_ENTRIES = 500
# Whisper's avg_logprob for a turn; 0 is certain, below -1 is usually garbled.
# A shaky transcript still gets answered, it just never reaches the cache — a
# misheard phrase promoted once would be repeated back for good. Erring toward
# not caching costs nothing, so this sits above the worst observed misfire.
MIN_CACHE_CONFIDENCE = -0.75

_GREETING = "Hello. What can I get started for you?"
_IDENTITY = (
    "I'm Frontdesk. Tell me what you need and I hand it to the right agent in the "
    "office: writing, research, planning, analysis, or code."
)

SMALL_TALK = {
    "hi": _GREETING,
    "hey": _GREETING,
    "yo": _GREETING,
    "hello": _GREETING,
    "hi there": _GREETING,
    "hey there": _GREETING,
    "hello there": _GREETING,
    "morning": _GREETING,
    "good morning": _GREETING,
    "good afternoon": _GREETING,
    "good evening": _GREETING,
    "how are you": "Doing fine, thanks. What do you need?",
    "how are you doing": "Doing fine, thanks. What do you need?",
    "hows it going": "Going well. What can I put in front of an agent?",
    "how is it going": "Going well. What can I put in front of an agent?",
    "you there": "Still here. Go ahead.",
    "are you there": "Still here. Go ahead.",
    "can you hear me": "Loud and clear.",
    "thanks": "Any time.",
    "thank you": "Any time.",
    "thanks a lot": "Any time.",
    "thank you very much": "Any time.",
    "cheers": "Any time.",
    "ok": "Standing by.",
    "okay": "Standing by.",
    "cool": "Standing by.",
    "nice": "Standing by.",
    "great": "Standing by.",
    "perfect": "Standing by.",
    "alright": "Standing by.",
    "sure": "Standing by.",
    "got it": "Standing by.",
    "never mind": "No problem.",
    "nevermind": "No problem.",
    "who are you": _IDENTITY,
    "what are you": _IDENTITY,
    "what can you do": _IDENTITY,
    "what do you do": _IDENTITY,
    "whats your name": _IDENTITY,
    "what is your name": _IDENTITY,
}

SYSTEM_PROMPT = """You are Frontdesk, the voice receptionist for a small office of AI agents \
(writing, research, planning, analysis, code).

Classify the user's message:

- Small talk, greetings, thanks, or a question about you or the office: answer it \
yourself, warmly and briefly, in at most two sentences.
- Anything that needs actual work (research a topic, draft something, make a plan, \
analyse a decision, write code, answer a factual question): reply with exactly \
{sentinel} and nothing else.

Your reply is read aloud, so write plain spoken sentences with no markdown, lists, \
or code. When you are unsure which it is, reply {sentinel}.""".format(
    sentinel=TASK_SENTINEL
)


def _normalize(text: str) -> str:
    # Apostrophes drop out rather than splitting, so "how's" matches "hows".
    stripped = re.sub(r"['’]", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", stripped).split())


class Decision(NamedTuple):
    """`reply` is None when the turn belongs to an agent. `tier` says who decided."""

    reply: str | None
    tier: str  # table | cache | model | dispatch
    cached: bool = False


class ReplyCache:
    """Model-decided small talk, promoted to instant replies for next time.

    The expensive path answers a phrase once; every repeat is then a dict lookup.
    Entries are plain text in a plain JSON file, so a bad one is easy to spot and
    delete by hand.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else LEARNED_PATH
        self.entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # absent or corrupt: start empty rather than fail to boot
        if isinstance(raw, dict):
            self.entries = {
                k: v for k, v in raw.items() if isinstance(v, dict) and v.get("reply")
            }

    def get(self, key: str) -> str | None:
        entry = self.entries.get(key)
        return entry["reply"] if entry else None

    def cacheable(self, key: str) -> bool:
        return (
            bool(key)
            and key not in SMALL_TALK
            and len(key) <= MAX_LEARNED_CHARS
            and len(key.split()) <= MAX_LEARNED_WORDS
        )

    def remember(self, key: str, reply: str) -> bool:
        """Promote a model reply. Returns whether it was kept."""
        if not self.cacheable(key):
            return False
        self.entries[key] = {
            "reply": reply,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if len(self.entries) > MAX_LEARNED_ENTRIES:
            oldest = sorted(self.entries.items(), key=lambda kv: kv[1].get("at", ""))
            for key_to_drop, _ in oldest[: len(self.entries) - MAX_LEARNED_ENTRIES]:
                del self.entries[key_to_drop]
        try:
            atomic_write_json(self.path, self.entries)
        except OSError as e:
            print(f"[frontdesk] (could not save learned reply: {type(e).__name__}: {e})")
        return True


class Chatter:
    """Decides whether Frontdesk answers a turn itself."""

    def __init__(
        self,
        model: str = TRIAGE_MODEL,
        timeout: float = TRIAGE_TIMEOUT,
        cache: ReplyCache | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.cache = cache if cache is not None else ReplyCache()

    def reply(self, text: str, confidence: float | None = None) -> Decision:
        """Answer small talk, or return a dispatch decision for an agent.

        `confidence` is whisper's avg_logprob for the turn. It gates only whether
        the answer is remembered, never whether one is given.
        """
        key = _normalize(text)

        canned = SMALL_TALK.get(key)
        if canned:
            return Decision(canned, "table")

        learned = self.cache.get(key)
        if learned:
            return Decision(learned, "cache")

        answer = self._triage(text)
        if answer is None:
            return Decision(None, "dispatch")

        cached = False
        if confidence is None or confidence >= MIN_CACHE_CONFIDENCE:
            cached = self.cache.remember(key, answer)
        return Decision(answer, "model", cached)

    def _triage(self, text: str) -> str | None:
        try:
            proc = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--output-format", "json",
                    "--system-prompt", SYSTEM_PROMPT,
                    "--tools", "",
                    "--strict-mcp-config",
                    "--no-session-persistence",
                    "--model", self.model,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            payload = json.loads(proc.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

        if payload.get("is_error"):
            return None
        answer = str(payload.get("result", "")).strip()
        if not answer or TASK_SENTINEL in answer.upper():
            return None
        return answer
