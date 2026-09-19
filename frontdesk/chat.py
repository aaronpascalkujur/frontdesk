"""Tell small talk apart from real work.

A greeting should not cost a thirty-second agent run, so obvious openers get an
instant canned reply and everything else goes to a fast model that either
answers conversationally or hands the turn back for dispatch.
"""

import json
import re
import subprocess

TRIAGE_MODEL = "haiku"
TRIAGE_TIMEOUT = 20.0
TASK_SENTINEL = "TASK"

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


class Chatter:
    """Decides whether Frontdesk answers a turn itself."""

    def __init__(self, model: str = TRIAGE_MODEL, timeout: float = TRIAGE_TIMEOUT):
        self.model = model
        self.timeout = timeout

    def reply(self, text: str) -> str | None:
        """Return a spoken reply for small talk, or None to dispatch to an agent."""
        canned = SMALL_TALK.get(_normalize(text))
        if canned:
            return canned
        return self._triage(text)

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
