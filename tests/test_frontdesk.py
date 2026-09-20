"""Tests for journalling, reply caching, triage tiers and dispatch resilience.

Run with: python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import frontdesk.core as core
from frontdesk.backend import MystinError
from frontdesk.chat import SMALL_TALK, Chatter, ReplyCache, _normalize
from frontdesk.journal import Journal
from frontdesk.speech.stt import Heard


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


# --------------------------------------------------------------------------- journal


class TestJournal(Tmp):
    def test_one_parseable_json_line_per_write(self):
        j = Journal(self.tmp / "j.jsonl")
        j.write("turn", route="chat", heard="hi")
        j.write("result", turn=1, agent="Researcher")

        lines = (self.tmp / "j.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        first, second = (json.loads(line) for line in lines)
        self.assertEqual(first["event"], "turn")
        self.assertEqual(first["route"], "chat")
        self.assertEqual(second["agent"], "Researcher")
        self.assertIn("at", first)
        self.assertEqual(first["session"], second["session"])

    def test_turn_ids_increment(self):
        j = Journal(self.tmp / "j.jsonl")
        self.assertEqual([j.next_turn_id() for _ in range(3)], [1, 2, 3])

    def test_creates_missing_parent_directory(self):
        j = Journal(self.tmp / "deep" / "nested" / "j.jsonl")
        j.write("turn")
        self.assertTrue((self.tmp / "deep" / "nested" / "j.jsonl").exists())

    def test_unwritable_path_does_not_raise(self):
        blocker = self.tmp / "afile"
        blocker.write_text("not a directory")
        j = Journal(blocker / "sub" / "j.jsonl")
        j.write("turn", heard="hi")  # must degrade, not explode

    def test_non_serialisable_values_do_not_raise(self):
        j = Journal(self.tmp / "j.jsonl")
        j.write("turn", weird=object())
        self.assertEqual(len((self.tmp / "j.jsonl").read_text().strip().splitlines()), 1)


# ----------------------------------------------------------------------- reply cache


class TestReplyCache(Tmp):
    def cache(self):
        return ReplyCache(self.tmp / "learned.json")

    def test_remember_then_get(self):
        c = self.cache()
        self.assertTrue(c.remember("hey buddy", "Hello there."))
        self.assertEqual(c.get("hey buddy"), "Hello there.")

    def test_persists_across_instances(self):
        self.cache().remember("hey buddy", "Hello there.")
        self.assertEqual(self.cache().get("hey buddy"), "Hello there.")

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.cache().get("never seen"))

    def test_rejects_long_phrase(self):
        c = self.cache()
        long_phrase = " ".join(["word"] * 30)
        self.assertFalse(c.remember(long_phrase, "nope"))
        self.assertIsNone(c.get(long_phrase))

    def test_rejects_shadowing_a_builtin(self):
        c = self.cache()
        self.assertFalse(c.remember("hello", "Something else entirely."))
        self.assertIsNone(c.get("hello"))

    def test_rejects_empty_key(self):
        self.assertFalse(self.cache().remember("", "x"))

    def test_corrupt_file_starts_empty_rather_than_crashing(self):
        (self.tmp / "learned.json").write_text("{not json at all")
        self.assertEqual(self.cache().entries, {})

    def test_malformed_entries_are_dropped(self):
        (self.tmp / "learned.json").write_text(
            json.dumps({"good": {"reply": "ok"}, "bad": "a string", "empty": {}})
        )
        c = self.cache()
        self.assertEqual(c.get("good"), "ok")
        self.assertIsNone(c.get("bad"))
        self.assertIsNone(c.get("empty"))

    def test_evicts_oldest_at_cap(self):
        import frontdesk.chat as chat_mod

        original = chat_mod.MAX_LEARNED_ENTRIES
        chat_mod.MAX_LEARNED_ENTRIES = 3
        self.addCleanup(setattr, chat_mod, "MAX_LEARNED_ENTRIES", original)

        c = self.cache()
        for i in range(6):
            c.entries[f"phrase {i}"] = {"reply": str(i), "at": f"2026-01-0{i + 1}"}
        c.remember("newest one", "kept")
        self.assertLessEqual(len(c.entries), 3)
        self.assertEqual(c.get("newest one"), "kept")
        self.assertIsNone(c.get("phrase 0"))  # oldest gone

    def test_written_file_is_valid_json(self):
        c = self.cache()
        c.remember("hey buddy", "Hello there.")
        payload = json.loads((self.tmp / "learned.json").read_text())
        self.assertEqual(payload["hey buddy"]["reply"], "Hello there.")
        self.assertIn("at", payload["hey buddy"])


# ---------------------------------------------------------------------- triage tiers


class StubChatter(Chatter):
    """Chatter with the model call stubbed, so tiers are testable offline."""

    def __init__(self, answer, cache):
        super().__init__(cache=cache)
        self.answer = answer
        self.calls = 0

    def _triage(self, text):
        self.calls += 1
        return self.answer


class TestChatterTiers(Tmp):
    def cache(self):
        return ReplyCache(self.tmp / "learned.json")

    def test_builtin_table_answers_without_calling_model(self):
        c = StubChatter("should not be used", self.cache())
        d = c.reply("Hello.")
        self.assertEqual(d.tier, "table")
        self.assertEqual(d.reply, SMALL_TALK["hello"])
        self.assertEqual(c.calls, 0)

    def test_model_reply_is_promoted_then_served_from_cache(self):
        cache = self.cache()
        c = StubChatter("Not much, you?", cache)

        first = c.reply("what's new")
        self.assertEqual(first.tier, "model")
        self.assertEqual(c.calls, 1)

        second = c.reply("what's new")
        self.assertEqual(second.tier, "cache")
        self.assertEqual(second.reply, "Not much, you?")
        self.assertEqual(c.calls, 1)  # the whole point: no second model call

    def test_promotion_survives_restart(self):
        StubChatter("Not much, you?", self.cache()).reply("what's new")
        fresh = StubChatter("different answer", self.cache())
        d = fresh.reply("what's new")
        self.assertEqual(d.tier, "cache")
        self.assertEqual(d.reply, "Not much, you?")
        self.assertEqual(fresh.calls, 0)

    def test_task_falls_through_to_dispatch_and_caches_nothing(self):
        cache = self.cache()
        c = StubChatter(None, cache)
        d = c.reply("summarise last week's notes")
        self.assertIsNone(d.reply)
        self.assertEqual(d.tier, "dispatch")
        self.assertEqual(cache.entries, {})

    def test_shaky_transcript_is_answered_but_not_cached(self):
        """A misheard phrase must never be promoted, or it repeats forever."""
        cache = self.cache()
        c = StubChatter("Hi there, I'm Frontdesk.", cache)
        d = c.reply("you", confidence=-0.99)  # the real-world misfire
        self.assertEqual(d.reply, "Hi there, I'm Frontdesk.")  # still answered
        self.assertEqual(d.tier, "model")
        self.assertFalse(d.cached)
        self.assertEqual(cache.entries, {})

    def test_confident_transcript_is_cached(self):
        cache = self.cache()
        c = StubChatter("Quiet so far.", cache)
        d = c.reply("how was your day", confidence=-0.47)
        self.assertTrue(d.cached)
        self.assertEqual(cache.get("how was your day"), "Quiet so far.")

    def test_borderline_confidence_is_rejected(self):
        cache = self.cache()
        c = StubChatter("hello", cache)
        # -0.822 was observed on a genuinely misheard turn
        self.assertFalse(c.reply("you", confidence=-0.822).cached)

    def test_missing_confidence_still_caches(self):
        cache = self.cache()
        c = StubChatter("Sure thing.", cache)
        self.assertTrue(c.reply("what's new", confidence=None).cached)

    def test_long_chat_reply_is_answered_but_not_cached(self):
        cache = self.cache()
        c = StubChatter("Sure.", cache)
        phrase = "this is a very long conversational aside that will not recur"
        d = c.reply(phrase)
        self.assertEqual(d.tier, "model")
        self.assertIsNone(cache.get(_normalize(phrase)))


class TestNormalisation(unittest.TestCase):
    def test_contractions_and_punctuation(self):
        self.assertEqual(_normalize("How's it going?"), "hows it going")
        self.assertEqual(_normalize("  HELLO!!  "), "hello")

    def test_every_builtin_key_is_reachable(self):
        unreachable = [k for k in SMALL_TALK if _normalize(k) != k]
        self.assertEqual(unreachable, [])

    def test_quit_words_are_not_small_talk(self):
        for word in core.QUIT_WORDS:
            self.assertNotIn(word, SMALL_TALK)


class TestHeard(unittest.TestCase):
    def test_confidence_fields_default_to_none(self):
        h = Heard("hello")
        self.assertEqual(h.text, "hello")
        self.assertIsNone(h.avg_logprob)
        self.assertIsNone(h.no_speech_prob)


# ----------------------------------------------------------------------- turn routing


class FakeListener:
    def __init__(self, heard):
        self.heard = heard

    def record_until_enter(self):
        return b""

    def transcribe(self, audio):
        return self.heard


class FakeSpeaker:
    def __init__(self, boom=False):
        self.said = []
        self.boom = boom

    def say(self, text):
        if self.boom:
            raise RuntimeError("audio device gone")
        self.said.append(text)


class FakeOffice:
    def __init__(self, reply=None, exc=None):
        self.reply = reply
        self.exc = exc
        self.dispatched = []

    def run_task(self, agent_id, text):
        if self.exc:
            raise self.exc
        self.dispatched.append(text)
        return self.reply


class TurnCase(Tmp):
    def build(self, heard, answer=None, office=None, speaker=None):
        import builtins

        self._real_input = builtins.input
        builtins.input = lambda *a, **k: ""
        self.addCleanup(setattr, builtins, "input", self._real_input)

        self.journal = Journal(self.tmp / "j.jsonl")
        self.office = office or FakeOffice(
            reply={"agent": "Researcher", "result": "done", "file": "n.md"}
        )
        self.speaker = speaker or FakeSpeaker()
        chatter = StubChatter(answer, ReplyCache(self.tmp / "learned.json"))
        return core.Frontdesk(
            self.office, FakeListener(heard), self.speaker, chatter, self.journal
        )

    def records(self):
        text = (self.tmp / "j.jsonl").read_text().strip()
        return [json.loads(line) for line in text.splitlines()] if text else []


class TestTurnRouting(TurnCase):
    def _drain(self, fd):
        for w in fd._workers:
            w.join(timeout=5)

    def test_small_talk_answers_without_dispatching(self):
        fd = self.build(Heard("Hello.", -0.2, 0.01))
        self.assertTrue(fd.turn())
        self._drain(fd)
        self.assertEqual(self.office.dispatched, [])
        rec = self.records()[0]
        self.assertEqual(rec["route"], "chat")
        self.assertEqual(rec["tier"], "table")
        self.assertEqual(rec["confidence"], -0.2)
        self.assertEqual(rec["no_speech"], 0.01)

    def test_task_dispatches_and_logs_result(self):
        fd = self.build(Heard("summarise the notes", -0.3, 0.02), answer=None)
        self.assertTrue(fd.turn())
        self._drain(fd)
        self.assertEqual(self.office.dispatched, ["summarise the notes"])

        turn, result = self.records()
        self.assertEqual(turn["route"], "task")
        self.assertEqual(turn["tier"], "dispatch")
        self.assertEqual(result["event"], "result")
        self.assertEqual(result["agent"], "Researcher")
        self.assertEqual(result["turn"], turn["turn"])

    def test_quit_stops_the_loop_and_is_logged(self):
        fd = self.build(Heard("quit"))
        self.assertFalse(fd.turn())
        self.assertEqual(self.records()[0]["route"], "quit")

    def test_empty_transcript_is_logged_and_continues(self):
        fd = self.build(Heard(""))
        self.assertTrue(fd.turn())
        self.assertEqual(self.records()[0]["route"], "empty")
        self.assertEqual(self.speaker.said, ["I did not catch that."])

    def test_dispatch_error_is_logged(self):
        fd = self.build(
            Heard("do a thing"), answer=None, office=FakeOffice(exc=MystinError("down"))
        )
        fd.turn()
        self._drain(fd)
        result = self.records()[1]
        self.assertEqual(result["error"], "down")


class TestDispatchResilience(TurnCase):
    """A dispatch must always report back, whatever the office returns."""

    def run_dispatch(self, office, speaker=None):
        fd = self.build(Heard("x"), office=office, speaker=speaker)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()) as buf:
            fd._dispatch("task", turn_id=1)
        return fd, buf.getvalue()

    def test_every_broken_shape_still_reaches_the_user(self):
        broken = [
            FakeOffice(reply={"agent": "A", "file": "f"}),          # no result
            FakeOffice(reply={"result": "bare"}),                    # no agent/file
            FakeOffice(reply=[1, 2, 3]),                             # not a dict
            FakeOffice(reply=None),                                  # nothing at all
            FakeOffice(reply={"agent": "A", "result": 42, "file": "f"}),
            FakeOffice(exc=ValueError("boom")),                      # unexpected error
            FakeOffice(exc=MystinError("unreachable")),
        ]
        for office in broken:
            with self.subTest(office=office.reply or office.exc):
                fd, _ = self.run_dispatch(office)
                self.assertTrue(fd.speaker.said, "user was left in silence")

    def test_broken_speaker_degrades_to_stdout(self):
        office = FakeOffice(reply={"agent": "A", "result": "hi", "file": "f"})
        fd, out = self.run_dispatch(office, speaker=FakeSpeaker(boom=True))
        self.assertIn("[frontdesk]", out)
        self.assertFalse(fd._speech_lock.locked(), "lock left held")

    def test_long_result_is_truncated_at_a_word_boundary(self):
        office = FakeOffice(
            reply={"agent": "A", "result": "word " * 300, "file": "big.md"}
        )
        fd, _ = self.run_dispatch(office)
        spoken = fd.speaker.said[0]
        self.assertIn("big.md", spoken)
        self.assertLess(len(spoken), core.SPOKEN_RESULT_LIMIT + 150)


if __name__ == "__main__":
    unittest.main()
