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
from frontdesk.console import ConsoleUI
from frontdesk.events import Bus, ConsoleGate, Event, EventQueue, SignalGate
from frontdesk.journal import Journal
from frontdesk.speech.stt import Heard
from frontdesk.speech.tts import Speaker


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


class FakeGate:
    """Opens on demand. Counts waits so a turn can be shown to use two."""

    def __init__(self, allow=99):
        self.waits = 0
        self.allow = allow
        self.closed = False

    def wait(self):
        self.waits += 1
        return self.waits <= self.allow

    def close(self):
        self.closed = True


class FakeListener:
    def __init__(self, heard):
        self.heard = heard
        self.on_open = None
        self.on_level = None
        self.gates = []

    def record_until(self, gate):
        self.gates.append(gate)
        if self.on_open:
            self.on_open()
        gate.wait()
        return b""

    def transcribe(self, audio):
        return self.heard


class FakeSpeaker:
    def __init__(self, boom=False):
        self.said = []
        self.boom = boom
        self.on_level = None

    def say(self, text):
        if self.boom:
            raise RuntimeError("audio device gone")
        self.said.append(text)


class FakeOffice:
    def __init__(self, reply=None, exc=None, roster=None):
        self.reply = reply
        self.exc = exc
        self.roster = roster
        self.dispatched = []

    def agents(self):
        if self.roster is None:
            raise MystinError("cannot reach Mystin Office")
        return self.roster

    def run_task(self, agent_id, text):
        if self.exc:
            raise self.exc
        self.dispatched.append(text)
        return self.reply


class TurnCase(Tmp):
    def build(self, heard, answer=None, office=None, speaker=None, allow=99):
        self.journal = Journal(self.tmp / "j.jsonl")
        self.office = office or FakeOffice(
            reply={"agent": "Researcher", "result": "done", "file": "n.md"}
        )
        self.speaker = speaker or FakeSpeaker()
        self.listener = FakeListener(heard)
        self.gate = FakeGate(allow)
        self.bus = Bus()
        self.events = []
        self.bus.subscribe(self.events.append)
        chatter = StubChatter(answer, ReplyCache(self.tmp / "learned.json"))
        return core.Frontdesk(
            self.office, self.listener, self.speaker, chatter, self.journal,
            bus=self.bus, gate=self.gate,
        )

    def kinds(self, kind):
        return [e.data for e in self.events if e.kind == kind]

    def states(self):
        return [e.data["value"] for e in self.events if e.kind == "state"]

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
        # A real terminal is attached, because "the user is told" is a claim
        # about what reaches a frontend, not about what reaches the bus.
        ConsoleUI(fd.bus).attach()
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


# ------------------------------------------------------------------------- event bus


class TestBus(unittest.TestCase):
    def test_delivers_to_every_subscriber(self):
        bus, a, b = Bus(), [], []
        bus.subscribe(a.append)
        bus.subscribe(b.append)
        bus.emit("state", value="idle")
        self.assertEqual(a[0].kind, "state")
        self.assertEqual(b[0].data, {"value": "idle"})

    def test_unsubscribe_stops_delivery(self):
        bus, seen = Bus(), []
        off = bus.subscribe(seen.append)
        bus.emit("a")
        off()
        bus.emit("b")
        self.assertEqual([e.kind for e in seen], ["a"])

    def test_a_broken_frontend_cannot_break_the_loop(self):
        """The bus is called while a microphone is open. A frontend that throws
        must not propagate into the turn, nor starve the other subscribers."""
        bus, seen = Bus(), []

        def explode(_event):
            raise RuntimeError("ui is on fire")

        bus.subscribe(explode)
        bus.subscribe(seen.append)
        bus.emit("state", value="listening")  # must not raise
        self.assertEqual(len(seen), 1)


class TestEventQueue(unittest.TestCase):
    def test_amplitude_frames_are_dropped_when_full(self):
        q = EventQueue(maxsize=2)
        q.put(Event("heard", {"n": 1}))
        q.put(Event("heard", {"n": 2}))
        q.put(Event("level", {"value": 0.5}))
        kinds = [q.get(timeout=0).kind for _ in range(2)]
        self.assertEqual(kinds, ["heard", "heard"], "a level shouldered out a fact")

    def test_meaningful_events_evict_the_oldest_instead(self):
        q = EventQueue(maxsize=2)
        q.put(Event("heard", {"n": 1}))
        q.put(Event("heard", {"n": 2}))
        q.put(Event("heard", {"n": 3}))
        self.assertEqual([q.get(timeout=0).data["n"] for _ in range(2)], [2, 3])

    def test_get_returns_none_on_timeout(self):
        self.assertIsNone(EventQueue().get(timeout=0))


class TestGates(unittest.TestCase):
    def test_signal_gate_releases_a_waiting_thread(self):
        import threading as th

        gate, out = SignalGate(), []
        t = th.Thread(target=lambda: out.append(gate.wait()))
        t.start()
        gate.signal()
        t.join(timeout=2)
        self.assertEqual(out, [True])

    def test_signal_gate_rearms_between_presses(self):
        """A turn waits twice on one gate: once to start, once to stop."""
        gate = SignalGate()
        gate.signal()
        self.assertTrue(gate.wait())
        gate.signal()
        self.assertTrue(gate.wait())

    def test_closing_releases_the_waiter_with_false(self):
        gate = SignalGate()
        gate.close()
        self.assertFalse(gate.wait())
        self.assertTrue(gate.closed)

    def test_console_gate_treats_eof_as_quit(self):
        import builtins

        real = builtins.input
        builtins.input = lambda *a, **k: (_ for _ in ()).throw(EOFError())
        self.addCleanup(setattr, builtins, "input", real)
        self.assertFalse(ConsoleGate().wait())


# ----------------------------------------------------------------- turn state machine


class TestTurnStates(TurnCase):
    def test_states_follow_the_turn(self):
        """Ends on the state speaking borrowed, not idle: the next turn decides
        when the loop is waiting on the user again."""
        fd = self.build(Heard("hello"), answer="Hi there.")
        fd.turn()
        self.assertEqual(
            self.states(),
            ["idle", "listening", "transcribing", "thinking", "speaking", "thinking"],
        )

    def test_a_turn_takes_two_presses(self):
        """One to open the microphone, one to close it."""
        fd = self.build(Heard("hello"), answer="Hi.")
        fd.turn()
        self.assertEqual(self.gate.waits, 2)

    def test_listening_is_published_by_the_device_not_guessed(self):
        """The cue must mean the microphone is open. It is emitted from inside
        record_until, so a listener that never opens never claims to listen."""
        fd = self.build(Heard("hello"), answer="Hi.")
        fd.listener.on_open = None
        fd.turn()
        self.assertNotIn("listening", self.states())

    def test_idle_is_announced_once_per_turn(self):
        """The console renders idle as "Press Enter". The loop passes through it
        twice a turn, and printing both prompted the user twice for every reply."""
        fd = self.build(Heard("hello"), answer="Hi.")
        fd.turn()
        fd.turn()
        self.assertEqual(self.states().count("idle"), 2, "one prompt per turn")

    def test_no_prompt_is_printed_twice_in_a_row(self):
        import io

        fd = self.build(Heard("hello"), answer="Hi.", office=FakeOffice(roster=[]), allow=4)
        buf = io.StringIO()
        ConsoleUI(fd.bus, stream=buf).attach()
        fd.turn()
        self.assertEqual(buf.getvalue().count("Press Enter"), 1)

    def test_speaking_restores_the_previous_state(self):
        """An agent announcing a result mid-turn must not leave the UI saying
        the microphone is shut while it is still open."""
        fd = self.build(Heard("x"))
        fd._set_state("listening")
        fd.say("an agent finished")
        self.assertEqual(fd._state, "listening")

    def test_quit_stops_the_loop_and_closes_nothing_early(self):
        fd = self.build(Heard("quit"))
        self.assertFalse(fd.turn())
        self.assertIn("speaking", self.states())

    def test_gate_refusal_ends_the_turn_without_touching_the_microphone(self):
        fd = self.build(Heard("hello"), allow=0)
        self.assertFalse(fd.turn())
        self.assertEqual(fd.listener.gates, [], "recorded after the gate closed")

    def test_microphone_levels_reach_the_bus(self):
        fd = self.build(Heard("x"))
        fd.listener.on_level(0.42)
        self.assertEqual(self.kinds("level"), [{"value": 0.42, "source": "mic"}])

    def test_voice_levels_reach_the_bus(self):
        fd = self.build(Heard("x"))
        fd.speaker.on_level(0.3)
        self.assertEqual(self.kinds("level"), [{"value": 0.3, "source": "voice"}])

    def test_heard_and_route_are_published_for_the_feed(self):
        fd = self.build(Heard("summarise the notes", -0.2))
        fd.turn()
        for w in fd._workers:
            w.join(timeout=5)
        self.assertEqual(self.kinds("heard")[0]["text"], "summarise the notes")
        self.assertEqual(self.kinds("route")[0]["route"], "task")
        self.assertEqual(self.kinds("task")[0]["text"], "summarise the notes")
        self.assertEqual(self.kinds("result")[0]["agent"], "Researcher")

    def test_finished_workers_are_reaped(self):
        fd = self.build(Heard("do a thing"))
        for _ in range(3):
            fd.turn()
            for w in fd._workers:
                w.join(timeout=5)
        fd.turn()
        self.assertLessEqual(len(fd._workers), 2, "worker list grows without bound")


class TestStartup(TurnCase):
    def roster(self):
        return FakeOffice(
            reply={"agent": "R", "result": "done", "file": "n.md"},
            roster=[{"name": "Researcher"}, {"name": "Writer"}],
        )

    def test_a_dead_speaker_does_not_prevent_startup(self):
        """Greeting the user must not be the thing that kills the session."""
        fd = self.build(
            Heard("quit"), office=self.roster(), speaker=FakeSpeaker(boom=True), allow=2
        )
        fd.run()  # must not raise
        self.assertEqual(self.states()[-1], "offline")
        self.assertTrue(
            any("could not speak" in e["message"] for e in self.kinds("error"))
        )

    def test_roster_is_published_for_the_ui(self):
        fd = self.build(Heard("quit"), office=self.roster(), allow=2)
        fd.run()
        self.assertEqual(self.kinds("agents")[0]["names"], ["Researcher", "Writer"])

    def test_unreachable_office_is_flagged_fatal_and_stops(self):
        fd = self.build(Heard("hello"), office=FakeOffice())
        fd.run()
        self.assertTrue(self.kinds("error")[0]["fatal"])
        self.assertEqual(self.states()[-1], "offline")
        self.assertEqual(self.gate.waits, 0, "waited for a turn with no office")


# ----------------------------------------------------------------------- console UI


class TestConsoleUI(unittest.TestCase):
    def render(self, events, **kwargs):
        import io

        buf = io.StringIO()
        bus = Bus()
        ConsoleUI(bus, stream=buf, **kwargs).attach()
        for kind, data in events:
            bus.emit(kind, **data)
        return buf.getvalue()

    def test_prompts_are_suppressed_when_a_browser_drives(self):
        events = [("state", {"value": "idle"}), ("state", {"value": "listening"})]
        self.assertIn("Press Enter", self.render(events))
        self.assertNotIn("Press Enter", self.render(events, prompts=False))

    def test_transcript_still_prints_without_prompts(self):
        out = self.render([("heard", {"text": "hello", "ms": 210})], prompts=False)
        self.assertIn('"hello"', out)

    def test_amplitude_never_reaches_the_terminal(self):
        self.assertEqual(self.render([("level", {"value": 0.5, "source": "mic"})]), "")

    def test_unreachable_office_explains_the_fix(self):
        out = self.render([("error", {"message": "cannot reach", "fatal": True})])
        self.assertIn("npm start", out)


# --------------------------------------------------------------------------- ui server


class TestUIServer(unittest.TestCase):
    def build(self):
        from frontdesk.ui.server import UIServer

        bus, gate = Bus(), SignalGate()
        return bus, gate, UIServer(bus, gate, host="127.0.0.1", port=0)

    def test_a_late_client_is_caught_up_on_the_conversation(self):
        bus, _, server = self.build()
        bus.emit("heard", turn=1, text="hello")
        bus.emit("say", text="Hi there.")
        client = server._add_client()
        kinds = []
        while (event := client.get(timeout=0)) is not None:
            kinds.append(event.kind)
        self.assertEqual(kinds, ["heard", "say", "state"])

    def test_replay_never_includes_stale_amplitude(self):
        """Replaying a waveform would animate speech that finished minutes ago."""
        bus, _, server = self.build()
        bus.emit("level", value=0.9, source="voice")
        client = server._add_client()
        kinds = []
        while (event := client.get(timeout=0)) is not None:
            kinds.append(event.kind)
        self.assertEqual(kinds, ["state"])

    def test_current_state_is_replayed_last_so_it_wins(self):
        bus, _, server = self.build()
        bus.emit("state", value="thinking")
        bus.emit("heard", turn=1, text="hello")
        client = server._add_client()
        last = None
        while (event := client.get(timeout=0)) is not None:
            last = event
        self.assertEqual(last.kind, "state")
        self.assertEqual(last.data["value"], "thinking")

    def test_live_events_reach_a_connected_client(self):
        bus, _, server = self.build()
        client = server._add_client()
        while client.get(timeout=0) is not None:
            pass
        bus.emit("say", text="On it.")
        self.assertEqual(client.get(timeout=0).data["text"], "On it.")

    def test_a_dropped_client_stops_receiving(self):
        bus, _, server = self.build()
        client = server._add_client()
        server._drop_client(client)
        while client.get(timeout=0) is not None:
            pass
        bus.emit("say", text="nobody home")
        self.assertIsNone(client.get(timeout=0))

    def test_taken_port_surfaces_to_the_caller(self):
        """launch.py reports this to the user, so it must not die in a thread."""
        _, _, first = self.build()
        first.bind()
        from frontdesk.ui.server import UIServer

        clash = UIServer(Bus(), SignalGate(), host="127.0.0.1", port=first.port)
        with self.assertRaises(OSError):
            clash.bind()
        first.shutdown()


class TestUIServerHTTP(unittest.TestCase):
    """One end-to-end pass over the real socket."""

    def setUp(self):
        from frontdesk.ui.server import UIServer

        self.bus, self.gate = Bus(), SignalGate()
        self.server = UIServer(self.bus, self.gate, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.shutdown)

    def get(self, path):
        import urllib.request

        with urllib.request.urlopen(self.server.url.rstrip("/") + path, timeout=5) as r:
            return r.status, r.read()

    def post(self, path):
        import urllib.request

        req = urllib.request.Request(
            self.server.url.rstrip("/") + path, data=b"", method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status

    def test_serves_the_page_and_its_assets(self):
        for path, needle in [
            ("/", b"<canvas"),
            ("/static/app.js", b"EventSource"),
            ("/static/style.css", b"--accent"),
        ]:
            with self.subTest(path=path):
                status, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertIn(needle, body)

    def test_press_opens_the_gate(self):
        self.assertEqual(self.post("/press"), 204)
        self.assertTrue(self.gate.wait())

    def test_quit_closes_the_gate(self):
        self.assertEqual(self.post("/quit"), 204)
        self.assertFalse(self.gate.wait())

    def test_refuses_to_serve_outside_the_static_directory(self):
        import urllib.error

        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/static/../../chat.py")
        self.addCleanup(cm.exception.close)
        self.assertEqual(cm.exception.code, 404)


# ---------------------------------------------------------------- speech animation


class TestSpeechEnvelope(unittest.TestCase):
    """The talking animation is driven by the audio actually being played."""

    def envelope(self, audio, rate):
        import numpy as np

        return Speaker._envelope(np.asarray(audio, dtype="int16"), rate)

    def test_one_frame_per_thirtieth_of_a_second(self):
        rate = 22050
        audio = [0] * (rate // 2)  # half a second
        self.assertEqual(len(self.envelope(audio, rate)), 15)

    def test_loud_audio_approaches_full_scale(self):
        rate = 22050
        loud = self.envelope([32767] * rate, rate)
        self.assertGreater(loud.max(), 0.9)
        self.assertLessEqual(loud.max(), 1.0)

    def test_silence_reads_as_zero(self):
        self.assertEqual(self.envelope([0] * 22050, 22050).max(), 0.0)

    def test_audio_shorter_than_a_frame_yields_nothing(self):
        self.assertEqual(len(self.envelope([1, 2, 3], 22050)), 0)

    def test_a_throwing_frontend_is_not_asked_twice(self):
        """A dead UI must never take down the voice."""
        speaker = Speaker.__new__(Speaker)
        calls = []

        def boom(_level):
            calls.append(1)
            raise RuntimeError("gone")

        speaker.on_level = boom
        speaker._report(0.5)
        speaker._report(0.5)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(speaker.on_level)


if __name__ == "__main__":
    unittest.main()
