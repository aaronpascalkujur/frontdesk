"""Voice front-end for Mystin Office.

Push to talk, transcribe locally, hand the task to the agent office, and read the
deliverable back when it lands. Dispatch runs on a worker thread so you can keep
talking while an agent works.

The loop reports what it is doing on an event bus rather than printing, so the
same turn logic drives either the console or the browser UI.
"""

import threading
import time

from .backend import MystinError, MystinOffice
from .chat import Chatter
from .events import (
    IDLE,
    LISTENING,
    OFFLINE,
    SPEAKING,
    THINKING,
    TRANSCRIBING,
    Bus,
    ConsoleGate,
)
from .journal import Journal
from .speech import Listener, Speaker

SPOKEN_RESULT_LIMIT = 400
QUIT_WORDS = {"quit", "exit", "stop", "goodbye"}


class Frontdesk:
    def __init__(
        self,
        office: MystinOffice,
        listener: Listener,
        speaker: Speaker,
        chatter: Chatter,
        journal: Journal,
        bus: Bus | None = None,
        gate=None,
    ):
        self.office = office
        self.listener = listener
        self.speaker = speaker
        self.chatter = chatter
        self.journal = journal
        self.bus = bus if bus is not None else Bus()
        self.gate = gate if gate is not None else ConsoleGate()
        self._speech_lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._state: str | None = None  # nothing published until the loop starts
        self._wire_speech()

    def _wire_speech(self) -> None:
        """Let the microphone and the voice report straight to the bus.

        The state machine lives here rather than in the speech layer, but only
        the speech layer knows the honest moment the device opens or a syllable
        is played. Fakes in tests carry neither attribute, hence the guards.
        """
        if hasattr(self.listener, "on_open"):
            self.listener.on_open = lambda: self._set_state(LISTENING)
        if hasattr(self.listener, "on_level"):
            self.listener.on_level = lambda v: self.bus.emit(
                "level", value=v, source="mic"
            )
        if hasattr(self.speaker, "on_level"):
            self.speaker.on_level = lambda v: self.bus.emit(
                "level", value=v, source="voice"
            )

    def _set_state(self, state: str) -> None:
        """Publish a transition, never a repeat.

        Frontends treat idle as "it is your turn to speak", and the loop reaches
        idle twice in a row on every pass — once when an utterance finishes and
        again at the top of the next turn. Emitting both printed the console
        prompt twice for every single thing said.
        """
        if state == self._state:
            return
        self._state = state
        self.bus.emit("state", value=state)

    def say(self, text: str) -> None:
        """Speak, holding the floor so two finished agents cannot overlap."""
        with self._speech_lock:
            resume = self._state
            self._set_state(SPEAKING)
            self.bus.emit("say", text=text)
            try:
                self.speaker.say(text)
            finally:
                # Hand back exactly what was borrowed. Idle means "waiting for
                # you to speak", and only turn() knows when that is true, so
                # finishing an utterance must not claim it: an agent announcing
                # a result mid-turn would be telling the UI the microphone is
                # shut while it is still open.
                self._set_state(resume if resume is not None else IDLE)

    def _dispatch(self, task: str, turn_id: int = 0) -> None:
        started = time.monotonic()
        try:
            reply = self.office.run_task("auto", task)
            elapsed = time.monotonic() - started

            agent = reply.get("agent") or "The agent"
            filename = reply.get("file") or "a note"
            result = str(reply.get("result") or "").strip()
            self.bus.emit(
                "result",
                turn=turn_id,
                agent=agent,
                file=filename,
                elapsed=round(elapsed, 1),
                text=result,
            )
            self.journal.write(
                "result",
                turn=turn_id,
                agent=agent,
                file=filename,
                elapsed_s=round(elapsed, 1),
                result_chars=len(result),
            )

            if not result:
                self._announce(f"{agent} finished but sent nothing back.")
                return

            spoken = result
            if len(spoken) > SPOKEN_RESULT_LIMIT:
                spoken = spoken[:SPOKEN_RESULT_LIMIT].rsplit(" ", 1)[0]
                spoken += f". That is the first part. The full answer is saved as {filename}."
            self._announce(f"{agent} is done. {spoken}")
        except MystinError as e:
            self.journal.write("result", turn=turn_id, error=str(e))
            self.bus.emit("error", turn=turn_id, message=str(e))
            self._announce(f"That failed. {e}")
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            self.journal.write("result", turn=turn_id, error=detail)
            self.bus.emit("error", turn=turn_id, message=detail)
            self._announce(f"That task went wrong. {detail}")

    def _announce(self, text: str) -> None:
        """Speak, surviving a speaker that cannot.

        Every utterance goes through here. An audio device that is missing at
        startup or unplugged mid-session degrades the loop to a silent one that
        still journals, still dispatches, and still shows the words in the UI —
        rather than taking the whole session down on the way to saying hello.
        """
        try:
            self.say(text)
        except Exception as e:
            self.bus.emit("say", text=text, spoken=False)
            self.bus.emit("error", message=f"could not speak: {type(e).__name__}: {e}")

    def turn(self) -> bool:
        """Run one push-to-talk turn. Returns False when the user wants to quit."""
        self._set_state(IDLE)
        if not self.gate.wait():
            return False

        # LISTENING is published by the listener the moment the device is live.
        audio = self.listener.record_until(self.gate)
        self._set_state(TRANSCRIBING)

        t0 = time.monotonic()
        heard = self.listener.transcribe(audio)
        stt_ms = (time.monotonic() - t0) * 1000
        task = heard.text

        turn_id = self.journal.next_turn_id()
        entry = {
            "turn": turn_id,
            "heard": task,
            "stt_ms": round(stt_ms),
            "confidence": heard.avg_logprob,
            "no_speech": heard.no_speech_prob,
        }
        self.bus.emit(
            "heard",
            turn=turn_id,
            text=task,
            ms=round(stt_ms),
            confidence=heard.avg_logprob,
            no_speech=heard.no_speech_prob,
        )

        if not task:
            self.journal.write("turn", route="empty", **entry)
            self._announce("I did not catch that.")
            return True

        if task.lower().strip(" .!?") in QUIT_WORDS:
            self.journal.write("turn", route="quit", **entry)
            self._announce("Shutting down.")
            return False

        self._set_state(THINKING)
        t0 = time.monotonic()
        decision = self.chatter.reply(task, heard.avg_logprob)
        triage_ms = (time.monotonic() - t0) * 1000

        if decision.reply:
            self.journal.write(
                "turn", route="chat", tier=decision.tier, cached=decision.cached,
                triage_ms=round(triage_ms), reply=decision.reply, **entry
            )
            self.bus.emit(
                "route", turn=turn_id, route="chat", tier=decision.tier,
                cached=decision.cached, ms=round(triage_ms),
            )
            self._announce(decision.reply)
            return True

        self.journal.write(
            "turn", route="task", tier=decision.tier, triage_ms=round(triage_ms), **entry
        )
        self.bus.emit(
            "route", turn=turn_id, route="task", tier=decision.tier,
            ms=round(triage_ms),
        )
        self.bus.emit("task", turn=turn_id, text=task)
        self._announce("On it.")
        worker = threading.Thread(target=self._dispatch, args=(task, turn_id), daemon=True)
        worker.start()
        self._reap_workers()
        self._workers.append(worker)
        return True

    def _reap_workers(self) -> None:
        """Drop finished threads so a long session does not accumulate them."""
        self._workers = [w for w in self._workers if w.is_alive()]

    def stop(self) -> None:
        """Ask the loop to unwind from another thread, e.g. a closing window."""
        self.gate.close()

    def run(self) -> None:
        try:
            agents = self.office.agents()
        except MystinError as e:
            self.bus.emit("error", message=str(e), fatal=True)
            self._set_state(OFFLINE)
            return

        self.bus.emit(
            "agents",
            names=[a["name"] for a in agents],
            learned=len(self.chatter.cache.entries),
        )
        self._announce("Frontdesk online.")

        try:
            while self.turn():
                pass
        except (KeyboardInterrupt, EOFError):
            pass

        self._set_state(OFFLINE)
        pending = [w for w in self._workers if w.is_alive()]
        if pending:
            self.bus.emit("waiting", count=len(pending))
            for w in pending:
                w.join()


def main() -> None:
    from .console import ConsoleUI

    print("[frontdesk] loading speech models…")
    bus = Bus()
    ConsoleUI(bus).attach()
    frontdesk = Frontdesk(
        MystinOffice(), Listener(), Speaker(), Chatter(), Journal(), bus=bus
    )
    frontdesk.run()


if __name__ == "__main__":
    main()
