"""Voice front-end for Mystin Office.

Push to talk, transcribe locally, hand the task to the agent office, and read the
deliverable back when it lands. Dispatch runs on a worker thread so you can keep
talking while an agent works.
"""

import threading
import time

from .backend import MystinError, MystinOffice
from .chat import Chatter
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
    ):
        self.office = office
        self.listener = listener
        self.speaker = speaker
        self.chatter = chatter
        self.journal = journal
        self._speech_lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def say(self, text: str) -> None:
        with self._speech_lock:
            print(f"[frontdesk] {text}")
            self.speaker.say(text)

    def _dispatch(self, task: str, turn_id: int = 0) -> None:
        started = time.monotonic()
        try:
            reply = self.office.run_task("auto", task)
            elapsed = time.monotonic() - started

            agent = reply.get("agent") or "The agent"
            filename = reply.get("file") or "a note"
            result = str(reply.get("result") or "").strip()
            print(f"\n[{agent} · {elapsed:.1f}s · saved to {filename}]\n{result}\n")
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
            self._announce(f"That failed. {e}")
        except Exception as e:
            self.journal.write("result", turn=turn_id, error=f"{type(e).__name__}: {e}")
            self._announce(f"That task went wrong. {type(e).__name__}: {e}")

    def _announce(self, text: str) -> None:
        """Speak from a worker thread. A dispatch must never end in silence, so a
        broken speaker degrades to stdout instead of killing the thread."""
        try:
            self.say(text)
        except Exception as e:
            print(f"[frontdesk] {text}")
            print(f"[frontdesk] (could not speak: {type(e).__name__}: {e})")

    def turn(self) -> bool:
        """Run one push-to-talk turn. Returns False when the user wants to quit."""
        input("\nPress Enter when you are ready to speak. ")
        audio = self.listener.record_until_enter()

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

        if not task:
            self.journal.write("turn", route="empty", **entry)
            self.say("I did not catch that.")
            return True

        print(f'[heard in {stt_ms:.0f}ms] "{task}"')
        if task.lower().strip(" .!?") in QUIT_WORDS:
            self.journal.write("turn", route="quit", **entry)
            self.say("Shutting down.")
            return False

        t0 = time.monotonic()
        decision = self.chatter.reply(task)
        triage_ms = (time.monotonic() - t0) * 1000

        if decision.reply:
            self.journal.write(
                "turn", route="chat", tier=decision.tier,
                triage_ms=round(triage_ms), reply=decision.reply, **entry
            )
            self.say(decision.reply)
            return True

        self.journal.write(
            "turn", route="task", tier=decision.tier, triage_ms=round(triage_ms), **entry
        )
        self.say("On it.")
        worker = threading.Thread(target=self._dispatch, args=(task, turn_id), daemon=True)
        worker.start()
        self._workers.append(worker)
        return True

    def run(self) -> None:
        try:
            agents = self.office.agents()
        except MystinError as e:
            print(f"[frontdesk] {e}")
            print("[frontdesk] Start Mystin Office first: npm start")
            return

        names = ", ".join(a["name"] for a in agents)
        print(f"[frontdesk] Connected to Mystin Office. Agents: {names}")
        learned = len(self.chatter.cache.entries)
        if learned:
            print(f"[frontdesk] {learned} learned repl{'y' if learned == 1 else 'ies'} cached")
        self.say("Frontdesk online.")

        try:
            while self.turn():
                pass
        except (KeyboardInterrupt, EOFError):
            print()

        pending = [w for w in self._workers if w.is_alive()]
        if pending:
            print(f"[frontdesk] waiting for {len(pending)} task(s) to finish…")
            for w in pending:
                w.join()


def main() -> None:
    print("[frontdesk] loading speech models…")
    frontdesk = Frontdesk(
        MystinOffice(), Listener(), Speaker(), Chatter(), Journal()
    )
    frontdesk.run()


if __name__ == "__main__":
    main()
