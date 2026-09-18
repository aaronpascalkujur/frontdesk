"""Voice front-end for Mystin Office.

Push to talk, transcribe locally, hand the task to the agent office, and read the
deliverable back when it lands. Dispatch runs on a worker thread so you can keep
talking while an agent works.
"""

import threading
import time

from .backend import MystinError, MystinOffice
from .speech import Listener, Speaker

SPOKEN_RESULT_LIMIT = 400


class Jarvis:
    def __init__(self, office: MystinOffice, listener: Listener, speaker: Speaker):
        self.office = office
        self.listener = listener
        self.speaker = speaker
        self._speech_lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def say(self, text: str) -> None:
        with self._speech_lock:
            print(f"[jarvis] {text}")
            self.speaker.say(text)

    def _dispatch(self, task: str) -> None:
        started = time.monotonic()
        try:
            reply = self.office.run_task("auto", task)
        except MystinError as e:
            self.say(f"That failed. {e}")
            return
        elapsed = time.monotonic() - started

        result = reply["result"].strip()
        print(f"\n[{reply['agent']} · {elapsed:.1f}s · saved to {reply['file']}]\n{result}\n")

        spoken = result
        if len(spoken) > SPOKEN_RESULT_LIMIT:
            spoken = spoken[:SPOKEN_RESULT_LIMIT].rsplit(" ", 1)[0]
            spoken += f". That is the first part. The full answer is saved as {reply['file']}."
        self.say(f"{reply['agent']} is done. {spoken}")

    def turn(self) -> bool:
        """Run one push-to-talk turn. Returns False when the user wants to quit."""
        input("\nPress Enter to speak, then Enter again to send. ")
        audio = self.listener.record_until_enter()

        t0 = time.monotonic()
        task = self.listener.transcribe(audio)
        stt_ms = (time.monotonic() - t0) * 1000

        if not task:
            self.say("I did not catch that.")
            return True

        print(f'[heard in {stt_ms:.0f}ms] "{task}"')
        if task.lower().strip(" .!?") in {"quit", "exit", "stop", "goodbye"}:
            self.say("Shutting down.")
            return False

        self.say("On it.")
        worker = threading.Thread(target=self._dispatch, args=(task,), daemon=True)
        worker.start()
        self._workers.append(worker)
        return True

    def run(self) -> None:
        try:
            agents = self.office.agents()
        except MystinError as e:
            print(f"[jarvis] {e}")
            print("[jarvis] Start Mystin Office first: npm start")
            return

        names = ", ".join(a["name"] for a in agents)
        print(f"[jarvis] Connected to Mystin Office. Agents: {names}")
        self.say("Jarvis online.")

        try:
            while self.turn():
                pass
        except (KeyboardInterrupt, EOFError):
            print()

        pending = [w for w in self._workers if w.is_alive()]
        if pending:
            print(f"[jarvis] waiting for {len(pending)} task(s) to finish…")
            for w in pending:
                w.join()


def main() -> None:
    print("[jarvis] loading speech models…")
    jarvis = Jarvis(MystinOffice(), Listener(), Speaker())
    jarvis.run()


if __name__ == "__main__":
    main()
