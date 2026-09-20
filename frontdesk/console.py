"""The terminal frontend, rebuilt on top of the event bus.

This is the output the loop used to print inline. Routing it through a
subscriber keeps `core` free of formatting, and the lock fixes a real bug: two
agents finishing at once used to interleave their result blocks mid-line.
"""

import threading

from .events import IDLE, LISTENING, Event


class ConsoleUI:
    """Prints the loop to a terminal.

    `prompts` is off when a browser is driving the turn: the transcript is still
    worth having in the terminal, but telling someone to press Enter when the
    turn is taken by clicking would be a lie.
    """

    def __init__(self, bus, stream=None, prompts: bool = True):
        self.bus = bus
        self.stream = stream
        self.prompts = prompts
        self._lock = threading.Lock()

    def attach(self):
        return self.bus.subscribe(self.handle)

    def _print(self, text: str = "", end: str = "\n") -> None:
        with self._lock:
            print(text, end=end, flush=True, file=self.stream)

    def handle(self, event: Event) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler:
            handler(event.data)

    # `level` has no handler on purpose: 30 amplitude readings a second are for
    # drawing a waveform, not for scrolling a terminal.

    def _on_state(self, data: dict) -> None:
        state = data.get("value")
        if not self.prompts:
            return
        if state == IDLE:
            self._print("\nPress Enter when you are ready to speak. ", end="")
        elif state == LISTENING:
            self._print("[listening — speak now, then press Enter]")

    def _on_agents(self, data: dict) -> None:
        names = ", ".join(data.get("names") or [])
        self._print(f"[frontdesk] Connected to Mystin Office. Agents: {names}")
        learned = data.get("learned") or 0
        if learned:
            word = "reply" if learned == 1 else "replies"
            self._print(f"[frontdesk] {learned} learned {word} cached")

    def _on_heard(self, data: dict) -> None:
        if data.get("text"):
            self._print(f'[heard in {data.get("ms", 0):.0f}ms] "{data["text"]}"')

    def _on_say(self, data: dict) -> None:
        self._print(f'[frontdesk] {data.get("text", "")}')

    def _on_result(self, data: dict) -> None:
        head = (
            f'[{data.get("agent")} · {data.get("elapsed")}s '
            f'· saved to {data.get("file")}]'
        )
        self._print(f'\n{head}\n{data.get("text", "")}\n')

    def _on_error(self, data: dict) -> None:
        self._print(f'[frontdesk] {data.get("message")}')
        if data.get("fatal"):
            self._print("[frontdesk] Start Mystin Office first: npm start")

    def _on_waiting(self, data: dict) -> None:
        self._print(f'[frontdesk] waiting for {data.get("count")} task(s) to finish…')
