"""What the turn loop is doing, published so something can draw it.

The loop used to describe itself with bare `print()` calls, which only a
terminal can read. It now emits events instead; the console frontend prints
them and the web frontend animates them. Nothing in here knows about either.

Delivery is best-effort on purpose. A subscriber that raises, blocks, or falls
behind must never stall the loop that is holding a microphone open, so `emit`
swallows subscriber errors and `EventQueue` drops frames under pressure.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

# Where the loop is in a turn. The web UI maps each to an animation, so adding
# one here means teaching the frontend what it looks like.
IDLE = "idle"              # waiting for the user to start a turn
LISTENING = "listening"    # microphone open
TRANSCRIBING = "transcribing"
THINKING = "thinking"      # triage is deciding chat vs dispatch
SPEAKING = "speaking"      # Piper is talking
OFFLINE = "offline"        # shut down, or never reached the office


@dataclass
class Event:
    kind: str
    data: dict = field(default_factory=dict)


class Bus:
    """Fan-out to zero or more frontends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Register a listener. Returns a function that removes it again."""
        with self._lock:
            self._subscribers.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return unsubscribe

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        event = Event(kind, data)
        for fn in listeners:
            try:
                fn(event)
            except Exception:
                # A broken frontend is not worth dropping a turn over. It will
                # miss this event and, if it is the web UI, reconnect on its own.
                pass


class EventQueue:
    """A subscriber that buffers for a client reading at its own pace.

    `level` events arrive ~30 times a second to drive the waveform. If a client
    stalls, those are the frames worth discarding: the next one is along in
    33ms and it carries no meaning on its own. Everything else is a fact about
    the conversation, so the oldest is dropped only once the buffer is full.
    """

    LOSSY_KINDS = frozenset({"level"})

    def __init__(self, maxsize: int = 512):
        self._q: queue.Queue[Event] = queue.Queue(maxsize=maxsize)

    def put(self, event: Event) -> None:
        try:
            self._q.put_nowait(event)
            return
        except queue.Full:
            pass
        if event.kind in self.LOSSY_KINDS:
            return  # a stale amplitude is worth nothing; let it go
        try:
            self._q.get_nowait()  # make room by discarding the oldest
            self._q.put_nowait(event)
        except (queue.Empty, queue.Full):
            pass

    def get(self, timeout: float | None = None) -> Event | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class ConsoleGate:
    """Push-to-talk on a terminal: the user presses Enter."""

    def __init__(self, prompt: str = ""):
        self.prompt = prompt

    def wait(self) -> bool:
        try:
            input(self.prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        return True

    def close(self) -> None:
        pass


class SignalGate:
    """Push-to-talk from a frontend: the loop waits, the UI thread signals.

    `wait` returns False once closed so the turn loop can unwind instead of
    parking forever on a window that has gone away.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._closed = False

    def wait(self) -> bool:
        self._event.wait()
        self._event.clear()
        return not self._closed

    def signal(self) -> None:
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()  # release whoever is parked in wait()

    @property
    def closed(self) -> bool:
        return self._closed
