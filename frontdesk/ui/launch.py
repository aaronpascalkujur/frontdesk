"""Wire the turn loop to the browser frontend and run it."""

import threading
import webbrowser

from ..backend import MystinOffice
from ..chat import Chatter
from ..console import ConsoleUI
from ..core import Frontdesk
from ..events import Bus, SignalGate
from ..journal import Journal
from ..speech import Listener, Speaker
from .server import DEFAULT_HOST, DEFAULT_PORT, UIServer


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    bus = Bus()
    gate = SignalGate()
    # The terminal keeps a transcript, minus the press-Enter prompts that no
    # longer apply once a browser owns the turn.
    ConsoleUI(bus, prompts=False).attach()

    fatal = threading.Event()
    bus.subscribe(lambda e: e.kind == "error" and e.data.get("fatal") and fatal.set())

    server = UIServer(bus, gate, host, port)
    try:
        server.start()
    except OSError as e:
        print(f"[frontdesk] cannot serve the UI on {host}:{port}: {e}")
        return

    print(f"[frontdesk] UI at {server.url}")
    print("[frontdesk] loading speech models…")
    frontdesk = Frontdesk(
        MystinOffice(), Listener(), Speaker(), Chatter(), Journal(),
        bus=bus, gate=gate,
    )

    if open_browser:
        # Opened after the models are loaded, so the page does not sit on
        # "connecting" for the ten seconds whisper takes to come up.
        threading.Thread(target=webbrowser.open, args=(server.url,), daemon=True).start()

    try:
        frontdesk.run()
        if fatal.is_set():
            # The office was unreachable, so the loop returned at once. Keep
            # serving anyway: the browser that just opened should get to show
            # the reason rather than a dead page.
            print("[frontdesk] UI still up so you can read the error. Ctrl-C to quit.")
            threading.Event().wait()
    except KeyboardInterrupt:
        print()
    finally:
        gate.close()
        server.shutdown()
