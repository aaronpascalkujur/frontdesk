"""A local web frontend for the turn loop.

Server-Sent Events rather than WebSockets: the traffic is almost entirely one
way — the loop has state to push, the page has the occasional button press to
POST back — and SSE needs no dependency beyond the standard library. The browser
reconnects on its own if this process restarts.

The page is presentation only. The microphone, whisper and Piper all stay in
Python, so there are no browser audio permissions in the path.
"""

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..events import IDLE, Event, EventQueue

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4520
# Long enough to keep proxies and browsers from dropping an idle stream, short
# enough that a client that has gone away is noticed within a turn or two.
HEARTBEAT = 15.0
# Replayed to a browser that connects late or reloads mid-session, so the page
# comes back with the conversation instead of blank.
REPLAY_LIMIT = 60

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class UIServer:
    def __init__(self, bus, gate, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.bus = bus
        self.gate = gate
        self.host = host
        self.port = port
        self._clients: list[EventQueue] = []
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=REPLAY_LIMIT)
        self._state = Event("state", {"value": IDLE})
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        bus.subscribe(self._fan_out)

    def _fan_out(self, event: Event) -> None:
        # Amplitude is live-only. Replaying a waveform into a page that just
        # opened would animate speech that finished minutes ago.
        if event.kind == "state":
            self._state = event
        elif event.kind != "level":
            self._history.append(event)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.put(event)

    def _add_client(self) -> EventQueue:
        client = EventQueue()
        for event in list(self._history):
            client.put(event)
        client.put(self._state)  # last, so it wins over anything replayed
        with self._lock:
            self._clients.append(client)
        return client

    def _drop_client(self, client: EventQueue) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def bind(self) -> None:
        """Claim the port up front.

        Binding here rather than inside the serving thread means "that port is
        taken" surfaces to the caller as an OSError, instead of disappearing
        into a thread that dies while the loop carries on as if it had a UI.
        """
        self._httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # resolves port 0 to the real one

    def start(self) -> threading.Thread:
        if self._httpd is None:
            self.bind()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self._thread

    def shutdown(self) -> None:
        """Safe to call whether or not the server ever started serving.

        `shutdown()` waits for the serve_forever loop to acknowledge it, so
        calling it on a socket that was only ever bound would block forever —
        which is exactly what a failed startup does on its way out.
        """
        if not self._httpd:
            return
        if self._thread and self._thread.is_alive():
            self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"


def _make_handler(server: UIServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass  # the turn loop owns this terminal; request logs would bury it

        # ------------------------------------------------------------- helpers

        def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if not path.is_file() or STATIC not in path.parents:
                self._send(404, b"not found")
                return
            self._send(
                200,
                path.read_bytes(),
                CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
            )

        # ---------------------------------------------------------------- GET

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._static("index.html")
            elif self.path.startswith("/static/"):
                self._static(self.path[len("/static/"):].split("?")[0])
            elif self.path == "/events":
                self._stream()
            else:
                self._send(404, b"not found")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            client = server._add_client()
            try:
                while True:
                    event = client.get(timeout=HEARTBEAT)
                    if event is None:
                        # A comment frame. Its only job is to fail loudly on a
                        # socket whose browser has gone, so the client is reaped.
                        self.wfile.write(b": ping\n\n")
                    else:
                        payload = json.dumps({"kind": event.kind, **event.data})
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
                pass  # browser closed the tab or navigated away
            finally:
                server._drop_client(client)

        # --------------------------------------------------------------- POST

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)

            if self.path == "/press":
                server.gate.signal()
                self._send(204)
            elif self.path == "/quit":
                server.gate.close()
                self._send(204)
            else:
                self._send(404, b"not found")

    return Handler
