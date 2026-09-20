"""Append-only record of what Frontdesk heard and what it did about it.

Every turn lands here as one JSON object per line. Nothing reads it back yet —
it exists so that decisions about triage, transcription and routing can later be
judged against what actually happened rather than guessed at.
"""

import json
import os
import threading
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
JOURNAL_PATH = STATE_DIR / "journal.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    """Thread-safe JSONL writer. Worker threads log results as they land."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else JOURNAL_PATH
        self._lock = threading.Lock()
        self._ids = count(1)
        self._session = _now()

    def next_turn_id(self) -> int:
        return next(self._ids)

    def write(self, event: str, **fields) -> None:
        """Append one record. Journalling must never take the app down with it."""
        record = {"at": _now(), "session": self._session, "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as e:
            print(f"[frontdesk] (journal write failed: {type(e).__name__}: {e})")


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON without risking a half-written file if the process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
