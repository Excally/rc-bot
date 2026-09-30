"""Console timestamping and repeated-warning watchdog."""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from .config import CONFIG
from .exceptions import RepeatedOutputReset

class RepeatedOutputGuard:
    """Detect recurring error lines within a bounded time window."""

    _RESET_LINES = (
        "[+] Red square gone",
        "[+] Back action closed",
        "[+] System Back closed",
        "[+] Screen capture recovered",
        "[*] Target locked",
        "[*] Detected active target lock",
    )

    def __init__(self, limit: int = CONFIG.repeated_output_limit, window_seconds: float = 30.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self.counts: dict[str, tuple[int, float]] = {}

    def observe(self, message: str) -> None:
        line = message.strip()
        if not line:
            return
        if line.startswith(self._RESET_LINES):
            self.counts.clear()
            return

        # `[ ... ] Scanning ...` is expected during normal idle operation.
        should_watch = line.startswith(("[!]", "[-]"))
        if line.startswith("[~]"):
            lower = line.lower()
            should_watch = any(
                word in lower
                for word in ("failed", "skipped", "still", "unavailable", "could not", "retry")
            )
        if not should_watch:
            return

        pattern = re.sub(r"\d+(?:\.\d+)?", "<n>", line)
        pattern = re.sub(r"\s+", " ", pattern)
        now = time.monotonic()
        self.counts = {
            key: entry for key, entry in self.counts.items()
            if now - entry[1] <= self.window_seconds
        }
        previous_count, previous_time = self.counts.get(pattern, (0, 0.0))
        count = previous_count + 1 if now - previous_time <= self.window_seconds else 1
        self.counts[pattern] = (count, now)
        if count >= self.limit:
            self.counts.clear()
            raise RepeatedOutputReset(pattern)


class TimestampedOutputStream:
    """Add timestamps without losing normal stream behaviour."""

    def __init__(self, stream: Any, guard: Optional[RepeatedOutputGuard] = None):
        self.stream = stream
        self.guard = guard
        self.pending = ""

    def _write_line(self, line: str, ending: str) -> None:
        raw_line = line.rstrip("\r")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.stream.write(f"[{timestamp}] {raw_line}{ending}")
        if self.guard is not None:
            self.guard.observe(raw_line)

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._write_line(line, "\n")
        return len(text)

    def flush(self) -> None:
        if self.pending:
            pending, self.pending = self.pending, ""
            self._write_line(pending, "")
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)
