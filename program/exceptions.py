"""Internal control-flow exceptions used by the bot."""
from __future__ import annotations

class RepeatedOutputReset(Exception):
    """Raised when one warning/status line loops long enough to be unsafe."""

    def __init__(self, pattern: str):
        super().__init__(pattern)
        self.pattern = pattern


class BotQuit(Exception):
    """Internal signal used when the preview window receives ``q``."""
