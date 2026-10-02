"""Mutable state owned by the combat state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional

from .models import TargetMatch

@dataclass
class BotState:
    current_target: Optional[TargetMatch] = None
    target_click_time: float = 0.0
    confirmed_locked: bool = False
    last_log_time: float = 0.0
    last_ui_back_time: float = 0.0
    no_target_since: float = field(default_factory=time.monotonic)
    last_minimap_time: float = 0.0
    frame_count: int = 0
    blacklist: dict[tuple[int, int], float] = field(default_factory=dict)
    idle_mark_check_needed: bool = True
    next_idle_mark_check_at: float = 0.0
    ignored_exhausted_marker: Optional[tuple[int, int]] = None
    ignored_stale_marker: Optional[tuple[int, int]] = None
    last_capture_warning: float = 0.0
    pickup_pending: bool = False
    ui_recovery_interval: float = 1.5
    failed_recovery_state: Optional[str] = None
    failed_recovery_count: int = 0
    exhausted_caption_check_at: Optional[float] = None
    exhausted_caption_next_poll: float = 0.0
    exhausted_caption_target: Optional[TargetMatch] = None
    consecutive_exhausted: int = 0
    exhausted_travel_pending: bool = False
    last_ui_state: str = "other"
    next_ui_check_at: float = 0.0
