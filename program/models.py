"""Shared typed state structures."""
from __future__ import annotations

from typing import Optional, TypedDict

class TargetMatch(TypedDict, total=False):
    nx: int
    ny: int
    nw: int
    nh: int
    click_x: int
    click_y: int
    distance: float
    score: float
    scale: float
    last_entity_check: float
    last_entity_seen_time: float
    red_loss_since: Optional[float]
    red_last_seen: float
    red_observations: int


class RetryContext(TypedDict):
    x: int
    y: int
    grid: tuple[int, int]
    time: float
