"""Bot settings and repository-relative asset paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BotConfig:
    prefer_adb: bool = True
    window_title: str = "MSI App Player"
    show_preview: bool = False

    enable_minimap_walk: bool = True
    minimap_idle_delay: float = 0.9
    minimap_cooldown: float = 3.0
    minimap_safe_view_fraction: float = 0.70
    minimap_min_click_distance: float = 100
    minimap_max_click_distance: float = 260
    minimap_align_min_gap: float = 0.025
    # Partial views can score below 0.40; coverage and candidate separation
    # provide additional safeguards for this lower geometry threshold.
    minimap_geometry_min_score: float = 0.30
    # The map overlay is drawn over a changing game scene, so raw grayscale
    # template scores are not reliable.  Require a meaningful amount of the
    # reference linework to be visible before accepting geometry-only matches.
    minimap_min_visible_reference_fraction: float = 0.35
    minimap_min_visible_reference_pixels: int = 30000

    margin_top: float = 0.12
    margin_bottom: float = 0.06
    margin_left: float = 0.06
    margin_right: float = 0.06
    mob_body_y_offset: int = 72
    player_deadzone_radius: int = 90
    match_threshold: float = 0.60
    min_name_template_overlap: float = 0.58
    target_scan_width: int = 1600
    ui_match_threshold: float = 0.80
    back_icon_match_threshold: float = 0.86
    back_icon_center: tuple[int, int] = (1546, 44)
    back_icon_scales: tuple[float, ...] = (1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
    white_text_min_channel: int = 190
    white_text_max_channel_spread: int = 50

    target_lock_timeout: float = 1.5
    target_retry_limit: int = 2
    target_retry_blacklist_seconds: float = 4.0
    idle_mark_check_interval: float = 1.0
    target_position_refresh_interval: float = 0.25
    target_reacquire_radius: float = 90
    # A red lock marker can flicker during attack animations.  Use the
    # target nameplate as a second identity check so a stale marker cannot
    # hold a defeated target forever.
    target_red_loss_timeout: float = 240.0
    ui_check_interval: float = 0.35
    target_stall_timeout: float = 30.0
    target_stall_blacklist_seconds: float = 10.0
    exhausted_caption_confirm_seconds: float = 10.0
    exhausted_caption_poll_seconds: float = 0.25
    exhausted_mob_blacklist_seconds: float = 600.0
    exhausted_travel_trigger_count: int = 2
    exhausted_travel_hold_seconds: float = 10.0
    repeated_output_limit: int = 20


CONFIG = BotConfig()
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
