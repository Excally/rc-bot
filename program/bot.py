"""Combat, target tracking, and recovery state machine."""
from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig, CONFIG
from .device import InputDevice
from .exceptions import BotQuit, RepeatedOutputReset
from .models import TargetMatch
from .navigation import MinimapNavigator
from .profiles import ZoneProfile, load_profile_settings
from .state import BotState
from .vision import FrameVision, TemplateStore

class RucoyBot:
    def __init__(self, config: BotConfig = CONFIG, profile: Optional[ZoneProfile] = None):
        self.config = config
        if profile is None:
            settings = load_profile_settings()
            profile = settings.profiles[settings.active_profile]
        self.profile = profile
        self.templates = TemplateStore()
        self.device: Optional[InputDevice] = None
        self.vision = FrameVision(config, self.templates)
        self.navigator: Optional[MinimapNavigator] = None
        self.target_templates: list[tuple[np.ndarray, str]] = []
        self.state = BotState()

    def start(self) -> bool:
        images = self.templates.load(self.profile.template_keys)
        exact = images.get(self.profile.target_template_key)
        if exact is None:
            print(
                f"[-] Target template '{self.profile.target_template_key}' "
                f"for '{self.profile.target_name}' not found in {self.templates.directory}!"
            )
            return False
        if self.config.enable_minimap_walk and images.get(self.profile.minimap_template_key) is None:
            print(
                f"[-] Minimap template '{self.profile.minimap_template_key}' "
                f"for profile '{self.profile.key}' not found in {self.templates.directory}!"
            )
            return False
        self.target_templates = [(exact, "white")]
        self.device = InputDevice.connect(self.config)
        if self.device is None:
            return False
        self.navigator = MinimapNavigator(
            self.config, self.vision, self.device, self.templates,
            self.profile.minimap_template_key,
        )
        print("[+] Capturing initial frame...")
        frame = self.device.capture()
        if frame is None:
            print("[-] Failed to capture initial frame! Check if emulator is running.")
            return False
        self._verify_template(frame)
        self._print_startup()
        return True

    def _verify_template(self, frame: np.ndarray) -> None:
        fh, fw = frame.shape[:2]
        player = self.vision.player_center(frame)
        center_x, center_y = player if player is not None else (fw // 2, fh // 2)
        matches = self.vision.find_targets(frame, self.target_templates, center_x, center_y)
        if matches:
            strongest = max(matches, key=lambda match: match["score"])
            print(f"[+] Live target matcher found a nameplate (score: {strongest['score']:.2f}).")
        else:
            print("[~] Runtime matcher found no eligible target in the startup frame; scanning will continue.")

    def _print_startup(self) -> None:
        assert self.device is not None
        print(f"\n[+] Zone profile: '{self.profile.key}'")
        print(f"[+] Target: '{self.profile.target_name}'")
        print(f"[+] Name template: {self.profile.target_template_key}")
        print(f"[+] Minimap template: {self.profile.minimap_template_key}")
        print(f"[+] Method: {'ADB Background Tap (Free Mouse)' if self.device.use_adb else 'Win32 Click'}")
        print(f"[+] Background: {'YES (Can stay behind other windows)' if self.device.use_adb else 'Window visible'}")
        print(f"[+] Movement: Registered minimap waypoints ({'enabled' if self.config.enable_minimap_walk else 'disabled'})")
        print(
            f"[+] Minimap alignment limits: map>={self.config.minimap_geometry_min_score:.2f}, "
            f"coverage>={self.config.minimap_min_visible_reference_fraction:.2f}, "
            f"gap>={self.config.minimap_align_min_gap:.3f}, "
            f"visible pixels>={self.config.minimap_min_visible_reference_pixels}"
        )
        print(f"[+] Hitbox Offset: {self.config.mob_body_y_offset}px below nametag (lowered for clean hits)")
        print("[+] Target marker check: waiting for the exact marked.png outline; nearest visible Skeleton first")
        print(
            f"[+] Exhaustion failsafe: {self.config.exhausted_travel_trigger_count} consecutive exhausted targets "
            f"within {self.config.exhausted_travel_window_seconds:.0f}s -> farthest minimap point, "
            f"hold {self.config.exhausted_travel_hold_seconds:.0f}s"
        )
        preview_status = "ON (Press q in preview window to exit)" if self.config.show_preview else "OFF (Press Ctrl+C in terminal to exit)"
        print(f"[+] Preview: {preview_status}")
        print("\n[+] Hunting loop started! You can work while bot is AFK.\n")

    def run(self) -> None:
        if not self.start() or self.device is None:
            return
        try:
            while True:
                frame = self.device.capture()
                if frame is None:
                    self._capture_failed()
                    time.sleep(0.3)
                    continue
                if self.state.last_capture_warning:
                    print("[+] Screen capture recovered.")
                    self.state.last_capture_warning = 0.0
                self._process_frame(frame)
        except BotQuit:
            pass
        if self.config.show_preview:
            cv2.destroyAllWindows()
        print("[+] Bot stopped.")

    def _capture_failed(self) -> None:
        now = time.monotonic()
        if now - self.state.last_capture_warning >= 5.0:
            print("[!] Screen capture unavailable; retrying (ADB capture times out after 3s).")
            self.state.last_capture_warning = now

    def _process_frame(self, frame: np.ndarray) -> None:
        assert self.device is not None and self.navigator is not None
        state = self.state
        cfg = self.config
        fh, fw = frame.shape[:2]
        detected_player = self.vision.player_center(frame)
        cx, cy = detected_player if detected_player is not None else (fw // 2, fh // 2)
        now = time.monotonic()
        state.frame_count += 1
        # Fixed UI icon matching is considerably more expensive than the
        # target scan.  Recheck it often enough to recover from a panel, while
        # allowing intervening frames to go straight to combat detection.
        if state.frame_count == 1 or now >= state.next_ui_check_at:
            state.last_ui_state, _ = self.vision.ui_state(frame)
            state.next_ui_check_at = now + cfg.ui_check_interval
        ui_state = state.last_ui_state
        if ui_state == "combat":
            state.failed_recovery_state = None
            state.failed_recovery_count = 0
        else:
            self._recover_ui(frame, ui_state, now)
            time.sleep(0.08)
            return
        # Finish a pending loot tap before selecting the next mob. Sending it
        # after target acquisition can immediately steal the game's focus.
        if state.current_target is None and state.pickup_pending:
            pickup = self.vision.pickup_position(frame)
            if pickup is not None:
                print(f"[+] Pickup available after kill; tapping ({pickup[0]}, {pickup[1]}).")
                if self.device.click(*pickup):
                    state.pickup_pending = False
        if state.current_target is None:
            self._acquire_target(frame, cx, cy, now)
        self._check_exhausted_caption(frame, now)
        if state.exhausted_travel_pending:
            self._run_exhausted_travel_failsafe(now)
            self._preview(frame, cx, cy)
            return
        if state.current_target is not None:
            self._update_target(frame, cx, cy, now)
        if state.current_target is None:
            idle_duration = now - state.no_target_since
            if cfg.enable_minimap_walk and idle_duration >= cfg.minimap_idle_delay and now - state.last_minimap_time >= cfg.minimap_cooldown:
                print(f"[~] No '{self.profile.target_name}' in vision for {idle_duration:.1f}s -> registering minimap waypoint...")
                self.navigator.travel()
                state.last_minimap_time = time.monotonic()
                state.no_target_since = state.last_minimap_time
        self._preview(frame, cx, cy)

    def _recover_ui(self, frame: np.ndarray, ui_state: str, now: float) -> None:
        state = self.state
        interval = 4.0 if ui_state == "disconnected" else state.ui_recovery_interval
        if now - state.last_ui_back_time < interval:
            return
        print(f"[!] UI state is {ui_state}; recovering and resuming automatically...")
        recovery = self._recover_unexpected_ui(frame)
        state.ui_recovery_interval = 5.0 if recovery == "failed" else 1.5
        if recovery == "failed":
            if state.failed_recovery_state == ui_state:
                state.failed_recovery_count += 1
            else:
                state.failed_recovery_state, state.failed_recovery_count = ui_state, 1
            if state.failed_recovery_count >= self.config.repeated_output_limit:
                raise RepeatedOutputReset(f"UI recovery for '{ui_state}' failed {state.failed_recovery_count} times")
        else:
            state.failed_recovery_state = None
            state.failed_recovery_count = 0
        state.current_target = None
        state.confirmed_locked = False
        state.idle_mark_check_needed = True
        state.no_target_since = now
        state.last_ui_back_time = now

    def _recover_unexpected_ui(self, frame: np.ndarray) -> str:
        assert self.device is not None
        if self.vision.is_disconnected(frame):
            fh, fw = frame.shape[:2]
            if not self.device.click(int(fw * 0.3125), int(fh * 0.667)):
                print("[!] Reconnect tap failed; UI recovery will retry.")
                return "failed"
            print("[~] Disconnected screen detected; selected Reconnect and will resume automatically.")
            return "reconnect"
        back = self.vision.back_icon(frame)
        if back is not None and self.navigator is not None and self.navigator._close_map(back):
            print("[+] Back action closed the unexpected UI.")
            return "back"
        self.device.back()
        time.sleep(0.35)
        after = self.device.capture()
        if after is not None and self.vision.ui_state(after)[0] == "combat":
            print("[+] System Back closed the unexpected UI.")
            return "back"
        print("[!] System Back did not return to combat; recovery is backing off.")
        return "failed"

    def _acquire_target(self, frame: np.ndarray, cx: int, cy: int, now: float) -> None:
        assert self.device is not None
        state, cfg = self.state, self.config
        has_red, position = (False, None)
        marker_checked = False
        if state.idle_mark_check_needed or now >= state.next_idle_mark_check_at:
            has_red, position = self.vision.red_square(frame)
            marker_checked = True
            state.idle_mark_check_needed = False
            state.next_idle_mark_check_at = now + cfg.idle_mark_check_interval
        if marker_checked and state.ignored_stale_marker is not None:
            if has_red and position is not None:
                if np.hypot(position[0] - state.ignored_stale_marker[0], position[1] - state.ignored_stale_marker[1]) < 140:
                    has_red = False
                else:
                    state.ignored_stale_marker = None
            else:
                state.ignored_stale_marker = None
        mobs = self.vision.find_targets(frame, self.target_templates, cx, cy)
        if state.unconfirmed_target_position is not None:
            if now >= state.unconfirmed_target_until:
                state.unconfirmed_target_position = None
            else:
                ignored_x, ignored_y = state.unconfirmed_target_position
                mobs = [
                    mob for mob in mobs
                    if not (
                        abs(mob.get("nx", mob["click_x"]) - ignored_x) < 30
                        and abs(mob.get("ny", mob["click_y"]) - ignored_y) < 15
                    )
                ]
        if not mobs:
            if now - state.last_log_time > 5.0:
                print(f"[...] Scanning for '{self.profile.target_name}'... (frame #{state.frame_count}, res {frame.shape[1]}x{frame.shape[0]})")
                state.last_log_time = now
            return
        # find_targets sorts by player distance first; match score only breaks
        # distance ties, so a farther but sharper nameplate cannot win.
        best = mobs[0]
        bx, by = best["click_x"], best["click_y"]
        if has_red and position is not None:
            marker_gap = np.hypot(position[0] - bx, position[1] - by)
            if marker_gap <= cfg.target_reacquire_radius:
                state.ignored_stale_marker = None
                print(f"[*] Detected marked.png on the nearest Skeleton at ({position[0]}, {position[1]}); adopting target...")
                best["click_x"], best["click_y"] = position
                best["last_entity_check"], best["last_entity_seen_time"] = now, now
                state.current_target = best
                state.target_click_time = now
                state.confirmed_locked = True
                return
            # An exact mark on a farther/unknown mob must not override the
            # nearest-nameplate rule while starting or recovering the bot.
        if not self.device.click(bx, by):
            state.no_target_since = now
            print(f"[!] Could not tap the selected {self.profile.target_name} at ({bx}, {by}); retrying detection.")
            return
        print(f"[+] Found {len(mobs)} '{self.profile.target_name}' (selected nearest score:{best['score']:.2f} scale:{best['scale']:.1f}x)")
        print(f"    -> Clicking mob at ({bx}, {by}) [dist:{int(best['distance'])}px]")
        best["last_entity_check"], best["last_entity_seen_time"] = now, now
        best["lock_retry_count"] = 0
        state.current_target = best
        state.ignored_stale_marker = None
        state.target_click_time = now
        state.confirmed_locked = False
        state.idle_mark_check_needed = False
        state.no_target_since = now

    def _check_exhausted_caption(self, frame: np.ndarray, now: float) -> None:
        state, cfg = self.state, self.config
        if state.current_target is not state.exhausted_caption_target:
            state.exhausted_caption_target = state.current_target
            state.exhausted_caption_check_at = None
            state.exhausted_caption_next_poll = 0.0
        waiting_for_confirmation = state.exhausted_caption_check_at is not None
        due = state.current_target is not None and (
            now >= state.exhausted_caption_check_at
            if waiting_for_confirmation
            else now >= state.exhausted_caption_next_poll
        )
        caption_visible = False
        if due:
            caption_visible = self.vision.exhausted_caption(frame, self.templates.get("exhausted-caption"))
        if state.current_target is None:
            state.exhausted_caption_check_at = None
        elif waiting_for_confirmation and due:
            state.exhausted_caption_check_at = None
            state.exhausted_caption_next_poll = now + cfg.exhausted_caption_poll_seconds
            if caption_visible:
                target = state.current_target
                tx, ty = target["click_x"], target["click_y"]
                if (
                    state.exhausted_window_started_at is None
                    or now - state.exhausted_window_started_at > cfg.exhausted_travel_window_seconds
                ):
                    state.consecutive_exhausted = 0
                    state.exhausted_window_started_at = now
                state.consecutive_exhausted += 1
                print(
                    f"[~] Exhausted caption still visible after {cfg.exhausted_caption_confirm_seconds:.0f}s; "
                    f"releasing target at ({tx}, {ty}) without position blacklisting."
                )
                if state.consecutive_exhausted >= max(1, cfg.exhausted_travel_trigger_count):
                    state.exhausted_travel_pending = True
                    print(
                        f"[!] {state.consecutive_exhausted} consecutive exhausted targets within "
                        f"{cfg.exhausted_travel_window_seconds:.0f}s; "
                        f"starting farthest-waypoint minimap failsafe."
                    )
                state.current_target = None
                state.confirmed_locked = False
                state.idle_mark_check_needed = False
                state.no_target_since = now
            else:
                self._reset_exhaustion_streak()
                if state.confirmed_locked:
                    print("[+] Exhausted caption cleared; keeping the confirmed target.")
                else:
                    tx, ty = state.current_target["click_x"], state.current_target["click_y"]
                    print(
                        "[+] Exhausted caption cleared before red-square lock; "
                        f"releasing pending target at ({tx}, {ty}) and rescanning."
                    )
                    state.current_target = None
                    state.confirmed_locked = False
                    state.idle_mark_check_needed = True
                    state.no_target_since = now
        elif due:
            state.exhausted_caption_next_poll = now + cfg.exhausted_caption_poll_seconds
            if caption_visible:
                state.exhausted_caption_check_at = now + cfg.exhausted_caption_confirm_seconds
                print(f"[~] Exhausted caption detected; checking again in {cfg.exhausted_caption_confirm_seconds:.0f}s.")

    def _run_exhausted_travel_failsafe(self, now: float) -> None:
        """Move to the farthest safe visible map point after repeated exhaustion."""
        state, cfg = self.state, self.config
        state.exhausted_travel_pending = False
        state.no_target_since = now
        if not cfg.enable_minimap_walk or self.navigator is None:
            print("[!] Exhausted travel failsafe skipped because minimap walking is disabled.")
            self._reset_exhaustion_streak()
            return
        moved = self.navigator.travel(
            farthest=True,
            keep_open_seconds=max(0.0, cfg.exhausted_travel_hold_seconds),
        )
        state.last_minimap_time = time.monotonic()
        state.no_target_since = state.last_minimap_time
        self._reset_exhaustion_streak()
        if moved:
            # The old screen coordinate belongs to the exhausted mob.  After
            # travelling, allow a fresh scan in the new camera position.
            print("[+] Exhausted travel failsafe complete; resuming normal target scanning.")
        else:
            print("[!] Exhausted travel failsafe could not move; resuming normal scanning.")

    def _reset_exhaustion_streak(self) -> None:
        self.state.consecutive_exhausted = 0
        self.state.exhausted_window_started_at = None

    def _update_target(self, frame: np.ndarray, cx: int, cy: int, now: float) -> None:
        assert self.device is not None
        state, cfg = self.state, self.config
        target = state.current_target
        assert target is not None
        tx, ty = target["click_x"], target["click_y"]

        # Check the active lock locally first, then globally if the mob moved
        # outside the small crop. A confirmed marker stays locked as long as
        # the marker remains visible, regardless of fight duration.
        has_red, position = self.vision.red_square(
            frame, (tx, ty), radius=int(cfg.target_reacquire_radius)
        )
        if not has_red and state.confirmed_locked:
            has_red, position = self.vision.red_square(frame)

        if has_red and position is not None:
            target["click_x"], target["click_y"] = position
            tx, ty = position
            target["red_loss_since"] = None

        refresh_interval = max(0.05, cfg.target_position_refresh_interval)
        if now - target.get("last_entity_check", 0.0) >= refresh_interval:
            target["last_entity_check"] = now
            match = self._find_locked_entity(frame, cx, cy, target, now)
            if match is not None:
                target["last_entity_seen_time"] = now
                # When the red outline flickers, keep tracking from the
                # target's visible nameplate instead of its stale point.
                if not has_red:
                    target["click_x"], target["click_y"] = match["click_x"], match["click_y"]
                    target["nx"] = match.get("nx", target.get("nx", tx))
                    target["ny"] = match.get("ny", target.get("ny", ty))
                    tx, ty = target["click_x"], target["click_y"]
            elif (
                state.confirmed_locked
                and now - target.get("last_entity_seen_time", state.target_click_time)
                >= cfg.target_entity_loss_timeout
                and state.exhausted_caption_check_at is None
            ):
                print(
                    f"[+] Target nameplate missing for {cfg.target_entity_loss_timeout:.1f}s; "
                    f"releasing stale lock at ({tx}, {ty}) and searching again."
                )
                state.current_target = None
                state.confirmed_locked = False
                self._reset_exhaustion_streak()
                state.idle_mark_check_needed = False
                state.no_target_since = now
                state.pickup_pending = True
                state.ignored_stale_marker = (tx, ty)
                return
            elif (
                not state.confirmed_locked
                and not has_red
                and now - target.get("last_entity_seen_time", state.target_click_time)
                >= cfg.target_entity_loss_timeout
                and state.exhausted_caption_check_at is None
            ):
                print(
                    f"[+] Nearest Skeleton nameplate disappeared for "
                    f"{cfg.target_entity_loss_timeout:.1f}s before marked.png appeared; "
                    f"rescanning at ({tx}, {ty})."
                )
                state.current_target = None
                state.confirmed_locked = False
                self._reset_exhaustion_streak()
                state.idle_mark_check_needed = False
                state.no_target_since = now
                state.ignored_stale_marker = None
                return

        if has_red and position is not None:
            state.idle_mark_check_needed = False
            if not state.confirmed_locked:
                last_red = target.get("red_last_seen", now)
                observations = target.get("red_observations", 0)
                target["red_observations"] = observations + 1 if now - last_red <= 0.5 else 1
                target["red_last_seen"] = now
                # One weak/false detection is not a lock. Require two
                # consecutive observations from the clicked target.
                if target["red_observations"] < 2:
                    return
                state.confirmed_locked = True
                print(f"[*] Target locked with red square! Fighting at ({tx}, {ty})...")
            return

        if state.confirmed_locked:
            loss_since = target.get("red_loss_since")
            if loss_since is None:
                target["red_loss_since"] = now
                return
            if (
                now - loss_since < cfg.target_red_loss_timeout
                or state.exhausted_caption_check_at is not None
            ):
                return
            print(
                f"[+] Red square absent for {cfg.target_red_loss_timeout:.1f}s; "
                f"releasing target at ({tx}, {ty}) and searching again."
            )
            state.current_target = None
            state.confirmed_locked = False
            self._reset_exhaustion_streak()
            state.idle_mark_check_needed = False
            state.no_target_since = now
            state.pickup_pending = True
            state.ignored_stale_marker = (tx, ty)
            return

        # Keep the nearest visible Skeleton pending until its actual marker is
        # found. Retry one missed tap, then skip that nameplate briefly so the
        # bot can move on instead of waiting forever on an unmarked target.
        target["red_observations"] = 0
        target.pop("red_last_seen", None)
        if (
            not state.confirmed_locked
            and not has_red
            and state.exhausted_caption_check_at is None
            and now - state.target_click_time >= cfg.target_entity_loss_timeout
        ):
            retries = int(target.get("lock_retry_count", 0))
            if retries < 1:
                target["lock_retry_count"] = retries + 1
                state.target_click_time = now
                if self.device.click(tx, ty):
                    print(
                        f"[~] No red mark after {cfg.target_entity_loss_timeout:.1f}s; "
                        f"retrying the target click at ({tx}, {ty})."
                    )
                else:
                    print(f"[!] Target retry click failed at ({tx}, {ty}); will keep checking for its red mark.")
                return
            ignored_position = (
                int(target.get("nx", tx)), int(target.get("ny", ty))
            )
            state.unconfirmed_target_position = ignored_position
            state.unconfirmed_target_until = now + cfg.target_entity_loss_timeout
            print(
                f"[!] No red mark after retry; skipping the unconfirmed target "
                f"at {ignored_position} briefly and searching for another mob."
            )
            state.current_target = None
            state.confirmed_locked = False
            state.idle_mark_check_needed = True
            state.no_target_since = now
            self._reset_exhaustion_streak()

    def _find_locked_entity(
        self, frame: np.ndarray, cx: int, cy: int, target: TargetMatch, now: float,
    ) -> Optional[TargetMatch]:
        """Find this target's nameplate close to its current screen position."""
        assert self.vision is not None
        match_radius = max(60, int(self.config.target_reacquire_radius))
        search_radius = max(150, match_radius)
        tx, ty = target["click_x"], target["click_y"]
        matches = self.vision.find_targets(
            frame, self.target_templates, cx, cy,
            search_center=(tx, ty), search_radius=search_radius, include_deadzone=True,
        )
        if not matches:
            return None
        nearest = min(
            matches,
            key=lambda match: np.hypot(match["click_x"] - tx, match["click_y"] - ty),
        )
        distance = np.hypot(nearest["click_x"] - tx, nearest["click_y"] - ty)
        return nearest if distance <= match_radius else None

    def _preview(self, frame: np.ndarray, cx: int, cy: int) -> None:
        if not self.config.show_preview:
            if self.device is not None and not self.device.use_adb:
                time.sleep(0.01)
            return
        display = frame.copy()
        cv2.circle(display, (cx, cy), self.config.player_deadzone_radius, (255, 255, 0), 1)
        if self.state.current_target is not None:
            tx, ty = self.state.current_target["click_x"], self.state.current_target["click_y"]
            color = (0, 0, 255) if self.state.confirmed_locked else (0, 255, 0)
            cv2.circle(display, (tx, ty), 12, color, 2)
            cv2.line(display, (cx, cy), (tx, ty), color, 1)
        fh, fw = frame.shape[:2]
        scale = min(1.0, 1600 / fw, 900 / fh)
        cv2.imshow("Rucoy Online Bot - AFK Preview", cv2.resize(display, (int(fw * scale), int(fh * scale))))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[+] Quit key pressed.")
            raise BotQuit
