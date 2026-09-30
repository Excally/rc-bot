"""Combat, target tracking, and recovery state machine."""
from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig, CONFIG
from .device import InputDevice
from .exceptions import BotQuit, RepeatedOutputReset
from .models import RetryContext, TargetMatch
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
        purple_key = self.profile.target_purple_template_key
        purple = images.get(purple_key) if purple_key else None
        if purple is not None:
            self.target_templates.append((purple, "purple"))
        elif purple_key:
            print(f"[~] Optional purple target template '{purple_key}' is unavailable.")
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
        print(f"[+] Hitbox Offset: {self.config.mob_body_y_offset}px below nametag (lowered for clean hits)")
        print(f"[+] Target lock check: {self.config.target_lock_timeout:.1f}s; retry blacklist after {self.config.target_retry_limit} misses ({self.config.target_retry_blacklist_seconds:.0f}s)")
        print(f"[+] Exhaustion failsafe: {self.config.exhausted_travel_trigger_count} consecutive exhausted targets -> farthest minimap point, hold {self.config.exhausted_travel_hold_seconds:.0f}s")
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
        if state.blacklist and state.frame_count % 30 == 0:
            state.blacklist = {position: expiry for position, expiry in state.blacklist.items() if now < expiry}

        if state.current_target is None:
            self._acquire_target(frame, cx, cy, now)
        if state.pickup_pending:
            pickup = self.vision.pickup_position(frame)
            if pickup is not None:
                print(f"[+] Pickup available after kill; tapping ({pickup[0]}, {pickup[1]}).")
                if self.device.click(*pickup):
                    state.pickup_pending = False
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
        state.locked_since = None
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
        if state.idle_mark_check_needed or (state.ignored_exhausted_marker is None and now >= state.next_idle_mark_check_at):
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
        if has_red and position is not None and state.ignored_exhausted_marker is not None:
            if np.hypot(position[0] - state.ignored_exhausted_marker[0], position[1] - state.ignored_exhausted_marker[1]) < 140:
                has_red = False
        if has_red and position is not None:
            state.ignored_exhausted_marker = None
            state.ignored_stale_marker = None
            print(f"[*] Detected active target lock at ({position[0]}, {position[1]})! Adopting target...")
            state.current_target = {
                "click_x": position[0], "click_y": position[1],
                "distance": np.hypot(position[0] - cx, position[1] - cy),
                "score": 1.0, "scale": 1.0,
                "last_entity_check": now,
                "last_entity_seen_time": now,
            }
            state.target_click_time = now
            state.confirmed_locked = True
            state.locked_since = now
            state.retry_context = None
            return
        mobs = self.vision.find_targets(frame, self.target_templates, cx, cy, state.blacklist, now)
        if not mobs:
            if now - state.last_log_time > 5.0:
                print(f"[...] Scanning for '{self.profile.target_name}'... (frame #{state.frame_count}, res {frame.shape[1]}x{frame.shape[0]})")
                state.last_log_time = now
            return
        best = mobs[0]
        if state.retry_context is not None:
            retry = state.retry_context
            if now - retry["time"] <= 5.0:
                nearby = min(mobs, key=lambda candidate: np.hypot(candidate["click_x"] - retry["x"], candidate["click_y"] - retry["y"]))
                if np.hypot(nearby["click_x"] - retry["x"], nearby["click_y"] - retry["y"]) <= 180:
                    # Retrying a failed position is allowed only when it is
                    # still the closest visible mob.  This prevents a stale
                    # retry from overriding a newly detected nearer target.
                    if nearby["distance"] <= best["distance"]:
                        best = nearby
                        best["retry_grid"] = retry["grid"]
                else:
                    state.retry_context = None
            else:
                state.retry_context = None
        bx, by = best["click_x"], best["click_y"]
        best.setdefault("retry_grid", (int(bx // 40), int(by // 40)))
        if not self.device.click(bx, by):
            state.no_target_since = now
            print(f"[!] Could not tap the selected {self.profile.target_name} at ({bx}, {by}); retrying detection.")
            return
        print(f"[+] Found {len(mobs)} '{self.profile.target_name}' (selected nearest score:{best['score']:.2f} scale:{best['scale']:.1f}x)")
        print(f"    -> Clicking mob at ({bx}, {by}) [dist:{int(best['distance'])}px]")
        best["initial_click"], best["last_position_refresh"] = (bx, by), now
        best["last_entity_check"], best["last_entity_seen_time"] = now, now
        state.current_target = best
        state.ignored_stale_marker = None
        state.target_click_time = now
        state.confirmed_locked = False
        state.locked_since = None
        state.idle_mark_check_needed = False
        state.no_target_since = now
        state.retry_context = None

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
                expiry = now + cfg.exhausted_mob_blacklist_seconds
                for grid in ((int(tx // 40), int(ty // 40)), target.get("retry_grid"), tuple(int(value // 40) for value in target.get("initial_click", (tx, ty)))):
                    if grid is not None:
                        state.blacklist[grid] = max(state.blacklist.get(grid, 0.0), expiry)
                state.fail_counts.pop(target.get("retry_grid"), None)
                state.ignored_exhausted_marker = (tx, ty)
                state.retry_context = None
                state.consecutive_exhausted += 1
                print(f"[~] Exhausted caption still visible after {cfg.exhausted_caption_confirm_seconds:.0f}s; skipping mob at ({tx}, {ty}) for {cfg.exhausted_mob_blacklist_seconds:.0f}s.")
                if state.consecutive_exhausted >= max(1, cfg.exhausted_travel_trigger_count):
                    state.exhausted_travel_pending = True
                    print(
                        f"[!] {state.consecutive_exhausted} consecutive exhausted targets; "
                        f"starting farthest-waypoint minimap failsafe."
                    )
                state.current_target = None
                state.confirmed_locked = False
                state.locked_since = None
                state.idle_mark_check_needed = False
                state.no_target_since = now
            else:
                print("[+] Exhausted caption cleared; keeping the current target.")
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
            state.consecutive_exhausted = 0
            return
        moved = self.navigator.travel(
            farthest=True,
            keep_open_seconds=max(0.0, cfg.exhausted_travel_hold_seconds),
        )
        state.last_minimap_time = time.monotonic()
        state.no_target_since = state.last_minimap_time
        state.consecutive_exhausted = 0
        if moved:
            # The old screen coordinate belongs to the exhausted mob.  After
            # travelling, allow a fresh scan in the new camera position.
            state.ignored_exhausted_marker = None
            print("[+] Exhausted travel failsafe complete; resuming normal target scanning.")
        else:
            print("[!] Exhausted travel failsafe could not move; resuming normal scanning.")

    def _update_target(self, frame: np.ndarray, cx: int, cy: int, now: float) -> None:
        assert self.device is not None
        state, cfg = self.state, self.config
        target = state.current_target
        assert target is not None
        tx, ty = target["click_x"], target["click_y"]

        # Check the active lock locally first, then globally if the mob moved
        # outside the small crop. A missing marker is not a reason to adopt a
        # different mob; it starts the short red-loss timer below.
        if (
            state.confirmed_locked
            and state.locked_since is not None
            and now - state.locked_since >= cfg.target_stall_timeout
        ):
            self._abandon_stalled_target(tx, ty, now)
            return

        has_red, position = self.vision.red_square(
            frame, (tx, ty), radius=int(cfg.target_reacquire_radius)
        )
        if not has_red and state.confirmed_locked:
            has_red, position = self.vision.red_square(frame)

        if has_red and position is not None and state.ignored_exhausted_marker is not None:
            old_distance = np.hypot(position[0] - state.ignored_exhausted_marker[0], position[1] - state.ignored_exhausted_marker[1])
            target_distance = np.hypot(position[0] - tx, position[1] - ty)
            if old_distance < 140 and target_distance >= old_distance - 25:
                has_red, position = False, None

        if has_red and position is not None:
            state.idle_mark_check_needed = False
            target["click_x"], target["click_y"] = position
            tx, ty = position
            target["red_loss_since"] = None
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
                state.locked_since = now
                state.fail_counts.pop(target.get("retry_grid"), None)
                state.retry_context = None
                state.ignored_exhausted_marker = None
                print(f"[*] Target locked with red square! Fighting at ({tx}, {ty})...")
            return

        if state.confirmed_locked:
            loss_since = target.get("red_loss_since")
            if loss_since is None:
                target["red_loss_since"] = now
                return
            if now - loss_since < cfg.target_red_loss_timeout:
                return
            print(
                f"[+] Red square absent for {cfg.target_red_loss_timeout:.1f}s; "
                f"releasing target at ({tx}, {ty}) and searching again."
            )
            state.current_target = None
            state.confirmed_locked = False
            state.locked_since = None
            state.consecutive_exhausted = 0
            state.idle_mark_check_needed = False
            state.no_target_since = now
            state.pickup_pending = True
            state.ignored_stale_marker = (tx, ty)
            return

        # A click that never produces a stable red marker is a failed lock,
        # not a target worth waiting on.
        target["red_observations"] = 0
        target.pop("red_last_seen", None)
        if now - state.target_click_time > cfg.target_lock_timeout:
            grid = (int(tx // 40), int(ty // 40))
            retry_grid = target.get("retry_grid", grid)
            failures = state.fail_counts.get(retry_grid, 0) + 1
            state.fail_counts[retry_grid] = failures
            if failures >= cfg.target_retry_limit:
                expiry = now + cfg.target_retry_blacklist_seconds
                state.blacklist[grid] = state.blacklist[retry_grid] = expiry
                state.fail_counts.pop(retry_grid, None)
                state.retry_context = None
                print(f"[-] No target lock after {failures} fresh-position attempts; skipping nearby location for {cfg.target_retry_blacklist_seconds:.0f}s.")
            else:
                state.retry_context = {"x": tx, "y": ty, "grid": retry_grid, "time": now}
                print(f"[-] No target lock after {cfg.target_lock_timeout:.1f}s; checking a fresh mob position before retrying.")
            state.current_target = None
            state.confirmed_locked = False
            state.locked_since = None
            state.consecutive_exhausted = 0
            state.idle_mark_check_needed = False
            state.no_target_since = now
    def _abandon_stalled_target(self, tx: int, ty: int, now: float) -> None:
        """Release a lock that has survived too long without defeating its mob."""
        state, cfg = self.state, self.config
        expiry = now + cfg.target_stall_blacklist_seconds
        target = state.current_target or {}

        # The red marker is usually a few pixels below the click point.  Add
        # the adjacent vertical bucket as well so that the same mob cannot be
        # rediscovered immediately after a stall.
        grids: set[Optional[tuple[int, int]]] = set()

        def add_position(position: tuple[int, int] | tuple[float, float]) -> None:
            grid_x, grid_y = int(position[0] // 40), int(position[1] // 40)
            grids.update({(grid_x, grid_y), (grid_x, grid_y - 1), (grid_x, grid_y + 1)})

        add_position((tx, ty))
        retry_grid = target.get("retry_grid")
        if retry_grid is not None:
            grids.add(retry_grid)
        initial_click = target.get("initial_click")
        if initial_click is not None:
            add_position(initial_click)
        for grid in grids:
            if grid is not None:
                state.blacklist[grid] = max(state.blacklist.get(grid, 0.0), expiry)

        state.ignored_exhausted_marker = (tx, ty)
        state.retry_context = None
        state.current_target = None
        state.confirmed_locked = False
        state.locked_since = None
        state.consecutive_exhausted = 0
        state.idle_mark_check_needed = False
        state.no_target_since = now
        print(
            f"[!] Target lock stalled for {cfg.target_stall_timeout:.0f}s at "
            f"({tx}, {ty}); skipping it for "
            f"{cfg.target_stall_blacklist_seconds:.0f}s and searching again."
        )

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
