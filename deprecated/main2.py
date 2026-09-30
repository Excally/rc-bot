"""Readable rewrite of the Rucoy Online hunting bot.

The original script grew by accretion around one large ``main`` loop.  This
version keeps the same external behaviour, but gives each concern a home:

* :class:`InputDevice` hides ADB and Win32 input/capture differences.
* :class:`TemplateStore` owns image assets.
* :class:`FrameVision` contains stateless image recognition.
* :class:`MinimapNavigator` owns route registration and waypoint state.
* :class:`RucoyBot` is the combat/recovery state machine.

Run this file directly for the desktop bot.  ``main.py`` is left untouched so
the existing launcher and the phone helper remain backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import glob
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BotConfig:
    target_name: str = "Skeleton Lv.75"
    target_template_key: str = "skeleton_lv75"
    target_purple_template_key: str = "purple_name_skeleton_lv75"
    minimap_reference_key: str = "minimap-skeleton-layout"
    prefer_adb: bool = True
    window_title: str = "MSI App Player"
    show_preview: bool = False

    enable_minimap_walk: bool = True
    minimap_idle_delay: float = 0.9
    minimap_cooldown: float = 3.0
    minimap_safe_view_fraction: float = 0.70
    minimap_min_click_distance: float = 100
    minimap_max_click_distance: float = 260
    # Retained as a compatibility setting for callers; raw match score is
    # reported but is not an acceptance gate (the overlay background changes).
    minimap_align_min_score: float = 0.45
    minimap_align_min_gap: float = 0.025
    minimap_geometry_min_score: float = 0.40
    # The map overlay is drawn over a changing game scene, so raw grayscale
    # template scores are not reliable.  Require a meaningful amount of the
    # reference linework to be visible before accepting geometry-only matches.
    minimap_min_visible_reference_fraction: float = 0.35
    minimap_min_visible_reference_pixels: int = 30000

    margin_top: float = 0.12
    margin_bottom: float = 0.06
    margin_left: float = 0.06
    margin_right: float = 0.06
    mob_body_y_offset: int = 52
    player_deadzone_radius: int = 60
    grid_tile_size: int = 100

    # The saved Skeleton Lv.75 label matches the live 1600x900 frames at
    # roughly 0.71 because the in-game outline and antialiasing vary.
    match_threshold: float = 0.70
    min_name_template_overlap: float = 0.65
    target_scan_width: int = 1100
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
    target_reacquire_radius: float = 180
    ui_check_interval: float = 0.35
    target_stall_timeout: float = 30.0
    target_stall_blacklist_seconds: float = 10.0
    target_defeated_grace_time: float = 0.30
    exhausted_caption_confirm_seconds: float = 10.0
    exhausted_caption_poll_seconds: float = 0.25
    exhausted_mob_blacklist_seconds: float = 600.0
    exhausted_travel_trigger_count: int = 2
    exhausted_travel_hold_seconds: float = 10.0
    repeated_output_limit: int = 20


CONFIG = BotConfig()
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# Console watchdog
# ---------------------------------------------------------------------------


class RepeatedOutputReset(Exception):
    """Raised when one warning/status line loops long enough to be unsafe."""

    def __init__(self, pattern: str):
        super().__init__(pattern)
        self.pattern = pattern


class BotQuit(Exception):
    """Internal signal used when the preview window receives ``q``."""


class RepeatedOutputGuard:
    """Detect a stuck recovery loop from normalized console output."""

    _RESET_LINES = (
        "[+] Red square gone",
        "[+] Back action closed",
        "[+] System Back closed",
        "[+] Screen capture recovered",
        "[*] Target locked",
        "[*] Detected active target lock",
    )

    def __init__(self, limit: int = CONFIG.repeated_output_limit):
        self.limit = limit
        self.counts: dict[str, int] = {}

    def observe(self, message: str) -> None:
        line = message.strip()
        if not line:
            return
        if line.startswith(self._RESET_LINES):
            self.counts.clear()
            return

        should_watch = line.startswith(("[!]", "[-]", "[...]"))
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
        self.counts[pattern] = self.counts.get(pattern, 0) + 1
        if self.counts[pattern] >= self.limit:
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


# ---------------------------------------------------------------------------
# Input and capture backends
# ---------------------------------------------------------------------------


def _find_bluestacks_adb() -> Optional[str]:
    for candidate in (
        r"C:\Program Files\BlueStacks_msi5\HD-Adb.exe",
        r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_msi5\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return shutil.which("adb")


def _find_adb_port() -> int:
    patterns = (
        r"C:\ProgramData\BlueStacks_msi5\bluestacks.conf",
        r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf",
        r"C:\ProgramData\BlueStacks*\**\*.conf",
    )
    for pattern in patterns:
        for filename in glob.glob(pattern, recursive=True):
            if not os.path.isfile(filename):
                continue
            try:
                text = Path(filename).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if "status.adb_port" not in line:
                    continue
                value = line.split("=", 1)[-1].strip().strip("\"' ")
                if value.isdigit():
                    return int(value)
    return 5555


class InputDevice:
    """Unified screen capture and tap interface used by the bot loop."""

    def __init__(self, adb_path: Optional[str], serial: Optional[str], hwnd: Any, use_adb: bool):
        self.adb_path = adb_path
        self.serial = serial
        self.hwnd = hwnd
        self.use_adb = use_adb
        self._raw_capture_supported: Optional[bool] = None

    @classmethod
    def connect(cls, config: BotConfig) -> Optional["InputDevice"]:
        if config.prefer_adb:
            adb_path, serial = cls._connect_adb()
            if adb_path and serial:
                print("[+] MODE: ADB (100% True Background - No mouse takeover!)")
                return cls(adb_path, serial, None, True)
            print("[!] ADB unavailable. Falling back to Win32 window mode...")

        try:
            top, title = cls._find_window(config.window_title)
            hwnd = cls._find_game_child(top)
            print(f"[+] Found emulator window: '{title}' (HWND: {hwnd})")
            print("[+] MODE: Win32 Window Fallback")
            return cls(None, None, hwnd, False)
        except Exception as error:
            print(f"[-] {error}")
            return None

    @staticmethod
    def _connect_adb() -> tuple[Optional[str], Optional[str]]:
        adb_path = _find_bluestacks_adb()
        if not adb_path:
            return None, None
        serial = f"127.0.0.1:{_find_adb_port()}"
        print(f"[+] Found ADB executable: {adb_path}")
        print(f"[+] Connecting to emulator instance at {serial}...")
        try:
            subprocess.run(
                [adb_path, "connect", serial], capture_output=True, text=True, timeout=15
            )
            result = subprocess.run(
                [adb_path, "devices"], capture_output=True, text=True, timeout=5
            )
            if serial in result.stdout and "device" in result.stdout:
                print(f"[+] ADB connected successfully to {serial}!")
                return adb_path, serial
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[!] ADB connection failed: {error}")
        return None, None

    @staticmethod
    def _find_window(title: str) -> tuple[Any, str]:
        import win32gui

        exact: list[Any] = [None, ""]

        def exact_callback(hwnd: Any, _: Any) -> bool:
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == title:
                exact[0], exact[1] = hwnd, title
                return False
            return True

        try:
            win32gui.EnumWindows(exact_callback, None)
        except Exception:
            pass
        if exact[0]:
            return exact[0], exact[1]

        def fuzzy_callback(hwnd: Any, _: Any) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            text = win32gui.GetWindowText(hwnd)
            cls = win32gui.GetClassName(hwnd)
            if title.lower() in text.lower() and "CabinetW" not in cls and "Chrome" not in cls:
                exact[0], exact[1] = hwnd, text
                return False
            return True

        try:
            win32gui.EnumWindows(fuzzy_callback, None)
        except Exception:
            pass
        if exact[0]:
            return exact[0], exact[1]
        raise RuntimeError(f"Window '{title}' not found!")

    @staticmethod
    def _find_game_child(parent: Any) -> Any:
        import win32gui

        found: list[Any] = [None]

        def callback(hwnd: Any, _: Any) -> bool:
            if "BlueStacks" in win32gui.GetClassName(hwnd):
                found[0] = hwnd
                return False
            return True

        try:
            win32gui.EnumChildWindows(parent, callback, None)
        except Exception:
            pass
        return found[0] or parent

    def capture(self) -> Optional[np.ndarray]:
        if self.use_adb:
            try:
                # BlueStacks returns a fixed-size RGBA framebuffer when the
                # PNG flag is omitted.  Avoiding PNG compression/decompression
                # saves roughly 100 ms per frame on the target emulator.
                if self._raw_capture_supported is not False:
                    result = subprocess.run(
                        [self.adb_path, "-s", self.serial, "exec-out", "screencap"],
                        capture_output=True, timeout=3,
                    )
                    raw = result.stdout
                    if len(raw) >= 16:
                        width, height, pixel_format, _ = np.frombuffer(raw[:16], dtype="<u4")
                        expected_bytes = int(width) * int(height) * 4
                        if pixel_format == 1 and len(raw) >= 16 + expected_bytes:
                            self._raw_capture_supported = True
                            rgba = np.frombuffer(raw, dtype=np.uint8, offset=16, count=expected_bytes)
                            rgba = rgba.reshape((int(height), int(width), 4))
                            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    self._raw_capture_supported = False

                # Keep a PNG fallback for devices whose screencap command does
                # not expose the raw RGBA format.
                png = subprocess.run(
                    [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=3,
                )
                if png.stdout:
                    return cv2.imdecode(np.frombuffer(png.stdout, np.uint8), cv2.IMREAD_COLOR)
            except (OSError, subprocess.SubprocessError):
                pass
            return None

        try:
            import win32gui
            from mss import MSS

            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                return None
            screen_x, screen_y = win32gui.ClientToScreen(self.hwnd, (0, 0))
            with MSS() as screen:
                shot = np.array(screen.grab({
                    "left": screen_x, "top": screen_y,
                    "width": width, "height": height,
                }))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def click(self, x: int | float, y: int | float, synchronous: bool = False) -> bool:
        x, y = int(x), int(y)
        if self.use_adb:
            command = [self.adb_path, "-s", self.serial, "shell", "input", "tap", str(x), str(y)]
            try:
                if synchronous:
                    result = subprocess.run(
                        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0
                    )
                    if result.returncode:
                        print(f"[!] ADB tap failed at ({x}, {y}), exit code {result.returncode}.")
                        return False
                else:
                    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, subprocess.SubprocessError) as error:
                print(f"[!] ADB tap failed at ({x}, {y}): {error}")
                return False

        try:
            import win32api
            import win32con
            import win32gui

            screen_x, screen_y = win32gui.ClientToScreen(self.hwnd, (x, y))
            original = win32api.GetCursorPos()
            try:
                win32api.SetCursorPos((screen_x, screen_y))
                time.sleep(0.03)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.06)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.03)
                win32api.SetCursorPos(original)
            except Exception:
                lparam = (y << 16) | (x & 0xFFFF)
                win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
                time.sleep(0.05)
                win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, lparam)
            return True
        except Exception as error:
            print(f"[!] Win32 click error: {error}")
            return False

    def back(self) -> None:
        if self.use_adb:
            try:
                subprocess.Popen(
                    [self.adb_path, "-s", self.serial, "shell", "input", "keyevent", "4"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                print(f"[!] Could not dismiss unexpected UI: {error}")
            return
        try:
            import win32con
            import win32gui

            win32gui.PostMessage(self.hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
            win32gui.PostMessage(self.hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
        except Exception as error:
            print(f"[!] Could not dismiss unexpected UI: {error}")


# ---------------------------------------------------------------------------
# Templates and stateless frame vision
# ---------------------------------------------------------------------------


class TemplateStore:
    REQUIRED = {
        "skeleton_lv75", "purple_name_skeleton_lv75", "minimapicon", "backicon",
        "minimap-skeleton-layout", "pickup-available", "pickup-not-yet", "exhausted-caption",
    }

    def __init__(self, directory: Path = TEMPLATE_DIR):
        self.directory = directory
        self.images: dict[str, np.ndarray] = {}

    def load(self) -> dict[str, np.ndarray]:
        self.images.clear()
        if not self.directory.exists():
            return self.images
        for filename in self.directory.iterdir():
            if filename.suffix.lower() not in {".png", ".jpg"} or filename.stem not in self.REQUIRED:
                continue
            mode = cv2.IMREAD_UNCHANGED if filename.stem in {"backicon", "minimapicon"} else cv2.IMREAD_COLOR
            image = cv2.imread(str(filename), mode)
            if image is not None:
                self.images[filename.stem] = image
        return self.images

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.images.get(name)


class FrameVision:
    """Pure image operations.  Methods do not send taps or mutate bot state."""

    def __init__(self, config: BotConfig = CONFIG, templates: Optional[TemplateStore] = None):
        self.config = config
        self.templates = templates
        self._marked_template_image: Optional[np.ndarray] = None
        self._red_frame: Any = None
        self._red_mask: Optional[np.ndarray] = None
        self._caption_cache: Any = None
        self._target_mask_frame: Any = None
        self._target_masks: dict[str, np.ndarray] = {}

    @staticmethod
    def minimap_white_mask(image: np.ndarray) -> np.ndarray:
        channels = image.astype(np.int16)
        spread = channels.max(axis=2) - channels.min(axis=2)
        return ((channels.min(axis=2) >= 220) & (spread <= 35)).astype(np.uint8) * 255

    @staticmethod
    def is_disconnected(frame: Optional[np.ndarray]) -> bool:
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        title = gray[int(fh * 0.18):int(fh * 0.32), int(fw * 0.32):int(fw * 0.68)]
        reconnect = gray[int(fh * 0.58):int(fh * 0.74), int(fw * 0.14):int(fw * 0.49)]
        return (
            np.mean(gray < 24) > 0.82
            and np.count_nonzero(title > 175) > 250
            and np.count_nonzero(reconnect > 175) > 180
        )

    def find_fixed_icon(
        self, frame: Optional[np.ndarray], icon: Optional[np.ndarray], expected_center: tuple[int, int],
        threshold: Optional[float] = None, search_radius: tuple[int, int] = (90, 70),
        scales: tuple[float, ...] = (1.0, 0.9, 0.85, 0.8, 0.75, 1.1),
    ) -> Optional[tuple[int, int]]:
        if frame is None or icon is None:
            return None
        threshold = self.config.ui_match_threshold if threshold is None else threshold
        alpha = icon[:, :, 3] if icon.ndim == 3 and icon.shape[2] == 4 else None
        if alpha is not None:
            icon = icon[:, :, :3]

        fh, fw = frame.shape[:2]
        sx, sy = fw / 1600.0, fh / 900.0
        center_x, center_y = int(expected_center[0] * sx), int(expected_center[1] * sy)
        radius_x, radius_y = int(search_radius[0] * sx), int(search_radius[1] * sy)
        best_score, best_center = -1.0, None
        match_threshold = max(threshold, self.config.ui_match_threshold) if alpha is not None else threshold

        for adjustment in scales:
            scale = sx * adjustment
            scaled = cv2.resize(icon, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ih, iw = scaled.shape[:2]
            if ih >= fh or iw >= fw:
                continue
            left = max(0, center_x - radius_x - iw // 2)
            top = max(0, center_y - radius_y - ih // 2)
            right = min(fw, center_x + radius_x + iw // 2)
            bottom = min(fh, center_y + radius_y + ih // 2)
            region = frame[top:bottom, left:right]
            if region.shape[0] < ih or region.shape[1] < iw:
                continue
            if alpha is not None:
                scaled_alpha = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_NEAREST)
                mask = np.where(scaled_alpha > 0, 255, 0).astype(np.uint8)
                result = cv2.matchTemplate(region, scaled, cv2.TM_SQDIFF_NORMED, mask=mask)
                minimum, _, location, _ = cv2.minMaxLoc(result)
                score = 1.0 - minimum
            else:
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_center = (left + location[0] + iw // 2, top + location[1] + ih // 2)
            if score >= match_threshold + 0.02:
                return best_center
        return best_center if best_score >= match_threshold else None

    def back_icon(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        return self.find_fixed_icon(
            frame, self.templates.get("backicon") if self.templates else None,
            self.config.back_icon_center, self.config.back_icon_match_threshold,
            search_radius=(35, 30), scales=self.config.back_icon_scales,
        )

    def ui_state(self, frame: np.ndarray) -> tuple[str, Optional[tuple[int, int]]]:
        if self.is_disconnected(frame):
            return "disconnected", None
        back = self.back_icon(frame)
        if back is not None:
            return "minimap", back
        map_button = self.find_fixed_icon(
            frame, self.templates.get("minimapicon") if self.templates else None, (1150, 50)
        )
        if map_button is not None:
            return "combat", map_button
        return "other", None

    def white_text_mask(self, image: np.ndarray) -> np.ndarray:
        blue, green, red = cv2.split(image)
        bright = cv2.inRange(
            image, (self.config.white_text_min_channel,) * 3, (255, 255, 255)
        )
        spread_bg = cv2.inRange(cv2.absdiff(blue, green), 0, self.config.white_text_max_channel_spread)
        spread_gr = cv2.inRange(cv2.absdiff(green, red), 0, self.config.white_text_max_channel_spread)
        spread_br = cv2.inRange(cv2.absdiff(blue, red), 0, self.config.white_text_max_channel_spread)
        cv2.bitwise_and(bright, spread_bg, dst=bright)
        cv2.bitwise_and(bright, spread_gr, dst=bright)
        cv2.bitwise_and(bright, spread_br, dst=bright)
        return bright

    @staticmethod
    def purple_text_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            (hsv[:, :, 0] >= 138) & (hsv[:, :, 0] <= 154)
            & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 170)
        ).astype(np.uint8) * 255

    def player_center(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """Find the player's bright-green body near the camera center.

        Green health bars and the green player name are much thinner than the
        body sprite, so connected-component size and shape separate the player
        from those UI elements.  The result is used for target ranking only;
        the red lock marker remains the authority once combat starts.
        """
        fh, fw = frame.shape[:2]
        roi_x1, roi_x2 = int(fw * 0.30), int(fw * 0.70)
        roi_y1, roi_y2 = int(fh * 0.20), int(fh * 0.80)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 45) & (hsv[:, :, 0] <= 85)
            & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 120)
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(green, 8)
        screen_center = np.array([fw / 2.0, fh / 2.0])
        choices: list[tuple[float, tuple[int, int]]] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not (500 <= area <= 7000 and 25 <= width <= 130 and 25 <= height <= 130):
                continue
            if not (0.35 <= width / max(1, height) <= 2.5):
                continue
            point = centroids[index] + np.array([roi_x1, roi_y1])
            distance = float(np.linalg.norm(point - screen_center))
            choices.append((distance, (int(round(point[0])), int(round(point[1])))))
        return min(choices)[1] if choices else None

    def find_targets(
        self, frame: np.ndarray, templates: list[tuple[np.ndarray, str]], center_x: int, center_y: int,
        blacklist: Optional[dict[tuple[int, int], float]] = None, now: float = 0.0,
    ) -> list[dict[str, Any]]:
        # Match every name at native scale first.  The slower scale fallback is
        # used only when the complete primary pass found nothing, just like the
        # original bot's two-tier scanner.
        matches: list[dict[str, Any]] = []
        for template, color in templates:
            matches.extend(self._find_one_target(frame, template, color, center_x, center_y, blacklist, now, allow_fallback=False))
        if not matches:
            for template, color in templates:
                matches.extend(self._find_one_target(frame, template, color, center_x, center_y, blacklist, now, allow_fallback=True, fallback_only=True))
        # White and purple versions can describe the same nametag.
        unique: list[dict[str, Any]] = []
        for candidate in sorted(matches, key=lambda item: item["distance"]):
            if any(abs(candidate["nx"] - item["nx"]) < 30 and abs(candidate["ny"] - item["ny"]) < 15 for item in unique):
                continue
            unique.append(candidate)
        # Distance from the detected player center is the primary rule.  A
        # coarse tile bucket can put a farther mob ahead of a closer one near
        # a bucket boundary, so it is kept only as diagnostic metadata.
        return sorted(unique, key=lambda item: (item["distance"], -item["score"]))

    def _find_one_target(
        self, frame: np.ndarray, template: np.ndarray, name_color: str, center_x: int, center_y: int,
        blacklist: Optional[dict[tuple[int, int], float]], now: float,
        allow_fallback: bool = True, fallback_only: bool = False,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        fh, fw = frame.shape[:2]
        top, bottom = int(fh * cfg.margin_top), int(fh * (1.0 - cfg.margin_bottom))
        left, right = int(fw * cfg.margin_left), int(fw * (1.0 - cfg.margin_right))
        playfield = frame[top:bottom, left:fw]
        if playfield.size == 0:
            return []

        processing_scale = min(1.0, cfg.target_scan_width / fw)
        mask_builder = self.purple_text_mask if name_color == "purple" else self.white_text_mask
        if self._target_mask_frame is not frame:
            self._target_mask_frame = frame
            self._target_masks = {}
        text_mask = self._target_masks.get(name_color)
        if text_mask is None:
            search_image = (
                cv2.resize(playfield, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_NEAREST)
                if processing_scale < 1.0 else playfield
            )
            text_mask = mask_builder(search_image)
            self._target_masks[name_color] = text_mask
        if cv2.countNonZero(text_mask) == 0:
            return []

        matches: list[dict[str, Any]] = []

        def scan(scale: float) -> None:
            combined = scale * processing_scale
            scaled_color = (
                cv2.resize(template, None, fx=combined, fy=combined, interpolation=cv2.INTER_NEAREST)
                if combined != 1.0 else template
            )
            scaled = mask_builder(scaled_color)
            th, tw = scaled.shape[:2]
            if th >= text_mask.shape[0] or tw >= text_mask.shape[1] or th < 3 or tw < 3:
                return
            result = cv2.matchTemplate(text_mask, scaled, cv2.TM_CCOEFF_NORMED)
            template_pixels = np.count_nonzero(scaled)
            for px, py in zip(*np.where(result >= cfg.match_threshold)[::-1]):
                patch = text_mask[py:py + th, px:px + tw]
                shared = np.count_nonzero(cv2.bitwise_and(patch, scaled))
                observed = cv2.countNonZero(patch)
                if (
                    template_pixels == 0 or observed == 0
                    or shared / template_pixels < cfg.min_name_template_overlap
                    or shared / observed < cfg.min_name_template_overlap
                ):
                    continue
                x = int(round(px / processing_scale)) + left
                y = int(round(py / processing_scale)) + top
                native_w = max(1, int(round(template.shape[1] * scale)))
                native_h = max(1, int(round(template.shape[0] * scale)))
                if any(abs(x - item["nx"]) < 30 and abs(y - item["ny"]) < 15 for item in matches):
                    continue
                click_x = x + native_w // 2
                click_y = y + native_h + int(cfg.mob_body_y_offset * scale)
                distance = float(np.hypot(click_x - center_x, click_y - center_y))
                if distance < cfg.player_deadzone_radius:
                    continue
                tile_distance = max(
                    round(abs(click_x - center_x) / cfg.grid_tile_size),
                    round(abs(click_y - center_y) / cfg.grid_tile_size),
                )
                grid = (int(click_x // 40), int(click_y // 40))
                if blacklist and grid in blacklist and now < blacklist[grid]:
                    continue
                if left <= click_x <= right and top <= click_y <= bottom:
                    matches.append({
                        "nx": x, "ny": y, "nw": native_w, "nh": native_h,
                        "click_x": click_x, "click_y": click_y, "distance": distance,
                        "tile_distance": tile_distance,
                        "score": float(result[py, px]), "scale": scale,
                    })

        if not fallback_only:
            scan(1.0)
        if allow_fallback and not matches:
            for scale in (0.85, 1.15, 0.75, 1.25):
                scan(scale)
                if matches:
                    break
        return sorted(matches, key=lambda item: item["distance"])

    def pickup_position(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        if not self.templates:
            return None
        available = self.templates.get("pickup-available")
        unavailable = self.templates.get("pickup-not-yet")
        if available is None or unavailable is None:
            return None
        fh, fw = frame.shape[:2]
        left, right, top, bottom = int(fw * 0.82), fw, int(fh * 0.10), int(fh * 0.55)
        region = frame[top:bottom, left:right]
        best_available: tuple[float, Optional[tuple[int, int]]] = (-1.0, None)
        best_unavailable = -1.0
        for scale in (0.75, 0.9, 1.0, 1.1, 1.25):
            scaled = cv2.resize(available, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ah, aw = scaled.shape[:2]
            if ah >= region.shape[0] or aw >= region.shape[1]:
                continue
            _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED))
            if score > best_available[0]:
                best_available = (score, (left + location[0] + aw // 2, top + location[1] + ah // 2))
            scaled_unavailable = cv2.resize(unavailable, (aw, ah), interpolation=cv2.INTER_NEAREST)
            _, unavailable_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(region, scaled_unavailable, cv2.TM_CCOEFF_NORMED))
            best_unavailable = max(best_unavailable, unavailable_score)
        score, position = best_available
        return position if position is not None and score >= 0.86 and score >= best_unavailable + 0.08 else None

    def _marked_template(self) -> np.ndarray:
        if self._marked_template_image is not None:
            return self._marked_template_image
        for filename in ("marked.png", "mob-current-marked.png"):
            path = TEMPLATE_DIR / filename
            image = cv2.imread(str(path)) if path.exists() else None
            if image is not None:
                difference = np.abs(image.astype(int) - [50, 50, 207])
                self._marked_template_image = (np.all(difference <= 20, axis=2)).astype(np.uint8) * 255
                return self._marked_template_image
        box = np.zeros((72, 72), dtype=np.uint8)
        box[:7, :] = box[-7:, :] = 255
        box[:, :7] = box[:, -7:] = 255
        self._marked_template_image = box
        return box

    def _red_marker_mask(self, frame: np.ndarray) -> np.ndarray:
        if self._red_frame is frame and self._red_mask is not None:
            return self._red_mask
        mask = cv2.inRange(frame, (28, 28, 185), (72, 72, 229))
        fh, fw = frame.shape[:2]
        mask[int(fh * 0.75):, :int(fw * 0.18)] = 0
        mask[int(fh * 0.45):int(fh * 0.75), int(fw * 0.90):] = 0
        self._red_frame, self._red_mask = frame, mask
        return mask

    @staticmethod
    def _match_outline(mask: np.ndarray, template: np.ndarray, threshold: float = 0.50) -> Optional[tuple[int, int]]:
        th, tw = template.shape[:2]
        if mask.shape[0] < th or mask.shape[1] < tw:
            return None
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score < threshold:
            return None
        x, y = location
        visible = mask[y:y + th, x:x + tw] > 0
        border = template > 0
        border_pixels = max(1, int(np.count_nonzero(border)))
        border_hit = np.count_nonzero(visible & border) / border_pixels
        thickness = max(3, min(10, int(round(min(th, tw) * 0.07))))
        template_sides = (
            border[:thickness, thickness:-thickness], border[-thickness:, thickness:-thickness],
            border[thickness:-thickness, :thickness], border[thickness:-thickness, -thickness:],
        )
        visible_sides = (
            visible[:thickness, thickness:-thickness], visible[-thickness:, thickness:-thickness],
            visible[thickness:-thickness, :thickness], visible[thickness:-thickness, -thickness:],
        )
        hits = [np.count_nonzero(current & expected) / max(1, int(np.count_nonzero(expected))) for expected, current in zip(template_sides, visible_sides)]
        if border_hit < 0.35 or sum(hit >= 0.22 for hit in hits) < 2:
            return None
        return x + tw // 2, y + th // 2

    def red_square(
        self, frame: np.ndarray, target: Optional[tuple[int, int]] = None, radius: int = 120
    ) -> tuple[bool, Optional[tuple[int, int]]]:
        fh, fw = frame.shape[:2]
        top, bottom, left, right = int(fh * 0.13), int(fh * 0.85), int(fw * 0.08), int(fw * 0.90)
        template = self._marked_template()
        if target is not None:
            tx, ty = target
            x1, x2 = max(left, tx - radius), min(right, tx + radius)
            y1, y2 = max(top, ty - radius), min(bottom, ty + radius)
            th, tw = template.shape[:2]
            if x2 - x1 >= tw and y2 - y1 >= th:
                # An active target only needs a small crop. Building a red
                # mask for the whole 1600x900 frame delays the next click.
                local_mask = cv2.inRange(frame[y1:y2, x1:x2], (28, 28, 185), (72, 72, 229))
                center = self._match_outline(local_mask, template)
                if center is not None:
                    return True, (x1 + center[0], y1 + center[1])
            return False, None
        marker_mask = self._red_marker_mask(frame)
        playfield = marker_mask[top:bottom, left:right]
        if cv2.countNonZero(playfield) < 40:
            return False, None
        center = self._match_outline(playfield, template)
        return (True, (left + center[0], top + center[1])) if center is not None else (False, None)

    def exhausted_caption(self, frame: np.ndarray, template: Optional[np.ndarray]) -> bool:
        if frame is None or template is None:
            return False
        fh, fw = frame.shape[:2]
        if self._caption_cache is None or self._caption_cache[0] is not template:
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            mask = (((hue <= 10) | (hue >= 170)) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 90)).astype(np.uint8) * 255
            points = cv2.findNonZero(mask)
            if points is None:
                return False
            x, y, width, height = cv2.boundingRect(points)
            mask = mask[y:y + height, x:x + width]
            density = cv2.countNonZero(mask) / (width * height)
            self._caption_cache = (template, width, height, density)
        _, template_w, template_h, density = self._caption_cache
        x0, x1, y0, y1 = int(fw * 0.05), int(fw * 0.95), int(fh * 0.50), int(fh * 0.85)
        region = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        red_mask = (((hue <= 10) | (hue >= 170)) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 90)).astype(np.uint8) * 255
        screen_scale = fw / 1600.0
        processing_scale = min(0.5, 800.0 / max(1, region.shape[1]))
        if processing_scale < 1.0:
            red_mask = cv2.resize(red_mask, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_NEAREST)
        close_width = max(5, int(round(15 * screen_scale * processing_scale)))
        joined = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3, close_width), dtype=np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        expected_w, expected_h = template_w * screen_scale * processing_scale, template_h * screen_scale * processing_scale
        min_w, max_w, min_h, max_h = expected_w * 0.50, expected_w * 1.20, expected_h * 0.45, expected_h * 1.50
        min_aspect, max_aspect = (template_w / template_h) * 0.55, (template_w / template_h) * 1.70
        min_density, max_density = density * 0.45, min(0.78, density * 1.80)
        for x, y, width, height, _ in stats[1:count]:
            if not (min_w <= width <= max_w and min_h <= height <= max_h):
                continue
            aspect = width / max(1, height)
            if not (min_aspect <= aspect <= max_aspect):
                continue
            area_density = cv2.countNonZero(red_mask[y:y + height, x:x + width]) / (width * height)
            if min_density <= area_density <= max_density:
                return True
        return False


# ---------------------------------------------------------------------------
# Minimap route registration
# ---------------------------------------------------------------------------


class MinimapNavigator:
    """Register the visible map against the immutable full-map reference."""

    def __init__(self, config: BotConfig, vision: FrameVision, device: InputDevice, templates: TemplateStore):
        self.config = config
        self.vision = vision
        self.device = device
        self.templates = templates
        self.safe_mask: Optional[np.ndarray] = None
        self.waypoints: list[tuple[int, int]] = []
        self.reference_id: Optional[int] = None
        self.route: Optional[dict[str, Any]] = None
        self.last_route_target: Optional[tuple[int, int]] = None

    @staticmethod
    def _build_safe_mask(reference: np.ndarray) -> np.ndarray:
        walls = FrameVision.minimap_white_mask(reference)
        contours, hierarchy = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(walls.shape, dtype=np.uint8)
        outer_index = max(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]))
        outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(contours[outer_index])
        margin = max(30, int(round(min(reference.shape[:2]) * 0.035)))
        safe = np.zeros(walls.shape, dtype=np.uint8)
        x1, y1, x2, y2 = outer_x + margin, outer_y + margin, outer_x + outer_w - margin, outer_y + outer_h - margin
        if x2 <= x1 or y2 <= y1:
            return safe
        safe[y1:y2, x1:x2] = 255
        blocked = walls.copy()
        if hierarchy is not None:
            for index, contour in enumerate(contours):
                parent = int(hierarchy[0][index][3])
                if parent < 0 or cv2.contourArea(contour) < 1000 or cv2.contourArea(contours[parent]) < 5000:
                    continue
                cv2.drawContours(blocked, [contour], -1, 255, thickness=-1)
        blocked = cv2.dilate(blocked, np.ones((9, 9), np.uint8), iterations=1)
        safe[blocked > 0] = 0
        return cv2.erode(safe, np.ones((5, 5), np.uint8), iterations=1)

    @staticmethod
    def build_waypoints(safe_mask: np.ndarray, grid_step: int = 120) -> list[tuple[int, int]]:
        if safe_mask is None or safe_mask.size == 0:
            return []
        height, width = safe_mask.shape[:2]
        waypoints: list[tuple[int, int]] = []
        radius, half_step = max(12, grid_step // 3), max(1, grid_step // 2)
        # Sweep columns instead of rows.  The reference contains long
        # horizontal rooms; a row-first sweep can keep the character moving
        # sideways for many taps before it changes level.  Alternating column
        # direction still covers the whole safe area while forcing regular
        # vertical progress.
        for column, x in enumerate(range(half_step, width, grid_step)):
            column_points: list[tuple[int, int]] = []
            for y in range(half_step, height, grid_step):
                x1, y1, x2, y2 = max(0, x - radius), max(0, y - radius), min(width, x + radius + 1), min(height, y + radius + 1)
                local = safe_mask[y1:y2, x1:x2]
                local_y, local_x = np.where(local > 0)
                if len(local_x) == 0:
                    continue
                distances = (local_x + x1 - x) ** 2 + (local_y + y1 - y) ** 2
                nearest = int(np.argmin(distances))
                column_points.append((int(local_x[nearest] + x1), int(local_y[nearest] + y1)))
            if column % 2:
                column_points.reverse()
            for point in column_points:
                if not waypoints or point != waypoints[-1]:
                    waypoints.append(point)
        return waypoints

    @staticmethod
    def _geometry_score(frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]) -> float:
        fh, fw = frame_mask.shape[:2]
        sh, sw = scaled_reference.shape[:2]
        ox, oy = origin
        x1, y1, x2, y2 = max(0, ox), max(0, oy), min(fw, ox + sw), min(fh, oy + sh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        expected = (scaled_reference[y1 - oy:y2 - oy, x1 - ox:x2 - ox] > 0).astype(np.uint8)
        observed = (frame_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        if cv2.countNonZero(expected) < 3000 or cv2.countNonZero(observed) < 3000:
            return 0.0
        kernel = np.ones((7, 7), np.uint8)
        expected_near, observed_near = cv2.dilate(expected, kernel, iterations=1) > 0, cv2.dilate(observed, kernel, iterations=1) > 0
        recall = float(np.count_nonzero(expected & observed_near)) / max(1, np.count_nonzero(expected))
        precision = float(np.count_nonzero(observed & expected_near)) / max(1, np.count_nonzero(observed))
        return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0

    @staticmethod
    def _visible_reference_pixels(
        frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]
    ) -> tuple[int, int]:
        """Return expected white pixels visible in the frame and total reference pixels.

        A partial view is normal because the full route is larger than the
        1600x900 game viewport.  This metric lets us reject tiny repeated
        patches without requiring the whole reference to be on screen.
        """
        fh, fw = frame_mask.shape[:2]
        sh, sw = scaled_reference.shape[:2]
        ox, oy = origin
        x1, y1, x2, y2 = max(0, ox), max(0, oy), min(fw, ox + sw), min(fh, oy + sh)
        if x2 <= x1 or y2 <= y1:
            return 0, int(cv2.countNonZero(scaled_reference))
        visible = scaled_reference[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
        return int(cv2.countNonZero(visible)), int(cv2.countNonZero(scaled_reference))

    def _refine_y(self, frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]) -> tuple[tuple[int, int], float]:
        ox, oy = origin
        def candidate(offset_y: int) -> tuple[float, int]:
            candidate_origin = (ox, offset_y)
            visible, _ = self._visible_reference_pixels(frame_mask, scaled_reference, candidate_origin)
            if visible < self.config.minimap_min_visible_reference_pixels:
                return 0.0, offset_y
            return self._geometry_score(frame_mask, scaled_reference, candidate_origin), offset_y

        coarse = [candidate(oy + offset) for offset in range(-360, 361, 16)]
        _, best_y = max(coarse)
        fine = [candidate(candidate_y) for candidate_y in range(best_y - 16, best_y + 17, 2)]
        geometry, refined_y = max(fine)
        return (ox, refined_y), geometry

    def align(self, frame: np.ndarray, reference: np.ndarray) -> Optional[dict[str, Any]]:
        frame_mask = self.vision.minimap_white_mask(frame)
        if cv2.countNonZero(frame_mask) < 12000:
            return None
        downsample = 0.25
        search = cv2.resize(frame_mask, None, fx=downsample, fy=downsample, interpolation=cv2.INTER_NEAREST)
        reference_mask = self.vision.minimap_white_mask(reference)
        patch_w, patch_h = min(180, search.shape[1] - 4), min(180, search.shape[0] - 4)
        if patch_w < 120 or patch_h < 120:
            return None
        raw: list[tuple[float, float, tuple[int, int]]] = []
        for scale in (0.92, 0.96, 1.0, 1.04, 1.08):
            scaled = cv2.resize(reference_mask, None, fx=scale * downsample, fy=scale * downsample, interpolation=cv2.INTER_NEAREST)
            scaled_h, scaled_w = scaled.shape[:2]
            if scaled_w < patch_w or scaled_h < patch_h:
                continue
            x_starts = sorted({0, max(0, (scaled_w - patch_w) // 2), max(0, scaled_w - patch_w)})
            y_remaining = max(0, scaled_h - patch_h)
            y_starts = sorted({int(round(y_remaining * fraction)) for fraction in (0.0, 0.5, 1.0)})
            for crop_y in y_starts:
                for crop_x in x_starts:
                    patch = scaled[crop_y:crop_y + patch_h, crop_x:crop_x + patch_w]
                    if cv2.countNonZero(patch) < 3000:
                        continue
                    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
                    origin = (int(round((location[0] - crop_x) / downsample)), int(round((location[1] - crop_y) / downsample)))
                    raw.append((float(score), float(scale), origin))
        if not raw:
            return None
        raw.sort(key=lambda item: item[0], reverse=True)
        candidates: list[tuple[float, float, tuple[int, int], float]] = []
        for score, scale, origin in raw:
            if any(abs(origin[0] - kept[2][0]) <= 10 and abs(origin[1] - kept[2][1]) <= 10 and abs(scale - kept[1]) <= 0.035 for kept in candidates):
                continue
            scaled_reference = cv2.resize(reference_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            geometry = self._geometry_score(frame_mask, scaled_reference, origin)
            visible_pixels, total_reference_pixels = self._visible_reference_pixels(frame_mask, scaled_reference, origin)
            visible_fraction = visible_pixels / max(1, total_reference_pixels)
            if (
                geometry > 0
                and visible_pixels >= self.config.minimap_min_visible_reference_pixels
                and visible_fraction >= self.config.minimap_min_visible_reference_fraction
            ):
                candidates.append((score, scale, origin, geometry))
                if len(candidates) >= 8:
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[3], item[0]), reverse=True)
        score, scale, origin, geometry = candidates[0]
        second_geometry = candidates[1][3] if len(candidates) > 1 else 0.0
        scaled_reference = cv2.resize(reference_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        origin, geometry = self._refine_y(frame_mask, scaled_reference, origin)
        gap = geometry - second_geometry
        visible_pixels, total_reference_pixels = self._visible_reference_pixels(frame_mask, scaled_reference, origin)
        visible_fraction = visible_pixels / max(1, total_reference_pixels)
        # `score` is kept for diagnostics, but it compares screenshots whose
        # backgrounds can differ dramatically.  Geometry plus coverage is the
        # authoritative test for the white route linework.
        valid = (
            geometry >= self.config.minimap_geometry_min_score
            and gap >= self.config.minimap_align_min_gap
            and visible_pixels >= self.config.minimap_min_visible_reference_pixels
            and visible_fraction >= self.config.minimap_min_visible_reference_fraction
        )
        result = {
            "valid": valid,
            "score": score,
            "geometry": geometry,
            "gap": gap,
            "coverage": visible_fraction,
            "visible_pixels": visible_pixels,
            "scale": scale,
            "origin": origin,
        }
        return result

    @staticmethod
    def player_marker(frame: np.ndarray) -> Optional[tuple[int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        marker = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 100)).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(marker, 8)
        fh, fw = frame.shape[:2]
        center = np.array([fw / 2.0, fh / 2.0])
        choices: list[tuple[float, tuple[int, int]]] = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if not (200 <= area <= 3000 and 14 <= w <= 100 and 14 <= h <= 100):
                continue
            point = centroids[index]
            distance = float(np.linalg.norm(point - center))
            if distance <= min(fw, fh) * 0.18:
                choices.append((distance, (int(round(point[0])), int(round(point[1])))))
        return min(choices)[1] if choices else None

    def destination(
        self,
        frame: np.ndarray,
        alignment: dict[str, Any],
        safe_mask: np.ndarray,
        player: tuple[int, int],
        waypoints: list[tuple[int, int]],
        farthest: bool = False,
    ) -> Optional[tuple[int, int]]:
        if not alignment or not alignment.get("valid") or safe_mask is None or player is None or not waypoints:
            return None
        scale, (ox, oy) = alignment["scale"], alignment["origin"]
        player_x, player_y = player
        map_x, map_y = (player_x - ox) / scale, (player_y - oy) / scale
        ref_h, ref_w = safe_mask.shape[:2]
        if not (0 <= map_x < ref_w and 0 <= map_y < ref_h and safe_mask[int(map_y), int(map_x)] > 0):
            return None
        step = 8
        ys, xs = np.where(safe_mask[::step, ::step] > 0)
        if len(xs) == 0:
            return None
        ref_xs, ref_ys = xs * step + step // 2, ys * step + step // 2
        screen_xs, screen_ys = ox + ref_xs * scale, oy + ref_ys * scale
        fh, fw = frame.shape[:2]
        margin_x, margin_y = fw * (1.0 - self.config.minimap_safe_view_fraction) / 2.0, fh * (1.0 - self.config.minimap_safe_view_fraction) / 2.0
        visible = ((screen_xs >= margin_x) & (screen_xs <= fw - margin_x) & (screen_ys >= margin_y) & (screen_ys <= fh - margin_y))
        distances = np.hypot(screen_xs - player_x, screen_ys - player_y)
        visible &= distances >= self.config.minimap_min_click_distance * fw / 1600.0
        if not farthest:
            visible &= distances <= self.config.minimap_max_click_distance * fw / 1600.0
        if not np.any(visible):
            return None

        signature = (safe_mask.shape, len(waypoints), waypoints[0], waypoints[-1])
        points = np.asarray(waypoints, dtype=np.float32)
        if self.route is None or self.route["signature"] != signature:
            nearest = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
            self.route = {"signature": signature, "waypoints": tuple(waypoints), "index": nearest}
        else:
            # Camera movement can carry the player a long way from the old
            # route cursor, especially after a failsafe jump.  Re-anchor it
            # instead of continuing a stale same-row sweep.
            current_index = int(self.route["index"])
            if np.hypot(points[current_index, 0] - map_x, points[current_index, 1] - map_y) > max(150.0, self.config.minimap_max_click_distance):
                self.route["index"] = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
        if farthest:
            index = int(np.argmax(np.where(visible, distances, -np.inf)))
            chosen_ref = np.array([ref_xs[index], ref_ys[index]], dtype=np.float32)
            self.route["index"] = int(np.argmin(np.hypot(points[:, 0] - chosen_ref[0], points[:, 1] - chosen_ref[1])))
            self.last_route_target = (int(round(ref_xs[index])), int(round(ref_ys[index])))
            return int(round(screen_xs[index])), int(round(screen_ys[index]))
        points = np.asarray(self.route["waypoints"], dtype=np.float32)
        current = int(self.route["index"])
        target_index = (current + 1) % len(points)
        target_x, target_y = points[target_index]
        if np.hypot(target_x - map_x, target_y - map_y) < max(70.0, self.config.minimap_min_click_distance / max(scale, 0.1)):
            current = target_index
            self.route["index"] = current
            target_index = (current + 1) % len(points)
            target_x, target_y = points[target_index]
        candidate_ref_xs, candidate_ref_ys = (screen_xs - ox) / scale, (screen_ys - oy) / scale
        before = np.hypot(target_x - map_x, target_y - map_y)
        after = np.hypot(target_x - candidate_ref_xs, target_y - candidate_ref_ys)
        preferred = (self.config.minimap_min_click_distance + self.config.minimap_max_click_distance) * 0.5 * fw / 1600.0
        scores = before - after
        scores = scores * 1000.0 - np.abs(distances - preferred)
        scores[~visible] = -np.inf
        index = int(np.argmax(scores))
        self.last_route_target = (int(round(target_x)), int(round(target_y)))
        return int(round(screen_xs[index])), int(round(screen_ys[index]))

    def travel(self, *, farthest: bool = False, keep_open_seconds: float = 0.0, force: bool = False) -> bool:
        reference = self.templates.get(self.config.minimap_reference_key)
        if reference is None:
            print("[!] Minimap travel disabled: full reference map is missing.")
            return False
        frame = self.device.capture()
        if frame is None:
            return False
        state, icon = self.vision.ui_state(frame)
        if state == "combat":
            target_templates: list[tuple[np.ndarray, str]] = []
            for name, color in (
                (self.config.target_template_key, "white"),
                (self.config.target_purple_template_key, "purple"),
            ):
                image = self.templates.get(name)
                if image is not None:
                    target_templates.append((image, color))
            fh, fw = frame.shape[:2]
            if not force and (self.vision.red_square(frame)[0] or self.vision.find_targets(frame, target_templates, fw // 2, fh // 2)):
                print("[~] Minimap travel skipped: a target appeared in the latest frame.")
                return False
            if icon is None or not self.device.click(*icon):
                return False
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                time.sleep(0.04)
                frame = self.device.capture()
                if frame is None:
                    continue
                state, icon = self.vision.ui_state(frame)
                if state in ("minimap", "disconnected"):
                    break
        if state != "minimap":
            print(f"[!] Minimap travel skipped: UI state is {state}.")
            return False
        alignment = self.align(frame, reference)
        if not alignment or not alignment.get("valid"):
            details = alignment or {}
            return self._fail(icon, f"[!] Minimap alignment rejected (match {details.get('score', 0.0):.2f}, geometry {details.get('geometry', 0.0):.2f}, coverage {details.get('coverage', 0.0):.2f}, gap {details.get('gap', 0.0):.2f}); no movement tap sent.")
        player = self.player_marker(frame)
        if player is None:
            return self._fail(icon, "[!] Minimap player marker was not found; no movement tap sent.")
        if self.reference_id != id(reference):
            self.safe_mask = self._build_safe_mask(reference)
            self.waypoints = self.build_waypoints(self.safe_mask)
            self.reference_id = id(reference)
        destination = self.destination(frame, alignment, self.safe_mask, player, self.waypoints, farthest=farthest)
        if destination is None:
            return self._fail(icon, "[!] No safe visible minimap waypoint was found; no movement tap sent.")
        mode = "farthest visible" if farthest else "route"
        print(f"[~] Minimap {mode} waypoint {destination} from player {player} toward reference {self.last_route_target or '?'} (match {alignment['score']:.2f}, geometry {alignment['geometry']:.2f}, coverage {alignment['coverage']:.2f}, gap {alignment['gap']:.2f}, scale {alignment['scale']:.2f}).")
        sent = self.device.click(*destination, synchronous=True)
        if sent:
            time.sleep(0.07)
            if keep_open_seconds > 0:
                print(f"[~] Keeping minimap open for {keep_open_seconds:.0f}s after movement tap.")
                deadline = time.monotonic() + keep_open_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
                # The game can still be applying the map tap when the hold
                # expires.  Give that transition a moment to settle before
                # sending the Back action.
                time.sleep(0.8)
        closed = self._close_map(icon)
        if not closed and keep_open_seconds > 0:
            # A late UI frame can leave the first Back verification stale.
            # Re-match the live icon and make one final delayed close attempt.
            time.sleep(0.8)
            latest = self.device.capture()
            retry_icon = self.vision.back_icon(latest) if latest is not None else icon
            closed = self._close_map(retry_icon)
        return bool(sent and closed)

    def _fail(self, icon: Optional[tuple[int, int]], message: str) -> bool:
        print(message)
        self._close_map(icon)
        return False

    def _close_map(self, icon: Optional[tuple[int, int]]) -> bool:
        back_template = self.templates.get("backicon")
        if icon is None:
            self.device.back()
            time.sleep(0.45)
        else:
            for attempt in range(2):
                if not self.device.click(*icon, synchronous=True):
                    print(f"[!] Back-icon tap {attempt + 1} failed.")
                time.sleep(0.45)
                frame = self.device.capture()
                if frame is None:
                    print("[!] Could not verify the panel closed after tapping Back.")
                    break
                if back_template is None:
                    print("[!] Back icon template is unavailable; trying the system Back action.")
                    break
                refreshed = self.vision.find_fixed_icon(frame, back_template, self.config.back_icon_center, self.config.back_icon_match_threshold, search_radius=(35, 30), scales=self.config.back_icon_scales)
                if refreshed is None:
                    return True
                if attempt == 0:
                    icon = refreshed
                    print(f"[!] Back icon remains visible at {icon}; retrying its fresh match.")
                else:
                    print("[!] Back icon remains visible after retry; trying the system Back action.")
        self.device.back()
        time.sleep(0.45)
        frame = self.device.capture()
        if frame is not None and back_template is not None and self.vision.find_fixed_icon(frame, back_template, self.config.back_icon_center, self.config.back_icon_match_threshold, search_radius=(35, 30), scales=self.config.back_icon_scales) is None:
            return True
        print("[!] Panel is still open after Back-icon and system Back attempts.")
        return False


# ---------------------------------------------------------------------------
# Combat state machine
# ---------------------------------------------------------------------------


@dataclass
class BotState:
    current_target: Optional[dict[str, Any]] = None
    target_click_time: float = 0.0
    confirmed_locked: bool = False
    last_red_seen_time: float = 0.0
    last_log_time: float = 0.0
    last_ui_back_time: float = 0.0
    no_target_since: float = field(default_factory=time.time)
    last_minimap_time: float = 0.0
    frame_count: int = 0
    fail_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    blacklist: dict[tuple[int, int], float] = field(default_factory=dict)
    retry_context: Optional[dict[str, Any]] = None
    idle_mark_check_needed: bool = True
    next_idle_mark_check_at: float = 0.0
    ignored_exhausted_marker: Optional[tuple[int, int]] = None
    last_capture_warning: float = 0.0
    pickup_pending: bool = False
    ui_recovery_interval: float = 1.5
    failed_recovery_state: Optional[str] = None
    failed_recovery_count: int = 0
    exhausted_caption_check_at: Optional[float] = None
    exhausted_caption_next_poll: float = 0.0
    exhausted_caption_target: Any = None
    consecutive_exhausted: int = 0
    exhausted_travel_pending: bool = False
    last_ui_state: str = "other"
    next_ui_check_at: float = 0.0
    locked_since: Optional[float] = None


class RucoyBot:
    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.templates = TemplateStore()
        self.device: Optional[InputDevice] = None
        self.vision = FrameVision(config, self.templates)
        self.navigator: Optional[MinimapNavigator] = None
        self.target_templates: list[tuple[np.ndarray, str]] = []
        self.state = BotState()

    def start(self) -> bool:
        images = self.templates.load()
        exact = images.get(self.config.target_template_key)
        if exact is None:
            print(f"[-] Exact {self.config.target_name} template not found in {TEMPLATE_DIR}!")
            return False
        self.target_templates = [(exact, "white")]
        purple = images.get(self.config.target_purple_template_key)
        if purple is not None:
            self.target_templates.append((purple, "purple"))
        self.device = InputDevice.connect(self.config)
        if self.device is None:
            return False
        self.navigator = MinimapNavigator(self.config, self.vision, self.device, self.templates)
        print("[+] Capturing initial frame...")
        frame = self.device.capture()
        if frame is None:
            print("[-] Failed to capture initial frame! Check if emulator is running.")
            return False
        self._verify_template(frame, exact)
        self._print_startup()
        return True

    def _verify_template(self, frame: np.ndarray, template: np.ndarray) -> None:
        gray_frame, gray_template = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        best = 0.0
        for scale in (0.4, 0.5, 0.6, 0.75, 1.0, 1.25):
            scaled = cv2.resize(gray_template, None, fx=scale, fy=scale) if scale != 1.0 else gray_template
            if scaled.shape[0] >= gray_frame.shape[0] or scaled.shape[1] >= gray_frame.shape[1]:
                continue
            _, score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(gray_frame, scaled, cv2.TM_CCOEFF_NORMED))
            best = max(best, score)
        if best < self.config.match_threshold:
            print(f"[!] Level-specific template score is {best:.2f}; keeping the saved {self.config.target_name} template.")
        else:
            print(f"[+] Template match verified (score: {best:.2f})")

    def _print_startup(self) -> None:
        assert self.device is not None
        print(f"\n[+] Target: '{self.config.target_name}'")
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
                if self._process_frame(frame):
                    break
        except BotQuit:
            pass
        if self.config.show_preview:
            cv2.destroyAllWindows()
        print("[+] Bot stopped.")

    def _capture_failed(self) -> None:
        now = time.time()
        if now - self.state.last_capture_warning >= 5.0:
            print("[!] Screen capture unavailable; retrying (ADB capture times out after 3s).")
            self.state.last_capture_warning = now

    def _process_frame(self, frame: np.ndarray) -> bool:
        assert self.device is not None and self.navigator is not None
        state = self.state
        cfg = self.config
        fh, fw = frame.shape[:2]
        detected_player = self.vision.player_center(frame)
        cx, cy = detected_player if detected_player is not None else (fw // 2, fh // 2)
        now = time.time()
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
        elif not self._recover_ui(frame, ui_state, now):
            time.sleep(0.08)
            return False
        if state.blacklist and state.frame_count % 30 == 0:
            state.blacklist = {position: expiry for position, expiry in state.blacklist.items() if now < expiry}

        if state.current_target is None:
            self._acquire_target(frame, cx, cy, now)
        if state.pickup_pending:
            pickup = self.vision.pickup_position(frame)
            if pickup is not None:
                print(f"[+] Pickup available after kill; tapping ({pickup[0]}, {pickup[1]}).")
                self.device.click(*pickup)
                state.pickup_pending = False
        self._check_exhausted_caption(frame, now)
        if state.exhausted_travel_pending:
            self._run_exhausted_travel_failsafe(now)
            self._preview(frame, cx, cy)
            return False
        if state.current_target is not None:
            self._update_target(frame, cx, cy, now)
        if state.current_target is None:
            idle_duration = now - state.no_target_since
            if cfg.enable_minimap_walk and idle_duration >= cfg.minimap_idle_delay and now - state.last_minimap_time >= cfg.minimap_cooldown:
                print(f"[~] No '{cfg.target_name}' in vision for {idle_duration:.1f}s -> registering minimap waypoint...")
                self.navigator.travel()
                state.last_minimap_time = time.time()
                state.no_target_since = state.last_minimap_time
        self._preview(frame, cx, cy)
        return False

    def _recover_ui(self, frame: np.ndarray, ui_state: str, now: float) -> bool:
        state = self.state
        interval = 4.0 if ui_state == "disconnected" else state.ui_recovery_interval
        if now - state.last_ui_back_time < interval:
            return False
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
        return False

    def _recover_unexpected_ui(self, frame: np.ndarray) -> str:
        assert self.device is not None
        if self.vision.is_disconnected(frame):
            fh, fw = frame.shape[:2]
            self.device.click(int(fw * 0.3125), int(fh * 0.667))
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
        if state.idle_mark_check_needed or (state.ignored_exhausted_marker is None and now >= state.next_idle_mark_check_at):
            has_red, position = self.vision.red_square(frame)
            state.idle_mark_check_needed = False
            state.next_idle_mark_check_at = now + cfg.idle_mark_check_interval
        if has_red and position is not None and state.ignored_exhausted_marker is not None:
            if np.hypot(position[0] - state.ignored_exhausted_marker[0], position[1] - state.ignored_exhausted_marker[1]) < 140:
                has_red = False
        if has_red and position is not None:
            state.ignored_exhausted_marker = None
            print(f"[*] Detected active target lock at ({position[0]}, {position[1]})! Adopting target...")
            state.current_target = {"click_x": position[0], "click_y": position[1], "distance": np.hypot(position[0] - cx, position[1] - cy), "score": 1.0, "scale": 1.0}
            state.target_click_time = state.last_red_seen_time = now
            state.confirmed_locked = True
            state.locked_since = now
            state.retry_context = None
            return
        mobs = self.vision.find_targets(frame, self.target_templates, cx, cy, state.blacklist, now)
        if not mobs:
            if now - state.last_log_time > 5.0:
                print(f"[...] Scanning for '{cfg.target_name}'... (frame #{state.frame_count}, res {frame.shape[1]}x{frame.shape[0]})")
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
        self.device.click(bx, by)
        print(f"[+] Found {len(mobs)} '{cfg.target_name}' (score:{best['score']:.2f} scale:{best['scale']:.1f}x)")
        print(f"    -> Clicking mob at ({bx}, {by}) [dist:{int(best['distance'])}px]")
        best["initial_click"], best["last_position_refresh"] = (bx, by), now
        state.current_target = best
        state.target_click_time = state.last_red_seen_time = now
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
        due = state.current_target is not None and (now >= state.exhausted_caption_next_poll or (state.exhausted_caption_check_at is not None and now >= state.exhausted_caption_check_at))
        caption_visible = False
        if due:
            caption_visible = self.vision.exhausted_caption(frame, self.templates.get("exhausted-caption"))
            state.exhausted_caption_next_poll = now + cfg.exhausted_caption_poll_seconds
        if state.current_target is None:
            state.exhausted_caption_check_at = None
        elif state.exhausted_caption_check_at is None and due and caption_visible:
            state.exhausted_caption_check_at = now + cfg.exhausted_caption_confirm_seconds
            print(f"[~] Exhausted caption detected; checking again in {cfg.exhausted_caption_confirm_seconds:.0f}s.")
        elif state.exhausted_caption_check_at is not None and now >= state.exhausted_caption_check_at and due:
            state.exhausted_caption_check_at = None
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
            force=True,
        )
        state.last_minimap_time = time.time()
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
        has_red, position = self.vision.red_square(frame, (tx, ty))
        if has_red and position is not None and state.ignored_exhausted_marker is not None:
            old_distance = np.hypot(position[0] - state.ignored_exhausted_marker[0], position[1] - state.ignored_exhausted_marker[1])
            target_distance = np.hypot(position[0] - tx, position[1] - ty)
            if old_distance < 140 and target_distance >= old_distance - 25:
                has_red, position = False, None
        if not has_red:
            has_red, position = self.vision.red_square(frame)
            state.next_idle_mark_check_at = now + cfg.idle_mark_check_interval
            if has_red and position is not None and state.ignored_exhausted_marker is not None:
                old_distance = np.hypot(position[0] - state.ignored_exhausted_marker[0], position[1] - state.ignored_exhausted_marker[1])
                new_distance = np.hypot(position[0] - tx, position[1] - ty)
                if old_distance < 140 and new_distance >= old_distance - 25:
                    has_red, position = False, None
        if has_red:
            state.last_red_seen_time = now
            state.idle_mark_check_needed = False
            if position is not None:
                target["click_x"], target["click_y"] = position
                tx, ty = position
            if not state.confirmed_locked:
                state.confirmed_locked = True
                state.locked_since = now
                state.fail_counts.pop(target.get("retry_grid"), None)
                state.retry_context = None
                state.ignored_exhausted_marker = None
                print(f"[*] Target locked with red square! Fighting at ({tx}, {ty})...")
            elif state.locked_since is None:
                # Defensive recovery for targets created by older state or a
                # future caller that forgot to initialize the timestamp.
                state.locked_since = now
            elif now - state.locked_since >= cfg.target_stall_timeout:
                self._abandon_stalled_target(tx, ty, now)
            return
        if state.confirmed_locked:
            if now - state.last_red_seen_time >= cfg.target_defeated_grace_time:
                print(f"[+] Red square gone for {now - state.last_red_seen_time:.1f}s! Mob defeated. Searching next...")
                state.current_target = None
                state.confirmed_locked = False
                state.locked_since = None
                state.consecutive_exhausted = 0
                state.idle_mark_check_needed = False
                state.no_target_since = now
                state.pickup_pending = True
            return
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
            return
        if now - target.get("last_position_refresh", state.target_click_time) >= cfg.target_position_refresh_interval:
            target["last_position_refresh"] = now
            fresh = self.vision.find_targets(frame, self.target_templates, cx, cy, state.blacklist, now)
            if fresh:
                nearest = min(fresh, key=lambda item: np.hypot(item["click_x"] - tx, item["click_y"] - ty))
                if np.hypot(nearest["click_x"] - tx, nearest["click_y"] - ty) <= cfg.target_reacquire_radius:
                    for key in ("nx", "ny", "nw", "nh", "click_x", "click_y", "distance", "score", "scale"):
                        target[key] = nearest[key]

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
        scale = min(640 / fw, 480 / fh)
        cv2.imshow("Rucoy Online Bot - AFK Preview", cv2.resize(display, (int(fw * scale), int(fh * scale))))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[+] Quit key pressed.")
            raise BotQuit


def run_bot() -> None:
    """Install the output watchdog and restart the engine once if it loops."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    guard = RepeatedOutputGuard()
    output = original_stdout if isinstance(original_stdout, TimestampedOutputStream) else TimestampedOutputStream(original_stdout, guard)
    error = original_stderr if isinstance(original_stderr, TimestampedOutputStream) else TimestampedOutputStream(original_stderr)
    sys.stdout, sys.stderr = output, error
    print(f"[+] Repeated warning/status watchdog armed at {CONFIG.repeated_output_limit} matches.")
    resets = 0
    try:
        while True:
            try:
                RucoyBot().run()
                break
            except RepeatedOutputReset as problem:
                resets += 1
                if resets > 1:
                    sys.stdout.write("[!] The same output loop returned after an automatic reset; stopping to prevent repeated restarts.\n")
                    sys.stdout.flush()
                    break
                sys.stdout.write(f"\n[!] Output pattern repeated {CONFIG.repeated_output_limit} times; resetting the bot engine: {problem.pattern}\n")
                sys.stdout.flush()
                guard.counts.clear()
            except KeyboardInterrupt:
                sys.stdout.write("\n[+] Interrupt received; bot stopped cleanly.\n")
                sys.stdout.flush()
                break
    finally:
        output.guard = None
        output.flush()
        error.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if CONFIG.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_bot()
