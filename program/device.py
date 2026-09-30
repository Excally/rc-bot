"""ADB and Win32 input/screen-capture backends."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig

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
            device_states: dict[str, str] = {}
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 2:
                    device_states[fields[0]] = fields[1]
            if result.returncode == 0 and device_states.get(serial) == "device":
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
