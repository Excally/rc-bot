import os
import re
import glob
import time
import sys
import subprocess
import shutil
import cv2
import numpy as np

# ================================================================================
# CONFIGURATION
# ==============================================================================
# Target mob name to hunt
TARGET_MOB_NAME = "Zombie Lv.65"

# True Background Mode via ADB (Recommended):
# - 100% background: Emulator can stay behind your work/browser/games.
# - ZERO mouse hijacking: Your physical mouse cursor is never touched!
# - Direct touch taps sent straight into Android.
PREFER_ADB = True

# Window title of the emulator (used for Win32 fallback)
WINDOW_TITLE = "MSI App Player"

# Templates directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Show small preview window (press 'q' in preview window to quit)
SHOW_PREVIEW = False

# Minimap registration travel. The reference image remains read-only; travel
# is allowed only after the visible map registers to it with clear confidence.
ENABLE_MINIMAP_WALK = True
MINIMAP_IDLE_DELAY = 0.9
MINIMAP_COOLDOWN = 3.0
MINIMAP_SAFE_VIEW_FRACTION = 0.70
MINIMAP_MIN_CLICK_DISTANCE = 100
MINIMAP_MAX_CLICK_DISTANCE = 260
MINIMAP_ALIGN_MIN_SCORE = 0.45
MINIMAP_ALIGN_MIN_GAP = 0.025
MINIMAP_GEOMETRY_MIN_SCORE = 0.40

# Safety margins (fraction of game canvas) to exclude UI elements
MARGIN_TOP = 0.12
MARGIN_BOTTOM = 0.06
MARGIN_LEFT = 0.06
MARGIN_RIGHT = 0.06

# Offset below the nametag to click the mob body (pixels, auto-scaled)
# Increased to 52 to click lower on the mob body/hitbox and avoid clicking too high
MOB_BODY_Y_OFFSET = 52

# Deadzone radius around screen center to ignore player's own character
PLAYER_DEADZONE_RADIUS = 60

# Template matching minimum score
MATCH_THRESHOLD = 0.55
TARGET_SCAN_WIDTH = 1100
UI_MATCH_THRESHOLD = 0.68
BACK_ICON_CENTER = (1546, 44)
BACK_ICON_SCALES = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)
WHITE_TEXT_MIN_CHANNEL = 190
WHITE_TEXT_MAX_CHANNEL_SPREAD = 50

# How long (seconds) to wait for red target square after clicking mob (gives time to walk/lock)
TARGET_LOCK_TIMEOUT = 2.5

# Grace period (seconds) before confirming mob is defeated when red square drops (prevents flicker drops)
TARGET_DEFEATED_GRACE_TIME = 0.70
EXHAUSTED_MOB_BLACKLIST_SECONDS = 600.0
REPEATED_OUTPUT_LIMIT = 20
# ==============================================================================

# ==============================================================================
class RepeatedOutputReset(Exception):
    def __init__(self, pattern):
        super().__init__(pattern)
        self.pattern = pattern

class RepeatedOutputGuard:
    """Request one engine reset when the same warning/status pattern loops."""
    def __init__(self, limit=REPEATED_OUTPUT_LIMIT):
        self.limit = limit
        self.counts = {}

    def observe(self, message):
        line = message.strip()
        if not line:
            return
        if line.startswith((
            "[+] Red square gone", "[+] Back action closed",
            "[+] System Back closed", "[+] Screen capture recovered",
            "[*] Target locked", "[*] Detected active target lock",
        )):
            self.counts.clear()
            return

        watch = line.startswith(("[!]", "[-]", "[...]"))
        if line.startswith("[~]"):
            lower_line = line.lower()
            watch = any(word in lower_line for word in (
                "failed", "skipped", "still", "unavailable", "could not", "retry",
            ))
        if not watch:
            return

        pattern = re.sub(r"\d+(?:\.\d+)?", "<n>", line)
        pattern = re.sub(r"\s+", " ", pattern)
        self.counts[pattern] = self.counts.get(pattern, 0) + 1
        if self.counts[pattern] >= self.limit:
            self.counts.clear()
            raise RepeatedOutputReset(pattern)

class TimestampedOutputStream:
    """Prefix each console line while passing raw lines to the loop watchdog."""
    def __init__(self, stream, guard=None):
        self.stream = stream
        self.guard = guard
        self.pending = ""

    def _write_line(self, line, ending):
        raw_line = line.rstrip("\r")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.stream.write(f"[{timestamp}] {raw_line}{ending}")
        if self.guard is not None:
            self.guard.observe(raw_line)

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._write_line(line, "\n")
        return len(text)

    def flush(self):
        if self.pending:
            pending, self.pending = self.pending, ""
            self._write_line(pending, "")
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

# ADB ENGINE (TRUE BACKGROUND HUNTING)
# ==============================================================================

def get_bluestacks_adb_info():
    """Find HD-Adb.exe and active ADB port for MSI App Player / BlueStacks."""
    candidates = [
        r"C:\Program Files\BlueStacks_msi5\HD-Adb.exe",
        r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_msi5\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
    ]
    adb_path = None
    for c in candidates:
        if os.path.exists(c):
            adb_path = c
            break
    if not adb_path:
        adb_path = shutil.which("adb")

    # Read live port from bluestacks.conf
    port = None
    conf_patterns = [
        r"C:\ProgramData\BlueStacks_msi5\bluestacks.conf",
        r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf",
        r"C:\ProgramData\BlueStacks*\**\*.conf"
    ]
    for pattern in conf_patterns:
        for conf_file in glob.glob(pattern, recursive=True):
            if os.path.isfile(conf_file):
                try:
                    with open(conf_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if "status.adb_port" in line:
                                parts = line.split("=")
                                if len(parts) == 2:
                                    val = parts[1].strip().strip('"\' \r\n')
                                    if val.isdigit():
                                        port = int(val)
                                        break
                except Exception:
                    pass
            if port:
                break
        if port:
            break

    return adb_path, port or 5555

def init_adb():
    """Connect to MSI App Player via ADB for background capture and input."""
    adb_path, port = get_bluestacks_adb_info()
    if not adb_path:
        return None, None

    device_serial = f"127.0.0.1:{port}"
    print(f"[+] Found ADB executable: {adb_path}")
    print(f"[+] Connecting to emulator instance at {device_serial}...")

    try:
        res = subprocess.run(
            [adb_path, "connect", device_serial],
            capture_output=True,
            text=True,
            timeout=15
        )
        # Check devices
        dev_res = subprocess.run(
            [adb_path, "devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if device_serial in dev_res.stdout and "device" in dev_res.stdout:
            print(f"[+] ADB connected successfully to {device_serial}!")
            return adb_path, device_serial
    except Exception as e:
        print(f"[!] ADB connection failed: {e}")

    return None, None

def adb_capture(adb_path, device_serial):
    """
    Capture the game screen directly from Android framebuffer.
    Works 100% in the background even if emulator is behind other windows or minimized!
    """
    try:
        p = subprocess.run(
            [adb_path, "-s", device_serial, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=3
        )
        if not p.stdout:
            return None
        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def adb_click(adb_path, device_serial, x, y, synchronous=False):
    """Send an ADB tap, optionally waiting for injection to finish."""
    command = [
        adb_path, "-s", device_serial, "shell", "input", "tap",
        str(int(x)), str(int(y)),
    ]
    try:
        if synchronous:
            result = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0,
            )
            if result.returncode != 0:
                print(f"[!] ADB tap failed at ({int(x)}, {int(y)}), exit code {result.returncode}.")
                return False
        else:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[!] ADB tap failed at ({int(x)}, {int(y)}): {e}")
        return False

# ==============================================================================
# WIN32 FALLBACK ENGINE
# ==============================================================================

def find_window_by_title(title):
    """Find a window by title (exact or substring)."""
    import win32gui
    result = [None, ""]

    def cb(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            if t == title:
                result[0] = h
                result[1] = t
                return False
        return True

    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass

    if result[0]:
        return result[0], result[1]

    def cb2(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            cls = win32gui.GetClassName(h)
            if title.lower() in t.lower() and "CabinetW" not in cls and "Chrome" not in cls:
                result[0] = h
                result[1] = t
                return False
        return True

    try:
        win32gui.EnumWindows(cb2, None)
    except Exception:
        pass

    if result[0]:
        return result[0], result[1]

    raise Exception(f"Window '{title}' not found!")

def find_game_child(parent_hwnd):
    """Find BlueStacksApp child window inside MSI App Player."""
    import win32gui
    result = [None]
    def cb(c, _):
        cls = win32gui.GetClassName(c)
        if "BlueStacks" in cls:
            result[0] = c
            return False
        return True
    try:
        win32gui.EnumChildWindows(parent_hwnd, cb, None)
    except Exception:
        pass

    if result[0]:
        return result[0]

    return parent_hwnd

def win32_capture(hwnd):
    """Win32 mss fallback screen capture."""
    import win32gui
    from mss import MSS

    r = win32gui.GetClientRect(hwnd)
    w, h = r[2] - r[0], r[3] - r[1]
    if w <= 0 or h <= 0:
        return None

    sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
    region = {"left": sx, "top": sy, "width": w, "height": h}
    with MSS() as sct:
        shot = np.array(sct.grab(region))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

def win32_click(hwnd, x, y):
    """Safely click on window via Win32 cursor with error handling."""
    import win32gui
    import win32api
    import win32con

    try:
        scr_x, scr_y = win32gui.ClientToScreen(hwnd, (int(x), int(y)))
        orig = win32api.GetCursorPos()
        try:
            win32api.SetCursorPos((int(scr_x), int(scr_y)))
            time.sleep(0.03)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.06)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.03)
            win32api.SetCursorPos(orig)
        except Exception:
            # Fallback to PostMessage if SetCursorPos is restricted
            lparam = (int(y) << 16) | (int(x) & 0xFFFF)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
            time.sleep(0.05)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
    except Exception as e:
        print(f"[!] Win32 click error: {e}")

def capture_screen(adb_path, device_serial, game_hwnd, use_adb):
    return adb_capture(adb_path, device_serial) if use_adb else win32_capture(game_hwnd)

def send_click(adb_path, device_serial, game_hwnd, use_adb, x, y, synchronous=False):
    if use_adb:
        return adb_click(adb_path, device_serial, x, y, synchronous=synchronous)
    win32_click(game_hwnd, x, y)
    return True

# ==============================================================================
# VISION & HUNTING LOGIC
# ==============================================================================

def load_templates():
    """Load only templates used by combat targeting and UI recovery."""
    required = {
        "zombie_lv65",
        "zombie",
        "purple_name_zombie_lv65",
        "minimapicon",
        "backicon",
        "minimap-zombie-layout",
        "pickup-available",
        "pickup-not-yet",
        "exhausted-caption",
    }
    templates = {}
    if os.path.exists(TEMPLATE_DIR):
        for f in os.listdir(TEMPLATE_DIR):
            if f.lower().endswith((".png", ".jpg")):
                name = os.path.splitext(f)[0]
                if name not in required:
                    continue
                img = cv2.imread(os.path.join(TEMPLATE_DIR, f))
                if img is not None:
                    templates[name] = img
    return templates

def find_exhausted_caption(frame, template):
    """Detect the template's red banner using its color, aspect, and density."""
    if frame is None or template is None:
        return False

    fh, fw = frame.shape[:2]
    screen_scale = fw / 1600.0
    cached = getattr(find_exhausted_caption, "_template_cache", None)
    if cached is None or cached[0] is not template:
        template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        hue = template_hsv[:, :, 0]
        template_mask = (
            ((hue <= 10) | (hue >= 170)) &
            (template_hsv[:, :, 1] >= 110) &
            (template_hsv[:, :, 2] >= 90)
        ).astype(np.uint8) * 255
        points = cv2.findNonZero(template_mask)
        if points is None:
            return False
        _, _, template_w, template_h = cv2.boundingRect(points)
        density = cv2.countNonZero(template_mask) / (template_w * template_h)
        cached = (template, template_w, template_h, density)
        find_exhausted_caption._template_cache = cached

    template_w, template_h, density = cached[1:]
    x0, x1 = int(fw * 0.12), int(fw * 0.88)
    y1 = int(fh * 0.50)
    region = frame[:y1, x0:x1]
    processing_scale = min(0.5, 800.0 / max(1, region.shape[1]))
    frame_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = frame_hsv[:, :, 0]
    red_mask = (
        ((hue <= 10) | (hue >= 170)) &
        (frame_hsv[:, :, 1] >= 110) &
        (frame_hsv[:, :, 2] >= 90)
    ).astype(np.uint8) * 255
    if processing_scale < 1.0:
        red_mask = cv2.resize(
            red_mask, None, fx=processing_scale, fy=processing_scale,
            interpolation=cv2.INTER_NEAREST,
        )
    close_width = max(5, int(round(15 * screen_scale * processing_scale)))
    joined = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, np.ones((3, close_width), dtype=np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    expected_w = template_w * screen_scale * processing_scale
    expected_h = template_h * screen_scale * processing_scale
    min_w, max_w = expected_w * 0.50, expected_w * 1.20
    min_h, max_h = expected_h * 0.45, expected_h * 1.50
    min_aspect, max_aspect = (template_w / template_h) * 0.55, (template_w / template_h) * 1.70
    min_density, max_density = density * 0.45, min(0.78, density * 1.80)

    for x, y, width, height, _ in stats[1:count]:
        if not (min_w <= width <= max_w and min_h <= height <= max_h):
            continue
        aspect = width / max(1, height)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        red_pixels = cv2.countNonZero(red_mask[y:y + height, x:x + width])
        region_density = red_pixels / (width * height)
        if min_density <= region_density <= max_density:
            return True
    return False

def find_fixed_ui_icon(frame, icon, expected_center, threshold=UI_MATCH_THRESHOLD, search_radius=(90, 70), scale_adjustments=(0.75, 0.8, 0.85, 0.9, 1.0, 1.1)):
    """Match a fixed UI icon near its known screen position."""
    if frame is None or icon is None:
        return None

    fh, fw = frame.shape[:2]
    sx, sy = fw / 1600.0, fh / 900.0
    center_x, center_y = int(expected_center[0] * sx), int(expected_center[1] * sy)
    radius_x, radius_y = int(search_radius[0] * sx), int(search_radius[1] * sy)
    best_score, best_center = -1.0, None

    # Game UI icons can be rendered smaller than their saved templates at
    # different emulator UI scales, even when their screen position is fixed.
    for scale_adjustment in scale_adjustments:
        scale = sx * scale_adjustment
        scaled = cv2.resize(icon, None, fx=scale, fy=scale)
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

        result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_center = (left + location[0] + iw // 2, top + location[1] + ih // 2)

    return best_center if best_score >= threshold else None

def find_back_icon(frame, templates):
    return find_fixed_ui_icon(
        frame, templates.get("backicon"), BACK_ICON_CENTER,
        threshold=0.40, search_radius=(35, 30),
        scale_adjustments=BACK_ICON_SCALES,
    )

def detect_game_ui(frame, templates):
    """Recognize the map panel before combat, then treat other panels separately."""
    if is_disconnected_screen(frame):
        return "disconnected", None
    back = find_back_icon(frame, templates)
    if back is not None:
        return "minimap", back
    map_button = find_fixed_ui_icon(frame, templates.get("minimapicon"), (1150, 50))
    if map_button is not None:
        return "combat", map_button
    return "other", None

def tap_back_icon_and_confirm(adb_path, device_serial, game_hwnd, use_adb, back, back_icon):
    """Tap Back once, allow the UI to react, then use system Back if needed."""
    tapped = send_click(
        adb_path, device_serial, game_hwnd, use_adb, *back, synchronous=True
    )
    if not tapped:
        print("[!] Back-icon tap failed; trying the system Back action.")
    else:
        time.sleep(0.35)
        frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)
        if frame is None:
            print("[!] Could not verify the panel closed after tapping Back.")
            return False
        if back_icon is None:
            print("[!] Back icon template is unavailable; panel closure could not be verified.")
            return False
        if find_fixed_ui_icon(
            frame, back_icon, BACK_ICON_CENTER, threshold=0.40,
            search_radius=(35, 30), scale_adjustments=BACK_ICON_SCALES,
        ) is None:
            return True
        print("[!] Back icon remains visible; trying the system Back action.")

    press_ui_back(adb_path, device_serial, game_hwnd, use_adb)
    time.sleep(0.35)
    frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)
    if frame is not None and back_icon is not None and find_fixed_ui_icon(
        frame, back_icon, BACK_ICON_CENTER, threshold=0.40,
        search_radius=(35, 30), scale_adjustments=BACK_ICON_SCALES,
    ) is None:
        return True

    print("[!] Panel is still open after Back-icon and system Back attempts.")
    return False

def fail_minimap_travel(adb_path, device_serial, game_hwnd, use_adb, back, templates, message):
    print(message)
    tap_back_icon_and_confirm(
        adb_path, device_serial, game_hwnd, use_adb,
        back, templates.get("backicon"),
    )
    return False

def minimap_white_mask(image):
    """Extract bright map geometry while ignoring colored game objects."""
    channels = image.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    return (
        (channels.min(axis=2) >= 220) &
        (spread <= 35)
    ).astype(np.uint8) * 255

def build_minimap_safe_mask(reference):
    """Build an interior floor mask from the immutable full-map reference."""
    walls = minimap_white_mask(reference)
    contours, hierarchy = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(walls.shape, dtype=np.uint8)

    outer_index = max(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]))
    outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(contours[outer_index])
    margin = max(30, int(round(min(reference.shape[:2]) * 0.035)))

    safe = np.zeros(walls.shape, dtype=np.uint8)
    x1 = outer_x + margin
    y1 = outer_y + margin
    x2 = outer_x + outer_w - margin
    y2 = outer_y + outer_h - margin
    if x2 <= x1 or y2 <= y1:
        return safe
    safe[y1:y2, x1:x2] = 255

    blocked = walls.copy()
    # The large outlined rectangles are obstacles. Fill their enclosed holes
    # so the selector cannot click inside a tomb or similar structure.
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            parent = int(hierarchy[0][index][3])
            if parent < 0:
                continue
            if cv2.contourArea(contour) < 1000:
                continue
            if cv2.contourArea(contours[parent]) < 5000:
                continue
            cv2.drawContours(blocked, [contour], -1, 255, thickness=-1)

    blocked = cv2.dilate(blocked, np.ones((9, 9), np.uint8), iterations=1)
    safe[blocked > 0] = 0
    safe = cv2.erode(safe, np.ones((5, 5), np.uint8), iterations=1)
    return safe

def build_minimap_waypoints(safe_mask, grid_step=120):
    """Build a deterministic serpentine route over the complete safe map."""
    if safe_mask is None or safe_mask.size == 0:
        return []

    height, width = safe_mask.shape[:2]
    waypoints = []
    radius = max(12, grid_step // 3)
    half_step = max(1, grid_step // 2)

    # Snap every grid cell to its nearest safe pixel. This keeps the route
    # inside the reference while still covering rooms around walls/obstacles.
    for row, y in enumerate(range(half_step, height, grid_step)):
        row_points = []
        for x in range(half_step, width, grid_step):
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(width, x + radius + 1)
            y2 = min(height, y + radius + 1)
            local = safe_mask[y1:y2, x1:x2]
            local_y, local_x = np.where(local > 0)
            if len(local_x) == 0:
                continue
            distances = (local_x + x1 - x) ** 2 + (local_y + y1 - y) ** 2
            nearest = int(np.argmin(distances))
            row_points.append((int(local_x[nearest] + x1), int(local_y[nearest] + y1)))

        # Alternating rows make the route sweep the map instead of repeatedly
        # returning to one side after each horizontal pass.
        if row % 2:
            row_points.reverse()
        for point in row_points:
            if not waypoints or point != waypoints[-1]:
                waypoints.append(point)

    return waypoints

def _minimap_geometry_score(frame_mask, scaled_reference, origin):
    """Score a proposed reference placement using overlap in both directions."""
    fh, fw = frame_mask.shape[:2]
    sh, sw = scaled_reference.shape[:2]
    ox, oy = origin
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(fw, ox + sw), min(fh, oy + sh)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    expected = (scaled_reference[y1 - oy:y2 - oy, x1 - ox:x2 - ox] > 0).astype(np.uint8)
    observed = (frame_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if cv2.countNonZero(expected) < 3000 or cv2.countNonZero(observed) < 3000:
        return 0.0

    kernel = np.ones((7, 7), np.uint8)
    expected_near = cv2.dilate(expected, kernel, iterations=1) > 0
    observed_near = cv2.dilate(observed, kernel, iterations=1) > 0
    recall = float(np.count_nonzero(expected & observed_near)) / max(1, np.count_nonzero(expected))
    precision = float(np.count_nonzero(observed & expected_near)) / max(1, np.count_nonzero(observed))
    if recall + precision == 0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def _refine_minimap_y(frame_mask, scaled_reference, origin):
    """Correct repeated-row ambiguity by optimizing the vertical placement."""
    ox, oy = origin
    coarse = []
    for offset in range(-360, 361, 16):
        candidate_y = oy + offset
        coarse.append((
            _minimap_geometry_score(frame_mask, scaled_reference, (ox, candidate_y)),
            candidate_y,
        ))
    _, coarse_y = max(coarse)

    fine = []
    for candidate_y in range(coarse_y - 16, coarse_y + 17, 2):
        fine.append((
            _minimap_geometry_score(frame_mask, scaled_reference, (ox, candidate_y)),
            candidate_y,
        ))
    geometry, refined_y = max(fine)
    return (ox, refined_y), geometry


def align_minimap_to_reference(frame, reference):
    """Register a partial transparent minimap to the full reference image."""
    if frame is None or reference is None:
        return None

    frame_mask = minimap_white_mask(frame)
    if cv2.countNonZero(frame_mask) < 12000:
        return None

    # Downsampling makes the one-time travel registration inexpensive while
    # preserving the large walls and repeated room rectangles.
    downsample = 0.25
    search_frame = cv2.resize(
        frame_mask, None, fx=downsample, fy=downsample,
        interpolation=cv2.INTER_NEAREST,
    )
    reference_mask = minimap_white_mask(reference)
    raw_candidates = []
    patch_w = min(180, search_frame.shape[1] - 4)
    patch_h = min(180, search_frame.shape[0] - 4)
    if patch_w < 120 or patch_h < 120:
        return None

    scales = (0.92, 0.96, 1.0, 1.04, 1.08)
    for scale in scales:
        scaled = cv2.resize(
            reference_mask, None, fx=scale * downsample, fy=scale * downsample,
            interpolation=cv2.INTER_NEAREST,
        )
        scaled_h, scaled_w = scaled.shape[:2]
        if scaled_w < patch_w or scaled_h < patch_h:
            continue

        x_starts = sorted(set((
            0,
            max(0, (scaled_w - patch_w) // 2),
            max(0, scaled_w - patch_w),
        )))
        y_remaining = max(0, scaled_h - patch_h)
        y_starts = sorted(set(
            int(round(y_remaining * fraction))
            for fraction in (0.0, 0.5, 1.0)
        ))
        for crop_y in y_starts:
            for crop_x in x_starts:
                patch = scaled[crop_y:crop_y + patch_h, crop_x:crop_x + patch_w]
                if cv2.countNonZero(patch) < 3000:
                    continue
                result = cv2.matchTemplate(
                    search_frame, patch, cv2.TM_CCOEFF_NORMED
                )
                _, score, _, location = cv2.minMaxLoc(result)
                origin = (
                    int(round((location[0] - crop_x) / downsample)),
                    int(round((location[1] - crop_y) / downsample)),
                )
                raw_candidates.append((float(score), float(scale), origin))

    if not raw_candidates:
        return None

    raw_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    candidates = []
    for candidate in raw_candidates:
        score, scale, origin = candidate
        if any(
            abs(origin[0] - kept[2][0]) <= 10 and
            abs(origin[1] - kept[2][1]) <= 10 and
            abs(scale - kept[1]) <= 0.035
            for kept in candidates
        ):
            continue
        scaled_reference = cv2.resize(
            reference_mask, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        geometry = _minimap_geometry_score(frame_mask, scaled_reference, origin)
        if geometry > 0:
            candidates.append((score, scale, origin, geometry))
            if len(candidates) >= 8:
                break
    if not candidates:
        return None
    # Repeated rooms and rows can make a local template patch look excellent
    # at the wrong vertical offset. Full visible geometry is the authority.
    candidates.sort(key=lambda candidate: (candidate[3], candidate[0]), reverse=True)
    best = candidates[0]
    second_geometry = candidates[1][3] if len(candidates) > 1 else 0.0
    score, scale, origin, geometry = best
    scaled_reference = cv2.resize(
        reference_mask, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    origin, geometry = _refine_minimap_y(frame_mask, scaled_reference, origin)
    gap = geometry - second_geometry
    if (
        score < MINIMAP_ALIGN_MIN_SCORE or
        geometry < MINIMAP_GEOMETRY_MIN_SCORE or
        gap < MINIMAP_ALIGN_MIN_GAP
    ):
        return {
            "valid": False,
            "score": score,
            "geometry": geometry,
            "gap": gap,
            "scale": scale,
            "origin": origin,
        }

    return {
        "valid": True,
        "score": score,
        "geometry": geometry,
        "gap": gap,
        "scale": scale,
        "origin": origin,
    }

def find_minimap_player(frame):
    """Find the fixed 20px cyan player marker on the open map."""
    if frame is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    marker = (
        (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115) &
        (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 100)
    ).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(marker, 8)
    fh, fw = frame.shape[:2]
    center = np.array([fw / 2.0, fh / 2.0])
    choices = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (200 <= area <= 3000 and 14 <= w <= 100 and 14 <= h <= 100):
            continue
        point = centroids[index]
        distance = float(np.linalg.norm(point - center))
        if distance > min(fw, fh) * 0.18:
            continue
        choices.append((distance, (int(round(point[0])), int(round(point[1])))))
    if not choices:
        return None
    choices.sort(key=lambda choice: choice[0])
    return choices[0][1]

def choose_minimap_destination(frame, alignment, safe_mask, player_screen, waypoints=None):
    """Choose a visible point that advances through the full-map route."""
    if not alignment or not alignment.get("valid") or safe_mask is None or player_screen is None:
        return None

    if waypoints is None:
        waypoints = build_minimap_waypoints(safe_mask)
    if not waypoints:
        return None

    scale = alignment["scale"]
    ox, oy = alignment["origin"]
    player_x, player_y = player_screen
    map_x = (player_x - ox) / scale
    map_y = (player_y - oy) / scale
    ref_h, ref_w = safe_mask.shape[:2]
    if not (0 <= map_x < ref_w and 0 <= map_y < ref_h and safe_mask[int(map_y), int(map_x)] > 0):
        return None

    step = 8
    ys, xs = np.where(safe_mask[::step, ::step] > 0)
    if len(xs) == 0:
        return None
    ref_xs = xs * step + step // 2
    ref_ys = ys * step + step // 2
    screen_xs = ox + ref_xs * scale
    screen_ys = oy + ref_ys * scale
    fh, fw = frame.shape[:2]
    margin_x = fw * (1.0 - MINIMAP_SAFE_VIEW_FRACTION) / 2.0
    margin_y = fh * (1.0 - MINIMAP_SAFE_VIEW_FRACTION) / 2.0
    visible = (
        (screen_xs >= margin_x) & (screen_xs <= fw - margin_x) &
        (screen_ys >= margin_y) & (screen_ys <= fh - margin_y)
    )
    distances = np.hypot(screen_xs - player_x, screen_ys - player_y)
    visible &= (
        distances >= MINIMAP_MIN_CLICK_DISTANCE * fw / 1600.0
    ) & (
        distances <= MINIMAP_MAX_CLICK_DISTANCE * fw / 1600.0
    )
    if not np.any(visible):
        return None

    route_signature = (
        safe_mask.shape,
        len(waypoints),
        waypoints[0],
        waypoints[-1],
    )
    route = getattr(choose_minimap_destination, "route", None)
    if route is None or route["signature"] != route_signature:
        # Start at the nearest route point, then move to the next point. This
        # avoids forcing a newly started bot to cross the whole map first.
        waypoint_array = np.asarray(waypoints, dtype=np.float32)
        nearest_index = int(np.argmin(
            np.hypot(waypoint_array[:, 0] - map_x, waypoint_array[:, 1] - map_y)
        ))
        route = {
            "signature": route_signature,
            "waypoints": tuple(waypoints),
            "index": nearest_index,
        }
        choose_minimap_destination.route = route

    waypoint_array = np.asarray(route["waypoints"], dtype=np.float32)
    current_index = int(route["index"])
    target_index = (current_index + 1) % len(waypoint_array)
    target_x, target_y = waypoint_array[target_index]
    distance_to_target = float(np.hypot(target_x - map_x, target_y - map_y))
    if distance_to_target < max(70.0, MINIMAP_MIN_CLICK_DISTANCE / max(scale, 0.1)):
        current_index = target_index
        route["index"] = current_index
        target_index = (current_index + 1) % len(waypoint_array)
        target_x, target_y = waypoint_array[target_index]

    # A target may be outside the partial minimap. Choose the visible safe
    # point that makes the most progress toward it, with a stable distance tie
    # breaker so the same frame produces the same click.
    candidate_ref_xs = (screen_xs - ox) / scale
    candidate_ref_ys = (screen_ys - oy) / scale
    before = np.hypot(target_x - map_x, target_y - map_y)
    after = np.hypot(target_x - candidate_ref_xs, target_y - candidate_ref_ys)
    progress = before - after
    preferred_distance = (
        MINIMAP_MIN_CLICK_DISTANCE + MINIMAP_MAX_CLICK_DISTANCE
    ) * 0.5 * fw / 1600.0
    score = progress * 1000.0 - np.abs(distances - preferred_distance)
    score[~visible] = -np.inf
    index = int(np.argmax(score))
    choose_minimap_destination.last_route_target = (
        int(round(target_x)), int(round(target_y))
    )
    return int(round(screen_xs[index])), int(round(screen_ys[index]))

def travel_via_minimap(adb_path, device_serial, game_hwnd, use_adb, templates):
    """Register the visible minimap, click one safe nearby waypoint, and close it."""
    reference = templates.get("minimap-zombie-layout")
    if reference is None:
        print("[!] Minimap travel disabled: full reference map is missing.")
        return False

    map_frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)
    if map_frame is None:
        return False

    state, icon_position = detect_game_ui(map_frame, templates)
    if state == "combat":
        target_template = templates.get("zombie_lv65")
        if target_template is None:
            target_template = templates.get("zombie")
        visible_targets = []
        if target_template is not None:
            visible_targets.append((target_template, "white"))
        purple_template = templates.get("purple_name_zombie_lv65")
        if purple_template is not None:
            visible_targets.append((purple_template, "purple"))
        fh, fw = map_frame.shape[:2]
        if has_red_square(map_frame)[0] or find_target_mobs(
            map_frame, visible_targets, fw // 2, fh // 2, MATCH_THRESHOLD
        ):
            print("[~] Minimap travel skipped: a target appeared in the latest frame.")
            return False

        if not send_click(
            adb_path, device_serial, game_hwnd, use_adb, *icon_position
        ):
            return False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            time.sleep(0.04)
            map_frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)
            if map_frame is None:
                continue
            state, icon_position = detect_game_ui(map_frame, templates)
            if state in ("minimap", "disconnected"):
                break

    if state != "minimap":
        print(f"[!] Minimap travel skipped: UI state is {state}.")
        return False

    alignment = align_minimap_to_reference(map_frame, reference)
    if not alignment or not alignment.get("valid"):
        details = alignment or {}
        return fail_minimap_travel(
            adb_path, device_serial, game_hwnd, use_adb, icon_position, templates,
            "[!] Minimap alignment rejected "
            f"(match {details.get('score', 0.0):.2f}, "
            f"geometry {details.get('geometry', 0.0):.2f}, "
            f"gap {details.get('gap', 0.0):.2f}); no movement tap sent."
        )

    player_screen = find_minimap_player(map_frame)
    if player_screen is None:
        return fail_minimap_travel(
            adb_path, device_serial, game_hwnd, use_adb, icon_position, templates,
            "[!] Minimap player marker was not found; no movement tap sent."
        )

    route_cache = getattr(travel_via_minimap, "route_cache", None)
    if route_cache is None or route_cache[0] != id(reference):
        safe_mask = build_minimap_safe_mask(reference)
        waypoints = build_minimap_waypoints(safe_mask)
        travel_via_minimap.route_cache = (id(reference), safe_mask, waypoints)
    else:
        _, safe_mask, waypoints = route_cache
    destination = choose_minimap_destination(
        map_frame, alignment, safe_mask, player_screen, waypoints
    )
    if destination is None:
        return fail_minimap_travel(
            adb_path, device_serial, game_hwnd, use_adb, icon_position, templates,
            "[!] No safe visible minimap waypoint was found; no movement tap sent."
        )

    print(
        f"[~] Minimap waypoint {destination} from player {player_screen} "
        f"toward reference {getattr(choose_minimap_destination, 'last_route_target', '?')} "
        f"(match {alignment['score']:.2f}, geometry {alignment['geometry']:.2f}, "
        f"gap {alignment['gap']:.2f}, scale {alignment['scale']:.2f})."
    )
    sent = send_click(
        adb_path, device_serial, game_hwnd, use_adb,
        *destination, synchronous=True,
    )
    if sent:
        time.sleep(0.07)
    closed = tap_back_icon_and_confirm(
        adb_path, device_serial, game_hwnd, use_adb,
        icon_position, templates.get("backicon"),
    )
    return bool(sent and closed)

def is_disconnected_screen(frame):
    """Recognize the game's dark disconnect screen without changing templates."""
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape
    title = gray[int(fh * 0.18):int(fh * 0.32), int(fw * 0.32):int(fw * 0.68)]
    reconnect = gray[int(fh * 0.58):int(fh * 0.74), int(fw * 0.14):int(fw * 0.49)]
    return (
        np.mean(gray < 24) > 0.82 and
        np.count_nonzero(title > 175) > 250 and
        np.count_nonzero(reconnect > 175) > 180
    )

def recover_unexpected_ui(frame, adb_path, device_serial, game_hwnd, use_adb, templates):
    """Dismiss an unexpected panel or reconnect the game, keeping this process alive."""
    if is_disconnected_screen(frame):
        fh, fw = frame.shape[:2]
        point = (int(fw * 0.3125), int(fh * 0.667))
        send_click(adb_path, device_serial, game_hwnd, use_adb, *point)
        print("[~] Disconnected screen detected; selected Reconnect and will resume automatically.")
        return "reconnect"

    back = find_back_icon(frame, templates)
    if back is not None:
        if tap_back_icon_and_confirm(
            adb_path, device_serial, game_hwnd, use_adb,
            back, templates.get("backicon"),
        ):
            print("[+] Back action closed the unexpected UI.")
            return "back"
        print("[!] Could not close the unexpected UI; recovery is backing off.")
        return "failed"

    press_ui_back(adb_path, device_serial, game_hwnd, use_adb)
    time.sleep(0.35)
    after = capture_screen(adb_path, device_serial, game_hwnd, use_adb)
    if after is not None and detect_game_ui(after, templates)[0] == "combat":
        print("[+] System Back closed the unexpected UI.")
        return "back"
    print("[!] System Back did not return to combat; recovery is backing off.")
    return "failed"

def press_ui_back(adb_path, device_serial, game_hwnd, use_adb):
    """Dismiss an unexpected in-game panel without restarting the bot."""
    if use_adb:
        subprocess.Popen(
            [adb_path, "-s", device_serial, "shell", "input", "keyevent", "4"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    try:
        import win32con
        import win32gui
        win32gui.PostMessage(game_hwnd, win32con.WM_KEYDOWN, win32con.VK_ESCAPE, 0)
        win32gui.PostMessage(game_hwnd, win32con.WM_KEYUP, win32con.VK_ESCAPE, 0)
    except Exception as e:
        print(f"[!] Could not dismiss unexpected UI: {e}")

def white_text_mask(image):
    """Keep only near-white pixels used by the mob name template."""
    channels = image.astype(np.int16)
    return (
        (channels.min(axis=2) >= WHITE_TEXT_MIN_CHANNEL) &
        (channels.max(axis=2) - channels.min(axis=2) <= WHITE_TEXT_MAX_CHANNEL_SPREAD)
    ).astype(np.uint8) * 255

def purple_text_mask(image):
    """Keep the saturated purple pixels used by the elite mob name template."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 0] >= 138) & (hsv[:, :, 0] <= 154) &
        (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 170)
    ).astype(np.uint8) * 255

def find_pickup_available(frame, templates):
    """Locate the orange pickup button and reject its gray unavailable state."""
    available = templates.get("pickup-available")
    unavailable = templates.get("pickup-not-yet")
    if frame is None or available is None or unavailable is None:
        return None

    fh, fw = frame.shape[:2]
    left, right = int(fw * 0.82), fw
    top, bottom = int(fh * 0.10), int(fh * 0.55)
    region = frame[top:bottom, left:right]
    best_available = (-1.0, None)
    best_unavailable = -1.0

    for scale in (0.75, 0.9, 1.0, 1.1, 1.25):
        scaled_available = cv2.resize(available, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ah, aw = scaled_available.shape[:2]
        if ah >= region.shape[0] or aw >= region.shape[1]:
            continue
        scores = cv2.matchTemplate(region, scaled_available, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        if score > best_available[0]:
            best_available = (
                score,
                (left + location[0] + aw // 2, top + location[1] + ah // 2),
            )

        scaled_unavailable = cv2.resize(unavailable, (aw, ah), interpolation=cv2.INTER_NEAREST)
        _, unavailable_score, _, _ = cv2.minMaxLoc(
            cv2.matchTemplate(region, scaled_unavailable, cv2.TM_CCOEFF_NORMED)
        )
        best_unavailable = max(best_unavailable, unavailable_score)

    score, position = best_available
    if position is not None and score >= 0.86 and score >= best_unavailable + 0.08:
        return position
    return None

def find_target_mobs(frame, template, center_x, center_y, threshold=MATCH_THRESHOLD, blacklist=None, curr_time=0.0, name_color="white", allow_scale_fallback=True, fallback_only=False):
    """
    Find mobs matching the template nametag.
    Uses fast two-tier matching (1.0x primary) to minimize CPU usage and prevent hardware heat.
    Returns list of targets sorted by distance (closest to center first).
    """
    if isinstance(template, (list, tuple)):
        matches = []
        def collect(candidate_template, candidate_color, scale_fallback, fallback_pass=False):
            found = find_target_mobs(
                frame, candidate_template, center_x, center_y, threshold,
                blacklist=blacklist, curr_time=curr_time, name_color=candidate_color,
                allow_scale_fallback=scale_fallback, fallback_only=fallback_pass,
            )
            for candidate in found:
                if any(
                    abs(candidate["nx"] - existing["nx"]) < 30 and
                    abs(candidate["ny"] - existing["ny"]) < 15
                    for existing in matches
                ):
                    continue
                matches.append(candidate)

        for candidate_template, candidate_color in template:
            collect(candidate_template, candidate_color, False)
        if not matches:
            for candidate_template, candidate_color in template:
                collect(candidate_template, candidate_color, True, True)
        matches.sort(key=lambda match: match["distance"])
        return matches

    fh, fw = frame.shape[:2]
    top = int(fh * MARGIN_TOP)
    bot = int(fh * (1.0 - MARGIN_BOTTOM))
    left = int(fw * MARGIN_LEFT)
    right = int(fw * (1.0 - MARGIN_RIGHT))

    playfield = frame[top:bot, left:right]
    if playfield.size == 0:
        return []

    processing_scale = min(1.0, TARGET_SCAN_WIDTH / fw)
    make_text_mask = purple_text_mask if name_color == "purple" else white_text_mask
    if getattr(find_target_mobs, "_mask_frame", None) is not frame:
        find_target_mobs._mask_frame = frame
        find_target_mobs._text_masks = {}
    text_pf = find_target_mobs._text_masks.get(name_color)
    if text_pf is None:
        if processing_scale < 1.0:
            search_image = cv2.resize(
                playfield, None, fx=processing_scale, fy=processing_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            search_image = playfield
        text_pf = make_text_mask(search_image)
        find_target_mobs._text_masks[name_color] = text_pf
    if cv2.countNonZero(text_pf) == 0:
        return []

    matches = []

    def scan_scale(scale):
        combined_scale = scale * processing_scale
        scaled_color = (
            cv2.resize(template, None, fx=combined_scale, fy=combined_scale)
            if combined_scale != 1.0 else template
        )
        scaled = make_text_mask(scaled_color)
        th, tw = scaled.shape[:2]
        if th >= text_pf.shape[0] or tw >= text_pf.shape[1] or th < 3 or tw < 3:
            return

        res = cv2.matchTemplate(text_pf, scaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        template_pixels = np.count_nonzero(scaled)

        for pt in zip(*loc[::-1]):
            px, py = pt
            frame_patch = text_pf[py:py + th, px:px + tw]
            shared_pixels = np.count_nonzero(cv2.bitwise_and(frame_patch, scaled))
            if template_pixels == 0 or shared_pixels / template_pixels < 0.35:
                continue

            x = int(round(px / processing_scale)) + left
            y = int(round(py / processing_scale)) + top
            native_tw = max(1, int(round(template.shape[1] * scale)))
            native_th = max(1, int(round(template.shape[0] * scale)))

            # Dedup close matches
            if any(abs(x - m["nx"]) < 30 and abs(y - m["ny"]) < 15 for m in matches):
                continue

            click_x = x + native_tw // 2
            # Lower the click further into the mob body/feet (hitbox)
            click_y = y + native_th + int(MOB_BODY_Y_OFFSET * scale)

            dist = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
            if dist < PLAYER_DEADZONE_RADIUS:
                continue

            # Check if this position is temporarily blacklisted (unreachable / repeatedly missed)
            if blacklist:
                grid_pos = (int(click_x // 40), int(click_y // 40))
                if grid_pos in blacklist and curr_time < blacklist[grid_pos]:
                    continue

            if left <= click_x <= right and top <= click_y <= bot:
                matches.append({
                    "nx": x, "ny": y, "nw": native_tw, "nh": native_th,
                    "click_x": click_x, "click_y": click_y,
                    "distance": dist, "score": float(res[py, px]),
                    "scale": scale
                })

    # Tier 1: Check native 1.0x scale first (instant & covers ~99% of matches on ADB)
    if not fallback_only:
        scan_scale(1.0)

    # Tier 2: Only test fallback scales if no mobs found at 1.0x (saves ~85% CPU power)
    if allow_scale_fallback and not matches:
        for scale in [0.85, 1.15, 0.75, 1.25]:
            scan_scale(scale)
            if matches:
                break

    matches.sort(key=lambda m: m["distance"])
    return matches

def get_marked_template():
    """Load templates/marked.png and extract the binary red border template."""
    if hasattr(get_marked_template, "_tpl"):
        return get_marked_template._tpl
    for fname in ["marked.png", "mob-current-marked.png"]:
        p = os.path.join(TEMPLATE_DIR, fname)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                diff = np.abs(img.astype(int) - [50, 50, 207])
                tpl_border = (np.all(diff <= 20, axis=2)).astype(np.uint8) * 255
                get_marked_template._tpl = tpl_border
                return tpl_border
    # Fallback: create 72x72 hollow square template programmatically
    box = np.zeros((72, 72), dtype=np.uint8)
    box[:7, :] = 255
    box[-7:, :] = 255
    box[:, :7] = 255
    box[:, -7:] = 255
    get_marked_template._tpl = box
    return box

def _match_marked_outline(mask, tpl, threshold=0.50):
    """Return the matched marker center only when the red outline is present."""
    th, tw = tpl.shape[:2]
    if mask.shape[0] < th or mask.shape[1] < tw:
        return None

    result = cv2.matchTemplate(mask, tpl, cv2.TM_CCOEFF_NORMED)
    _, match, _, location = cv2.minMaxLoc(result)
    if match < threshold:
        return None

    x, y = location
    visible = mask[y:y + th, x:x + tw] > 0
    border = tpl > 0
    border_pixels = max(1, int(np.count_nonzero(border)))
    border_hit = np.count_nonzero(visible & border) / border_pixels

    # A mob or nametag can hide two sides. Require a strong template alignment,
    # enough total border pixels, and two substantial sides to reject red mobs.
    thickness = max(3, min(10, int(round(min(th, tw) * 0.07))))
    template_sides = (
        border[:thickness, thickness:-thickness],
        border[-thickness:, thickness:-thickness],
        border[thickness:-thickness, :thickness],
        border[thickness:-thickness, -thickness:],
    )
    visible_sides = (
        visible[:thickness, thickness:-thickness],
        visible[-thickness:, thickness:-thickness],
        visible[thickness:-thickness, :thickness],
        visible[thickness:-thickness, -thickness:],
    )
    side_hits = []
    for side, visible_side in zip(template_sides, visible_sides):
        side_pixels = max(1, int(np.count_nonzero(side)))
        side_hits.append(np.count_nonzero(visible_side & side) / side_pixels)

    if border_hit < 0.35 or sum(hit >= 0.22 for hit in side_hits) < 2:
        return None
    return x + tw // 2, y + th // 2

def has_red_square(frame, tx=None, ty=None, radius=120):
    """
    Check if Rucoy target lock marker (red square under/around mob, templates/marked.png) exists.
    Uses templates/marked.png template matching to strictly reject red mobs, red armor, and UI buttons.
    Returns:
      (True, (center_x, center_y)) if target square is confirmed,
      (False, None) otherwise.
    """
    fh, fw = frame.shape[:2]
    top = int(fh * 0.13)
    bot = int(fh * 0.85)
    left = int(fw * 0.08)
    right = int(fw * 0.90)

    tpl = get_marked_template()

    # Exact Rucoy marker color: BGR [50, 50, 207] (strict tolerance to reject red mobs / orange tones)
    diff = np.abs(frame.astype(np.int32) - np.array([50, 50, 207], dtype=np.int32))
    marker_red = (np.all(diff <= 22, axis=2)).astype(np.uint8) * 255

    # Exclude UI areas (potions/spells bottom-left, action buttons right edge)
    marker_red[int(fh * 0.75):, :int(fw * 0.18)] = 0
    marker_red[int(fh * 0.45):int(fh * 0.75), int(fw * 0.90):] = 0

    # 1. Local check around actively targeted mob (tx, ty)
    if tx is not None and ty is not None:
        x1 = max(left, tx - radius)
        x2 = min(right, tx + radius)
        y1 = max(top, ty - radius)
        y2 = min(bot, ty + radius)
        th, tw = tpl.shape[:2]
        if x2 - x1 >= tw and y2 - y1 >= th:
            crop_red = marker_red[y1:y2, x1:x2]
            marker_center = _match_marked_outline(crop_red, tpl)
            if marker_center is not None:
                return True, (x1 + marker_center[0], y1 + marker_center[1])
        return False, None

    # 2. Global scan (current_target is None - checking if a mob is ALREADY locked on screen)
    pf_red = marker_red[top:bot, left:right]
    if cv2.countNonZero(pf_red) < 40:
        return False, None

    marker_center = _match_marked_outline(pf_red, tpl)
    if marker_center is not None:
        cx = left + marker_center[0]
        cy = top + marker_center[1]
        return True, (cx, cy)

    return False, None

# ==============================================================================
# MAIN ENGINE
# ==============================================================================

def main():
    print("=" * 65)
    print("    Rucoy Online Bot - True Background AFK Engine v4.0")
    print("=" * 65)

    # 1. Load templates
    templates = load_templates()
    template = templates.get("zombie_lv65")
    if template is None:
        template = templates.get("zombie")

    if template is None:
        print(f"[-] No template found in {TEMPLATE_DIR}!")
        return

    target_templates = [(template, "white")]
    purple_template = templates.get("purple_name_zombie_lv65")
    if purple_template is not None:
        target_templates.append((purple_template, "purple"))

    # 2. Determine mode (ADB vs Win32)
    use_adb = False
    adb_path = None
    device_serial = None
    game_hwnd = None

    if PREFER_ADB:
        adb_path, device_serial = init_adb()
        if adb_path and device_serial:
            use_adb = True
            print("[+] MODE: ADB (100% True Background - No mouse takeover!)")
        else:
            print("[!] ADB unavailable. Falling back to Win32 window mode...")

    if not use_adb:
        try:
            top_hwnd, full_title = find_window_by_title(WINDOW_TITLE)
            game_hwnd = find_game_child(top_hwnd)
            print(f"[+] Found emulator window: '{full_title}' (HWND: {game_hwnd})")
            print("[+] MODE: Win32 Window Fallback")
        except Exception as e:
            print(f"[-] {e}")
            return

    # 3. Capture test frame & calibrate template
    print("[+] Capturing initial frame...")
    test_frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)

    if test_frame is None:
        print("[-] Failed to capture initial frame! Check if emulator is running.")
        return

    # Verify template match
    gf = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    gt = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best_score = 0
    for s in [0.4, 0.5, 0.6, 0.75, 1.0, 1.25]:
        st = cv2.resize(gt, None, fx=s, fy=s) if s != 1.0 else gt
        if st.shape[0] >= gf.shape[0] or st.shape[1] >= gf.shape[1]:
            continue
        _, mx, _, _ = cv2.minMaxLoc(cv2.matchTemplate(gf, st, cv2.TM_CCOEFF_NORMED))
        if mx > best_score:
            best_score = mx

    if best_score < MATCH_THRESHOLD:
        print(f"[!] Level-specific template score is {best_score:.2f}; keeping the saved Zombie Lv.65 template.")
    else:
        print(f"[+] Template match verified (score: {best_score:.2f})")

    # State tracking
    current_target = None
    target_click_time = 0.0
    confirmed_locked = False
    last_red_seen_time = 0.0
    last_log_time = 0.0
    last_ui_back_time = 0.0
    no_target_since = time.time()
    last_minimap_time = 0.0
    frame_count = 0
    fail_counts = {}
    blacklist = {}
    last_capture_warning = 0.0
    pickup_pending = False
    ui_recovery_interval = 1.5
    failed_recovery_state = None
    failed_recovery_count = 0
    exhausted_caption_armed = True
    exhausted_caption_clear_frames = 0

    print(f"\n[+] Target: '{TARGET_MOB_NAME}'")
    print(f"[+] Method: {'ADB Background Tap (Free Mouse)' if use_adb else 'Win32 Click'}")
    print(f"[+] Background: {'YES (Can stay behind other windows)' if use_adb else 'Window visible'}")
    print(f"[+] Movement: Registered minimap waypoints ({'enabled' if ENABLE_MINIMAP_WALK else 'disabled'})")
    print(f"[+] Hitbox Offset: {MOB_BODY_Y_OFFSET}px below nametag (lowered for clean hits)")
    print(f"[+] Response Time: {TARGET_LOCK_TIMEOUT}s retry window (fast & responsive)")
    if SHOW_PREVIEW:
        print("[+] Preview: ON (Press 'q' in preview window to exit)")
    else:
        print("[+] Preview: OFF (Press Ctrl+C in terminal to exit)")
    print(f"\n[+] Hunting loop started! You can work while bot is AFK.\n")

    while True:
        # Capture frame
        frame = capture_screen(adb_path, device_serial, game_hwnd, use_adb)

        if frame is None:
            if time.time() - last_capture_warning >= 5.0:
                print("[!] Screen capture unavailable; retrying (ADB capture times out after 3s).")
                last_capture_warning = time.time()
            time.sleep(0.3)
            continue
        if last_capture_warning:
            print("[+] Screen capture recovered.")
            last_capture_warning = 0.0

        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        curr_time = time.time()
        frame_count += 1

        ui_state, _ = detect_game_ui(frame, templates)
        if ui_state == "combat":
            failed_recovery_state = None
            failed_recovery_count = 0
        if ui_state != "combat":
            if curr_time - last_ui_back_time >= (
                4.0 if ui_state == "disconnected" else ui_recovery_interval
            ):
                print(f"[!] UI state is {ui_state}; recovering and resuming automatically...")
                recovery = recover_unexpected_ui(
                    frame, adb_path, device_serial, game_hwnd, use_adb, templates
                )
                ui_recovery_interval = 5.0 if recovery == "failed" else 1.5
                if recovery == "failed":
                    if failed_recovery_state == ui_state:
                        failed_recovery_count += 1
                    else:
                        failed_recovery_state = ui_state
                        failed_recovery_count = 1
                    if failed_recovery_count >= REPEATED_OUTPUT_LIMIT:
                        raise RepeatedOutputReset(
                            f"UI recovery for '{ui_state}' failed "
                            f"{failed_recovery_count} times"
                        )
                else:
                    failed_recovery_state = None
                    failed_recovery_count = 0
                current_target = None
                confirmed_locked = False
                no_target_since = curr_time
                last_ui_back_time = curr_time
            time.sleep(0.08)
            continue

        # Periodic cleanup of expired blacklist entries
        if blacklist and frame_count % 30 == 0:
            blacklist = {pos: exp for pos, exp in blacklist.items() if curr_time < exp}

        if pickup_pending:
            pickup_position = find_pickup_available(frame, templates)
            if pickup_position is not None:
                print(f"[+] Pickup available after kill; tapping ({pickup_position[0]}, {pickup_position[1]}).")
                send_click(
                    adb_path, device_serial, game_hwnd, use_adb, *pickup_position
                )
                pickup_pending = False

        # 1. If currently attacking a target, check for red square
        caption_visible = False
        if current_target is not None or not exhausted_caption_armed:
            caption_visible = find_exhausted_caption(
                frame, templates.get("exhausted-caption")
            )
            if caption_visible:
                exhausted_caption_clear_frames = 0
            else:
                exhausted_caption_clear_frames += 1
                if exhausted_caption_clear_frames >= 2:
                    exhausted_caption_armed = True

        if current_target is not None and exhausted_caption_armed and caption_visible:
            tx, ty = current_target["click_x"], current_target["click_y"]
            grid_pos = (int(tx // 40), int(ty // 40))
            blacklist[grid_pos] = max(
                blacklist.get(grid_pos, 0.0),
                curr_time + EXHAUSTED_MOB_BLACKLIST_SECONDS,
            )
            fail_counts.pop(grid_pos, None)
            print(
                f"[~] Exhausted caption found; skipping mob at ({tx}, {ty}) "
                f"for {EXHAUSTED_MOB_BLACKLIST_SECONDS:.0f}s. Looking for another."
            )
            current_target = None
            confirmed_locked = False
            no_target_since = curr_time
            exhausted_caption_armed = False
            exhausted_caption_clear_frames = 0

        if current_target is not None:
            tx, ty = current_target["click_x"], current_target["click_y"]
            has_red, updated_pos = has_red_square(frame, tx, ty)

            if has_red:
                last_red_seen_time = curr_time
                if updated_pos is not None:
                    # Dynamically follow moving mob
                    current_target["click_x"], current_target["click_y"] = updated_pos
                    tx, ty = updated_pos

                if not confirmed_locked:
                    confirmed_locked = True
                    grid_pos = (int(tx // 40), int(ty // 40))
                    fail_counts.pop(grid_pos, None)
                    print(f"[*] Target locked with red square! Fighting at ({tx}, {ty})...")
            else:
                if confirmed_locked:
                    # Grace period: require red square to be absent for at least TARGET_DEFEATED_GRACE_TIME
                    if curr_time - last_red_seen_time >= TARGET_DEFEATED_GRACE_TIME:
                        print(f"[+] Red square gone for {curr_time - last_red_seen_time:.1f}s! Mob defeated. Searching next...")
                        current_target = None
                        confirmed_locked = False
                        no_target_since = curr_time
                        pickup_pending = True
                elif curr_time - target_click_time > TARGET_LOCK_TIMEOUT:
                    print(
                        f"[-] No target lock after {TARGET_LOCK_TIMEOUT}s. "
                        "Skipping this click location for 30.0s to avoid toggling its mark..."
                    )
                    grid_pos = (int(tx // 40), int(ty // 40))
                    blacklist[grid_pos] = curr_time + 30.0
                    fail_counts.pop(grid_pos, None)
                    current_target = None
                    confirmed_locked = False
                    no_target_since = curr_time

        # 2. If idle, search for new mob
        if current_target is None:
            # First check if a mob is ALREADY targeted/marked in the playfield!
            # In Rucoy, tapping an already-targeted mob CANCELS the attack, so never re-click it.
            has_red, active_pos = has_red_square(frame)
            active_grid = (
                (int(active_pos[0] // 40), int(active_pos[1] // 40))
                if active_pos is not None else None
            )
            if (
                has_red and active_pos is not None and
                curr_time >= blacklist.get(active_grid, 0.0)
            ):
                ax, ay = active_pos
                print(f"[*] Detected active target lock at ({ax}, {ay})! Adopting target...")
                current_target = {
                    "click_x": ax,
                    "click_y": ay,
                    "distance": np.hypot(ax - cx, ay - cy),
                    "score": 1.0,
                    "scale": 1.0
                }
                target_click_time = curr_time
                last_red_seen_time = curr_time
                confirmed_locked = True
                continue

            mobs = find_target_mobs(
                frame, target_templates, cx, cy, MATCH_THRESHOLD,
                blacklist=blacklist, curr_time=curr_time,
            )

            if mobs:
                best = mobs[0]
                bx, by = best["click_x"], best["click_y"]
                dist = int(best["distance"])
                score = best["score"]
                scale = best["scale"]

                send_click(adb_path, device_serial, game_hwnd, use_adb, bx, by)
                print(f"[+] Found {len(mobs)} '{TARGET_MOB_NAME}' (score:{score:.2f} scale:{scale:.1f}x)")
                print(f"    -> Clicking mob at ({bx}, {by}) [dist:{dist}px]")

                current_target = best
                target_click_time = curr_time
                last_red_seen_time = curr_time
                confirmed_locked = False
                no_target_since = curr_time
                time.sleep(0.04)
            else:
                if curr_time - last_log_time > 5.0:
                    print(f"[...] Scanning for '{TARGET_MOB_NAME}'... (frame #{frame_count}, res {fw}x{fh})")
                    last_log_time = curr_time

                idle_duration = curr_time - no_target_since
                if (
                    ENABLE_MINIMAP_WALK and
                    idle_duration >= MINIMAP_IDLE_DELAY and
                    curr_time - last_minimap_time >= MINIMAP_COOLDOWN
                ):
                    print(
                        f"[~] No '{TARGET_MOB_NAME}' in vision for "
                        f"{idle_duration:.1f}s -> registering minimap waypoint..."
                    )
                    travel_via_minimap(
                        adb_path, device_serial, game_hwnd, use_adb, templates
                    )
                    now = time.time()
                    last_minimap_time = now
                    no_target_since = now

        # 3. Preview window
        if SHOW_PREVIEW:
            disp = frame.copy()
            cv2.circle(disp, (cx, cy), PLAYER_DEADZONE_RADIUS, (255, 255, 0), 1)
            if current_target:
                tx, ty = current_target["click_x"], current_target["click_y"]
                color = (0, 0, 255) if confirmed_locked else (0, 255, 0)
                cv2.circle(disp, (tx, ty), 12, color, 2)
                cv2.line(disp, (cx, cy), (tx, ty), color, 1)

            scale_f = min(640 / fw, 480 / fh)
            mini = cv2.resize(disp, (int(fw * scale_f), int(fh * scale_f)))
            cv2.imshow("Rucoy Online Bot - AFK Preview", mini)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[+] Quit key pressed.")
                break
        else:
            time.sleep(0.05)

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print("[+] Bot stopped.")

def run_bot():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    repeat_guard = RepeatedOutputGuard()
    if isinstance(original_stdout, TimestampedOutputStream):
        output_stream = original_stdout
        output_stream.guard = repeat_guard
    else:
        output_stream = TimestampedOutputStream(original_stdout, repeat_guard)
    error_stream = (
        original_stderr
        if isinstance(original_stderr, TimestampedOutputStream)
        else TimestampedOutputStream(original_stderr)
    )
    sys.stdout = output_stream
    sys.stderr = error_stream
    print(f"[+] Repeated warning/status watchdog armed at {REPEATED_OUTPUT_LIMIT} matches.")
    auto_resets = 0
    try:
        while True:
            try:
                main()
                break
            except RepeatedOutputReset as error:
                auto_resets += 1
                if auto_resets > 1:
                    sys.stdout.write(
                        "[!] The same output loop returned after an automatic reset; "
                        "stopping to prevent repeated restarts.\n"
                    )
                    sys.stdout.flush()
                    break
                sys.stdout.write(
                    f"\n[!] Output pattern repeated {REPEATED_OUTPUT_LIMIT} times; "
                    f"resetting the bot engine: {error.pattern}\n"
                )
                sys.stdout.flush()
                repeat_guard.counts.clear()
            except KeyboardInterrupt:
                sys.stdout.write("\n[+] Interrupt received; bot stopped cleanly.\n")
                sys.stdout.flush()
                break
    finally:
        output_stream.guard = None
        output_stream.flush()
        error_stream.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bot()
