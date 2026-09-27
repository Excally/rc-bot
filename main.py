import os
import sys
import glob
import time
import math
import random
import shutil
import ctypes
import subprocess
import cv2
import numpy as np

# ==============================================================================
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

# Minimap Travel (Open minimap to travel when no mobs in vision)
ENABLE_MINIMAP_WALK = True
MINIMAP_IDLE_DELAY = 1.0     # Seconds with no visible mobs before opening minimap
MINIMAP_COOLDOWN = 3.5       # Minimum seconds between minimap travels (lets character walk)
MINIMAP_WALK_MIN_DIST = 160  # Min travel distance from player (pixels, avoids micro-steps)
MINIMAP_WALK_MAX_DIST = 420  # Max travel distance from player (pixels, keeps within safe zone)

# Periodic Ground Walking / Patrol (Fallback if minimap travel is disabled)
ENABLE_WANDER = False
WANDER_IDLE_DELAY = 1.0      # Seconds with no visible mobs before taking a step
WANDER_COOLDOWN = 1.6        # Minimum seconds between wander steps
WANDER_RADIUS_MIN = 160      # Min distance from player (pixels, safely outside player deadzone)
WANDER_RADIUS_MAX = 340      # Max distance from player (larger area for exploration, still 100% safe from UI)

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

# How long (seconds) to wait for red target square after clicking mob (gives time to walk/lock)
TARGET_LOCK_TIMEOUT = 2.5

# Grace period (seconds) before confirming mob is defeated when red square drops (prevents flicker drops)
TARGET_DEFEATED_GRACE_TIME = 0.8
# ==============================================================================



# ==============================================================================
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


def adb_click(adb_path, device_serial, x, y):
    """
    Send tap to Android via ADB.
    Non-blocking: DOES NOT touch the Windows mouse cursor at all!
    """
    try:
        subprocess.Popen(
            [adb_path, "-s", device_serial, "shell", "input", "tap", str(int(x)), str(int(y))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
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


# ==============================================================================
# VISION & HUNTING LOGIC
# ==============================================================================

def load_templates():
    """Load nametag templates from disk."""
    templates = {}
    if os.path.exists(TEMPLATE_DIR):
        for f in os.listdir(TEMPLATE_DIR):
            if f.lower().endswith((".png", ".jpg")):
                name = os.path.splitext(f)[0]
                img = cv2.imread(os.path.join(TEMPLATE_DIR, f))
                if img is not None:
                    templates[name] = img
    return templates


def find_target_mobs(frame, template, center_x, center_y, threshold=MATCH_THRESHOLD, blacklist=None, curr_time=0.0):
    """
    Find mobs matching the template nametag.
    Uses fast two-tier matching (1.0x primary) to minimize CPU usage and prevent hardware heat.
    Returns list of targets sorted by distance (closest to center first).
    """
    fh, fw = frame.shape[:2]
    top = int(fh * MARGIN_TOP)
    bot = int(fh * (1.0 - MARGIN_BOTTOM))
    left = int(fw * MARGIN_LEFT)
    right = int(fw * (1.0 - MARGIN_RIGHT))

    playfield = frame[top:bot, left:right]
    if playfield.size == 0:
        return []

    gray_pf = cv2.cvtColor(playfield, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    matches = []

    def scan_scale(scale):
        scaled = cv2.resize(gray_tpl, None, fx=scale, fy=scale) if scale != 1.0 else gray_tpl
        th, tw = scaled.shape[:2]
        if th >= gray_pf.shape[0] or tw >= gray_pf.shape[1] or th < 3 or tw < 3:
            return

        res = cv2.matchTemplate(gray_pf, scaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            px, py = pt
            x = px + left
            y = py + top

            # Dedup close matches
            if any(abs(x - m["nx"]) < 30 and abs(y - m["ny"]) < 15 for m in matches):
                continue

            click_x = x + tw // 2
            # Lower the click further into the mob body/feet (hitbox)
            click_y = y + th + int(MOB_BODY_Y_OFFSET * scale)

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
                    "nx": x, "ny": y, "nw": tw, "nh": th,
                    "click_x": click_x, "click_y": click_y,
                    "distance": dist, "score": float(res[py, px]),
                    "scale": scale
                })

    # Tier 1: Check native 1.0x scale first (instant & covers ~99% of matches on ADB)
    scan_scale(1.0)

    # Tier 2: Only test fallback scales if no mobs found at 1.0x (saves ~85% CPU power)
    if not matches:
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
        if x2 - x1 >= 72 and y2 - y1 >= 72:
            crop_red = marker_red[y1:y2, x1:x2]
            res = cv2.matchTemplate(crop_red, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v >= 0.28:
                cx = x1 + max_l[0] + 36
                cy = y1 + max_l[1] + 36
                return True, (cx, cy)

            cnt = cv2.countNonZero(crop_red)
            if cnt >= 35:
                ys, xs = np.where(crop_red > 0)
                cx = x1 + int(np.mean(xs))
                cy = y1 + int(np.mean(ys))
                return True, (cx, cy)
        return False, None

    # 2. Global scan (current_target is None - checking if a mob is ALREADY locked on screen)
    pf_red = marker_red[top:bot, left:right]
    if cv2.countNonZero(pf_red) < 40:
        return False, None

    res = cv2.matchTemplate(pf_red, tpl, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
    # Require strong correlation with templates/marked.png (rejects solid red mobs and noise)
    if max_v >= 0.35:
        cx = left + max_l[0] + 36
        cy = top + max_l[1] + 36
        return True, (cx, cy)

    return False, None


def auto_extract_template(frame):
    """Extract a fresh nametag template from the current frame if needed."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh, fw = frame.shape[:2]
    candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > h * 2 and 40 < w < 400 and 8 < h < 50 and y > fh * 0.10:
            candidates.append((x, y, w, h))

    if candidates:
        candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
        x, y, w, h = candidates[0]
        return frame[y:y+h, x:x+w]
    return None


def find_player_on_minimap(map_frame):
    """
    Detect player's position marker on the open minimap.
    In Rucoy Online, the player pin is a bright cyan/blue marker
    (~[252, 188, 60] BGR or HSV H:90..118, S:80..255, V:80..255).
    Falls back to screen center (800, 450) if marker is obscured.
    """
    fh, fw = map_frame.shape[:2]
    cx, cy = fw // 2, fh // 2

    hsv = cv2.cvtColor(map_frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 118) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 80)

    # Exclude UI headers/borders
    top_ui = int(90 * (fh / 900.0))
    bot_ui = int(810 * (fh / 900.0))
    left_ui = int(120 * (fw / 1600.0))
    right_ui = int(1480 * (fw / 1600.0))
    mask[:top_ui, :] = False
    mask[bot_ui:, :] = False
    mask[:, :left_ui] = False
    mask[:, right_ui:] = False

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best_dist = 999999
    best_pt = (cx, cy)
    found = False
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            c_x, c_y = centroids[i]
            d = (c_x - cx) ** 2 + (c_y - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_pt = (int(round(c_x)), int(round(c_y)))
                found = True
    return best_pt, found


def get_zombie_zone_walkable(map_frame, layout_img):
    """
    Align minimap-zombie-layout.png with open minimap frame.
    The Rucoy minimap centers on the player, so the layout can appear at any
    vertical/horizontal position depending on where the player is in the dungeon.
    Uses multi-crop sliding window matching to handle all player positions.

    Returns:
      valid_mask: binary mask of walkable corridor pixels strictly inside the zombie farm
      zone_info: dict with offset, scale, bbox, and match score
    """
    fh, fw = map_frame.shape[:2]
    res_scale = fw / 1600.0
    base_scale = 6.944444 * res_scale

    # 1. Screen corridor mask (~[243, 243, 243])
    diff = np.abs(map_frame.astype(np.int16) - np.array([243, 243, 243]))
    fc = ((np.max(diff, axis=-1) <= 8) & (map_frame[:, :, 0] > 200)).astype(np.uint8) * 255
    top_ui = int(90 * (fh / 900.0))
    bot_ui = int(810 * (fh / 900.0))
    left_ui = int(120 * res_scale)
    right_ui = int(1480 * res_scale)
    fc[:top_ui, :] = 0
    fc[bot_ui:, :] = 0
    fc[:, :left_ui] = 0
    fc[:, right_ui:] = 0

    if layout_img is None:
        # Fallback if no layout template provided
        return fc, {"offset": (0, 0), "scale": 1.0, "score": 0.0}

    # 2. Layout corridors
    diff_l = np.abs(layout_img.astype(np.int16) - np.array([243, 243, 243]))
    lc = ((np.max(diff_l, axis=-1) <= 6) & (layout_img[:, :, 0] > 200)).astype(np.uint8)

    # Safety clipping on layout edges:
    # Only mask actual exit tunnels (south exit tunnel at rows 144..147, cols 60..80;
    # and northeast connector at rows 0..8, cols 137+)
    # Preserves the bottom main road (rows 140..143) and east road (cols 134..137) so the bot
    # never gets trapped in corners or treats farm edges as out-of-bounds.
    lc_safe = lc.copy()
    if lc_safe.shape[0] >= 144:
        lc_safe[144:, 60:80] = 0  # South exit tunnel leading out
    if lc_safe.shape[1] >= 137:
        lc_safe[:8, 137:] = 0    # Northeast connection leading out

    best_score = 0
    best_origin = (int(81 * res_scale), int(148 * (fh / 900.0)))  # fallback default
    best_scale = base_scale

    # 3. Multi-scale, multi-crop template matching
    #    The minimap scrolls with the player, so the layout can be at any vertical
    #    position on screen. We slide a matching window down the full scaled layout
    #    to find the best alignment regardless of where the player is in the zone.
    min_crop_h = int(300 * (fh / 900.0))   # minimum window height for reliable match
    max_crop_h = int(650 * (fh / 900.0))   # max window height (fits within screen)
    crop_step = int(80 * (fh / 900.0))     # step between vertical window positions

    for s in [base_scale, base_scale * 0.99, base_scale * 1.01]:
        sw = int(round(layout_img.shape[1] * s))
        sh = int(round(layout_img.shape[0] * s))
        if sw <= 0 or sh <= 0:
            continue
        scaled_lc = cv2.resize(lc_safe, (sw, sh), interpolation=cv2.INTER_NEAREST) * 255

        # Slide vertical window across the full scaled layout
        for crop_start in range(0, max(1, sh - min_crop_h + 1), crop_step):
            crop_end = min(crop_start + max_crop_h, sh)
            if crop_end - crop_start < min_crop_h:
                continue

            tpl = scaled_lc[crop_start:crop_end, :]
            if tpl.shape[0] >= fc.shape[0] or tpl.shape[1] >= fc.shape[1]:
                continue

            res = cv2.matchTemplate(fc, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)

            if max_v > best_score:
                best_score = max_v
                # max_l is screen position where crop_start row of layout appears
                # Layout origin (0,0) is at: (max_l[0], max_l[1] - crop_start)
                best_origin = (max_l[0], max_l[1] - crop_start)
                best_scale = s

    ox, oy = best_origin
    sw = int(round(layout_img.shape[1] * best_scale))
    sh = int(round(layout_img.shape[0] * best_scale))
    scaled_safe = cv2.resize(lc_safe, (sw, sh), interpolation=cv2.INTER_NEAREST)

    safe_mask = np.zeros((fh, fw), dtype=np.uint8)
    # Handle negative offsets (layout extends beyond screen top/left when player is deep in zone)
    src_x1, src_y1 = max(0, -ox), max(0, -oy)
    dst_x1, dst_y1 = max(0, ox), max(0, oy)
    src_x2 = min(sw, fw - ox)
    src_y2 = min(sh, fh - oy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        safe_mask[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_safe[src_y1:src_y2, src_x1:src_x2]

    # Intersection: pixels that are white in the game frame AND part of the zombie layout
    valid_mask = ((fc > 0) & (safe_mask > 0)).astype(np.uint8)

    zone_info = {
        "offset": (ox, oy),
        "size": (sw, sh),
        "scale": best_scale,
        "score": best_score,
        "bbox": (dst_x1, dst_y1, dst_x2 - dst_x1, dst_y2 - dst_y1)
    }
    return valid_mask, zone_info


def travel_via_minimap(adb_path, device_serial, game_hwnd=None, use_adb=True, templates=None):
    """
    Open minimap, recognize player location and zombie layout to prevent leaving the farm zone,
    tap an optimal safe corridor destination, and close the minimap.
    """
    minimap_icon = templates.get("minimapicon") if templates else None
    back_icon = templates.get("backicon") if templates else None
    layout_img = templates.get("minimap-zombie-layout") if templates else None
    if layout_img is None:
        layout_path = os.path.join(TEMPLATE_DIR, "minimap-zombie-layout.png")
        if os.path.exists(layout_path):
            layout_img = cv2.imread(layout_path)

    # Step 1: Click minimap button (default ~1150, 50 or dynamic template match)
    open_x, open_y = 1150, 50
    if use_adb:
        adb_click(adb_path, device_serial, open_x, open_y)
    else:
        win32_click(game_hwnd, open_x, open_y)
    time.sleep(0.55)

    # Step 2: Capture open minimap frame
    if use_adb:
        map_frame = adb_capture(adb_path, device_serial)
    else:
        map_frame = win32_capture(game_hwnd)

    if map_frame is None:
        return False

    fh, fw = map_frame.shape[:2]

    # Dynamic back icon location (default ~1546, 44)
    back_x, back_y = int(1546 * (fw / 1600.0)), int(44 * (fh / 900.0))
    if back_icon is not None:
        res_back = cv2.matchTemplate(map_frame, back_icon, cv2.TM_CCOEFF_NORMED)
        _, bv, _, bl = cv2.minMaxLoc(res_back)
        if bv > 0.65:
            back_x = bl[0] + back_icon.shape[1] // 2
            back_y = bl[1] + back_icon.shape[0] // 2

    # Step 3: Recognize player position and zombie farm layout
    (player_x, player_y), player_found = find_player_on_minimap(map_frame)
    valid_mask, zone_info = get_zombie_zone_walkable(map_frame, layout_img)

    # Detect if character is stuck at same minimap spot
    hist = getattr(travel_via_minimap, '_history', {'last_pos': None, 'stuck_count': 0})
    stuck_detected = False
    if hist.get('last_pos') is not None and player_found:
        dist_from_last = np.hypot(player_x - hist['last_pos'][0], player_y - hist['last_pos'][1])
        if dist_from_last < 25:
            hist['stuck_count'] = hist.get('stuck_count', 0) + 1
            if hist['stuck_count'] >= 1:
                stuck_detected = True
        else:
            hist['stuck_count'] = 0
    hist['last_pos'] = (player_x, player_y)
    travel_via_minimap._history = hist

    # Screen corridor connectivity check (ensure candidate destinations are physically reachable)
    diff = np.abs(map_frame.astype(np.int16) - np.array([243, 243, 243]))
    fc = ((np.max(diff, axis=-1) <= 8) & (map_frame[:, :, 0] > 200)).astype(np.uint8) * 255
    top_ui = int(90 * (fh / 900.0))
    bot_ui = int(810 * (fh / 900.0))
    left_ui = int(120 * (fw / 1600.0))
    right_ui = int(1480 * (fw / 1600.0))
    fc[:top_ui, :] = 0
    fc[bot_ui:, :] = 0
    fc[:, :left_ui] = 0
    fc[:, right_ui:] = 0

    kernel = np.ones((7, 7), np.uint8)
    fc_connected = cv2.dilate(fc, kernel)
    ys_fc, xs_fc = np.where(fc > 0)

    reachable_mask = valid_mask.copy()
    if len(xs_fc) > 0:
        d_p = np.hypot(xs_fc - player_x, ys_fc - player_y)
        near_idx = np.argmin(d_p)
        _, labels = cv2.connectedComponents(fc_connected)
        player_label = labels[ys_fc[near_idx], xs_fc[near_idx]]
        connected_corridors = ((labels == player_label) & (valid_mask > 0)).astype(np.uint8)
        if np.sum(connected_corridors > 0) >= 50:
            reachable_mask = connected_corridors

    ys, xs = np.where(reachable_mask > 0)
    if len(xs) == 0:
        ys, xs = np.where(valid_mask > 0)

    if len(xs) > 0:
        # Distance from player to all safe zombie corridor pixels
        dists = np.sqrt((xs - player_x) ** 2 + (ys - player_y) ** 2)
        min_dist_to_zone = np.min(dists)

        # Center of zombie farm zone in screen coordinates
        ox, oy = zone_info["offset"]
        sw, sh = zone_info["size"]
        zone_cx = ox + sw // 2
        zone_cy = oy + sh // 2

        # Outside threshold: ~80px away from any valid zombie farm corridor
        outside_threshold = 80 * (fw / 1600.0)
        is_outside = min_dist_to_zone > outside_threshold

        if is_outside:
            # Player wandered outside the zombie farm (e.g. skeleton area or exit)!
            # Return immediately to the closest safe corridor in the zombie zone
            print(f"[!] Minimap: Player is OUTSIDE Zombie Farm zone at ({player_x}, {player_y}) [dist: {min_dist_to_zone:.1f}px]!")
            print(f"    -> Guiding character back to Zombie Farm safe corridor...")
            candidates = np.where(dists <= min_dist_to_zone + 60 * (fw / 1600.0))[0]
            chosen = random.choice(candidates)
        elif stuck_detected:
            # Player is stuck at a corner or obstacle! Pick a direct reachable step inward toward farm center
            print(f"[!] Minimap: Stuck at corner ({player_x}, {player_y})! Forcing escape route toward farm center...")
            min_walk = 120 * (fw / 1600.0)
            max_walk = 250 * (fw / 1600.0)
            in_range = np.where((dists >= min_walk) & (dists <= max_walk))[0]
            if len(in_range) > 0:
                dist_to_center = np.hypot(xs[in_range] - zone_cx, ys[in_range] - zone_cy)
                chosen = in_range[np.argmin(dist_to_center)]
            else:
                chosen = np.argmin(dists)
        else:
            # Player is safely inside the zombie farm!
            # Pick a destination 150px - 380px away to explore / patrol corridors
            min_walk = 150 * (fw / 1600.0)
            max_walk = 380 * (fw / 1600.0)
            in_range = np.where((dists >= min_walk) & (dists <= max_walk))[0]
            if len(in_range) > 0:
                # If player is near the edge/corner (>300px from farm center), bias toward center
                player_dist_to_center = np.hypot(player_x - zone_cx, player_y - zone_cy)
                if player_dist_to_center > 280:
                    cand_dists_to_center = np.hypot(xs[in_range] - zone_cx, ys[in_range] - zone_cy)
                    # Pick from the top 30% of candidates that pull inward toward farm center
                    sorted_indices = np.argsort(cand_dists_to_center)
                    top_pool = sorted_indices[:max(1, len(sorted_indices) // 3)]
                    chosen = in_range[random.choice(top_pool)]
                else:
                    chosen = random.choice(in_range)
            else:
                chosen = np.argmin(np.abs(dists - 240 * (fw / 1600.0)))

        dest_x = int(xs[chosen])
        dest_y = int(ys[chosen])
        step_dist = int(np.hypot(dest_x - player_x, dest_y - player_y))
        status_desc = "Returning to Farm" if is_outside else ("Escaping Corner" if stuck_detected else "Patrolling Farm")
        p_tag = f"({player_x}, {player_y})" if player_found else f"({player_x}, {player_y}) [center fallback]"
        print(f"[~] Minimap: {status_desc} -> Tapped ({dest_x}, {dest_y}) [player: {p_tag}, dist: {step_dist}px, zone: {zone_info['score']:.2f}, safe: {len(xs)}]...")

        if use_adb:
            adb_click(adb_path, device_serial, dest_x, dest_y)
        else:
            win32_click(game_hwnd, dest_x, dest_y)
        time.sleep(0.25)
    else:
        # Fallback if no layout corridor detected: click safe area near screen center
        rx = int(fw // 2 + random.randint(-120, 120))
        ry = int(fh // 2 + random.randint(-80, 80))
        print(f"[!] Minimap: No zombie corridors detected! Tapped fallback center ({rx}, {ry})...")
        if use_adb:
            adb_click(adb_path, device_serial, rx, ry)
        else:
            win32_click(game_hwnd, rx, ry)
        time.sleep(0.25)

    # Step 4: Close minimap via back icon
    if use_adb:
        adb_click(adb_path, device_serial, back_x, back_y)
    else:
        win32_click(game_hwnd, back_x, back_y)
    time.sleep(0.4)
    return True



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
    if template is None and templates:
        template = next(iter(templates.values()))

    if template is None:
        print(f"[-] No template found in {TEMPLATE_DIR}!")
        return

    # Check Zombie Zone layout template for minimap travel
    zombie_layout = templates.get("minimap-zombie-layout")
    if zombie_layout is None:
        layout_f = os.path.join(TEMPLATE_DIR, "minimap-zombie-layout.png")
        if os.path.exists(layout_f):
            zombie_layout = cv2.imread(layout_f)
            templates["minimap-zombie-layout"] = zombie_layout

    if zombie_layout is not None:
        print(f"[+] Loaded Zombie Farm layout: templates/minimap-zombie-layout.png ({zombie_layout.shape[1]}x{zombie_layout.shape[0]})")
    else:
        print("[!] Warning: templates/minimap-zombie-layout.png not found! Minimap safety bounds disabled.")

    # 2. Determine mode (ADB vs Win32)
    use_adb = False
    adb_path = None
    device_serial = None
    game_hwnd = None
    match_threshold = MATCH_THRESHOLD

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
    if use_adb:
        test_frame = adb_capture(adb_path, device_serial)
    else:
        test_frame = win32_capture(game_hwnd)

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

    if best_score < match_threshold:
        print(f"[!] Initial template score is {best_score:.2f}. Auto-extracting fresh template...")
        fresh = auto_extract_template(test_frame)
        if fresh is not None:
            template = fresh
            cv2.imwrite(os.path.join(TEMPLATE_DIR, "zombie_lv65.png"), fresh)
            print(f"[+] Fresh template extracted and saved ({fresh.shape})")
        else:
            match_threshold = max(0.42, best_score - 0.05)
            print(f"[+] Adjusted match threshold to {match_threshold:.2f}")
    else:
        print(f"[+] Template match verified (score: {best_score:.2f})")

    # State tracking
    current_target = None
    target_click_time = 0.0
    confirmed_locked = False
    last_red_seen_time = 0.0
    last_log_time = 0.0
    last_target_seen_time = time.time()
    last_wander_time = 0.0
    last_minimap_time = 0.0
    patrol_heading = None       # Directional momentum (prevents back-and-forth ping pong)
    patrol_step_count = 0
    frame_count = 0
    fail_counts = {}
    blacklist = {}

    print(f"\n[+] Target: '{TARGET_MOB_NAME}'")
    print(f"[+] Method: {'ADB Background Tap (Free Mouse)' if use_adb else 'Win32 Click'}")
    print(f"[+] Background: {'YES (Can stay behind other windows)' if use_adb else 'Window visible'}")
    print(f"[+] Movement: {'Minimap Auto-Travel (Zombie Farm Boundary Locked)' if ENABLE_MINIMAP_WALK else 'Ground Patrol'}")
    print(f"[+] Hitbox Offset: {MOB_BODY_Y_OFFSET}px below nametag (lowered for clean hits)")
    print(f"[+] Response Time: {TARGET_LOCK_TIMEOUT}s retry window (fast & responsive)")
    if SHOW_PREVIEW:
        print("[+] Preview: ON (Press 'q' in preview window to exit)")
    else:
        print("[+] Preview: OFF (Press Ctrl+C in terminal to exit)")
    print(f"\n[+] Hunting loop started! You can work while bot is AFK.\n")

    while True:
        # Capture frame
        if use_adb:
            frame = adb_capture(adb_path, device_serial)
        else:
            frame = win32_capture(game_hwnd)

        if frame is None:
            time.sleep(0.3)
            continue

        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        curr_time = time.time()
        frame_count += 1

        # Periodic cleanup of expired blacklist entries
        if blacklist and frame_count % 30 == 0:
            blacklist = {pos: exp for pos, exp in blacklist.items() if curr_time < exp}

        # 1. If currently attacking a target, check for red square
        if current_target is not None:
            last_target_seen_time = curr_time
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
                        last_target_seen_time = curr_time
                elif curr_time - target_click_time > TARGET_LOCK_TIMEOUT:
                    print(f"[-] No target lock after {TARGET_LOCK_TIMEOUT}s. Retrying...")
                    grid_pos = (int(tx // 40), int(ty // 40))
                    fail_counts[grid_pos] = fail_counts.get(grid_pos, 0) + 1
                    if fail_counts[grid_pos] >= 2:
                        print(f"[!] Target at ({tx}, {ty}) unreachable or missed twice. Skipping for 4.0s...")
                        blacklist[grid_pos] = curr_time + 4.0
                        fail_counts[grid_pos] = 0
                    current_target = None
                    confirmed_locked = False
                    last_target_seen_time = curr_time

        # 2. If idle, search for new mob
        if current_target is None:
            # First check if a mob is ALREADY targeted/marked in the playfield!
            # In Rucoy, tapping an already-targeted mob CANCELS the attack, so never re-click it.
            has_red, active_pos = has_red_square(frame)
            if has_red and active_pos is not None:
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

            mobs = find_target_mobs(frame, template, cx, cy, match_threshold, blacklist=blacklist, curr_time=curr_time)

            if mobs:
                last_target_seen_time = curr_time
                patrol_heading = None
                patrol_step_count = 0
                best = mobs[0]
                bx, by = best["click_x"], best["click_y"]
                dist = int(best["distance"])
                score = best["score"]
                scale = best["scale"]

                print(f"[+] Found {len(mobs)} '{TARGET_MOB_NAME}' (score:{score:.2f} scale:{scale:.1f}x)")
                print(f"    -> Clicking mob at ({bx}, {by}) [dist:{dist}px]")

                if use_adb:
                    adb_click(adb_path, device_serial, bx, by)
                else:
                    win32_click(game_hwnd, bx, by)

                current_target = best
                target_click_time = curr_time
                last_red_seen_time = curr_time
                confirmed_locked = False
                time.sleep(0.12)
            else:
                if curr_time - last_log_time > 5.0:
                    print(f"[...] Scanning for '{TARGET_MOB_NAME}'... (frame #{frame_count}, res {fw}x{fh})")
                    last_log_time = curr_time

                # Movement when no targets in vision:
                idle_duration = curr_time - last_target_seen_time

                # 1. Primary: Minimap Auto-Travel (Open minimap, tap corridor destination, close)
                if ENABLE_MINIMAP_WALK and (idle_duration >= MINIMAP_IDLE_DELAY) and (curr_time - last_minimap_time >= MINIMAP_COOLDOWN):
                    print(f"[~] No zombies in vision for {idle_duration:.1f}s -> Opening minimap to travel...")
                    travel_via_minimap(adb_path, device_serial, game_hwnd=game_hwnd, use_adb=use_adb, templates=templates)
                    curr_time = time.time()
                    last_minimap_time = curr_time
                    last_target_seen_time = curr_time

                # 2. Fallback: Periodic ground walking / patrol
                elif ENABLE_WANDER and (idle_duration >= WANDER_IDLE_DELAY) and (curr_time - last_wander_time >= WANDER_COOLDOWN):
                    # Pick an initial heading on first wander step, or maintain general direction
                    if patrol_heading is None:
                        patrol_heading = random.uniform(0, 2 * math.pi)
                        patrol_step_count = 0

                    # Keep general heading with slight natural variation (+/- 20 degrees)
                    angle = patrol_heading + random.uniform(-0.35, 0.35)
                    r = random.randint(WANDER_RADIUS_MIN, WANDER_RADIUS_MAX)

                    # 16:9 vertical scale: 0.65 keeps vertical steps proportional and far from top/bottom UI
                    wx = int(cx + r * math.cos(angle))
                    wy = int(cy + (r * 0.65) * math.sin(angle))

                    # Strict inner playfield bounds (keeps at least 380px from left/right and 180px from top/bottom)
                    max_x_offset = min(WANDER_RADIUS_MAX, 400)
                    max_y_offset = int(max_x_offset * 0.65)
                    clamped_x = max(cx - max_x_offset, min(cx + max_x_offset, wx))
                    clamped_y = max(cy - max_y_offset, min(cy + max_y_offset, wy))

                    # If we reached the boundary of the safe zone, rotate heading smoothly away
                    if clamped_x != wx or clamped_y != wy:
                        patrol_heading = (patrol_heading + random.choice([1.2, -1.2])) % (2 * math.pi)
                        wx, wy = clamped_x, clamped_y
                    else:
                        wx, wy = clamped_x, clamped_y

                    patrol_step_count += 1
                    # After 3-4 consecutive steps forward in this direction, curve towards a new area
                    if patrol_step_count >= 3:
                        patrol_heading = (patrol_heading + random.uniform(0.8, 1.4)) % (2 * math.pi)
                        patrol_step_count = 0

                    heading_deg = int(math.degrees(patrol_heading))
                    print(f"[~] No target in vision for {idle_duration:.1f}s -> Patrolling to ({wx}, {wy}) [heading: {heading_deg}°]...")
                    if use_adb:
                        adb_click(adb_path, device_serial, wx, wy)
                    else:
                        win32_click(game_hwnd, wx, wy)

                    last_wander_time = curr_time
                    time.sleep(0.2)

        # 3. Preview window
        if SHOW_PREVIEW:
            disp = frame.copy()
            cv2.circle(disp, (cx, cy), PLAYER_DEADZONE_RADIUS, (255, 255, 0), 1)
            if ENABLE_WANDER:
                cv2.ellipse(disp, (cx, cy), (WANDER_RADIUS_MAX, int(WANDER_RADIUS_MAX * 0.65)), 0, 0, 360, (60, 60, 60), 1)
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


if __name__ == "__main__":
    main()