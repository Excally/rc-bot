# Rucoy Bot

A fully autonomous AFK hunting bot for **Rucoy Online**, built with Python and OpenCV. It detects mobs by nameplate, locks targets with pixel-precise red-outline tracking, handles loot pickups, recovers from UI interruptions, and navigates farm zones using minimap template matching — all running silently in the background via ADB.

> **Zero mouse takeover.** The bot sends taps through ADB so your mouse stays free. Minimize the emulator and keep working.

---

## Features

| Feature | How it works |
|---|---|
| **Target Detection** | Finds mob nameplates via template matching on white/purple text masks |
| **Combat Lock** | Tracks the red selection outline (`marked.png`) with sub-frame persistence |
| **Combat Classes** | Supports `melee`, `ranged`, and `magic`; auto-steps closer to mobs for ranged/magic |
| **Smart Retargeting** | Nearest-first selection, retry taps on missed locks, auto-skip unresponsive targets |
| **Loot Pickup** | Detects and taps the pickup prompt after kills |
| **Minimap Navigation** | Aligns the visible minimap overlay to a full-map reference using integral-image scoring, then follows a sweep route through safe walkable cells |
| **Exhaustion Failsafe** | After repeated "exhausted" targets, jumps to the farthest visible map point to escape depleted areas |
| **UI Recovery** | Auto-dismisses disconnect screens, unexpected panels, and overlay menus |
| **Auto Potion** | Monitors HP/MP bar fill via vectorized pixel scanning and auto-taps potions with cooldown protection |
| **Output Watchdog** | Detects infinite warning loops and restarts the bot engine automatically |
| **Zone Profiles** | Switch between mob types and maps via `zone_profiles.json` |

---

## Auto-Potion & HP Detection Guide

The bot monitors character vitals in real time and automatically triggers potion taps via ADB when health or mana drops below configured thresholds.

### Detection Mechanism

- **HP Bar**: Scans the top-left red health bar (`y=4..45, x=4..460`, color `#cf3232` / BGR `50, 50, 207`)
- **MP Bar**: Scans the top-left blue mana bar (`y=50..62, x=4..460`, color `#3cbcfc` / BGR `252, 188, 60`)
- **Fill Calculation**: Uses vectorized NumPy column scanning to detect the rightmost filled column, avoiding single-pixel sampling errors caused by text or UI artifacts.

### HP Bar Text Overlay Behavior

In Rucoy Online, character current/max HP text (e.g., `475/475`) is rendered directly over the center of the HP bar, spanning columns `x ≈ 174` to `x ≈ 329` (roughly 37% to 72% of total bar width). Because the text digits replace the red bar pixels vertically across those columns, the rightmost detected red column behaves as follows:

| Threshold Setting | Effective Trigger Point | Accuracy | Notes |
|---|---|---|---|
| `0.75` - `0.95` | Exact (75% - 95%) | High | Health bar edge is to the right of text overlay. Reliable for high-risk hunting. |
| `0.70` (default) | ~72% | High | Triggers slightly early as the bar enters the text zone. Safe and conservative. |
| `0.40` - `0.65` | ~72% | Fixed | The text overlay covers the edge within this range. Any threshold in this bracket triggers at the right text boundary (~72%). |
| `0.10` - `0.35` | Exact (10% - 35%) | High | Health bar edge is to the left of text overlay. Suitable for emergency-only potion use. |

> **Note on Mana (MP):** The MP bar does **not** have numeric text overlay. Thresholds for `mp_potion_threshold` (default `0.50`) are 100% linear and precise from `0.01` to `1.00`.

### Configuration Options

Potion parameters can be adjusted in `program/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `auto_potion_enabled` | `True` | Master toggle for potion detection and usage |
| `hp_potion_threshold` | `0.70` | HP ratio below which HP potion is tapped (triggers at ~72% actual) |
| `mp_potion_threshold` | `0.50` | MP ratio below which Mana potion is tapped (exact linear scale) |
| `pvp_mode` | `False` | Toggles higher polling frequency for PvP or dangerous zones |
| `potion_check_interval`| `0.50` | Check interval in seconds during normal farming |
| `pvp_potion_check_interval` | `0.15` | Check interval in seconds when `pvp_mode` is enabled |
| `potion_cooldown` | `0.40` | Minimum delay in seconds between potion taps |
| `hp_potion_tap` | `(60, 745)` | Screen coordinates (x, y) for HP potion button |
| `mana_potion_tap` | `(60, 610)` | Screen coordinates (x, y) for Mana potion button |

---

## Quick Start

**Requirements:** Python 3.10+, BlueStacks/MSI App Player at **1600×900**, 240 DPI, 100% interface scale.

```bash
# Install dependencies
pip install -r requirements.txt

# See available profiles
python bot.py --list-profiles

# Run the bot (defaults to melee)
python bot.py --profile skeleton-lv75

# Select combat class: melee, ranged, or magic
python bot.py --profile skeleton-lv75 melee
python bot.py --profile skeleton-lv75 ranged
python bot.py --profile skeleton-lv75 magic

# Or use the package directly
python -m program --profile skeleton-lv75 ranged
```

### Combat Classes & Approach Mechanics
- **Melee (Knight)**: Target click automatically moves the character adjacent to the mob via Rucoy's native pathfinding.
- **Ranged (Archer) & Magic (Mage)**: In Rucoy, targeting a distant mob does not cause the character to walk. When running `ranged` or `magic`, the bot locks the mob and automatically steps closer by tapping halfway on the ground until adjacent or in close range (<100px), then fires.

Press `Ctrl+C` in the terminal to stop cleanly.

---

## Architecture

```
bot.py                  ← Entry point
program/
├── cli.py              ← Profile selection & CLI args
├── runner.py           ← Supervised bot restart + output watchdog
├── bot.py              ← Combat state machine (acquire → lock → track → release)
├── vision.py           ← All image recognition (stateless, never sends input)
├── navigation.py       ← Minimap alignment, routing, and travel
├── device.py           ← ADB / Win32 screen capture and tap input
├── config.py           ← Frozen runtime tuning parameters
├── state.py            ← Mutable bot state (current target, timers, flags)
├── models.py           ← Typed structures (TargetMatch)
├── profiles.py         ← Zone profile loader & validator
├── diagnostics.py      ← Timestamped output + repeated-warning detection
└── exceptions.py       ← Control-flow exceptions (BotQuit, RepeatedOutputReset)
```

Key design rules:
- **`vision.py` never sends input** — pure recognition
- **`device.py` never does CV** — pure I/O
- **`bot.py` orchestrates** — state machine driving vision + device
- **`config.py` is frozen** — immutable, separate from mutable state

---

## Minimap Templates

Templates live in `templates/` and are loaded at runtime.

**Cell-grid minimap templates** must follow these rules:
- Dimensions divisible by **21px** (each cell is a solid 21×21 block, no antialiasing)
- **Black** cells = walkable floor (route destinations)
- **White** `(243,243,243)` cells = walls (used for alignment)
- **Red** `(0,0,255)` cells = context walls from neighboring maps (alignment only, not walkable)
- **Gray** `(132,132,132)` cells = unwalkable terrain
- **Yellow** `(0,255,234)` cells = annotations (ignored by alignment and navigation)
- Must have a closed wall boundary enclosing the farm area

---

## Emulator Setup

The bot connects via **local BlueStacks ADB** first, falling back to Win32 window capture.

**Recommended BlueStacks settings:**
| Setting | Value |
|---|---|
| Display resolution | 1600 × 900 |
| DPI | 240 |
| Interface scale | 100% |
| Graphics renderer | OpenGL |
| Interface renderer | Auto |
| ASTC texture | Software |
| GPU preference | Dedicated |

---

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover target locking logic, exhaustion behavior, minimap grid validation, alignment registration, and route progress tracking — all with synthetic frames and mocked devices (no emulator needed).

---

## Releases

Release notes and version history are published on [GitHub Releases](https://github.com/Excally/rc-bot/releases).

---

## Disclaimer

Image matching and simulated tests cannot guarantee that every in-game tap will succeed. **Always observe behavior in-game before leaving the bot unattended.**
