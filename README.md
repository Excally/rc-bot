# 🤖 Rucoy Bot

A fully autonomous AFK hunting bot for **Rucoy Online**, built with Python and OpenCV. It detects mobs by nameplate, locks targets with pixel-precise red-outline tracking, handles loot pickups, recovers from UI interruptions, and navigates farm zones using minimap template matching — all running silently in the background via ADB.

> **Zero mouse takeover.** The bot sends taps through ADB so your mouse stays free. Minimize the emulator and keep working.

---

## ✨ Features

| Feature | How it works |
|---|---|
| **Target Detection** | Finds mob nameplates via template matching on white/purple text masks |
| **Combat Lock** | Tracks the red selection outline (`marked.png`) with sub-frame persistence |
| **Smart Retargeting** | Nearest-first selection, retry taps on missed locks, auto-skip unresponsive targets |
| **Loot Pickup** | Detects and taps the pickup prompt after kills |
| **Minimap Navigation** | Aligns the visible minimap overlay to a full-map reference using integral-image scoring, then follows a sweep route through safe walkable cells |
| **Exhaustion Failsafe** | After repeated "exhausted" targets, jumps to the farthest visible map point to escape depleted areas |
| **UI Recovery** | Auto-dismisses disconnect screens, unexpected panels, and overlay menus |
| **Output Watchdog** | Detects infinite warning loops and restarts the bot engine automatically |
| **Zone Profiles** | Switch between mob types and maps via `zone_profiles.json` |

---

## 🚀 Quick Start

**Requirements:** Python 3.10+, BlueStacks/MSI App Player at **1600×900**, 240 DPI, 100% interface scale.

```bash
# Install dependencies
pip install -r requirements.txt

# See available profiles
python bot.py --list-profiles

# Run the bot
python bot.py --profile skeleton-lv75
python bot.py --profile zombie-lv65

# Or use the package directly
python -m program --profile skeleton-lv75
```

Press `Ctrl+C` in the terminal to stop cleanly.

---

## 🏗️ Architecture

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

## 🗺️ Minimap Templates

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

## ⚙️ Emulator Setup

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

## 🧪 Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover target locking logic, exhaustion behavior, minimap grid validation, alignment registration, and route progress tracking — all with synthetic frames and mocked devices (no emulator needed).

---

## 📦 Releases

Release notes and version history are published on [GitHub Releases](https://github.com/Excally/rc-bot/releases).

---

## ⚠️ Disclaimer

Image matching and simulated tests cannot guarantee that every in-game tap will succeed. **Always observe behavior in-game before leaving the bot unattended.**
