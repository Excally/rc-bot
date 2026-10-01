# Rucoy Bot

Rucoy Bot is a Python screen-recognition and input project for automating target hunting in Rucoy Online. It scans the visible game frame for a configured mob nameplate, attempts to select and track that target, responds to selected in-game UI states, and can navigate with the minimap when a partial live view can be registered against a saved full-map reference.

## Project Overview

The active implementation is the `program/` Python package. It is split by responsibility so that target recognition, screen input, minimap navigation, configuration, and the main combat loop can be worked on separately instead of being tangled in one large script.

- `program/bot.py` owns the hunting loop, target acquisition, target tracking, pickup handling, and unexpected-UI recovery.
- `program/vision.py` loads image references and detects mob nameplates, target outlines, pickup prompts, UI icons, and exhausted captions.
- `program/navigation.py` reads minimap templates as a 21-pixel cell grid, aligns visible white wall cells, excludes red outside cells, and routes only through the enclosed safe area.
- `program/device.py` captures frames and sends taps or Back input through the supported emulator/window backends.
- `program/profiles.py` validates the zone profiles in `zone_profiles.json`; `program/config.py` contains shared runtime tuning.
- `program/state.py`, `program/models.py`, and `program/exceptions.py` contain state and small types used by the bot.
- `program/diagnostics.py` timestamps output and watches for repeated warning patterns; `program/runner.py` starts the selected profile and supervises the bot loop.
- `program/cli.py` implements profile selection and inspection commands.

`bot.py` is the main launcher. `skeleton.py` is a compatibility launcher for the same profile-driven program; the selected profile, not the launcher's filename, controls the target and map. The default profile is `skeleton-lv75`. `zombie-lv65` is also configured.

Run from the repository folder with Python 3.10 or newer. Install dependencies with `python -m pip install -r requirements.txt`, then use:

```powershell
python bot.py --list-profiles
python bot.py --show-config
python bot.py --profile skeleton-lv75
python bot.py --profile zombie-lv65
```

The current device connector first tries local BlueStacks ADB and falls back to the MSI App Player window backend. The older standalone Android Wireless Debugging phone runner is kept at `deprecated/mobile.py`; it is not connected to the active profile-based launcher, so `python bot.py` does not currently connect to a phone over Wi-Fi ADB.

Files in `templates/` are visual references used at runtime. Cell-grid minimap templates must have dimensions divisible by 21, use solid 21×21 pixel blocks without antialiasing, and include a closed white outer wall boundary. White cells mark walls, red cells mark outside space, yellow cells are annotations ignored by registration and player detection, and other colors mark floor. `screenshot-stock/` holds saved screenshots and diagnostic captures, not runtime templates. Older implementations are retained under `deprecated/` for reference.

The active runner also has a repeated-output watchdog: after the same warning pattern reaches its configured threshold, it restarts the bot engine once. If the same pattern returns after that reset, the runner stops the program rather than restarting endlessly. Output lines include local date and time. Image matching and simulated tests cannot guarantee that every live-game tap will succeed, so observe behavior in-game before leaving it unattended.

## Changelog

- **2026-10-01 - Cell-grid minimap navigation and target reacquisition:** changed the Skeleton map template to an exact 21-pixel cell grid and register partial game views by visible white wall cells; red outside cells are excluded from safe routes, yellow annotation cells are ignored, and malformed/non-uniform map cells are rejected. Added unit tests for map parsing, partial alignment with occlusion, invalid geometry, player-marker color filtering, and target scanning near the player. Added local nameplate reacquisition for locked targets and release of stale locks after the configured nameplate-loss timeout. Added the editable Aseprite sources and fresh diagnostic screenshots.
- **e6ece65 - Refactor bot into profile-based modular app:** moved the active implementation into the `program/` package, added the `bot.py` and `skeleton.py` launchers, and introduced `zone_profiles.json` with Skeleton Lv.75 and Zombie Lv.65 profiles. Organized screenshots under `screenshot-stock/`, kept older scripts under `deprecated/`, and added the Skeleton target and minimap reference images.
- **5d729e2 - Improve minimap recovery and refresh templates:** updated minimap and Back-icon matching to respect transparent template pixels, tightened UI recognition, and retried a Back tap using a refreshed icon match before falling back to the system Back action. Refreshed the Back, minimap, and pickup reference images.
- **13154ba - Add mobile ADB runner and improve bot targeting:** added a separate Android Wireless Debugging launcher with connect-port discovery and pairing support, and improved target-lock recognition and response handling in the earlier bot implementation. The phone launcher is now historical code under `deprecated/`.
- **fa46b61 - Initial project upload:** published the original Rucoy bot project to this repository.
