# Rucoy Bot

Rucoy Bot is a Python automation project for hunting configured mobs in Rucoy Online. It finds and tracks targets from the live screen, handles pickup prompts and unexpected UI panels, and can travel within a farm zone by matching the visible part of the in-game minimap to a full-map template. Minimap routes stay inside the template's marked farm area and check whether movement actually occurred before continuing.

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

The active device connector first tries local BlueStacks ADB and falls back to the MSI App Player window backend with the display resolutin around 1600x900 with 240 DPI and interface seting 100%. with Graphic renderer OpenGL, interface renderer Auto, ASTC texture by software, and prefer dedicated GPU. The older Android Wireless Debugging runner is kept at `deprecated/mobile.py`; it is not connected to the active launcher, so `python bot.py` does not currently connect to a phone over Wi-Fi ADB.

Files in `templates/` are visual references used at runtime. Cell-grid minimap templates must have dimensions divisible by 21, use solid 21×21 pixel blocks without antialiasing, and include a closed white outer wall boundary. White cells mark walls, red cells mark outside space, yellow cells are annotations ignored by registration and player detection, and other colors mark floor. `screenshot-stock/` holds saved screenshots and diagnostic captures, not runtime templates. Older implementations are retained under `deprecated/` for reference.

The active runner also has a repeated-output watchdog: after the same warning pattern reaches its configured threshold, it restarts the bot engine once. If the same pattern returns after that reset, the runner stops the program rather than restarting endlessly. Output lines include local date and time. Image matching and simulated tests cannot guarantee that every live-game tap will succeed, so observe behavior in-game before leaving it unattended.

## Releases

Release notes and version history are published on [GitHub Releases](https://github.com/Excally/rc-bot/releases). The README stays focused on what the project does and how to run it.
