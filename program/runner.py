"""Process-level setup and supervised bot restart."""
from __future__ import annotations

import sys

import cv2

from .bot import RucoyBot
from .config import CONFIG
from .diagnostics import RepeatedOutputGuard, TimestampedOutputStream
from .exceptions import RepeatedOutputReset
from .profiles import ZoneProfile, load_profile_settings

def run_bot(profile: ZoneProfile | None = None, combat_class: str | None = None) -> None:
    """Install the output watchdog and restart the engine once if it loops."""
    if profile is None:
        settings = load_profile_settings()
        profile = settings.profiles[settings.active_profile]
    chosen_class = combat_class or getattr(profile, "combat_class", "melee") or "melee"
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
                RucoyBot(profile=profile, combat_class=chosen_class).run()
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
