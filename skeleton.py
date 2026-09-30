"""Compatibility launcher and public imports for the modular bot.

Run this file as before, or use python -m program.
"""
from program.bot import RucoyBot
from program.cli import main
from program.config import BotConfig, CONFIG, TEMPLATE_DIR
from program.device import InputDevice, _find_adb_port, _find_bluestacks_adb
from program.diagnostics import RepeatedOutputGuard, TimestampedOutputStream
from program.exceptions import BotQuit, RepeatedOutputReset
from program.models import RetryContext, TargetMatch
from program.navigation import MinimapNavigator, MinimapRoute
from program.runner import run_bot
from program.state import BotState
from program.vision import FrameVision, TemplateStore


if __name__ == "__main__":
    raise SystemExit(main())
