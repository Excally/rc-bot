import inspect
import unittest
from unittest.mock import patch

import numpy as np

from program.bot import RucoyBot
from program.config import BotConfig
from program.navigation import MinimapNavigator
from program.state import BotState
from program.vision import FrameVision


class _AlwaysExhaustedVision:
    def exhausted_caption(self, frame, template):
        return True


class _ClearedCaptionVision:
    def exhausted_caption(self, frame, template):
        return False


class _Templates:
    def get(self, key):
        return None


class _TravelRecorder:
    def __init__(self, result=True):
        self.calls = []
        self.result = result

    def travel(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _MapTemplates:
    def __init__(self, reference):
        self.reference = reference

    def get(self, key):
        return self.reference if key == "minimap" else None


class _MapVision:
    def ui_state(self, frame):
        return "minimap", (1546, 44)


class ExhaustionBehaviorTests(unittest.TestCase):
    def _bot(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.state = BotState()
        bot.vision = _AlwaysExhaustedVision()
        bot.templates = _Templates()
        bot.navigator = _TravelRecorder()
        return bot

    def _confirm_exhausted(self, bot, now):
        target = {"click_x": 700, "click_y": 450}
        bot.state.current_target = target
        bot.state.exhausted_caption_target = target
        bot.state.exhausted_caption_check_at = now
        bot._check_exhausted_caption(np.zeros((900, 1600, 3), dtype=np.uint8), now)

    def test_exhausted_targets_trigger_farthest_travel_with_ten_second_hold(self):
        bot = self._bot()

        self._confirm_exhausted(bot, 10.0)
        self._confirm_exhausted(bot, 69.0)

        self.assertEqual(bot.state.consecutive_exhausted, 2)
        self.assertTrue(bot.state.exhausted_travel_pending)
        self.assertFalse(hasattr(bot.state, "blacklist"))
        self.assertFalse(hasattr(bot.state, "ignored_exhausted_marker"))

        bot._run_exhausted_travel_failsafe(69.0)

        self.assertEqual(
            bot.navigator.calls,
            [{"farthest": True, "keep_open_seconds": 10.0}],
        )
        self.assertEqual(bot.state.consecutive_exhausted, 0)
        self.assertIsNone(bot.state.exhausted_window_started_at)

    def test_exhaustion_streak_resets_when_events_are_more_than_one_minute_apart(self):
        bot = self._bot()

        self._confirm_exhausted(bot, 10.0)
        self._confirm_exhausted(bot, 70.1)

        self.assertEqual(bot.state.consecutive_exhausted, 1)
        self.assertFalse(bot.state.exhausted_travel_pending)
        self.assertEqual(bot.state.exhausted_window_started_at, 70.1)

    def test_cleared_caption_releases_pending_unconfirmed_target(self):
        bot = self._bot()
        bot.vision = _ClearedCaptionVision()
        target = {"click_x": 800, "click_y": 354}
        bot.state.current_target = target
        bot.state.exhausted_caption_target = target
        bot.state.exhausted_caption_check_at = 25.0
        bot.state.consecutive_exhausted = 1
        bot.state.exhausted_window_started_at = 10.0

        bot._check_exhausted_caption(np.zeros((900, 1600, 3), dtype=np.uint8), 25.0)

        self.assertIsNone(bot.state.current_target)
        self.assertFalse(bot.state.confirmed_locked)
        self.assertTrue(bot.state.idle_mark_check_needed)
        self.assertEqual(bot.state.no_target_since, 25.0)
        self.assertEqual(bot.state.consecutive_exhausted, 0)
        self.assertIsNone(bot.state.exhausted_window_started_at)

    def test_cleared_caption_keeps_confirmed_target(self):
        bot = self._bot()
        bot.vision = _ClearedCaptionVision()
        target = {"click_x": 800, "click_y": 354}
        bot.state.current_target = target
        bot.state.exhausted_caption_target = target
        bot.state.exhausted_caption_check_at = 25.0
        bot.state.confirmed_locked = True

        bot._check_exhausted_caption(np.zeros((900, 1600, 3), dtype=np.uint8), 25.0)

        self.assertIs(bot.state.current_target, target)
        self.assertTrue(bot.state.confirmed_locked)

    def test_target_scanner_has_no_position_blacklist_parameter(self):
        self.assertNotIn("blacklist", inspect.signature(FrameVision.find_targets).parameters)
        self.assertNotIn("exhausted_mob_blacklist_seconds", BotConfig.__dataclass_fields__)

    def test_navigator_closes_map_after_ten_second_hold(self):
        reference = np.zeros((21, 21, 3), dtype=np.uint8)
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        clock = [0.0]

        class Device:
            def __init__(self):
                self.clicks = []

            def capture(self):
                return frame

            def click(self, x, y, synchronous=False):
                self.clicks.append((x, y, synchronous))
                return True

        device = Device()
        navigator = MinimapNavigator.__new__(MinimapNavigator)
        navigator.config = BotConfig()
        navigator.vision = _MapVision()
        navigator.device = device
        navigator.templates = _MapTemplates(reference)
        navigator.minimap_template_key = "minimap"
        navigator.cell_map = object()
        navigator.reference_id = id(reference)
        navigator.safe_mask = np.ones((1, 1), dtype=bool)
        navigator.waypoints = []
        navigator.route = None
        navigator.last_route_target = None
        navigator.align = lambda image, template: {
            "valid": True, "origin": (0, 0), "scale": 1.0,
            "score": 1.0, "geometry": 1.0, "coverage": 1.0, "gap": 1.0,
        }
        navigator.player_marker = lambda image: (100, 100)
        navigator.destination = lambda *args, **kwargs: (300, 300)
        closed_at = []
        navigator._close_map = lambda icon: closed_at.append(clock[0]) or True

        def sleep(seconds):
            clock[0] += seconds

        with patch("program.navigation.time.monotonic", lambda: clock[0]), \
                patch("program.navigation.time.sleep", sleep):
            moved_and_closed = navigator.travel(farthest=True, keep_open_seconds=10.0)

        self.assertTrue(moved_and_closed)
        self.assertEqual(device.clicks, [(300, 300, True)])
        self.assertEqual(len(closed_at), 1)
        self.assertGreaterEqual(closed_at[0], 10.8)


if __name__ == "__main__":
    unittest.main()
