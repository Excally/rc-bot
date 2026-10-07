import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from program.bot import RucoyBot
from program.cli import main
from program.config import BotConfig
from program.profiles import ZoneProfile
from program.state import BotState


class _ClickRecorder:
    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))
        return True


class CombatClassTests(unittest.TestCase):
    def test_cli_positional_melee(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "melee"])
            self.assertEqual(result, 0)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[1]["combat_class"], "melee")

    def test_cli_positional_ranged(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "ranged"])
            self.assertEqual(result, 0)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[1]["combat_class"], "ranged")

    def test_cli_positional_magic(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "magic"])
            self.assertEqual(result, 0)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[1]["combat_class"], "magic")

    def test_cli_flag_class(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "--class", "ranged"])
            self.assertEqual(result, 0)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[1]["combat_class"], "ranged")

    def test_cli_case_insensitive(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "RANGED"])
            self.assertEqual(result, 0)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[1]["combat_class"], "ranged")

    def test_cli_invalid_class_fails(self):
        with self.assertRaises(SystemExit):
            main(["--profile", "skeleton-lv75", "warrior"])

    def test_cli_show_config_includes_combat_class(self):
        out = io.StringIO()
        with redirect_stdout(out):
            result = main(["--profile", "skeleton-lv75", "ranged", "--show-config"])
        self.assertEqual(result, 0)
        output = out.getvalue()
        self.assertIn("Combat class: ranged", output)

    def test_zone_profile_default_melee(self):
        profile = ZoneProfile(
            key="test",
            target_name="Test Mob",
            target_template_key="test_target",
            minimap_template_key="test_map",
        )
        self.assertEqual(profile.combat_class, "melee")

    def test_melee_class_does_not_approach_automatically(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "melee"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450},
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 15.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

    def test_ranged_class_approaches_distant_target(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450},  # 400px away horizontally
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertTrue(moved)
        self.assertEqual(len(recorder.clicks), 1)
        tap_x, tap_y = recorder.clicks[0]
        # Halfway between 800 and 1200 is 1000, y is 450
        self.assertEqual(tap_x, 1000)
        self.assertEqual(tap_y, 450)
        self.assertEqual(bot.state.last_approach_time, 11.0)

    def test_magic_class_approaches_distant_target(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "magic"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 800, "click_y": 850},  # 400px away vertically
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertTrue(moved)
        self.assertEqual(len(recorder.clicks), 1)
        tap_x, tap_y = recorder.clicks[0]
        self.assertEqual(tap_x, 800)
        self.assertEqual(tap_y, 650)  # halfway between 450 and 850 is 650

    def test_close_range_target_does_not_trigger_approach(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 880, "click_y": 450},  # 80px away (<= close distance 100px)
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

    def test_approach_interval_cooldown(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(approach_interval=0.7)
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450},
            target_click_time=10.0,
            last_approach_time=11.0,
        )

        # Calling at 11.3s (only 0.3s passed, less than 0.7s interval)
        moved = bot._approach_target(800, 450, 11.3)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

        # Calling at 11.8s (0.8s passed, interval satisfied)
        moved = bot._approach_target(800, 450, 11.8)
        self.assertTrue(moved)
        self.assertEqual(len(recorder.clicks), 1)

    def test_initial_delay_after_target_click(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(approach_initial_delay=0.25)
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450},
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        # Immediately at 10.1s (0.1s after target click, < 0.25s)
        moved = bot._approach_target(800, 450, 10.1)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

        # At 10.3s (> 0.25s after target click)
        moved = bot._approach_target(800, 450, 10.3)
        self.assertTrue(moved)
        self.assertEqual(len(recorder.clicks), 1)

    def test_tap_stays_outside_player_deadzone(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(player_deadzone_radius=90)
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 970, "click_y": 450},  # 170px away
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertTrue(moved)
        tap_x, tap_y = recorder.clicks[0]
        step = tap_x - 800
        # step must be >= deadzone_radius + 15 (105px)
        self.assertGreaterEqual(step, 105)
        # and must not land on target (170px)
        self.assertLess(tap_x, 970)

    def test_confirmed_locked_ranged_does_not_approach(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450},
            confirmed_locked=True,
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

    def test_confirmed_locked_magic_does_not_approach(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "magic"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 800, "click_y": 850},
            confirmed_locked=True,
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

    def test_target_with_red_observations_does_not_approach(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        bot.state = BotState(
            current_target={"click_x": 1200, "click_y": 450, "red_observations": 1},
            confirmed_locked=False,
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        moved = bot._approach_target(800, 450, 11.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), 0)

    def test_approach_sets_approached_since_tap_and_waits(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        target = {"click_x": 1200, "click_y": 450}
        bot.state = BotState(
            current_target=target,
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        # First approach step succeeds and marks approached_since_tap
        moved = bot._approach_target(800, 450, 11.0)
        self.assertTrue(moved)
        self.assertTrue(target.get("approached_since_tap"))
        self.assertEqual(len(recorder.clicks), 1)

        # Subsequent approach call before mob tap is suppressed
        moved_again = bot._approach_target(800, 450, 12.0)
        self.assertFalse(moved_again)
        self.assertEqual(len(recorder.clicks), 1)

    def test_post_step_mob_tap_triggers_after_interval(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(approach_interval=0.7)
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        target = {
            "click_x": 1200,
            "click_y": 450,
            "approached_since_tap": True,
            "last_entity_check": 10.0,
            "last_entity_seen_time": 10.0,
        }
        bot.state = BotState(
            current_target=target,
            confirmed_locked=False,
            target_click_time=10.0,
            last_approach_time=10.0,
        )
        mock_vision = SimpleNamespace(
            red_square=lambda frame, target=None, radius=260: (False, None),
        )
        bot.vision = mock_vision
        bot._find_locked_entity = lambda frame, cx, cy, target, now: target
        bot._reset_exhaustion_streak = lambda: None

        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        # Calling update at 10.8s (0.8s after approach, > 0.7s interval)
        bot._update_target(frame, 800, 450, 10.8)

        # Should have tapped mob at (1200, 450)
        self.assertEqual(recorder.clicks, [(1200, 450)])
        self.assertFalse(target.get("approached_since_tap"))
        self.assertEqual(target.get("approach_steps"), 1)
        self.assertEqual(bot.state.target_click_time, 10.8)

    def test_once_locked_approaching_stops_until_mob_dies(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(approach_interval=0.7, target_red_loss_timeout=1.0)
        bot.combat_class = "ranged"
        recorder = _ClickRecorder()
        bot.device = recorder
        target = {
            "click_x": 1200,
            "click_y": 450,
            "last_entity_check": 10.0,
            "last_entity_seen_time": 10.0,
        }
        bot.state = BotState(
            current_target=target,
            confirmed_locked=False,
            target_click_time=10.0,
            last_approach_time=0.0,
        )

        has_marker = False

        def mock_red_square(frame, target=None, radius=260):
            if has_marker:
                return True, (1200, 450)
            return False, None

        bot.vision = SimpleNamespace(red_square=mock_red_square)
        bot._find_locked_entity = lambda frame, cx, cy, target, now: target
        bot._reset_exhaustion_streak = lambda: None
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        # 1. Unconfirmed: approaches mob by tapping ground
        moved = bot._approach_target(800, 450, 10.5)
        self.assertTrue(moved)
        self.assertEqual(recorder.clicks, [(1000, 450)])  # ground tap

        # 2. Step finished at 11.3s: taps mob
        bot._update_target(frame, 800, 450, 11.3)
        self.assertEqual(recorder.clicks, [(1000, 450), (1200, 450)])  # mob tap

        # 3. Mob now has red mark: lock confirmed across 2 observations
        has_marker = True
        bot._update_target(frame, 800, 450, 11.4)
        bot._update_target(frame, 800, 450, 11.5)
        self.assertTrue(bot.state.confirmed_locked)

        # 4. Once locked, approach returns False and produces NO ground clicks!
        clicks_before = len(recorder.clicks)
        moved = bot._approach_target(800, 450, 12.0)
        self.assertFalse(moved)
        moved = bot._approach_target(800, 450, 13.0)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), clicks_before)

        # 5. Mob dies (red mark absent for 1.0s timeout)
        has_marker = False
        bot._update_target(frame, 800, 450, 13.1)  # marks loss_since
        bot._update_target(frame, 800, 450, 14.2)  # exceeds 1.0s timeout
        self.assertFalse(bot.state.confirmed_locked)
        self.assertIsNone(bot.state.current_target)

        # 6. Target is dead: approach does not run on None target
        moved = bot._approach_target(800, 450, 14.3)
        self.assertFalse(moved)
        self.assertEqual(len(recorder.clicks), clicks_before)

    def test_skeleton_archer_rejected_when_targeting_skeleton_lv75(self):
        import cv2
        from program.vision import FrameVision, TemplateStore
        store = TemplateStore()
        store.load(("skeleton_lv75",))
        tpl = store.get("skeleton_lv75")
        self.assertIsNotNone(tpl, "skeleton_lv75 template must exist")
        frame = cv2.imread("screenshot-stock/debug_skeleton_live.png")
        if frame is None:
            self.skipTest("debug_skeleton_live.png not available")
        vision = FrameVision(BotConfig(), store)
        matches = vision.find_targets(frame, [(tpl, "white")], 800, 450)
        # In debug_skeleton_live.png, the two Skeleton Archers are at x=274 and x=1074.
        # Neither of them should be included in matches!
        for match in matches:
            self.assertNotEqual(match["nx"], 274, "Skeleton Archer at x=274 must not be matched")
            self.assertNotEqual(match["nx"], 1074, "Skeleton Archer at x=1074 must not be matched")
        # Exactly the 5 genuine Skeleton Lv.75 nameplates should be found
        self.assertEqual(len(matches), 5)


if __name__ == "__main__":
    unittest.main()
