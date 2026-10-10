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

    def test_cli_positional_mage_alias(self):
        with patch("program.cli.run_bot") as mock_run:
            result = main(["--profile", "skeleton-lv75", "mage"])
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

    def test_locked_target_never_released_when_red_square_is_present(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(target_red_loss_timeout=1.0, target_entity_loss_timeout=2.0)
        bot.combat_class = "melee"
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
            confirmed_locked=True,
            target_click_time=10.0,
        )
        # Red mark remains visible at (1200, 450)
        bot.vision = SimpleNamespace(
            red_square=lambda frame, target=None, radius=260: (True, (1200, 450))
        )
        # Nameplate is obscured by damage numbers throughout combat
        bot._find_locked_entity = lambda frame, cx, cy, target, now: None
        bot._reset_exhaustion_streak = lambda: None
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        # Multiple updates over 5 seconds (well beyond the 2s entity loss timeout)
        for t in (11.0, 12.0, 13.0, 14.0, 15.0):
            bot._update_target(frame, 800, 450, t)

        # Lock must remain firmly held, no clicks issued
        self.assertTrue(bot.state.confirmed_locked)
        self.assertIsNotNone(bot.state.current_target)
        self.assertEqual(len(recorder.clicks), 0)

    def test_locked_target_released_only_after_red_square_disappears_for_timeout(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(target_red_loss_timeout=1.0)
        bot.combat_class = "melee"
        recorder = _ClickRecorder()
        bot.device = recorder
        target = {
            "click_x": 1200,
            "click_y": 450,
        }
        bot.state = BotState(
            current_target=target,
            confirmed_locked=True,
            target_click_time=10.0,
        )
        bot.vision = SimpleNamespace(
            red_square=lambda frame, target=None, radius=260: (False, None)
        )
        bot._reset_exhaustion_streak = lambda: None
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        # First missing frame records red_loss_since
        bot._update_target(frame, 800, 450, 11.0)
        self.assertTrue(bot.state.confirmed_locked)
        self.assertEqual(bot.state.current_target["red_loss_since"], 11.0)
        self.assertEqual(len(recorder.clicks), 0)

        # Still missing before 1.0s timeout (at 11.8s) -> stays locked
        bot._update_target(frame, 800, 450, 11.8)
        self.assertTrue(bot.state.confirmed_locked)
        self.assertEqual(len(recorder.clicks), 0)

        # After 1.0s timeout (at 12.1s) -> mob defeated, releases lock
        bot._update_target(frame, 800, 450, 12.1)
        self.assertFalse(bot.state.confirmed_locked)
        self.assertIsNone(bot.state.current_target)
        self.assertTrue(bot.state.pickup_pending)
        self.assertEqual(len(recorder.clicks), 0)

    def test_unconfirmed_target_retries_mob_click_once_then_skips_without_ground_click(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig(target_entity_loss_timeout=2.0)
        bot.combat_class = "melee"
        recorder = _ClickRecorder()
        bot.device = recorder
        target = {
            "click_x": 1200,
            "click_y": 450,
            "nx": 1200,
            "ny": 412,
            "last_entity_check": 10.0,
            "last_entity_seen_time": 10.0,
        }
        bot.state = BotState(
            current_target=target,
            confirmed_locked=False,
            target_click_time=10.0,
        )
        bot.vision = SimpleNamespace(
            red_square=lambda frame, target=None, radius=260: (False, None)
        )
        bot._find_locked_entity = lambda frame, cx, cy, target, now: target
        bot._reset_exhaustion_streak = lambda: None
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        # At 12.1s (>= 2.0s timeout): retries clicking mob at (1200, 450)
        bot._update_target(frame, 800, 450, 12.1)
        self.assertEqual(recorder.clicks, [(1200, 450)])
        self.assertFalse(bot.state.confirmed_locked)
        self.assertIsNotNone(bot.state.current_target)

        # At 14.2s (another 2.0s without red square): skips mob without clicking ground
        bot._update_target(frame, 800, 450, 14.2)
        self.assertFalse(bot.state.confirmed_locked)
        self.assertIsNone(bot.state.current_target)
        self.assertEqual(recorder.clicks, [(1200, 450)])  # No additional clicks

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
