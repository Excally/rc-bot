import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from program.bot import RucoyBot
from program.config import BotConfig
from program.state import BotState
from program.vision import FrameVision, TemplateStore


class _MarkerVision:
    def __init__(self, marker_visible):
        self.marker_visible = marker_visible

    def red_square(self, frame, target=None, radius=260):
        if self.marker_visible:
            return True, (700, 450)
        return False, None


class _ClickRecorder:
    def __init__(self):
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))
        return True


class _LockedEntityVision:
    def __init__(self, matches):
        self.matches = matches
        self.scan_options = None

    def find_targets(self, frame, templates, center_x, center_y, *, search_center, search_radius, include_deadzone):
        self.scan_options = (center_x, center_y, search_center, search_radius, include_deadzone)
        return self.matches


class TargetLockTests(unittest.TestCase):
    def _locked_bot(self, marker_visible):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.state = BotState(
            current_target={
                "click_x": 700,
                "click_y": 450,
                "last_entity_check": 0.0,
                "last_entity_seen_time": 100.0,
                "red_loss_since": None,
            },
            target_click_time=100.0,
            confirmed_locked=True,
        )
        bot.device = _ClickRecorder()
        bot.vision = _MarkerVision(marker_visible)
        bot.target_templates = []
        bot._find_locked_entity = lambda frame, cx, cy, target, now: {
            "click_x": 700,
            "click_y": 450,
        }
        return bot

    def test_locked_entity_scan_uses_current_target_scanner_api(self):
        match = {"click_x": 715, "click_y": 450}
        vision = _LockedEntityVision([match])
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.vision = vision
        bot.target_templates = []
        target = {"click_x": 700, "click_y": 450}

        found = bot._find_locked_entity(
            np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, target, 12.0
        )

        self.assertIs(found, match)
        self.assertEqual(vision.scan_options, (800, 450, (700, 450), 150, True))

    def test_confirmed_marker_does_not_expire_after_three_minutes(self):
        bot = self._locked_bot(marker_visible=True)

        bot._update_target(np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, 181.0)

        self.assertTrue(bot.state.confirmed_locked)
        self.assertIsNotNone(bot.state.current_target)

    def test_missing_marker_releases_lock_after_short_grace_period(self):
        bot = self._locked_bot(marker_visible=False)
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        bot._update_target(frame, 800, 450, 10.0)
        self.assertTrue(bot.state.confirmed_locked)
        self.assertEqual(bot.state.current_target["red_loss_since"], 10.0)

        bot._update_target(frame, 800, 450, 11.6)
        self.assertFalse(bot.state.confirmed_locked)
        self.assertIsNone(bot.state.current_target)

    def test_missing_marker_keeps_nearest_visible_skeleton_pending(self):
        bot = self._locked_bot(marker_visible=False)
        bot.state.confirmed_locked = False
        bot.state.target_click_time = 1.0
        bot.state.current_target.update(last_entity_check=0.0, last_entity_seen_time=1.0)
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        bot._update_target(frame, 800, 450, 2.6)

        self.assertIsNotNone(bot.state.current_target)
        self.assertFalse(bot.state.confirmed_locked)

    def test_pending_nearest_target_keeps_waiting_for_marked_png(self):
        bot = self._locked_bot(marker_visible=False)
        bot.state.confirmed_locked = False
        bot.state.target_click_time = 1.0
        bot.state.current_target.update(last_entity_check=180.0, last_entity_seen_time=180.0)

        bot._update_target(np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, 181.0)

        self.assertIsNotNone(bot.state.current_target)
        self.assertFalse(bot.state.confirmed_locked)

    def test_pending_skeleton_is_released_after_its_nameplate_disappears(self):
        bot = self._locked_bot(marker_visible=False)
        bot.state.confirmed_locked = False
        bot.state.target_click_time = 1.0
        bot.state.current_target.update(last_entity_check=0.0, last_entity_seen_time=1.0)
        bot._find_locked_entity = lambda frame, cx, cy, target, now: None

        bot._update_target(np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, 3.1)

        self.assertIsNone(bot.state.current_target)
        self.assertFalse(bot.state.confirmed_locked)

    def test_acquisition_clicks_nearest_even_when_farther_nameplate_scores_higher(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.profile = SimpleNamespace(target_name="Skeleton Lv.75")
        bot.state = BotState(idle_mark_check_needed=True, next_idle_mark_check_at=100.0)
        bot.templates = None
        bot.target_templates = [(np.ones((3, 3), dtype=np.uint8), "white")]
        bot.device = _ClickRecorder()
        bot.vision = FrameVision()
        # This full marker belongs to the farther target. It must not outrank
        # the closer Skeleton just because an existing mark was found first.
        bot.vision.red_square = lambda frame, target=None, radius=260: (True, (950, 450))
        farther = {
            "click_x": 950, "click_y": 450, "distance": 150.0,
            "score": 0.99, "scale": 1.0, "nx": 950, "ny": 450,
        }
        nearest = {
            "click_x": 850, "click_y": 450, "distance": 50.0,
            "score": 0.61, "scale": 1.0, "nx": 850, "ny": 450,
        }
        bot.vision._find_one_target = lambda *args, **kwargs: [farther, nearest]

        bot._acquire_target(np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, 10.0)

        self.assertEqual(bot.device.clicks, [(850, 450)])

    def test_existing_mark_is_adopted_when_it_matches_the_nearest_skeleton(self):
        bot = RucoyBot.__new__(RucoyBot)
        bot.config = BotConfig()
        bot.profile = SimpleNamespace(target_name="Skeleton Lv.75")
        bot.state = BotState(idle_mark_check_needed=True, next_idle_mark_check_at=100.0)
        bot.templates = None
        bot.target_templates = [(np.ones((3, 3), dtype=np.uint8), "white")]
        bot.device = _ClickRecorder()
        bot.vision = FrameVision()
        bot.vision.red_square = lambda frame, target=None, radius=260: (True, (870, 450))
        nearest = {
            "click_x": 850, "click_y": 450, "distance": 50.0,
            "score": 0.61, "scale": 1.0, "nx": 850, "ny": 450,
        }
        bot.vision._find_one_target = lambda *args, **kwargs: [nearest]

        bot._acquire_target(np.zeros((900, 1600, 3), dtype=np.uint8), 800, 450, 10.0)

        self.assertEqual(bot.device.clicks, [])
        self.assertTrue(bot.state.confirmed_locked)
        self.assertEqual(
            (bot.state.current_target["click_x"], bot.state.current_target["click_y"]),
            (870, 450),
        )

    def test_actual_marked_png_is_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            templates = TemplateStore(Path(temp_dir))
            # A legacy marker image must not silently substitute for marked.png.
            cv2.imwrite(
                str(Path(temp_dir) / "mob-current-marked.png"),
                np.full((72, 72, 3), (50, 50, 207), dtype=np.uint8),
            )
            vision = FrameVision(templates=templates)

            found, position = vision.red_square(np.zeros((900, 1600, 3), dtype=np.uint8))

        self.assertFalse(found)
        self.assertIsNone(position)

    def test_exact_template_sized_marker_matches(self):
        vision = FrameVision()
        template = vision._marked_template()
        self.assertGreater(template.size, 0, "templates/marked.png must be present")
        th, tw = template.shape
        x, y = 37, 29
        mask = np.zeros((th + 80, tw + 80), dtype=np.uint8)
        mask[y:y + th, x:x + tw] = template

        center = vision._match_outline(mask, template)

        self.assertEqual(center, (x + tw // 2, y + th // 2))

    def test_target_anchored_match_survives_two_sides_of_occlusion(self):
        vision = FrameVision()
        template = vision._marked_template()
        th, tw = template.shape
        thickness = max(3, min(10, int(round(min(th, tw) * 0.07))))
        partial = np.zeros_like(template)
        partial[:thickness, :] = template[:thickness, :]
        partial[:, :thickness] = template[:, :thickness]
        x, y = 37, 29
        mask = np.zeros((th + 80, tw + 80), dtype=np.uint8)
        mask[y:y + th, x:x + tw] = partial

        local_center = vision._match_outline(
            mask,
            template,
            threshold=0.40,
            expected_center=(x + tw // 2, y + th // 2),
            anchor_radius=2,
        )
        global_center = vision._match_outline(mask, template)

        self.assertEqual(local_center, (x + tw // 2, y + th // 2))
        self.assertIsNone(global_center)


if __name__ == "__main__":
    unittest.main()
