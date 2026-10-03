import unittest

import cv2
import numpy as np

from program.config import BotConfig
from program.navigation import MAP_CELL_PIXELS, MinimapNavigator
from program.vision import FrameVision


class MinimapGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = cv2.imread("templates/minimap-skeleton-layout.png", cv2.IMREAD_COLOR)
        if cls.reference is None:
            raise RuntimeError("The skeleton minimap template could not be loaded.")
        cls.navigator = MinimapNavigator(
            BotConfig(), FrameVision(), None, None, "minimap-skeleton-layout"
        )

    def test_template_palette_keeps_destinations_on_black_walkable_path(self):
        cell_map = self.navigator._build_cell_map(self.reference)
        self.assertEqual(cell_map.walls.shape, (71, 113))
        self.assertEqual(int(cell_map.walls.sum()), 648)
        self.assertEqual(int(cell_map.context_walls.sum()), 471)
        self.assertEqual(int(cell_map.unwalkable.sum()), 4060)
        self.assertEqual(int(cell_map.annotations.sum()), 7)
        self.assertEqual(int(np.count_nonzero(cell_map.walkable)), 2837)
        self.assertEqual(int(np.count_nonzero(cell_map.match_walls)), 1119)
        self.assertFalse(np.any(cell_map.safe_mask[cell_map.context_walls]))
        self.assertFalse(np.any(cell_map.safe_mask[cell_map.unwalkable]))
        self.assertFalse(np.any(cell_map.safe_mask[cell_map.walls]))
        self.assertFalse(np.any(cell_map.safe_mask[cell_map.annotations]))
        self.assertFalse(np.any(cell_map.safe_mask & (cell_map.walkable == 0)))
        for x, y in self.navigator.build_waypoints(cell_map.safe_mask):
            self.assertGreater(cell_map.safe_mask[int(y), int(x)], 0)
            self.assertGreater(cell_map.walkable[int(y), int(x)], 0)

    def test_nonuniform_antialiased_cell_is_rejected(self):
        damaged = self.reference.copy()
        damaged[10, 10] = (1, 1, 1)
        with self.assertRaisesRegex(ValueError, "non-uniform"):
            self.navigator._build_cell_map(damaged)

    def test_unknown_uniform_palette_color_is_rejected(self):
        damaged = self.reference.copy()
        damaged[:MAP_CELL_PIXELS, :MAP_CELL_PIXELS] = (12, 34, 56)
        with self.assertRaisesRegex(ValueError, "documented color palette"):
            self.navigator._build_cell_map(damaged)

    def test_registration_recovers_shifted_cell_map_with_occlusion(self):
        cell_map = self.navigator._build_cell_map(self.reference)
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        origin_x, origin_y = -218, -548
        ys, xs = np.where(cell_map.match_walls)
        for cell_y, cell_x in zip(ys, xs):
            x1 = origin_x + int(cell_x) * MAP_CELL_PIXELS
            y1 = origin_y + int(cell_y) * MAP_CELL_PIXELS
            x2, y2 = x1 + MAP_CELL_PIXELS, y1 + MAP_CELL_PIXELS
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if cx2 > cx1 and cy2 > cy1:
                frame[cy1:cy2, cx1:cx2] = (243, 243, 243)
        # A moving sprite covers a small piece of the static wall drawing.
        frame[360:430, 740:805] = (38, 53, 17)
        # The saved comparison shrinks the live frame by half: its 7x7 cyan
        # marker is only about 35 pixels in the actual captured frame.
        cv2.circle(frame, (801, 450), 3, (255, 255, 0), -1)
        result = self.navigator.align(frame, self.reference)
        self.assertIsNotNone(result)
        self.assertTrue(result["valid"], result)
        self.assertGreater(result["geometry"], 0.70, result)
        self.assertEqual(result["score"], result["geometry"])
        self.assertEqual(result["positions_tested"], 2837, result)
        self.assertLessEqual(abs(result["origin"][0] - origin_x), 7, result)
        self.assertLessEqual(abs(result["origin"][1] - origin_y), 7, result)
        self.assertGreaterEqual(result["coverage"], 0.35, result)

    def test_small_cyan_player_marker_is_detected_near_screen_center(self):
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        cv2.circle(frame, (801, 450), 3, (255, 255, 0), -1)
        self.assertEqual(self.navigator.player_marker(frame), (801, 450))

    def test_unrelated_bright_scene_does_not_lock_a_map_position(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(20, 115, size=(900, 1600, 3), dtype=np.uint8)
        # Add a few bright HUD-like strokes without any map-shaped structure.
        cv2.rectangle(frame, (90, 30), (510, 70), (240, 240, 240), 2)
        cv2.putText(frame, "475/475", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (245, 245, 245), 2, cv2.LINE_8)
        result = self.navigator.align(frame, self.reference)
        self.assertIsNotNone(result)
        self.assertFalse(result["valid"], result)
        self.assertIn("player marker not detected", result["failed_checks"])

    def test_red_outside_color_is_not_a_player_marker(self):
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        frame[430:451, 790:811] = (0, 0, 255)
        self.assertIsNone(self.navigator.player_marker(frame))

    def test_skeleton_nameplate_over_player_center_is_not_deadzoned(self):
        nameplate = cv2.imread("templates/skeleton_lv75.png", cv2.IMREAD_COLOR)
        self.assertIsNotNone(nameplate)
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        height, width = nameplate.shape[:2]
        near = (800 - width // 2, 262)
        farther = (817, 337)
        for x, y in (near, farther):
            frame[y:y + height, x:x + width] = nameplate

        matches = self.navigator.vision.find_targets(
            frame, [(nameplate, "white")], 801, 417
        )
        self.assertGreaterEqual(len(matches), 2)
        self.assertLess(matches[0]["distance"], BotConfig().player_deadzone_radius)
        self.assertLess(matches[0]["distance"], matches[1]["distance"])
        self.assertLessEqual(abs(matches[0]["click_x"] - 800), 2)


if __name__ == "__main__":
    unittest.main()
