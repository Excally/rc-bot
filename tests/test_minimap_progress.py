import unittest

from program.config import BotConfig
from program.navigation import MinimapNavigator, MinimapRoute
from program.vision import FrameVision


class MinimapProgressTests(unittest.TestCase):
    def setUp(self):
        self.navigator = MinimapNavigator(
            BotConfig(), FrameVision(), None, None, "minimap-skeleton-layout"
        )
        self.route = MinimapRoute(
            signature=("skeleton-map",),
            waypoints=((10.0, 10.0), (16.0, 10.0), (22.0, 10.0)),
            index=0,
            last_player_position=(10.0, 10.0),
            last_target_index=1,
            last_target_position=(16, 10),
        )
        self.navigator.route = self.route

    def test_stationary_player_advances_past_the_tapped_route_target(self):
        self.navigator._check_route_progress((10.0, 10.0))
        self.assertEqual(self.route.index, 1)
        self.assertIsNone(self.route.last_player_position)
        self.assertIsNone(self.route.last_target_index)

    def test_observed_movement_keeps_the_current_route_target(self):
        self.navigator._check_route_progress((12.0, 10.0))
        self.assertEqual(self.route.index, 0)
        self.assertIsNone(self.route.last_player_position)
        self.assertIsNone(self.route.last_target_index)


if __name__ == "__main__":
    unittest.main()
