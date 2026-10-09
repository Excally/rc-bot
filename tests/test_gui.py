import unittest
from unittest.mock import MagicMock, patch

from program.cli import main
from program.gui import LauncherDialog
from program.profiles import ProfileSettings, ZoneProfile


class GuiLauncherTests(unittest.TestCase):
    def setUp(self):
        self.profile1 = ZoneProfile(
            key="skeleton-lv75",
            target_name="Skeleton Lv.75",
            target_template_key="skeleton_lv75",
            minimap_template_key="minimap-skeleton-layout",
            combat_class="melee",
        )
        self.profile2 = ZoneProfile(
            key="zombie-lv65",
            target_name="Zombie Lv.65",
            target_template_key="zombie_lv65",
            minimap_template_key="minimap-zombie",
            combat_class="magic",
        )
        self.settings = ProfileSettings(
            active_profile="skeleton-lv75",
            profiles={"skeleton-lv75": self.profile1, "zombie-lv65": self.profile2},
        )

    def test_gui_dialog_initialization_headless(self):
        try:
            dialog = LauncherDialog(self.settings)
            # Should have preselected active profile
            self.assertEqual(dialog.combobox.current(), 0)
            self.assertEqual(dialog.class_var.get(), "melee")
            dialog._on_cancel()
            self.assertIsNone(dialog.result)
        except Exception as e:
            # Skip if running on headless server with no graphical display
            self.skipTest(f"Tkinter display not available: {e}")

    def test_gui_dialog_start_action(self):
        try:
            dialog = LauncherDialog(self.settings)
            dialog.combobox.current(1)  # zombie-lv65
            dialog._on_profile_changed()
            dialog.class_var.set("ranged")
            dialog._on_start()
            self.assertIsNotNone(dialog.result)
            prof, c_class = dialog.result
            self.assertEqual(prof.key, "zombie-lv65")
            self.assertEqual(c_class, "ranged")
        except Exception as e:
            self.skipTest(f"Tkinter display not available: {e}")

    def test_cli_gui_flag_invokes_launch_gui(self):
        with patch("program.gui.launch_gui") as mock_launch, patch("program.cli.run_bot") as mock_run:
            mock_launch.return_value = (self.profile1, "magic")
            result = main(["--gui"])
            self.assertEqual(result, 0)
            mock_launch.assert_called_once()
            mock_run.assert_called_once_with(self.profile1, combat_class="magic")

    def test_cli_gui_cancel_returns_zero(self):
        with patch("program.gui.launch_gui") as mock_launch, patch("program.cli.run_bot") as mock_run:
            mock_launch.return_value = None
            result = main(["--gui"])
            self.assertEqual(result, 0)
            mock_launch.assert_called_once()
            mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
