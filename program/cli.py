"""Command-line profile selection for the bot."""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .profiles import ProfileSettings, load_profile_settings
from .runner import run_bot


def _load_settings(parser: argparse.ArgumentParser) -> ProfileSettings:
    try:
        return load_profile_settings()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"could not load zone_profiles.json: {error}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rucoy bot with selectable zone and template profiles.")
    parser.add_argument(
        "--profile", metavar="NAME", help="zone profile to use; defaults to active_profile in zone_profiles.json"
    )
    parser.add_argument("--list-profiles", action="store_true", help="list available profiles and exit")
    settings = _load_settings(parser)
    parser.add_argument("--show-config", action="store_true", help="show the selected templates and exit")
    args = parser.parse_args(argv)

    if args.list_profiles:
        for key, profile in settings.profiles.items():
            active = " (active)" if key == settings.active_profile else ""
            print(f"{key}{active}: {profile.target_name}")
        return 0

    selected_key = args.profile or settings.active_profile
    profile = settings.profiles.get(selected_key)
    if profile is None:
        parser.error(f"unknown profile '{selected_key}'; use --list-profiles to see available profiles")
    if args.show_config:
        print(f"Profile: {profile.key}")
        print(f"Target: {profile.target_name}")
        print(f"Name template: {profile.target_template_key}.png")
        if profile.target_purple_template_key:
            print(f"Purple name template: {profile.target_purple_template_key}.png")
        print(f"Minimap template: {profile.minimap_template_key}.png")
        return 0

    run_bot(profile)
    return 0
