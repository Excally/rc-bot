"""Load zone-specific target and minimap template selections."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROFILE_FILE = Path(__file__).resolve().parent.parent / "zone_profiles.json"


@dataclass(frozen=True)
class ZoneProfile:
    key: str
    target_name: str
    target_template_key: str
    minimap_template_key: str
    combat_class: str = "melee"

    @property
    def template_keys(self) -> tuple[str, ...]:
        return self.target_template_key, self.minimap_template_key


@dataclass(frozen=True)
class ProfileSettings:
    active_profile: str
    profiles: dict[str, ZoneProfile]


def _template_key(value: object, field_name: str, profile_key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Profile '{profile_key}' needs a non-empty '{field_name}'.")
    key = value.strip()
    if Path(key).name != key or Path(key).suffix:
        raise ValueError(
            f"Profile '{profile_key}' field '{field_name}' must be a template filename stem, without a path or extension."
        )
    return key


def load_profile_settings(path: Path = PROFILE_FILE) -> ProfileSettings:
    """Load profiles from JSON and validate all configured template keys."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The profile file must contain a JSON object.")
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("The profile file needs a non-empty 'profiles' object.")

    profiles: dict[str, ZoneProfile] = {}
    for key, raw in raw_profiles.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(raw, dict):
            raise ValueError("Each profile must have a non-empty string key and an object value.")
        key = key.strip()
        target_name = raw.get("target_name")
        if not isinstance(target_name, str) or not target_name.strip():
            raise ValueError(f"Profile '{key}' needs a non-empty 'target_name'.")
        raw_combat_class = raw.get("combat_class", "melee")
        combat_class = (
            raw_combat_class.strip().lower()
            if isinstance(raw_combat_class, str) and raw_combat_class.strip().lower() in {"melee", "ranged", "magic"}
            else "melee"
        )
        profiles[key] = ZoneProfile(
            key=key,
            target_name=target_name.strip(),
            target_template_key=_template_key(raw.get("name_template"), "name_template", key),
            minimap_template_key=_template_key(raw.get("minimap_template"), "minimap_template", key),
            combat_class=combat_class,
        )

    active_profile = data.get("active_profile")
    if not isinstance(active_profile, str) or active_profile not in profiles:
        raise ValueError("'active_profile' must name one of the profiles in the profile file.")
    return ProfileSettings(active_profile=active_profile, profiles=profiles)
