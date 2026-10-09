"""Graphical launcher dialog for selecting zone profile and combat class."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple

from .profiles import ProfileSettings, ZoneProfile


class LauncherDialog:
    """A clean, modern dialog to choose profile and combat class with zero typing."""

    def __init__(self, settings: ProfileSettings):
        self.settings = settings
        self.result: Optional[Tuple[ZoneProfile, str]] = None

        self.root = tk.Tk()
        self.root.title("Rucoy Bot - Launcher")
        self.root.geometry("480x430")
        self.root.resizable(False, False)

        # Center on screen
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - 480) // 2)
        pos_y = max(0, (screen_h - 430) // 2)
        self.root.geometry(f"480x430+{pos_x}+{pos_y}")

        # Dark theme palette
        self.bg_color = "#181825"
        self.card_bg = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.fg_dim = "#a6adc8"
        self.accent_blue = "#89b4fa"
        self.accent_green = "#a6e3a1"
        self.border_color = "#313244"

        self.root.configure(bg=self.bg_color)
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "Card.TFrame",
            background=self.card_bg,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabel",
            background=self.bg_color,
            foreground=self.fg_color,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Card.TLabel",
            background=self.card_bg,
            foreground=self.fg_color,
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardDim.TLabel",
            background=self.card_bg,
            foreground=self.fg_dim,
            font=("Segoe UI", 9),
        )
        style.configure(
            "TRadiobutton",
            background=self.bg_color,
            foreground=self.fg_color,
            font=("Segoe UI", 10),
            focuscolor=self.bg_color,
        )
        style.map(
            "TRadiobutton",
            foreground=[("active", self.accent_blue)],
            background=[("active", self.bg_color)],
        )

    def _build_ui(self) -> None:
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill="x", padx=24, pady=(18, 12))

        title_lbl = tk.Label(
            header_frame,
            text="⚔️ Rucoy Bot Launcher",
            font=("Segoe UI", 16, "bold"),
            bg=self.bg_color,
            fg=self.accent_blue,
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Choose your hunting zone and combat class to start.",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg=self.fg_dim,
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Profile selection
        prof_frame = tk.Frame(self.root, bg=self.bg_color)
        prof_frame.pack(fill="x", padx=24, pady=6)

        prof_title = tk.Label(
            prof_frame,
            text="Hunting Profile:",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_color,
            fg=self.fg_color,
        )
        prof_title.pack(anchor="w", pady=(0, 4))

        self.profile_keys = list(self.settings.profiles.keys())
        self.profile_display_names = [
            f"{k} — {self.settings.profiles[k].target_name}"
            for k in self.profile_keys
        ]

        active_idx = 0
        if self.settings.active_profile in self.profile_keys:
            active_idx = self.profile_keys.index(self.settings.active_profile)

        self.selected_profile_idx = tk.IntVar(value=active_idx)
        self.combobox = ttk.Combobox(
            prof_frame,
            values=self.profile_display_names,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.combobox.current(active_idx)
        self.combobox.pack(fill="x")
        self.combobox.bind("<<ComboboxSelected>>", self._on_profile_changed)

        # Combat class selection
        class_frame = tk.Frame(self.root, bg=self.bg_color)
        class_frame.pack(fill="x", padx=24, pady=(12, 6))

        class_title = tk.Label(
            class_frame,
            text="Combat Class:",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_color,
            fg=self.fg_color,
        )
        class_title.pack(anchor="w", pady=(0, 4))

        initial_profile = self.settings.profiles[self.profile_keys[active_idx]]
        initial_class = (initial_profile.combat_class or "melee").lower()

        self.class_var = tk.StringVar(value=initial_class)

        radio_row = tk.Frame(class_frame, bg=self.bg_color)
        radio_row.pack(fill="x")

        classes = [
            ("melee", "⚔️ Melee\n(Native Walk)"),
            ("ranged", "🏹 Ranged\n(Auto-Approach)"),
            ("magic", "🔮 Magic / Mage\n(Auto-Approach)"),
        ]

        for val, label in classes:
            rb = ttk.Radiobutton(
                radio_row,
                text=label,
                value=val,
                variable=self.class_var,
                command=self._on_class_changed,
            )
            rb.pack(side="left", expand=True, fill="x", padx=4)

        # Info card
        self.info_card = tk.Frame(
            self.root,
            bg=self.card_bg,
            highlightbackground=self.border_color,
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        self.info_card.pack(fill="x", padx=24, pady=(10, 16))

        self.info_target_lbl = tk.Label(
            self.info_card,
            text="",
            font=("Segoe UI", 9, "bold"),
            bg=self.card_bg,
            fg=self.fg_color,
        )
        self.info_target_lbl.pack(anchor="w")

        self.info_mechanic_lbl = tk.Label(
            self.info_card,
            text="",
            font=("Segoe UI", 9),
            bg=self.card_bg,
            fg=self.fg_dim,
            wraplength=410,
            justify="left",
        )
        self.info_mechanic_lbl.pack(anchor="w", pady=(2, 0))

        self._refresh_info_card()

        # Bottom Buttons
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(fill="x", padx=24, pady=(0, 16))

        self.start_btn = tk.Button(
            btn_frame,
            text="🚀 Start Bot",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green,
            fg="#11111b",
            activebackground="#86d380",
            activeforeground="#11111b",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._on_start,
        )
        self.start_btn.pack(side="right")

        self.cancel_btn = tk.Button(
            btn_frame,
            text="Exit",
            font=("Segoe UI", 10),
            bg=self.card_bg,
            fg=self.fg_color,
            activebackground=self.border_color,
            activeforeground=self.fg_color,
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side="right", padx=(0, 10))

    def _on_profile_changed(self, event=None) -> None:
        idx = self.combobox.current()
        if 0 <= idx < len(self.profile_keys):
            prof = self.settings.profiles[self.profile_keys[idx]]
            if prof.combat_class:
                self.class_var.set(prof.combat_class.lower())
        self._refresh_info_card()

    def _on_class_changed(self) -> None:
        self._refresh_info_card()

    def _refresh_info_card(self) -> None:
        idx = self.combobox.current()
        if idx < 0 or idx >= len(self.profile_keys):
            return
        prof_key = self.profile_keys[idx]
        profile = self.settings.profiles[prof_key]
        c_class = self.class_var.get()

        self.info_target_lbl.config(
            text=f"Selected: {profile.target_name} ({prof_key})"
        )

        if c_class == "melee":
            desc = "• Melee Mode: Character automatically approaches mobs via in-game movement."
        elif c_class in ("ranged", "magic"):
            name = "Ranged" if c_class == "ranged" else "Magic"
            desc = f"• {name} Mode: Bot automatically steps toward distant mobs until within range (<100px) or red square is confirmed."
        else:
            desc = f"• Mode: {c_class}"

        self.info_mechanic_lbl.config(text=desc)

    def _on_start(self) -> None:
        idx = self.combobox.current()
        if 0 <= idx < len(self.profile_keys):
            prof_key = self.profile_keys[idx]
            profile = self.settings.profiles[prof_key]
            chosen_class = self.class_var.get()
            self.result = (profile, chosen_class)
        self.root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def show(self) -> Optional[Tuple[ZoneProfile, str]]:
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.root.mainloop()
        return self.result


def launch_gui(settings: ProfileSettings) -> Optional[Tuple[ZoneProfile, str]]:
    """Open the GUI launcher. Returns (profile, combat_class) or None if cancelled."""
    dialog = LauncherDialog(settings)
    return dialog.show()
