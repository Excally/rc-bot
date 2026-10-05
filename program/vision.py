"""Template management and stateless frame recognition."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig, CONFIG, TEMPLATE_DIR
from .models import TargetMatch

class TemplateStore:
    BUILT_IN_NAMES = {
        "skeleton_lv75", "minimapicon", "backicon",
        "minimap-skeleton-layout", "minimap-zombie-layout",
        "pickup-available", "pickup-not-yet", "exhausted-caption",
    }

    def __init__(self, directory: Path = TEMPLATE_DIR):
        self.directory = directory
        self.images: dict[str, np.ndarray] = {}

    def load(self, additional_names: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        self.images.clear()
        if not self.directory.exists():
            return self.images
        allowed_names = self.BUILT_IN_NAMES | set(additional_names)
        for filename in self.directory.iterdir():
            if filename.suffix.lower() not in {".png", ".jpg"} or filename.stem not in allowed_names:
                continue
            mode = cv2.IMREAD_UNCHANGED if filename.stem in {"backicon", "minimapicon"} else cv2.IMREAD_COLOR
            image = cv2.imread(str(filename), mode)
            if image is not None:
                self.images[filename.stem] = image
        return self.images

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.images.get(name)


class FrameVision:
    """Image recognition with per-frame caches; never sends input actions."""

    def __init__(self, config: BotConfig = CONFIG, templates: Optional[TemplateStore] = None):
        self.config = config
        self.templates = templates
        self._marked_template_image: Optional[np.ndarray] = None
        self._red_frame: Any = None
        self._red_mask: Optional[np.ndarray] = None
        self._caption_cache: Any = None
        self._target_mask_frame: Any = None
        self._target_masks: dict[
            tuple[str, tuple[int, int, int, int]], np.ndarray
        ] = {}
        self._target_template_masks: dict[
            tuple[int, str, float], tuple[np.ndarray, np.ndarray]
        ] = {}

    # HP/MP bar geometry at 1600x900.
    _BAR_X_START = 4
    _BAR_X_END = 460
    _HP_BAR_Y_START = 4
    _HP_BAR_Y_END = 45
    _MP_BAR_Y_START = 50
    _MP_BAR_Y_END = 62
    _HP_COLOR_BGR = np.array([50, 50, 207], dtype=np.int16)   # #cf3232
    _MP_COLOR_BGR = np.array([252, 188, 60], dtype=np.int16)  # #3cbcfc
    _BAR_COLOR_TOLERANCE = 30

    def _bar_fill(self, frame: np.ndarray, y_start: int, y_end: int,
                  color: np.ndarray) -> float:
        """Return the fill fraction (0.0–1.0) of a horizontal bar.

        The text overlay (e.g. '475/475') blocks the bar color in the center,
        so we scan columns from the right edge inward.  A column counts as
        filled if ANY pixel in the vertical slice matches the bar color.
        """
        xs, xe = self._BAR_X_START, self._BAR_X_END
        # Extract the bar region as int16 for safe subtraction.
        strip = frame[y_start:y_end + 1, xs:xe + 1].astype(np.int16)
        # For each column, check if any row matches the bar color.
        diff = np.abs(strip - color)  # shape: (rows, cols, 3)
        match = np.all(diff <= self._BAR_COLOR_TOLERANCE, axis=2)  # (rows, cols)
        col_has_color = np.any(match, axis=0)  # (cols,)
        # Find the rightmost column with the bar color.
        indices = np.where(col_has_color)[0]
        if len(indices) == 0:
            return 0.0
        rightmost = int(indices[-1])
        return (rightmost + 1) / (xe - xs + 1)

    def hp_below_threshold(self, frame: np.ndarray, threshold: float) -> bool:
        """True if the HP bar has depleted past *threshold* (0.0–1.0)."""
        return self._bar_fill(
            frame, self._HP_BAR_Y_START, self._HP_BAR_Y_END, self._HP_COLOR_BGR,
        ) < threshold

    def mp_below_threshold(self, frame: np.ndarray, threshold: float) -> bool:
        """True if the MP bar has depleted past *threshold* (0.0–1.0)."""
        return self._bar_fill(
            frame, self._MP_BAR_Y_START, self._MP_BAR_Y_END, self._MP_COLOR_BGR,
        ) < threshold

    @staticmethod
    def is_disconnected(frame: Optional[np.ndarray]) -> bool:
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        title = gray[int(fh * 0.18):int(fh * 0.32), int(fw * 0.32):int(fw * 0.68)]
        reconnect = gray[int(fh * 0.58):int(fh * 0.74), int(fw * 0.14):int(fw * 0.49)]
        return (
            np.mean(gray < 24) > 0.82
            and np.count_nonzero(title > 175) > 250
            and np.count_nonzero(reconnect > 175) > 180
        )

    def find_fixed_icon(
        self, frame: Optional[np.ndarray], icon: Optional[np.ndarray], expected_center: tuple[int, int],
        threshold: Optional[float] = None, search_radius: tuple[int, int] = (90, 70),
        scales: tuple[float, ...] = (1.0, 0.9, 0.85, 0.8, 0.75, 1.1),
    ) -> Optional[tuple[int, int]]:
        if frame is None or icon is None:
            return None
        threshold = self.config.ui_match_threshold if threshold is None else threshold
        alpha = icon[:, :, 3] if icon.ndim == 3 and icon.shape[2] == 4 else None
        if alpha is not None:
            icon = icon[:, :, :3]

        fh, fw = frame.shape[:2]
        sx, sy = fw / 1600.0, fh / 900.0
        center_x, center_y = int(expected_center[0] * sx), int(expected_center[1] * sy)
        radius_x, radius_y = int(search_radius[0] * sx), int(search_radius[1] * sy)
        best_score, best_center = -1.0, None
        match_threshold = max(threshold, self.config.ui_match_threshold) if alpha is not None else threshold

        for adjustment in scales:
            scale = sx * adjustment
            scaled = cv2.resize(icon, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ih, iw = scaled.shape[:2]
            if ih >= fh or iw >= fw:
                continue
            left = max(0, center_x - radius_x - iw // 2)
            top = max(0, center_y - radius_y - ih // 2)
            right = min(fw, center_x + radius_x + iw // 2)
            bottom = min(fh, center_y + radius_y + ih // 2)
            region = frame[top:bottom, left:right]
            if region.shape[0] < ih or region.shape[1] < iw:
                continue
            if alpha is not None:
                scaled_alpha = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_NEAREST)
                mask = np.where(scaled_alpha > 0, 255, 0).astype(np.uint8)
                result = cv2.matchTemplate(region, scaled, cv2.TM_SQDIFF_NORMED, mask=mask)
                minimum, _, location, _ = cv2.minMaxLoc(result)
                score = 1.0 - minimum
            else:
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_center = (left + location[0] + iw // 2, top + location[1] + ih // 2)
            if score >= match_threshold + 0.02:
                return best_center
        return best_center if best_score >= match_threshold else None

    def back_icon(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        return self.find_fixed_icon(
            frame, self.templates.get("backicon") if self.templates else None,
            self.config.back_icon_center, self.config.back_icon_match_threshold,
            search_radius=(35, 30), scales=self.config.back_icon_scales,
        )

    def ui_state(self, frame: np.ndarray) -> tuple[str, Optional[tuple[int, int]]]:
        if self.is_disconnected(frame):
            return "disconnected", None
        back = self.back_icon(frame)
        if back is not None:
            return "minimap", back
        map_button = self.find_fixed_icon(
            frame, self.templates.get("minimapicon") if self.templates else None, (1150, 50)
        )
        if map_button is not None:
            return "combat", map_button
        return "other", None

    def white_text_mask(self, image: np.ndarray) -> np.ndarray:
        blue, green, red = cv2.split(image)
        bright = cv2.inRange(
            image, (self.config.white_text_min_channel,) * 3, (255, 255, 255)
        )
        spread_bg = cv2.inRange(cv2.absdiff(blue, green), 0, self.config.white_text_max_channel_spread)
        spread_gr = cv2.inRange(cv2.absdiff(green, red), 0, self.config.white_text_max_channel_spread)
        spread_br = cv2.inRange(cv2.absdiff(blue, red), 0, self.config.white_text_max_channel_spread)
        cv2.bitwise_and(bright, spread_bg, dst=bright)
        cv2.bitwise_and(bright, spread_gr, dst=bright)
        cv2.bitwise_and(bright, spread_br, dst=bright)
        return bright

    @staticmethod
    def purple_text_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (138, 90, 170), (154, 255, 255))

    def player_center(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """Find the player's bright green or cyan body near the camera center.

        The avatar palette spans green through cyan. Green health bars and the
        player name are much thinner than the body sprite, so connected-
        component size and shape separate the player from those UI elements.
        The result is used for target ranking only; the red lock marker remains
        the authority once combat starts.
        """
        fh, fw = frame.shape[:2]
        roi_x1, roi_x2 = int(fw * 0.30), int(fw * 0.70)
        roi_y1, roi_y2 = int(fh * 0.20), int(fh * 0.80)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        body_colors = cv2.inRange(hsv, (45, 120, 120), (125, 255, 255))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(body_colors, 8)
        scale_x, scale_y = fw / 1600.0, fh / 900.0
        min_area, max_area = 500 * scale_x * scale_y, 7000 * scale_x * scale_y
        min_width, max_width = 25 * scale_x, 130 * scale_x
        min_height, max_height = 25 * scale_y, 130 * scale_y
        center_x, center_y = fw / 2.0, fh / 2.0
        best_key = (float("inf"), float("inf"), float("inf"))
        best_point: Optional[tuple[int, int]] = None
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not (
                min_area <= area <= max_area
                and min_width <= width <= max_width
                and min_height <= height <= max_height
            ):
                continue
            if not (0.35 <= width / max(1, height) <= 2.5):
                continue
            point_x, point_y = centroids[index]
            screen_x, screen_y = point_x + roi_x1, point_y + roi_y1
            dx, dy = screen_x - center_x, screen_y - center_y
            candidate_key = (dx * dx + dy * dy, screen_x, screen_y)
            if candidate_key < best_key:
                best_key = candidate_key
                best_point = (int(round(screen_x)), int(round(screen_y)))
        return best_point

    def find_targets(
        self, frame: np.ndarray, templates: list[tuple[np.ndarray, str]], center_x: int, center_y: int,
        *, search_center: Optional[tuple[int, int]] = None, search_radius: Optional[int] = None,
        include_deadzone: bool = True,
    ) -> list[TargetMatch]:
        # Match every name at native scale first.  The slower scale fallback is
        # used only when the complete primary pass found nothing, just like the
        # original bot's two-tier scanner.
        matches: list[TargetMatch] = []
        for template, color in templates:
            matches.extend(self._find_one_target(
                frame, template, color, center_x, center_y,
                allow_fallback=False, search_center=search_center,
                search_radius=search_radius, include_deadzone=include_deadzone,
            ))
        if not matches:
            for template, color in templates:
                matches.extend(self._find_one_target(
                    frame, template, color, center_x, center_y,
                    allow_fallback=True, fallback_only=True,
                    search_center=search_center, search_radius=search_radius,
                    include_deadzone=include_deadzone,
                ))
        # White and purple versions can describe the same nametag.
        unique: list[TargetMatch] = []
        for candidate in sorted(matches, key=lambda item: item["distance"]):
            if any(abs(candidate["nx"] - item["nx"]) < 30 and abs(candidate["ny"] - item["ny"]) < 15 for item in unique):
                continue
            unique.append(candidate)
        # Distance from the detected player center is the primary rule.  A
        # coarse tile bucket can put a farther mob ahead of a closer one near
        # a bucket boundary, so it is kept only as diagnostic metadata.
        return sorted(unique, key=lambda item: (item["distance"], -item["score"]))

    def _find_one_target(
        self, frame: np.ndarray, template: np.ndarray, name_color: str, center_x: int, center_y: int,
        allow_fallback: bool = True, fallback_only: bool = False,
        search_center: Optional[tuple[int, int]] = None, search_radius: Optional[int] = None,
        include_deadzone: bool = True,
    ) -> list[TargetMatch]:
        cfg = self.config
        fh, fw = frame.shape[:2]
        top, bottom = int(fh * cfg.margin_top), int(fh * (1.0 - cfg.margin_bottom))
        left, right = int(fw * cfg.margin_left), int(fw * (1.0 - cfg.margin_right))
        scan_left, scan_top, scan_right, scan_bottom = left, top, fw, bottom
        if search_center is not None and search_radius is not None:
            search_x, search_y = search_center
            scan_left = max(scan_left, search_x - search_radius)
            scan_top = max(scan_top, search_y - search_radius)
            scan_right = min(scan_right, search_x + search_radius)
            scan_bottom = min(scan_bottom, search_y + search_radius)
        playfield = frame[scan_top:scan_bottom, scan_left:scan_right]
        if playfield.size == 0:
            return []

        processing_scale = min(1.0, cfg.target_scan_width / fw)
        mask_builder = self.purple_text_mask if name_color == "purple" else self.white_text_mask
        if self._target_mask_frame is not frame:
            self._target_mask_frame = frame
            self._target_masks = {}
        mask_key = (name_color, (scan_left, scan_top, scan_right, scan_bottom))
        text_mask = self._target_masks.get(mask_key)
        if text_mask is None:
            search_image = (
                cv2.resize(playfield, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_NEAREST)
                if processing_scale < 1.0 else playfield
            )
            text_mask = mask_builder(search_image)
            self._target_masks[mask_key] = text_mask
        if cv2.countNonZero(text_mask) == 0:
            return []

        matches: list[TargetMatch] = []

        def scan(scale: float) -> None:
            combined = scale * processing_scale
            cache_key = (id(template), name_color, combined)
            cached_template = self._target_template_masks.get(cache_key)
            scaled = cached_template[1] if cached_template is not None else None
            if scaled is None:
                scaled_color = (
                    cv2.resize(template, None, fx=combined, fy=combined, interpolation=cv2.INTER_NEAREST)
                    if combined != 1.0 else template
                )
                scaled = mask_builder(scaled_color)
                # Target templates are immutable during a run, so avoid
                # resizing and rebuilding their binary masks on every frame.
                if len(self._target_template_masks) >= 32:
                    self._target_template_masks.clear()
                self._target_template_masks[cache_key] = (template, scaled)
            th, tw = scaled.shape[:2]
            if th >= text_mask.shape[0] or tw >= text_mask.shape[1] or th < 3 or tw < 3:
                return
            result = cv2.matchTemplate(text_mask, scaled, cv2.TM_CCOEFF_NORMED)
            template_pixels = np.count_nonzero(scaled)
            for px, py in zip(*np.where(result >= cfg.match_threshold)[::-1]):
                patch = text_mask[py:py + th, px:px + tw]
                shared = np.count_nonzero(cv2.bitwise_and(patch, scaled))
                observed = cv2.countNonZero(patch)
                if (
                    template_pixels == 0 or observed == 0
                    or shared / template_pixels < cfg.min_name_template_overlap
                    or shared / observed < cfg.min_name_template_overlap
                ):
                    continue
                x = int(round(px / processing_scale)) + scan_left
                y = int(round(py / processing_scale)) + scan_top
                native_w = max(1, int(round(template.shape[1] * scale)))
                native_h = max(1, int(round(template.shape[0] * scale)))
                if any(abs(x - item["nx"]) < 30 and abs(y - item["ny"]) < 15 for item in matches):
                    continue
                click_x = x + native_w // 2
                click_y = y + native_h + int(cfg.mob_body_y_offset * scale)
                distance = float(np.hypot(click_x - center_x, click_y - center_y))
                if not include_deadzone and distance < cfg.player_deadzone_radius:
                    continue
                if left <= click_x <= right and top <= click_y <= bottom:
                    matches.append({
                        "nx": x, "ny": y, "nw": native_w, "nh": native_h,
                        "click_x": click_x, "click_y": click_y, "distance": distance,
                        "score": float(result[py, px]), "scale": scale,
                    })

        if not fallback_only:
            scan(1.0)
        if allow_fallback and not matches:
            for scale in (0.85, 1.15, 0.75, 1.25):
                scan(scale)
                if matches:
                    break
        return sorted(matches, key=lambda item: item["distance"])

    def pickup_position(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        if not self.templates:
            return None
        available = self.templates.get("pickup-available")
        unavailable = self.templates.get("pickup-not-yet")
        if available is None or unavailable is None:
            return None
        fh, fw = frame.shape[:2]
        left, right, top, bottom = int(fw * 0.82), fw, int(fh * 0.10), int(fh * 0.55)
        region = frame[top:bottom, left:right]
        best_available: tuple[float, Optional[tuple[int, int]]] = (-1.0, None)
        best_unavailable = -1.0
        for scale in (0.75, 0.9, 1.0, 1.1, 1.25):
            scaled = cv2.resize(available, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ah, aw = scaled.shape[:2]
            if ah >= region.shape[0] or aw >= region.shape[1]:
                continue
            _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED))
            if score > best_available[0]:
                best_available = (score, (left + location[0] + aw // 2, top + location[1] + ah // 2))
            scaled_unavailable = cv2.resize(unavailable, (aw, ah), interpolation=cv2.INTER_NEAREST)
            _, unavailable_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(region, scaled_unavailable, cv2.TM_CCOEFF_NORMED))
            best_unavailable = max(best_unavailable, unavailable_score)
        score, position = best_available
        return position if position is not None and score >= 0.86 and score >= best_unavailable + 0.08 else None

    def _marked_template(self) -> np.ndarray:
        if self._marked_template_image is not None:
            return self._marked_template_image
        directory = self.templates.directory if self.templates is not None else TEMPLATE_DIR
        path = directory / "marked.png"
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            self._marked_template_image = np.zeros((0, 0), dtype=np.uint8)
        else:
            difference = np.abs(image.astype(int) - [50, 50, 207])
            self._marked_template_image = (np.all(difference <= 20, axis=2)).astype(np.uint8) * 255
            if cv2.countNonZero(self._marked_template_image) == 0:
                self._marked_template_image = np.zeros((0, 0), dtype=np.uint8)
        return self._marked_template_image

    def _red_marker_mask(self, frame: np.ndarray) -> np.ndarray:
        if self._red_frame is frame and self._red_mask is not None:
            return self._red_mask
        mask = cv2.inRange(frame, (28, 28, 185), (72, 72, 229))
        fh, fw = frame.shape[:2]
        mask[int(fh * 0.75):, :int(fw * 0.18)] = 0
        mask[int(fh * 0.45):int(fh * 0.75), int(fw * 0.90):] = 0
        self._red_frame, self._red_mask = frame, mask
        return mask

    @staticmethod
    def _match_outline(
        mask: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.50,
        *,
        expected_center: Optional[tuple[int, int]] = None,
        anchor_radius: int = 0,
    ) -> Optional[tuple[int, int]]:
        th, tw = template.shape[:2]
        if th == 0 or tw == 0 or mask.shape[0] < th or mask.shape[1] < tw:
            return None
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        if expected_center is not None:
            expected_x = expected_center[0] - tw // 2
            expected_y = expected_center[1] - th // 2
            x1 = max(0, expected_x - anchor_radius)
            y1 = max(0, expected_y - anchor_radius)
            x2 = min(result.shape[1], expected_x + anchor_radius + 1)
            y2 = min(result.shape[0], expected_y + anchor_radius + 1)
            search = result[y1:y2, x1:x2]
            if search.size == 0:
                return None
            _, score, _, local_location = cv2.minMaxLoc(search)
            location = (x1 + local_location[0], y1 + local_location[1])
        else:
            _, score, _, location = cv2.minMaxLoc(result)
        if score < threshold:
            return None
        x, y = location
        visible = mask[y:y + th, x:x + tw] > 0
        border = template > 0
        border_pixels = max(1, int(np.count_nonzero(border)))
        border_hit = np.count_nonzero(visible & border) / border_pixels
        thickness = max(3, min(10, int(round(min(th, tw) * 0.07))))
        template_sides = (
            border[:thickness, thickness:-thickness], border[-thickness:, thickness:-thickness],
            border[thickness:-thickness, :thickness], border[thickness:-thickness, -thickness:],
        )
        visible_sides = (
            visible[:thickness, thickness:-thickness], visible[-thickness:, thickness:-thickness],
            visible[thickness:-thickness, :thickness], visible[thickness:-thickness, -thickness:],
        )
        hits = [np.count_nonzero(current & expected) / max(1, int(np.count_nonzero(expected))) for expected, current in zip(template_sides, visible_sides)]
        if expected_center is not None:
            # The clicked Skeleton anchors this local search, so accept a
            # strongly matching partial border when a sprite hides part of it.
            if border_hit < 0.35 or sum(hit >= 0.20 for hit in hits) < 2:
                return None
        else:
            # Global adoption has no target anchor and needs stronger evidence
            # from the exact fixed-size outline to reject red scene clutter.
            if border_hit < 0.55 or sum(hit >= 0.30 for hit in hits) < 3:
                return None
        return x + tw // 2, y + th // 2

    def red_square(
        self, frame: np.ndarray, target: Optional[tuple[int, int]] = None, radius: int = 260
    ) -> tuple[bool, Optional[tuple[int, int]]]:
        fh, fw = frame.shape[:2]
        top, bottom, left, right = int(fh * 0.13), int(fh * 0.85), int(fw * 0.08), int(fw * 0.90)
        template = self._marked_template()
        if template.size == 0:
            return False, None
        if target is not None:
            tx, ty = target
            x1, x2 = max(left, tx - radius), min(right, tx + radius)
            y1, y2 = max(top, ty - radius), min(bottom, ty + radius)
            th, tw = template.shape[:2]
            if x2 - x1 >= tw and y2 - y1 >= th:
                # An active target only needs a small crop. Building a red
                # mask for the whole 1600x900 frame delays the next click.
                local_mask = cv2.inRange(frame[y1:y2, x1:x2], (28, 28, 185), (72, 72, 229))
                center = self._match_outline(
                    local_mask,
                    template,
                    threshold=0.40,
                    expected_center=(tx - x1, ty - y1),
                    anchor_radius=max(40, int(radius * 0.55)),
                )
                if center is not None:
                    return True, (x1 + center[0], y1 + center[1])
            return False, None
        marker_mask = self._red_marker_mask(frame)
        playfield = marker_mask[top:bottom, left:right]
        if cv2.countNonZero(playfield) < 40:
            return False, None
        center = self._match_outline(playfield, template)
        return (True, (left + center[0], top + center[1])) if center is not None else (False, None)

    def exhausted_caption(self, frame: np.ndarray, template: Optional[np.ndarray]) -> bool:
        if frame is None or template is None:
            return False
        fh, fw = frame.shape[:2]
        if self._caption_cache is None or self._caption_cache[0] is not template:
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            low_red = cv2.inRange(hsv, (0, 110, 90), (10, 255, 255))
            high_red = cv2.inRange(hsv, (170, 110, 90), (179, 255, 255))
            mask = cv2.bitwise_or(low_red, high_red)
            points = cv2.findNonZero(mask)
            if points is None:
                return False
            x, y, width, height = cv2.boundingRect(points)
            mask = mask[y:y + height, x:x + width]
            density = cv2.countNonZero(mask) / (width * height)
            self._caption_cache = (template, width, height, density)
        _, template_w, template_h, density = self._caption_cache
        x0, x1, y0, y1 = int(fw * 0.05), int(fw * 0.95), int(fh * 0.50), int(fh * 0.85)
        region = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, (0, 110, 90), (10, 255, 255))
        high_red = cv2.inRange(hsv, (170, 110, 90), (179, 255, 255))
        red_mask = cv2.bitwise_or(low_red, high_red)
        screen_scale = fw / 1600.0
        processing_scale = min(0.5, 800.0 / max(1, region.shape[1]))
        if processing_scale < 1.0:
            red_mask = cv2.resize(red_mask, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_NEAREST)
        close_width = max(5, int(round(15 * screen_scale * processing_scale)))
        joined = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3, close_width), dtype=np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        expected_w, expected_h = template_w * screen_scale * processing_scale, template_h * screen_scale * processing_scale
        min_w, max_w, min_h, max_h = expected_w * 0.50, expected_w * 1.20, expected_h * 0.45, expected_h * 1.50
        min_aspect, max_aspect = (template_w / template_h) * 0.55, (template_w / template_h) * 1.70
        min_density, max_density = density * 0.45, min(0.78, density * 1.80)
        for x, y, width, height, _ in stats[1:count]:
            if not (min_w <= width <= max_w and min_h <= height <= max_h):
                continue
            aspect = width / max(1, height)
            if not (min_aspect <= aspect <= max_aspect):
                continue
            area_density = cv2.countNonZero(red_mask[y:y + height, x:x + width]) / (width * height)
            if min_density <= area_density <= max_density:
                return True
        return False
