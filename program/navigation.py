"""Minimap route registration and travel logic."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig
from .device import InputDevice
from .models import RetryContext, TargetMatch
from .vision import FrameVision, TemplateStore

@dataclass
class MinimapRoute:
    signature: tuple[Any, ...]
    waypoints: tuple[tuple[int, int], ...]
    index: int


class MinimapNavigator:
    """Register the visible map against the immutable full-map reference."""

    def __init__(
        self, config: BotConfig, vision: FrameVision, device: InputDevice,
        templates: TemplateStore, minimap_template_key: str,
    ):
        self.config = config
        self.vision = vision
        self.device = device
        self.templates = templates
        self.minimap_template_key = minimap_template_key
        self.safe_mask: Optional[np.ndarray] = None
        self.waypoints: list[tuple[int, int]] = []
        self.reference_id: Optional[int] = None
        self.route: Optional[MinimapRoute] = None
        self.last_route_target: Optional[tuple[int, int]] = None

    @staticmethod
    def _build_safe_mask(reference: np.ndarray) -> np.ndarray:
        walls = FrameVision.minimap_white_mask(reference)
        contours, hierarchy = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(walls.shape, dtype=np.uint8)
        outer_index = max(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]))
        outer_area = cv2.contourArea(contours[outer_index])
        # Some older reference captures contain disconnected white room outlines rather than one enclosing boundary.
        # In that layout the largest contour is only a small room, so filling it produces an empty/near-empty
        # route mask and every destination is rejected. Use the linework bounding box as the traversable map,
        # then remove the walls below.
        image_area = walls.shape[0] * walls.shape[1]
        if outer_area < image_area * 0.20:
            ys, xs = np.where(walls > 0)
            if len(xs) == 0:
                return np.zeros(walls.shape, dtype=np.uint8)
            pad = 12
            x1, x2 = max(0, int(xs.min()) - pad), min(walls.shape[1], int(xs.max()) + pad + 1)
            y1, y2 = max(0, int(ys.min()) - pad), min(walls.shape[0], int(ys.max()) + pad + 1)
            safe = np.zeros(walls.shape, dtype=np.uint8)
            safe[y1:y2, x1:x2] = 255
        else:
            safe = np.zeros(walls.shape, dtype=np.uint8)
            cv2.drawContours(safe, [contours[outer_index]], -1, 255, thickness=-1)
        blocked = cv2.dilate(walls, np.ones((9, 9), np.uint8), iterations=1)
        if outer_area >= image_area * 0.20 and hierarchy is not None:
            for index, contour in enumerate(contours):
                parent = int(hierarchy[0][index][3])
                area = cv2.contourArea(contour)
                if parent < 0 or area < 1000 or area >= outer_area * 0.5:
                    continue
                cv2.drawContours(blocked, [contour], -1, 255, thickness=-1)
        safe[blocked > 0] = 0
        return cv2.erode(safe, np.ones((5, 5), np.uint8), iterations=1)

    @staticmethod
    def build_waypoints(safe_mask: np.ndarray, grid_step: int = 120) -> list[tuple[int, int]]:
        if safe_mask is None or safe_mask.size == 0:
            return []
        height, width = safe_mask.shape[:2]
        waypoints: list[tuple[int, int]] = []
        radius, half_step = max(12, grid_step // 3), max(1, grid_step // 2)
        # Sweep columns instead of rows.  The reference contains long
        # horizontal rooms; a row-first sweep can keep the character moving
        # sideways for many taps before it changes level.  Alternating column
        # direction still covers the whole safe area while forcing regular
        # vertical progress.
        for column, x in enumerate(range(half_step, width, grid_step)):
            column_points: list[tuple[int, int]] = []
            for y in range(half_step, height, grid_step):
                x1, y1, x2, y2 = max(0, x - radius), max(0, y - radius), min(width, x + radius + 1), min(height, y + radius + 1)
                local = safe_mask[y1:y2, x1:x2]
                local_y, local_x = np.where(local > 0)
                if len(local_x) == 0:
                    continue
                distances = (local_x + x1 - x) ** 2 + (local_y + y1 - y) ** 2
                nearest = int(np.argmin(distances))
                column_points.append((int(local_x[nearest] + x1), int(local_y[nearest] + y1)))
            if column % 2:
                column_points.reverse()
            for point in column_points:
                if not waypoints or point != waypoints[-1]:
                    waypoints.append(point)
        return waypoints

    @staticmethod
    def _geometry_score(frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]) -> float:
        fh, fw = frame_mask.shape[:2]
        sh, sw = scaled_reference.shape[:2]
        ox, oy = origin
        x1, y1, x2, y2 = max(0, ox), max(0, oy), min(fw, ox + sw), min(fh, oy + sh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        expected = (scaled_reference[y1 - oy:y2 - oy, x1 - ox:x2 - ox] > 0).astype(np.uint8)
        observed = (frame_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        if cv2.countNonZero(expected) < 3000 or cv2.countNonZero(observed) < 3000:
            return 0.0
        kernel = np.ones((7, 7), np.uint8)
        expected_near, observed_near = cv2.dilate(expected, kernel, iterations=1) > 0, cv2.dilate(observed, kernel, iterations=1) > 0
        recall = float(np.count_nonzero(expected & observed_near)) / max(1, np.count_nonzero(expected))
        precision = float(np.count_nonzero(observed & expected_near)) / max(1, np.count_nonzero(observed))
        return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0

    @staticmethod
    def _visible_reference_pixels(
        frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]
    ) -> tuple[int, int]:
        """Return expected white pixels visible in the frame and total reference pixels.

        A partial view is normal because the full route is larger than the
        1600x900 game viewport.  This metric lets us reject tiny repeated
        patches without requiring the whole reference to be on screen.
        """
        fh, fw = frame_mask.shape[:2]
        sh, sw = scaled_reference.shape[:2]
        ox, oy = origin
        x1, y1, x2, y2 = max(0, ox), max(0, oy), min(fw, ox + sw), min(fh, oy + sh)
        if x2 <= x1 or y2 <= y1:
            return 0, int(cv2.countNonZero(scaled_reference))
        visible = scaled_reference[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
        return int(cv2.countNonZero(visible)), int(cv2.countNonZero(scaled_reference))

    def _refine_y(self, frame_mask: np.ndarray, scaled_reference: np.ndarray, origin: tuple[int, int]) -> tuple[tuple[int, int], float]:
        ox, oy = origin
        def candidate(offset_y: int) -> tuple[float, int]:
            candidate_origin = (ox, offset_y)
            visible, _ = self._visible_reference_pixels(frame_mask, scaled_reference, candidate_origin)
            if visible < self.config.minimap_min_visible_reference_pixels:
                return 0.0, offset_y
            return self._geometry_score(frame_mask, scaled_reference, candidate_origin), offset_y

        coarse = [candidate(oy + offset) for offset in range(-360, 361, 16)]
        _, best_y = max(coarse)
        fine = [candidate(candidate_y) for candidate_y in range(best_y - 16, best_y + 17, 2)]
        geometry, refined_y = max(fine)
        return (ox, refined_y), geometry

    def align(self, frame: np.ndarray, reference: np.ndarray) -> Optional[dict[str, Any]]:
        frame_mask = self.vision.minimap_white_mask(frame)
        if cv2.countNonZero(frame_mask) < 12000:
            return None
        downsample = 0.25
        search = cv2.resize(frame_mask, None, fx=downsample, fy=downsample, interpolation=cv2.INTER_NEAREST)
        reference_mask = self.vision.minimap_white_mask(reference)
        patch_w, patch_h = min(180, search.shape[1] - 4), min(180, search.shape[0] - 4)
        if patch_w < 120 or patch_h < 120:
            return None
        raw: list[tuple[float, float, tuple[int, int]]] = []
        for scale in (0.92, 0.96, 1.0, 1.04, 1.08):
            scaled = cv2.resize(reference_mask, None, fx=scale * downsample, fy=scale * downsample, interpolation=cv2.INTER_NEAREST)
            scaled_h, scaled_w = scaled.shape[:2]
            if scaled_w < patch_w or scaled_h < patch_h:
                continue
            x_starts = sorted({0, max(0, (scaled_w - patch_w) // 2), max(0, scaled_w - patch_w)})
            y_remaining = max(0, scaled_h - patch_h)
            y_starts = sorted({int(round(y_remaining * fraction)) for fraction in (0.0, 0.5, 1.0)})
            for crop_y in y_starts:
                for crop_x in x_starts:
                    patch = scaled[crop_y:crop_y + patch_h, crop_x:crop_x + patch_w]
                    if cv2.countNonZero(patch) < 3000:
                        continue
                    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
                    origin = (int(round((location[0] - crop_x) / downsample)), int(round((location[1] - crop_y) / downsample)))
                    raw.append((float(score), float(scale), origin))
        if not raw:
            return None
        raw.sort(key=lambda item: item[0], reverse=True)
        candidates: list[tuple[float, float, tuple[int, int], float]] = []
        for score, scale, origin in raw:
            if any(
                abs(origin[0] - kept[2][0]) <= 10
                and abs(origin[1] - kept[2][1]) <= 10
                and abs(scale - kept[1]) <= 0.045
                for kept in candidates
            ):
                continue
            scaled_reference = cv2.resize(reference_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            geometry = self._geometry_score(frame_mask, scaled_reference, origin)
            visible_pixels, total_reference_pixels = self._visible_reference_pixels(frame_mask, scaled_reference, origin)
            visible_fraction = visible_pixels / max(1, total_reference_pixels)
            if (
                geometry > 0
                and visible_pixels >= self.config.minimap_min_visible_reference_pixels
                and visible_fraction >= self.config.minimap_min_visible_reference_fraction
            ):
                candidates.append((score, scale, origin, geometry))
                if len(candidates) >= 8:
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[3], item[0]), reverse=True)
        refined_candidates: list[tuple[float, float, tuple[int, int], float]] = []
        for candidate_score, candidate_scale, candidate_origin, _ in candidates[:2]:
            candidate_reference = cv2.resize(
                reference_mask, None, fx=candidate_scale, fy=candidate_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            refined_origin, refined_geometry = self._refine_y(
                frame_mask, candidate_reference, candidate_origin
            )
            refined_candidates.append((
                candidate_score, candidate_scale, refined_origin, refined_geometry
            ))
        refined_candidates.sort(key=lambda item: (item[3], item[0]), reverse=True)
        score, scale, origin, geometry = refined_candidates[0]
        second_geometry = refined_candidates[1][3] if len(refined_candidates) > 1 else 0.0
        scaled_reference = cv2.resize(
            reference_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        gap = geometry - second_geometry
        visible_pixels, total_reference_pixels = self._visible_reference_pixels(frame_mask, scaled_reference, origin)
        visible_fraction = visible_pixels / max(1, total_reference_pixels)
        # `score` is kept for diagnostics, but it compares screenshots whose
        # backgrounds can differ dramatically.  Geometry plus coverage is the
        # authoritative test for the white route linework.
        valid = (
            geometry >= self.config.minimap_geometry_min_score
            and gap >= self.config.minimap_align_min_gap
            and visible_pixels >= self.config.minimap_min_visible_reference_pixels
            and visible_fraction >= self.config.minimap_min_visible_reference_fraction
        )
        result = {
            "valid": valid,
            "score": score,
            "geometry": geometry,
            "gap": gap,
            "coverage": visible_fraction,
            "visible_pixels": visible_pixels,
            "scale": scale,
            "origin": origin,
        }
        return result

    def _best_alignment(
        self, frame: np.ndarray, references: list[tuple[str, np.ndarray]]
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Try every known map layout and keep the strongest registration."""
        best: Optional[dict[str, Any]] = None
        best_name: Optional[str] = None
        for name, reference in references:
            candidate = self.align(frame, reference)
            if candidate is None:
                continue
            candidate_valid = bool(candidate.get("valid"))
            best_valid = bool(best and best.get("valid"))
            candidate_key = (
                candidate_valid,
                float(candidate.get("geometry", 0.0)),
                float(candidate.get("coverage", 0.0)),
                float(candidate.get("gap", 0.0)),
            )
            best_key = (
                best_valid,
                float(best.get("geometry", 0.0)) if best else 0.0,
                float(best.get("coverage", 0.0)) if best else 0.0,
                float(best.get("gap", 0.0)) if best else 0.0,
            )
            if best is None or candidate_key > best_key:
                best, best_name = candidate, name
        return best, best_name

    @staticmethod
    def player_marker(frame: np.ndarray) -> Optional[tuple[int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        marker = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 100)).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(marker, 8)
        fh, fw = frame.shape[:2]
        center = np.array([fw / 2.0, fh / 2.0])
        choices: list[tuple[float, tuple[int, int]]] = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if not (200 <= area <= 3000 and 14 <= w <= 100 and 14 <= h <= 100):
                continue
            point = centroids[index]
            distance = float(np.linalg.norm(point - center))
            if distance <= min(fw, fh) * 0.18:
                choices.append((distance, (int(round(point[0])), int(round(point[1])))))
        return min(choices)[1] if choices else None

    def destination(
        self,
        frame: np.ndarray,
        alignment: dict[str, Any],
        safe_mask: np.ndarray,
        player: tuple[int, int],
        waypoints: list[tuple[int, int]],
        farthest: bool = False,
    ) -> Optional[tuple[int, int]]:
        if not alignment or not alignment.get("valid") or safe_mask is None or player is None or not waypoints:
            return None
        scale, (ox, oy) = alignment["scale"], alignment["origin"]
        player_x, player_y = player
        map_x, map_y = (player_x - ox) / scale, (player_y - oy) / scale
        ref_h, ref_w = safe_mask.shape[:2]
        if not (0 <= map_x < ref_w and 0 <= map_y < ref_h and safe_mask[int(map_y), int(map_x)] > 0):
            return None
        step = 8
        ys, xs = np.where(safe_mask[::step, ::step] > 0)
        if len(xs) == 0:
            return None
        ref_xs, ref_ys = xs * step + step // 2, ys * step + step // 2
        screen_xs, screen_ys = ox + ref_xs * scale, oy + ref_ys * scale
        fh, fw = frame.shape[:2]
        margin_x, margin_y = fw * (1.0 - self.config.minimap_safe_view_fraction) / 2.0, fh * (1.0 - self.config.minimap_safe_view_fraction) / 2.0
        visible = ((screen_xs >= margin_x) & (screen_xs <= fw - margin_x) & (screen_ys >= margin_y) & (screen_ys <= fh - margin_y))
        distances = np.hypot(screen_xs - player_x, screen_ys - player_y)
        visible &= distances >= self.config.minimap_min_click_distance * fw / 1600.0
        if not farthest:
            visible &= distances <= self.config.minimap_max_click_distance * fw / 1600.0
        if not np.any(visible):
            return None

        signature = (safe_mask.shape, len(waypoints), waypoints[0], waypoints[-1])
        points = np.asarray(waypoints, dtype=np.float32)
        if self.route is None or self.route.signature != signature:
            nearest = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
            self.route = MinimapRoute(signature, tuple(waypoints), nearest)
        else:
            # Camera movement can carry the player a long way from the old
            # route cursor, especially after a failsafe jump.  Re-anchor it
            # instead of continuing a stale same-row sweep.
            current_index = int(self.route.index)
            if np.hypot(points[current_index, 0] - map_x, points[current_index, 1] - map_y) > max(150.0, self.config.minimap_max_click_distance):
                self.route.index = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
        if farthest:
            index = int(np.argmax(np.where(visible, distances, -np.inf)))
            chosen_ref = np.array([ref_xs[index], ref_ys[index]], dtype=np.float32)
            self.route.index = int(np.argmin(np.hypot(points[:, 0] - chosen_ref[0], points[:, 1] - chosen_ref[1])))
            self.last_route_target = (int(round(ref_xs[index])), int(round(ref_ys[index])))
            return int(round(screen_xs[index])), int(round(screen_ys[index]))
        points = np.asarray(self.route.waypoints, dtype=np.float32)
        current = int(self.route.index)
        target_index = (current + 1) % len(points)
        target_x, target_y = points[target_index]
        if np.hypot(target_x - map_x, target_y - map_y) < max(70.0, self.config.minimap_min_click_distance / max(scale, 0.1)):
            current = target_index
            self.route.index = current
            target_index = (current + 1) % len(points)
            target_x, target_y = points[target_index]
        before = np.hypot(target_x - map_x, target_y - map_y)
        after = np.hypot(target_x - ref_xs, target_y - ref_ys)
        preferred = (self.config.minimap_min_click_distance + self.config.minimap_max_click_distance) * 0.5 * fw / 1600.0
        scores = before - after
        scores = scores * 1000.0 - np.abs(distances - preferred)
        scores[~visible] = -np.inf
        index = int(np.argmax(scores))
        self.last_route_target = (int(round(target_x)), int(round(target_y)))
        return int(round(screen_xs[index])), int(round(screen_ys[index]))

    def travel(self, *, farthest: bool = False, keep_open_seconds: float = 0.0) -> bool:
        reference_image = self.templates.get(self.minimap_template_key)
        references = [(self.minimap_template_key, reference_image)] if reference_image is not None else []
        if not references:
            print(f"[!] Minimap travel disabled: reference '{self.minimap_template_key}' is missing.")
            return False
        frame = self.device.capture()
        if frame is None:
            return False
        state, icon = self.vision.ui_state(frame)
        alignment: Optional[dict[str, Any]] = None
        reference_name: Optional[str] = None
        if state == "combat":
            if icon is None or not self.device.click(*icon):
                return False
            # The map panel fades in over several frames.  Aligning the first
            # frame that reports the Back icon often sees a partially drawn
            # overlay and rejects an otherwise valid map.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                time.sleep(0.06)
                frame = self.device.capture()
                if frame is None:
                    continue
                state, icon = self.vision.ui_state(frame)
                if state == "disconnected":
                    break
                if state == "minimap":
                    candidate, candidate_name = self._best_alignment(frame, references)
                    if candidate is not None and candidate.get("valid"):
                        alignment = candidate
                        reference_name = candidate_name
                        break
        if state != "minimap":
            print(f"[!] Minimap travel skipped: UI state is {state}.")
            return False
        if alignment is None:
            alignment, reference_name = self._best_alignment(frame, references)
        if not alignment or not alignment.get("valid"):
            details = alignment or {}
            return self._fail(icon, f"[!] Minimap alignment rejected (match {details.get('score', 0.0):.2f}, geometry {details.get('geometry', 0.0):.2f}, coverage {details.get('coverage', 0.0):.2f}, gap {details.get('gap', 0.0):.2f}); no movement tap sent.")
        reference = self.templates.get(reference_name) if reference_name is not None else None
        if reference is None:
            return self._fail(icon, "[!] Minimap reference disappeared during registration; no movement tap sent.")
        player = self.player_marker(frame)
        if player is None:
            return self._fail(icon, "[!] Minimap player marker was not found; no movement tap sent.")
        if self.reference_id != id(reference):
            self.safe_mask = self._build_safe_mask(reference)
            self.waypoints = self.build_waypoints(self.safe_mask)
            self.reference_id = id(reference)
        destination = self.destination(frame, alignment, self.safe_mask, player, self.waypoints, farthest=farthest)
        if destination is None:
            return self._fail(icon, "[!] No safe visible minimap waypoint was found; no movement tap sent.")
        mode = "farthest visible" if farthest else "route"
        print(f"[~] Minimap {mode} waypoint {destination} on {reference_name} from player {player} toward reference {self.last_route_target or '?'} (match {alignment['score']:.2f}, geometry {alignment['geometry']:.2f}, coverage {alignment['coverage']:.2f}, gap {alignment['gap']:.2f}, scale {alignment['scale']:.2f}).")
        sent = self.device.click(*destination, synchronous=True)
        if sent:
            # Let the game register the map tap before closing the panel.
            time.sleep(0.30)
            if keep_open_seconds > 0:
                print(f"[~] Keeping minimap open for {keep_open_seconds:.0f}s after movement tap.")
                deadline = time.monotonic() + keep_open_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
                # The game can still be applying the map tap when the hold
                # expires.  Give that transition a moment to settle before
                # sending the Back action.
                time.sleep(0.8)
        closed = self._close_map(icon)
        if not closed and keep_open_seconds > 0:
            # A late UI frame can leave the first Back verification stale.
            # Re-match the live icon and make one final delayed close attempt.
            time.sleep(0.8)
            latest = self.device.capture()
            retry_icon = self.vision.back_icon(latest) if latest is not None else icon
            closed = self._close_map(retry_icon)
        return bool(sent and closed)

    def _fail(self, icon: Optional[tuple[int, int]], message: str) -> bool:
        print(message)
        self._close_map(icon)
        return False

    def _close_map(self, icon: Optional[tuple[int, int]]) -> bool:
        back_template = self.templates.get("backicon")

        def panel_closed(frame: np.ndarray) -> bool:
            if back_template is None:
                return self.vision.ui_state(frame)[0] == "combat"
            return self.vision.find_fixed_icon(
                frame, back_template, self.config.back_icon_center,
                self.config.back_icon_match_threshold, search_radius=(35, 30),
                scales=self.config.back_icon_scales,
            ) is None

        if icon is None:
            self.device.back()
            time.sleep(0.45)
            frame = self.device.capture()
            if frame is not None and panel_closed(frame):
                return True
            print("[!] System Back did not close the panel; no second Back action sent.")
            return False
        else:
            for attempt in range(2):
                if not self.device.click(*icon, synchronous=True):
                    print(f"[!] Back-icon tap {attempt + 1} failed.")
                time.sleep(0.45)
                frame = self.device.capture()
                if frame is None:
                    print("[!] Could not verify the panel closed after tapping Back.")
                    break
                if panel_closed(frame):
                    return True
                if back_template is None:
                    print("[!] Back icon template is unavailable; trying the system Back action.")
                    break
                refreshed = self.vision.find_fixed_icon(
                    frame, back_template, self.config.back_icon_center,
                    self.config.back_icon_match_threshold, search_radius=(35, 30),
                    scales=self.config.back_icon_scales,
                )
                if attempt == 0:
                    icon = refreshed
                    print(f"[!] Back icon remains visible at {icon}; retrying its fresh match.")
                else:
                    print("[!] Back icon remains visible after retry; trying the system Back action.")
        self.device.back()
        time.sleep(0.45)
        frame = self.device.capture()
        if frame is not None and panel_closed(frame):
            return True
        print("[!] Panel is still open after Back-icon and system Back attempts.")
        return False


# ---------------------------------------------------------------------------
# Combat state machine
# ---------------------------------------------------------------------------
