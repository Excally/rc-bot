"""Minimap route registration and travel logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import BotConfig
from .device import InputDevice
from .vision import FrameVision, TemplateStore

@dataclass
class MinimapRoute:
    signature: tuple[Any, ...]
    waypoints: tuple[tuple[float, float], ...]
    index: int
    last_player_position: Optional[tuple[float, float]] = None
    last_target_index: Optional[int] = None
    last_target_position: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class CellMap:
    """A color-coded map reduced to exact, uniform logical cells."""

    walls: np.ndarray
    context_walls: np.ndarray
    unwalkable: np.ndarray
    annotations: np.ndarray
    walkable: np.ndarray
    safe_mask: np.ndarray
    cell_pixels: int = 21

    @property
    def match_walls(self) -> np.ndarray:
        """Both wall layers help registration; only the black path is navigable."""
        return self.walls | self.context_walls


MAP_CELL_PIXELS = 21


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
        self.waypoints: list[tuple[float, float]] = []
        self.cell_map: Optional[CellMap] = None
        self.reference_id: Optional[int] = None
        self.route: Optional[MinimapRoute] = None
        self.last_route_target: Optional[tuple[int, int]] = None

    @staticmethod
    def _cell_labels(
        reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read the skeleton's exact 21px annotation palette.

        The reference uses black for the walkable path, gray for unwalkable
        terrain, white for this map's walls, and red for context walls from a
        neighboring map. Both wall layers aid image registration, while only
        black cells may be destinations. Yellow markers remain blocked.
        These annotation colors describe the reference; the live minimap is
        still recognized from its grayscale appearance.
        """
        if reference is None or reference.ndim != 3 or reference.shape[2] < 3:
            raise ValueError("The minimap template must be a BGR color image.")
        height, width = reference.shape[:2]
        if height % MAP_CELL_PIXELS or width % MAP_CELL_PIXELS:
            raise ValueError(
                f"The minimap template must use {MAP_CELL_PIXELS}x{MAP_CELL_PIXELS} pixel cells."
            )
        cell_blocks = reference.reshape(
            height // MAP_CELL_PIXELS, MAP_CELL_PIXELS,
            width // MAP_CELL_PIXELS, MAP_CELL_PIXELS, 3,
        )
        cell_colors = cell_blocks[:, MAP_CELL_PIXELS // 2, :, MAP_CELL_PIXELS // 2, :]
        uniform = np.all(
            cell_blocks == cell_colors[:, None, :, None, :], axis=(1, 3, 4)
        )
        if not np.all(uniform):
            bad = int(uniform.size - np.count_nonzero(uniform))
            raise ValueError(
                f"The minimap template has {bad} antialiased or non-uniform cells; "
                "each map cell must be a solid 21x21 color block."
            )

        # OpenCV stores colors in BGR order. Keep this palette exact so a new
        # annotation color cannot silently become a routeable floor cell.
        walls = np.all(cell_colors == (243, 243, 243), axis=2)
        context_walls = np.all(cell_colors == (0, 0, 255), axis=2)
        unwalkable = np.all(cell_colors == (132, 132, 132), axis=2)
        walkable = np.all(cell_colors == (0, 0, 0), axis=2)
        annotations = np.all(cell_colors == (0, 255, 234), axis=2)
        recognized = walls | context_walls | unwalkable | walkable | annotations
        if not np.all(recognized):
            raise ValueError("The minimap template contains cells outside its documented color palette.")
        return walls, context_walls, unwalkable, annotations, walkable

    @classmethod
    def _build_cell_map(cls, reference: np.ndarray) -> CellMap:
        walls, context_walls, unwalkable, annotations, walkable = cls._cell_labels(reference)
        if not np.any(walkable):
            raise ValueError("The minimap template contains no explicitly walkable black cells.")
        walkable_u8 = walkable.astype(np.uint8) * 255
        # Only black cells are routeable. Keep clicks a cell away from every
        # nonwalkable class, including gray terrain and both wall layers.
        blocked = (~walkable).astype(np.uint8) * 255
        clearance = cv2.dilate(blocked, np.ones((3, 3), np.uint8), iterations=1)
        safe = walkable_u8.copy()
        safe[clearance > 0] = 0
        if cv2.countNonZero(safe) == 0:
            raise ValueError("The minimap template contains no safe floor cells.")
        return CellMap(walls, context_walls, unwalkable, annotations, walkable_u8, safe)

    @staticmethod
    def build_waypoints(safe_mask: np.ndarray, grid_step: int = 6) -> list[tuple[float, float]]:
        """Create a map-cell sweep independent of the image's pixel dimensions."""
        if safe_mask is None or safe_mask.size == 0:
            return []
        height, width = safe_mask.shape[:2]
        waypoints: list[tuple[float, float]] = []
        radius = max(1, grid_step // 3)
        half_step = max(1, grid_step // 2)
        for column, x in enumerate(range(half_step, width, grid_step)):
            column_points: list[tuple[float, float]] = []
            for y in range(half_step, height, grid_step):
                x1, y1 = max(0, x - radius), max(0, y - radius)
                x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
                local_y, local_x = np.where(safe_mask[y1:y2, x1:x2] > 0)
                if not len(local_x):
                    continue
                distances = (local_x + x1 - x) ** 2 + (local_y + y1 - y) ** 2
                nearest = int(np.argmin(distances))
                column_points.append((
                    float(local_x[nearest] + x1) + 0.5,
                    float(local_y[nearest] + y1) + 0.5,
                ))
            if column % 2:
                column_points.reverse()
            for point in column_points:
                if not waypoints or point != waypoints[-1]:
                    waypoints.append(point)
        return waypoints

    @staticmethod
    def _screen_cell_grid(
        integral: np.ndarray, frame_shape: tuple[int, int], cell_px: float,
        phase_x: float, phase_y: float,
    ) -> np.ndarray:
        """Sample white occupancy once per logical cell using an integral image."""
        height, width = frame_shape
        cols = max(0, int(np.floor((width - phase_x) / cell_px)))
        rows = max(0, int(np.floor((height - phase_y) / cell_px)))
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=np.uint8)
        sample = max(3, int(round(cell_px * 0.72)))
        centers_x = phase_x + (np.arange(cols, dtype=np.float32) + 0.5) * cell_px
        centers_y = phase_y + (np.arange(rows, dtype=np.float32) + 0.5) * cell_px
        x1 = np.rint(centers_x - sample / 2).astype(np.int32)
        y1 = np.rint(centers_y - sample / 2).astype(np.int32)
        x2, y2 = x1 + sample, y1 + sample
        x1c, x2c = np.clip(x1, 0, width), np.clip(x2, 0, width)
        y1c, y2c = np.clip(y1, 0, height), np.clip(y2, 0, height)
        sums = (
            integral[y2c[:, None], x2c[None, :]]
            - integral[y1c[:, None], x2c[None, :]]
            - integral[y2c[:, None], x1c[None, :]]
            + integral[y1c[:, None], x1c[None, :]]
        )
        return (sums >= sample * sample * 0.30).astype(np.uint8)

    @staticmethod
    def _score_origin(
        integral: np.ndarray, frame_shape: tuple[int, int], cell_map: CellMap,
        cell_px: float, origin: tuple[float, float],
    ) -> tuple[float, float, int]:
        """Compare map-wall cells with a candidate screen placement."""
        frame_h, frame_w = frame_shape
        map_h, map_w = cell_map.walls.shape
        sample = max(3, int(round(cell_px * 0.72)))
        xs = np.rint(origin[0] + (np.arange(map_w) + 0.5) * cell_px).astype(np.int32)
        ys = np.rint(origin[1] + (np.arange(map_h) + 0.5) * cell_px).astype(np.int32)
        x1, y1 = xs - sample // 2, ys - sample // 2
        x2, y2 = x1 + sample, y1 + sample
        visible_x = (x1 >= 0) & (x2 <= frame_w)
        visible_y = (y1 >= 0) & (y2 <= frame_h)
        visible = visible_y[:, None] & visible_x[None, :]
        if not np.any(visible):
            return 0.0, 0.0, 0
        x1c, x2c = np.clip(x1, 0, frame_w), np.clip(x2, 0, frame_w)
        y1c, y2c = np.clip(y1, 0, frame_h), np.clip(y2, 0, frame_h)
        sums = (
            integral[y2c[:, None], x2c[None, :]]
            - integral[y1c[:, None], x2c[None, :]]
            - integral[y2c[:, None], x1c[None, :]]
            + integral[y1c[:, None], x1c[None, :]]
        )
        observed = sums >= sample * sample * 0.30
        match_walls = cell_map.match_walls
        expected = match_walls & visible
        expected_count = int(np.count_nonzero(expected))
        total_walls = max(1, int(np.count_nonzero(match_walls)))
        if expected_count == 0:
            return 0.0, 0.0, 0
        hits = int(np.count_nonzero(expected & observed))
        precision_area = visible & ~cell_map.annotations
        observed_count = int(np.count_nonzero(observed & precision_area))
        recall = hits / expected_count
        precision = hits / max(1, observed_count)
        geometry = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
        return geometry, expected_count / total_walls, expected_count

    def align(self, frame: np.ndarray, reference: np.ndarray) -> Optional[dict[str, Any]]:
        """Register both annotated wall layers against bright live minimap walls.

        Matching runs on a small logical-cell grid, so frame resolution does
        not multiply template work. Gray unwalkable cells and yellow markers
        do not vote as wall features. Annotation colors are never expected in
        the live frame.
        """
        if self.reference_id == id(reference) and self.cell_map is not None:
            cell_map = self.cell_map
        else:
            try:
                cell_map = self._build_cell_map(reference)
            except ValueError as error:
                print(f"[!] Minimap grid rejected: {error}")
                return None
            self.cell_map = cell_map
            self.safe_mask = cell_map.safe_mask
            self.waypoints = self.build_waypoints(cell_map.safe_mask)
            self.reference_id = id(reference)
        white_mask = cv2.inRange(frame, (220, 220, 220), (255, 255, 255))
        if cv2.countNonZero(white_mask) < 120:
            return None
        integral = cv2.integral(white_mask // 255, sdepth=cv2.CV_32S)
        frame_h, frame_w = frame.shape[:2]
        match_walls = cell_map.match_walls
        map_h, map_w = match_walls.shape
        patch_h = min(28, map_h - 2)
        patch_w = min(32, map_w - 2)
        if patch_h < 12 or patch_w < 12:
            return None

        # Distributed patches cast translation votes. Several separated
        # patches must agree, which rejects repeated nameplates and HUD tiles.
        patch_positions = sorted({
            (x, y)
            for y in (0, max(0, (map_h - patch_h) // 2), map_h - patch_h)
            for x in (0, max(0, (map_w - patch_w) // 2), map_w - patch_w)
        })
        patches: list[tuple[int, int, np.ndarray]] = []
        for x, y in patch_positions:
            patch = match_walls[y:y + patch_h, x:x + patch_w].astype(np.float32)
            wall_count = int(np.count_nonzero(patch))
            if wall_count >= 10 and wall_count <= patch.size * 0.55:
                patches.append((x, y, patch))
        if not patches:
            return None

        # The game and the authored skeleton map both use exactly 21 pixels
        # per cell; keeping that fixed avoids unnecessary scale searches.
        scales = (float(MAP_CELL_PIXELS),)
        votes: dict[tuple[int, int, int], list[float]] = {}
        candidate_origins: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        for scale_index, cell_px in enumerate(scales):
            phase_values = (cell_px * 0.20, cell_px * 0.70)
            for phase_y in phase_values:
                for phase_x in phase_values:
                    grid = self._screen_cell_grid(
                        integral, (frame_h, frame_w), cell_px, phase_x, phase_y
                    )
                    if grid.shape[0] < patch_h or grid.shape[1] < patch_w:
                        continue
                    for ref_x, ref_y, patch in patches:
                        response = cv2.matchTemplate(
                            grid.astype(np.float32), patch, cv2.TM_CCOEFF_NORMED
                        )
                        for _ in range(1):
                            _, score, _, location = cv2.minMaxLoc(response)
                            if not np.isfinite(score):
                                break
                            origin_x = phase_x + (location[0] - ref_x) * cell_px
                            origin_y = phase_y + (location[1] - ref_y) * cell_px
                            bin_size = max(5.0, cell_px / 3.0)
                            key = (
                                scale_index,
                                int(round(origin_x / bin_size)),
                                int(round(origin_y / bin_size)),
                            )
                            votes.setdefault(key, []).append(float(score))
                            candidate_origins[key] = (origin_x, origin_y, cell_px)
                            lx, ly = location
                            x1, x2 = max(0, lx - 3), min(response.shape[1], lx + 4)
                            y1, y2 = max(0, ly - 3), min(response.shape[0], ly + 4)
                            response[y1:y2, x1:x2] = -1.0

        if not votes:
            return None
        ranked = sorted(
            votes,
            key=lambda key: (len(votes[key]), float(np.mean(votes[key]))),
            reverse=True,
        )[:8]
        candidates: list[dict[str, Any]] = []
        for key in ranked:
            approx_x, approx_y, cell_px = candidate_origins[key]
            # Refine translation around the grid vote at two-pixel steps.
            best_local: Optional[dict[str, Any]] = None
            for offset_y in (-3, 0, 3):
                for offset_x in (-3, 0, 3):
                    origin = (approx_x + offset_x, approx_y + offset_y)
                    geometry, coverage, visible_cells = self._score_origin(
                        integral, (frame_h, frame_w), cell_map, cell_px, origin
                    )
                    candidate = {
                        "geometry": geometry,
                        "coverage": coverage,
                        "visible_cells": visible_cells,
                        "origin": origin,
                        "scale": cell_px,
                        "patch_score": float(np.mean(votes[key])),
                    }
                    if best_local is None or geometry > best_local["geometry"]:
                        best_local = candidate
            if best_local is not None:
                # Multiple patch votes at essentially the same placement are
                # one alignment, not competing matches for the confidence gap.
                if any(
                    abs(best_local["origin"][0] - item["origin"][0]) < cell_px * 0.45
                    and abs(best_local["origin"][1] - item["origin"][1]) < cell_px * 0.45
                    and abs(cell_px - item["scale"]) < 1.0
                    for item in candidates
                ):
                    continue
                candidates.append(best_local)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["geometry"], item["patch_score"]), reverse=True)
        best = candidates[0]
        second_geometry = candidates[1]["geometry"] if len(candidates) > 1 else 0.0
        gap = best["geometry"] - second_geometry
        minimum_cells = max(20, int(np.ceil(
            self.config.minimap_min_visible_reference_pixels / (MAP_CELL_PIXELS ** 2)
        )))
        valid = (
            best["geometry"] >= self.config.minimap_geometry_min_score
            and gap >= self.config.minimap_align_min_gap
            and best["visible_cells"] >= minimum_cells
            and best["coverage"] >= self.config.minimap_min_visible_reference_fraction
        )
        return {
            "valid": valid,
            # The public score is the full visible-map geometric F-score.
            # Patch correlation is only a coarse candidate-voting signal and
            # should not be reported as the map's final match accuracy.
            "score": best["geometry"],
            "patch_score": best["patch_score"],
            "geometry": best["geometry"],
            "gap": gap,
            "coverage": best["coverage"],
            "visible_pixels": best["visible_cells"] * MAP_CELL_PIXELS ** 2,
            "scale": best["scale"],
            "origin": tuple(int(round(value)) for value in best["origin"]),
        }

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
        waypoints: list[tuple[float, float]],
        farthest: bool = False,
    ) -> Optional[tuple[int, int]]:
        if not alignment or not alignment.get("valid") or safe_mask is None or self.cell_map is None or player is None or not waypoints:
            return None
        scale, (ox, oy) = alignment["scale"], alignment["origin"]
        player_x, player_y = player
        map_x, map_y = (player_x - ox) / scale, (player_y - oy) / scale
        ref_h, ref_w = safe_mask.shape[:2]
        if not (0 <= map_x < ref_w and 0 <= map_y < ref_h and self.cell_map.walkable[int(map_y), int(map_x)] > 0):
            return None
        _, components = cv2.connectedComponents(
            (self.cell_map.walkable > 0).astype(np.uint8), connectivity=4
        )
        component = int(components[int(map_y), int(map_x)])
        if component == 0:
            return None
        walkable = components == component
        ys, xs = np.where((safe_mask > 0) & walkable)
        if len(xs) == 0:
            return None
        ref_xs, ref_ys = xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5
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

        points = np.asarray([
            point for point in waypoints
            if walkable[int(point[1]), int(point[0])] and safe_mask[int(point[1]), int(point[0])] > 0
        ], dtype=np.float32)
        if len(points) == 0:
            return None
        signature = (safe_mask.shape, component, len(points), tuple(points[0]), tuple(points[-1]))
        if self.route is None or self.route.signature != signature:
            nearest = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
            self.route = MinimapRoute(signature, tuple(points), nearest)
        else:
            self._check_route_progress((map_x, map_y))
            # Camera movement can carry the player a long way from the old
            # route cursor, especially after a failsafe jump.  Re-anchor it
            # instead of continuing a stale same-row sweep.
            current_index = int(self.route.index)
            reanchor_distance = max(
                8.0,
                self.config.minimap_max_click_distance * fw / 1600.0 / max(scale, 0.1),
            )
            if np.hypot(points[current_index, 0] - map_x, points[current_index, 1] - map_y) > reanchor_distance:
                self.route.index = int(np.argmin(np.hypot(points[:, 0] - map_x, points[:, 1] - map_y)))
        if farthest:
            waypoint_screen_xs = ox + points[:, 0] * scale
            waypoint_screen_ys = oy + points[:, 1] * scale
            waypoint_distances = np.hypot(waypoint_screen_xs - player_x, waypoint_screen_ys - player_y)
            waypoint_visible = (
                (waypoint_screen_xs >= margin_x) & (waypoint_screen_xs <= fw - margin_x)
                & (waypoint_screen_ys >= margin_y) & (waypoint_screen_ys <= fh - margin_y)
                & (waypoint_distances >= self.config.minimap_min_click_distance * fw / 1600.0)
            )
            if not np.any(waypoint_visible):
                return None
            index = int(np.argmax(np.where(waypoint_visible, waypoint_distances, -np.inf)))
            self.route.index = index
            self.last_route_target = (int(round(points[index, 0])), int(round(points[index, 1])))
            return int(round(waypoint_screen_xs[index])), int(round(waypoint_screen_ys[index]))
        points = np.asarray(self.route.waypoints, dtype=np.float32)
        current = int(self.route.index)
        target_index = (current + 1) % len(points)
        target_x, target_y = points[target_index]
        arrival_distance = max(
            2.0,
            self.config.minimap_min_click_distance * fw / 1600.0 / max(scale, 0.1),
        )
        if np.hypot(target_x - map_x, target_y - map_y) < arrival_distance:
            current = target_index
            self.route.index = current
            target_index = (current + 1) % len(points)
            target_x, target_y = points[target_index]
        before = np.hypot(target_x - map_x, target_y - map_y)
        after = np.hypot(target_x - ref_xs, target_y - ref_ys)
        preferred = (self.config.minimap_min_click_distance + self.config.minimap_max_click_distance) * 0.5 * fw / 1600.0
        scores = before - after
        scores = scores * scale * 1000.0 - np.abs(distances - preferred)
        scores[~visible] = -np.inf
        index = int(np.argmax(scores))
        self.last_route_target = (int(round(target_x)), int(round(target_y)))
        return int(round(screen_xs[index])), int(round(screen_ys[index]))

    def _check_route_progress(self, player_position: tuple[float, float]) -> None:
        route = self.route
        if route is None or route.last_player_position is None:
            return
        moved_cells = float(np.hypot(
            player_position[0] - route.last_player_position[0],
            player_position[1] - route.last_player_position[1],
        ))
        failed_target_index = route.last_target_index
        previous_target = route.last_target_position
        route.last_player_position = None
        route.last_target_index = None
        route.last_target_position = None
        if moved_cells < 0.75:
            if failed_target_index is not None:
                route.index = failed_target_index
            print(
                f"[!] No minimap progress after tapping waypoint {previous_target}; "
                f"advancing the route ({moved_cells:.2f} map cells moved)."
            )
        else:
            print(f"[+] Minimap progress confirmed: {moved_cells:.2f} map cells moved.")

    def travel(self, *, farthest: bool = False, keep_open_seconds: float = 0.0) -> bool:
        reference_image = self.templates.get(self.minimap_template_key)
        if reference_image is None:
            print(f"[!] Minimap travel disabled: reference '{self.minimap_template_key}' is missing.")
            return False
        frame = self.device.capture()
        if frame is None:
            return False
        state, icon = self.vision.ui_state(frame)
        alignment: Optional[dict[str, Any]] = None
        reference_name = self.minimap_template_key
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
                    candidate = self.align(frame, reference_image)
                    if candidate is not None and candidate.get("valid"):
                        alignment = candidate
                        break
        if state != "minimap":
            print(f"[!] Minimap travel skipped: UI state is {state}.")
            return False
        if alignment is None:
            alignment = self.align(frame, reference_image)
        if not alignment or not alignment.get("valid"):
            details = alignment or {}
            self._save_alignment_comparison(frame, reference_image, alignment)
            return self._fail(icon, f"[!] Minimap alignment rejected (map score {details.get('score', 0.0):.2f}, patch vote {details.get('patch_score', 0.0):.2f}, coverage {details.get('coverage', 0.0):.2f}, gap {details.get('gap', 0.0):.2f}); no movement tap sent.")
        reference = self.templates.get(reference_name) if reference_name is not None else None
        if reference is None:
            return self._fail(icon, "[!] Minimap reference disappeared during registration; no movement tap sent.")
        player = self.player_marker(frame)
        if player is None:
            return self._fail(icon, "[!] Minimap player marker was not found; no movement tap sent.")
        if self.reference_id != id(reference) or self.cell_map is None:
            try:
                self.cell_map = self._build_cell_map(reference)
            except ValueError as error:
                return self._fail(icon, f"[!] Minimap grid rejected: {error}")
            self.safe_mask = self.cell_map.safe_mask
            self.waypoints = self.build_waypoints(self.safe_mask)
            self.reference_id = id(reference)
        destination = self.destination(frame, alignment, self.safe_mask, player, self.waypoints, farthest=farthest)
        if destination is None:
            return self._fail(icon, "[!] No safe visible minimap waypoint was found; no movement tap sent.")
        player_map_position = (
            (player[0] - alignment["origin"][0]) / alignment["scale"],
            (player[1] - alignment["origin"][1]) / alignment["scale"],
        )
        mode = "farthest visible" if farthest else "route"
        patch_score = alignment.get("patch_score", alignment["score"])
        print(f"[~] Minimap {mode} waypoint {destination} on {reference_name} from player {player} at map position ({player_map_position[0]:.1f}, {player_map_position[1]:.1f}) toward reference {self.last_route_target or '?'} (map score {alignment['score']:.2f}, patch vote {patch_score:.2f}, coverage {alignment['coverage']:.2f}, gap {alignment['gap']:.2f}, scale {alignment['scale']:.2f}).")
        sent = self.device.click(*destination, synchronous=True)
        if sent and not farthest and self.route is not None and self.route.waypoints:
            self.route.last_player_position = player_map_position
            self.route.last_target_index = (self.route.index + 1) % len(self.route.waypoints)
            self.route.last_target_position = self.last_route_target
        elif sent and farthest and self.route is not None:
            self.route.last_player_position = None
            self.route.last_target_index = None
            self.route.last_target_position = None
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
        print(f"[~] Minimap movement tap sent={sent}; map closed={closed}.")
        if not closed and keep_open_seconds > 0:
            # A late UI frame can leave the first Back verification stale.
            # Re-match the live icon and make one final delayed close attempt.
            time.sleep(0.8)
            latest = self.device.capture()
            retry_icon = self.vision.back_icon(latest) if latest is not None else icon
            closed = self._close_map(retry_icon)
        return bool(sent and closed)

    def _save_alignment_comparison(
        self,
        frame: np.ndarray,
        reference: np.ndarray,
        alignment: Optional[dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Save the exact rejected frame beside its projected map comparison."""
        try:
            cell_map = self.cell_map or self._build_cell_map(reference)
            target = cell_map.walls
            context = cell_map.context_walls
            expected = cell_map.match_walls
            map_h, map_w = expected.shape
            comparison = np.full((map_h, map_w, 3), (32, 32, 32), dtype=np.uint8)
            overlay = frame.copy()
            if alignment is not None and "origin" in alignment and "scale" in alignment:
                scale = float(alignment["scale"])
                ox, oy = alignment["origin"]
                fh, fw = frame.shape[:2]
                sample = max(3, int(round(scale * 0.72)))
                center_x = np.rint(ox + (np.arange(map_w) + 0.5) * scale).astype(np.int32)
                center_y = np.rint(oy + (np.arange(map_h) + 0.5) * scale).astype(np.int32)
                x1, y1 = center_x - sample // 2, center_y - sample // 2
                x2, y2 = x1 + sample, y1 + sample
                visible = ((x1 >= 0) & (x2 <= fw))[None, :] & ((y1 >= 0) & (y2 <= fh))[:, None]
                x1c, x2c = np.clip(x1, 0, fw), np.clip(x2, 0, fw)
                y1c, y2c = np.clip(y1, 0, fh), np.clip(y2, 0, fh)
                white = cv2.inRange(frame, (220, 220, 220), (255, 255, 255)) // 255
                integral = cv2.integral(white, sdepth=cv2.CV_32S)
                sums = (
                    integral[y2c[:, None], x2c[None, :]]
                    - integral[y1c[:, None], x2c[None, :]]
                    - integral[y2c[:, None], x1c[None, :]]
                    + integral[y1c[:, None], x1c[None, :]]
                )
                observed = (sums >= sample * sample * 0.30) & visible
                comparison[~visible] = (72, 72, 72)
                comparison[target & visible] = (0, 0, 220)  # target wall missed
                comparison[target & observed] = (0, 210, 0)  # target wall hit
                comparison[context & visible] = (0, 125, 255)  # context wall missed
                comparison[context & observed] = (255, 220, 0)  # context wall hit
                comparison[observed & visible & ~expected & ~cell_map.annotations] = (255, 0, 0)

                for mask, color in (
                    (target & observed, (0, 255, 0)),
                    (target & visible & ~observed, (0, 0, 255)),
                    (context & observed, (255, 255, 0)),
                    (context & visible & ~observed, (0, 150, 255)),
                    (observed & visible & ~expected & ~cell_map.annotations, (255, 0, 0)),
                ):
                    ys, xs = np.where(mask)
                    for x, y in zip(xs, ys):
                        px = int(round(ox + (x + 0.5) * scale))
                        py = int(round(oy + (y + 0.5) * scale))
                        cv2.circle(overlay, (px, py), max(2, int(scale * 0.13)), color, -1)

                player = self.player_marker(frame)
                if player is not None:
                    cv2.circle(overlay, player, 9, (0, 255, 255), 2)

            def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
                ratio = min(width / image.shape[1], height / image.shape[0])
                size = (int(round(image.shape[1] * ratio)), int(round(image.shape[0] * ratio)))
                return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

            panel_w, panel_h, gap = 565, 450, 18
            live_w = 800
            header_h = 86
            canvas = np.full((header_h + panel_h + 28, panel_w * 2 + live_w + gap * 2, 3), 20, dtype=np.uint8)
            details = alignment or {}
            summary = (
                f"valid={details.get('valid', False)}  map={details.get('score', 0.0):.2f}  "
                f"patch={details.get('patch_score', 0.0):.2f}  coverage={details.get('coverage', 0.0):.2f}  "
                f"gap={details.get('gap', 0.0):.2f}  origin={details.get('origin', '?')}  scale={details.get('scale', '?')}"
            )
            cv2.putText(canvas, "Minimap alignment comparison (saved from this exact frame)", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, summary, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
            left_x, middle_x, right_x = 0, panel_w + gap, panel_w + gap + live_w + gap
            labels = ((left_x + 12, "REFERENCE PALETTE"), (middle_x + 12, "LIVE FRAME + PROJECTED WALL CELLS"), (right_x + 12, "CELL-BY-CELL MATCH"))
            for x, label in labels:
                cv2.putText(canvas, label, (x, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
            y0 = header_h
            ref_view = fit(reference, panel_w - 12, panel_h - 14)
            live_view = fit(overlay, live_w, panel_h - 14)
            compare_view = fit(cv2.resize(comparison, (map_w * 5, map_h * 5), interpolation=cv2.INTER_NEAREST), panel_w - 12, panel_h - 14)
            canvas[y0:y0 + ref_view.shape[0], left_x + (panel_w - ref_view.shape[1]) // 2:left_x + (panel_w + ref_view.shape[1]) // 2] = ref_view
            canvas[y0:y0 + live_view.shape[0], middle_x + (live_w - live_view.shape[1]) // 2:middle_x + (live_w + live_view.shape[1]) // 2] = live_view
            canvas[y0:y0 + compare_view.shape[0], right_x + (panel_w - compare_view.shape[1]) // 2:right_x + (panel_w + compare_view.shape[1]) // 2] = compare_view
            footer_y = header_h + panel_h + 16
            cv2.putText(canvas, "Match grid: green=target wall hit, red=miss; cyan=context wall hit, orange=miss; blue=unexpected bright cell; gray=off-screen.", (14, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)
            path = output_path or (Path(__file__).resolve().parents[1] / "minimap_alignment_latest.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), canvas):
                return None
            print(f"[!] Saved minimap alignment comparison: {path}")
            return path
        except (OSError, cv2.error, ValueError) as error:
            print(f"[!] Could not save minimap alignment comparison: {error}")
            return None

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
