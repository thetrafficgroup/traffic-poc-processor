"""
Crosswalk crossing detection using a state machine per (entity_id, crosswalk_name).

States:
  UNCROSSED       - hasn't crossed either line
  CROSSED_FIRST   - crossed one line, waiting for the other
  COUNTED         - crossed both lines, count recorded

Transitions:
  UNCROSSED -> CROSSED_FIRST:  entity crosses either line (record which)
  CROSSED_FIRST -> COUNTED:    entity crosses the other line
    -> direction = first_line_name -> second_line_name

Timeout:
  CROSSED_FIRST for >60 seconds -> reset to UNCROSSED
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any


class CrossingState(Enum):
    UNCROSSED = "uncrossed"
    CROSSED_FIRST = "crossed_first"
    COUNTED = "counted"


# Direction resolution based on crosswalk orientation and line crossing order
DIRECTION_MAP = {
    ("NORTH_CROSSWALK", "east_line", "west_line"): "Westbound",
    ("NORTH_CROSSWALK", "west_line", "east_line"): "Eastbound",
    ("SOUTH_CROSSWALK", "east_line", "west_line"): "Westbound",
    ("SOUTH_CROSSWALK", "west_line", "east_line"): "Eastbound",
    ("EAST_CROSSWALK", "north_line", "south_line"): "Southbound",
    ("EAST_CROSSWALK", "south_line", "north_line"): "Northbound",
    ("WEST_CROSSWALK", "north_line", "south_line"): "Southbound",
    ("WEST_CROSSWALK", "south_line", "north_line"): "Northbound",
}

CROSSING_TIMEOUT_SECONDS = 60
DIST_THRESHOLD = 15  # Slightly more generous than vehicle tracking (pedestrians walk slower)


class EntityCrossingState:
    """Tracks the crossing state for a single entity at a single crosswalk."""

    __slots__ = ("state", "first_line", "first_frame")

    def __init__(self):
        self.state: CrossingState = CrossingState.UNCROSSED
        self.first_line: Optional[str] = None
        self.first_frame: int = 0

    def reset(self):
        self.state = CrossingState.UNCROSSED
        self.first_line = None
        self.first_frame = 0


class CrosswalkLine:
    """A single crosswalk boundary line with crossing detection."""

    __slots__ = ("name", "pt1", "pt2")

    def __init__(self, name: str, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        self.name = name
        self.pt1 = pt1
        self.pt2 = pt2


class Crosswalk:
    """A crosswalk with two boundary lines."""

    def __init__(self, name: str, lines: List[CrosswalkLine]):
        self.name = name
        self.lines = lines


def _ensure_int_coords(point) -> Tuple[int, int]:
    if isinstance(point, dict):
        return (int(round(point["x"])), int(round(point["y"])))
    elif isinstance(point, (list, tuple)):
        return (int(round(point[0])), int(round(point[1])))
    return point


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance from point to line segment."""
    px, py = float(px), float(py)
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    dx = px - xx
    dy = py - yy
    return (dx**2 + dy**2) ** 0.5


class CrosswalkProcessor:
    """
    Processes pedestrian/bicycle detections against crosswalk line pairs.
    Uses a state machine to detect when entities cross both lines of a crosswalk.
    """

    def __init__(self, crosswalks_config: List[Dict[str, Any]], fps: float = 30.0):
        self.crosswalks: List[Crosswalk] = []
        self.fps = fps
        self._parse_config(crosswalks_config)

        # State per (entity_id, crosswalk_name) -> EntityCrossingState
        self._entity_states: Dict[Tuple[int, str], EntityCrossingState] = {}

        # Previous positions for crossing detection: entity_id -> (x, y)
        self._prev_positions: Dict[int, Tuple[float, float]] = {}

        # Counted entities per crosswalk to prevent duplicate counting
        self._counted_ids: Dict[str, Set[int]] = {cw.name: set() for cw in self.crosswalks}

        # Results accumulator: crosswalk_name -> class_name -> direction -> count
        self.results: Dict[str, Dict[str, Dict[str, int]]] = {}
        for cw in self.crosswalks:
            self.results[cw.name] = {}

        print(f"🚶 CrosswalkProcessor initialized with {len(self.crosswalks)} crosswalk(s)")
        for cw in self.crosswalks:
            line_names = [line.name for line in cw.lines]
            print(f"   └─ {cw.name}: lines {line_names}")

    def _parse_config(self, config: List[Dict[str, Any]]):
        """Parse crosswalk configuration from metadata."""
        for cw_data in config:
            name = cw_data["name"]
            lines = []

            # N/S crosswalks have east_line and west_line
            # E/W crosswalks have north_line and south_line
            is_ns = name.startswith("NORTH") or name.startswith("SOUTH")

            if is_ns:
                line_keys = ["east_line", "west_line"]
            else:
                line_keys = ["north_line", "south_line"]

            for key in line_keys:
                line_data = cw_data.get(key)
                if line_data:
                    pt1 = _ensure_int_coords(line_data["pt1"])
                    pt2 = _ensure_int_coords(line_data["pt2"])
                    lines.append(CrosswalkLine(name=key, pt1=pt1, pt2=pt2))

            if len(lines) == 2:
                self.crosswalks.append(Crosswalk(name=name, lines=lines))
            else:
                print(f"⚠️ Crosswalk {name} has {len(lines)} lines (expected 2), skipping")

    def process_detection(
        self,
        entity_id: int,
        class_name: str,
        cx: float,
        cy: float,
        frame_number: int,
    ) -> Optional[Dict[str, str]]:
        """
        Process a single detection (pedestrian or bicycle) against all crosswalks.

        Args:
            entity_id: Namespaced track ID
            class_name: "pedestrian" or "bicycle"
            cx: X coordinate (center)
            cy: Y coordinate (bottom of bbox = feet/wheels)
            frame_number: Current frame number

        Returns:
            Dict with crossing info if entity completed a crossing, None otherwise.
        """
        prev_pos = self._prev_positions.get(entity_id)
        self._prev_positions[entity_id] = (cx, cy)

        if prev_pos is None:
            return None

        result = None
        prev_x, prev_y = prev_pos

        for crosswalk in self.crosswalks:
            # Skip if already counted at this crosswalk
            if entity_id in self._counted_ids[crosswalk.name]:
                continue

            state_key = (entity_id, crosswalk.name)
            if state_key not in self._entity_states:
                self._entity_states[state_key] = EntityCrossingState()

            entity_state = self._entity_states[state_key]

            # Timeout check: reset if stuck in CROSSED_FIRST too long (frame-based)
            if (
                entity_state.state == CrossingState.CROSSED_FIRST
                and (frame_number - entity_state.first_frame) / self.fps > CROSSING_TIMEOUT_SECONDS
            ):
                entity_state.reset()

            # Check each line of this crosswalk for crossing
            for line in crosswalk.lines:
                x1, y1 = line.pt1
                x2, y2 = line.pt2

                dist = _point_line_distance(cx, cy, x1, y1, x2, y2)
                prev_dist = _point_line_distance(prev_x, prev_y, x1, y1, x2, y2)

                crossed = dist < DIST_THRESHOLD and prev_dist > DIST_THRESHOLD

                if not crossed:
                    continue

                if entity_state.state == CrossingState.UNCROSSED:
                    # First line crossed
                    entity_state.state = CrossingState.CROSSED_FIRST
                    entity_state.first_line = line.name
                    entity_state.first_frame = frame_number

                elif entity_state.state == CrossingState.CROSSED_FIRST:
                    # Must be a DIFFERENT line to complete the crossing
                    if line.name == entity_state.first_line:
                        continue

                    # Crossing complete
                    entity_state.state = CrossingState.COUNTED
                    self._counted_ids[crosswalk.name].add(entity_id)

                    # Resolve direction
                    direction_key = (crosswalk.name, entity_state.first_line, line.name)
                    direction = DIRECTION_MAP.get(direction_key, "Unknown")

                    # Record the count
                    if class_name not in self.results[crosswalk.name]:
                        self.results[crosswalk.name][class_name] = {}
                    dir_counts = self.results[crosswalk.name][class_name]
                    dir_counts[direction] = dir_counts.get(direction, 0) + 1

                    result = {
                        "crosswalk": crosswalk.name,
                        "class": class_name,
                        "direction": direction,
                        "entity_id": entity_id,
                        "frame": frame_number,
                    }

        return result

    def get_results(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get accumulated crosswalk results."""
        return self.results

    def get_totals(self) -> Dict[str, int]:
        """Get total counts by class across all crosswalks."""
        totals: Dict[str, int] = {}
        for cw_name, classes in self.results.items():
            for class_name, directions in classes.items():
                total = sum(directions.values())
                totals[class_name] = totals.get(class_name, 0) + total
        return totals

    def get_current_minute_results(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get results (same as get_results, used for minute tracking snapshots)."""
        return self.results

    def reset_state(self):
        """Reset all tracking state (for trim period boundaries)."""
        self._entity_states.clear()
        self._prev_positions.clear()
        # Don't clear _counted_ids or results — counts persist across periods
        print("🧹 CrosswalkProcessor state reset (prev positions and entity states cleared)")

    def cleanup_old_entities(self, active_ids: Set[int]):
        """Remove tracking state for entities no longer being tracked."""
        stale_keys = [
            key for key in self._entity_states
            if key[0] not in active_ids and self._entity_states[key].state != CrossingState.COUNTED
        ]
        for key in stale_keys:
            del self._entity_states[key]

        stale_positions = [
            eid for eid in self._prev_positions if eid not in active_ids
        ]
        for eid in stale_positions:
            del self._prev_positions[eid]
