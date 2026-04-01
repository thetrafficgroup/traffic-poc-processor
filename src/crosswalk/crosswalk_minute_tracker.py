"""
Minute-by-minute crosswalk tracking for pedestrian/bicycle counts.
Separate from the existing MinuteTracker (different data shape).

Data structure per minute:
{
    "crosswalks": {
        "NORTH_CROSSWALK": {
            "pedestrian": {"Westbound": 3, "Eastbound": 1},
            "bicycle": {"Westbound": 0, "Eastbound": 1}
        },
        ...
    }
}
"""

from typing import Dict, Any, Optional, Set


class CrosswalkMinuteTracker:
    """
    Tracks crosswalk crossing events on a per-minute basis.
    Designed to merge with the vehicle MinuteTracker's batch data.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self.current_minute = 0

        # Per-minute crosswalk data: minute_number -> crosswalk results snapshot
        self.minute_data: Dict[int, Dict[str, Any]] = {}

        # Track which crossings have been assigned to a minute already
        # Key: (entity_id, crosswalk_name), Value: minute_number
        self._counted_crossings: Dict[tuple, int] = {}

        print(f"🚶 CrosswalkMinuteTracker initialized with FPS={fps}")

    def calculate_minute_from_frame(self, frame_number: int) -> int:
        """Calculate which minute a frame belongs to (0-based)."""
        seconds = frame_number / self.fps
        return int(seconds / 60)

    def record_crossing(
        self,
        frame_number: int,
        entity_id: int,
        crosswalk_name: str,
        class_name: str,
        direction: str,
    ) -> None:
        """
        Record a completed crosswalk crossing for minute tracking.

        Args:
            frame_number: Frame where crossing was completed
            entity_id: Namespaced entity track ID
            crosswalk_name: e.g. "NORTH_CROSSWALK"
            class_name: "pedestrian" or "bicycle"
            direction: e.g. "Westbound", "Eastbound"
        """
        crossing_key = (entity_id, crosswalk_name)

        # Deduplicate: each entity counted once per crosswalk
        if crossing_key in self._counted_crossings:
            return

        video_minute = self.calculate_minute_from_frame(frame_number)
        self._counted_crossings[crossing_key] = video_minute

        # Initialize minute data if needed
        if video_minute not in self.minute_data:
            self.minute_data[video_minute] = {"crosswalks": {}}

        crosswalks = self.minute_data[video_minute]["crosswalks"]
        if crosswalk_name not in crosswalks:
            crosswalks[crosswalk_name] = {}

        if class_name not in crosswalks[crosswalk_name]:
            crosswalks[crosswalk_name][class_name] = {}

        dir_counts = crosswalks[crosswalk_name][class_name]
        dir_counts[direction] = dir_counts.get(direction, 0) + 1

        self.current_minute = max(self.current_minute, video_minute)

    def get_minute_result(self, minute_number: int) -> Dict[str, Any]:
        """
        Get crosswalk results for a specific minute.
        Returns empty crosswalks dict if no data for that minute.
        """
        if minute_number in self.minute_data:
            return self.minute_data[minute_number]
        return {"crosswalks": {}}

    def get_all_minute_data(self) -> Dict[int, Dict[str, Any]]:
        """Get all minute data for final batch merge."""
        return self.minute_data

    def finalize(self) -> Dict[str, Any]:
        """Get summary of crosswalk tracking."""
        total_crossings = len(self._counted_crossings)
        minutes_with_data = len(self.minute_data)
        print(f"🚶 CrosswalkMinuteTracker finalized: {total_crossings} crossings across {minutes_with_data} minutes")
        return {
            "total_crossings": total_crossings,
            "minutes_with_data": minutes_with_data,
        }
