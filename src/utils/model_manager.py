"""
model_manager.py - Adaptive model manager for day/night model switching.

Automatically selects between day and night YOLO models based on:
1. Time-based rules: 7 PM - 5 AM always uses night model
2. Brightness-based fallback: If brightness < 70 during daytime, use night model
3. Re-evaluates every 15 minutes to handle light transitions

Usage:
    manager = AdaptiveModelManager(day_model_path, night_model_path, video_start_time, fps)

    # In frame processing loop:
    model = manager.get_model_for_frame(frame, frame_count)
    results = model.predict(frame, ...)  # or model.track(...)
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from typing import Optional, Tuple


# Time-based rules (24-hour format)
NIGHT_START_HOUR = 19  # 7 PM
NIGHT_END_HOUR = 5     # 5 AM

# Brightness threshold for daytime fallback to night model
BRIGHTNESS_THRESHOLD = 70

# Re-evaluation interval in minutes
REEVALUATION_INTERVAL_MINUTES = 15


class AdaptiveModelManager:
    """
    Manages day/night model selection based on time and brightness conditions.
    """

    def __init__(
        self,
        day_model_path: str,
        night_model_path: Optional[str],
        video_start_time: Optional[datetime],
        fps: float,
        verbose: bool = True
    ):
        """
        Initialize the adaptive model manager.

        Args:
            day_model_path: Path to the daytime model (best.pt)
            night_model_path: Path to the night model (best_night.pt), or None for day-only
            video_start_time: Start datetime of the video (for time-based switching)
            fps: Video frames per second
            verbose: Whether to print model switching logs
        """
        self.day_model_path = day_model_path
        self.night_model_path = night_model_path
        self.video_start_time = video_start_time
        self.fps = fps
        self.verbose = verbose

        # Load models
        self.day_model = YOLO(day_model_path)
        self.night_model = None

        if night_model_path:
            try:
                self.night_model = YOLO(night_model_path)
                if verbose:
                    print(f"[ModelManager] Loaded night model: {night_model_path}")
            except Exception as e:
                print(f"[ModelManager] WARNING: Could not load night model: {e}")
                print(f"[ModelManager] Falling back to day model only")
                self.night_model = None

        if verbose:
            print(f"[ModelManager] Loaded day model: {day_model_path}")
            print(f"[ModelManager] Video start time: {video_start_time}")
            print(f"[ModelManager] Night hours: {NIGHT_START_HOUR}:00 - {NIGHT_END_HOUR}:00")
            print(f"[ModelManager] Brightness threshold: {BRIGHTNESS_THRESHOLD}")
            print(f"[ModelManager] Re-evaluation interval: {REEVALUATION_INTERVAL_MINUTES} minutes")

        # State tracking
        self._current_model_type = None  # 'day' or 'night'
        self._last_evaluation_minute = -1

        # Statistics
        self.stats = {
            "day_frames": 0,
            "night_frames": 0,
            "switches": 0,
            "brightness_triggered_switches": 0
        }

    def _calculate_frame_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate the mean brightness of a frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Mean brightness value (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _get_time_at_frame(self, frame_count: int) -> Optional[datetime]:
        """
        Calculate the wall-clock time at a given frame.

        Args:
            frame_count: Current frame number

        Returns:
            datetime at this frame, or None if video_start_time not available
        """
        if self.video_start_time is None:
            return None

        seconds_elapsed = frame_count / self.fps
        return self.video_start_time + timedelta(seconds=seconds_elapsed)

    def _is_night_by_time(self, current_time: datetime) -> bool:
        """
        Check if the given time falls within night hours.

        Night is defined as 7 PM (19:00) to 5 AM (05:00).

        Args:
            current_time: The datetime to check

        Returns:
            True if it's nighttime, False otherwise
        """
        hour = current_time.hour
        # Night: 19:00 - 23:59 OR 00:00 - 04:59
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR

    def _should_reevaluate(self, frame_count: int) -> bool:
        """
        Check if we should re-evaluate model selection.

        Re-evaluates every 15 minutes of video time.

        Args:
            frame_count: Current frame number

        Returns:
            True if we should re-evaluate
        """
        # Always evaluate on first frame
        if self._current_model_type is None:
            return True

        # Calculate current minute in video
        current_minute = int((frame_count / self.fps) / 60)

        # Re-evaluate every REEVALUATION_INTERVAL_MINUTES
        evaluation_period = current_minute // REEVALUATION_INTERVAL_MINUTES
        last_evaluation_period = self._last_evaluation_minute // REEVALUATION_INTERVAL_MINUTES

        if evaluation_period > last_evaluation_period:
            self._last_evaluation_minute = current_minute
            return True

        return False

    def _select_model(self, frame: np.ndarray, frame_count: int) -> Tuple[str, str]:
        """
        Select the appropriate model based on time and brightness.

        Args:
            frame: Current video frame
            frame_count: Current frame number

        Returns:
            Tuple of (model_type, reason) where model_type is 'day' or 'night'
        """
        # If no night model available, always use day
        if self.night_model is None:
            return 'day', 'no_night_model'

        # Get current video time
        current_time = self._get_time_at_frame(frame_count)

        # Check time-based rule first
        if current_time is not None:
            if self._is_night_by_time(current_time):
                return 'night', f'time_based ({current_time.strftime("%H:%M")})'

        # For daytime hours (or unknown time), check brightness
        brightness = self._calculate_frame_brightness(frame)

        if brightness < BRIGHTNESS_THRESHOLD:
            return 'night', f'brightness ({brightness:.1f} < {BRIGHTNESS_THRESHOLD})'
        else:
            return 'day', f'brightness ({brightness:.1f} >= {BRIGHTNESS_THRESHOLD})'

    def get_model_for_frame(self, frame: np.ndarray, frame_count: int) -> YOLO:
        """
        Get the appropriate model for the current frame.

        This is the main method to call in the processing loop.
        Re-evaluates every 15 minutes of video time.

        Args:
            frame: Current video frame (BGR)
            frame_count: Current frame number

        Returns:
            YOLO model instance to use for this frame
        """
        # Check if we need to re-evaluate
        if self._should_reevaluate(frame_count):
            new_model_type, reason = self._select_model(frame, frame_count)

            # Log if model changed
            if new_model_type != self._current_model_type:
                if self._current_model_type is not None:
                    self.stats["switches"] += 1
                    if "brightness" in reason:
                        self.stats["brightness_triggered_switches"] += 1

                if self.verbose:
                    current_time = self._get_time_at_frame(frame_count)
                    time_str = current_time.strftime("%H:%M:%S") if current_time else "unknown"
                    video_minute = int((frame_count / self.fps) / 60)
                    print(f"[ModelManager] Switching to {new_model_type.upper()} model at frame {frame_count} "
                          f"(video minute {video_minute}, time {time_str}) - reason: {reason}")

                self._current_model_type = new_model_type

        # Update stats
        if self._current_model_type == 'day':
            self.stats["day_frames"] += 1
        else:
            self.stats["night_frames"] += 1

        # Return appropriate model
        if self._current_model_type == 'night' and self.night_model is not None:
            return self.night_model
        return self.day_model

    def get_current_model_type(self) -> str:
        """Get the current model type being used ('day' or 'night')."""
        return self._current_model_type or 'day'

    def get_stats(self) -> dict:
        """Get usage statistics."""
        total_frames = self.stats["day_frames"] + self.stats["night_frames"]
        return {
            **self.stats,
            "total_frames": total_frames,
            "day_percentage": (self.stats["day_frames"] / total_frames * 100) if total_frames > 0 else 0,
            "night_percentage": (self.stats["night_frames"] / total_frames * 100) if total_frames > 0 else 0,
        }

    def print_summary(self):
        """Print a summary of model usage."""
        stats = self.get_stats()
        print(f"\n[ModelManager] Summary:")
        print(f"  Total frames processed: {stats['total_frames']:,}")
        print(f"  Day model frames: {stats['day_frames']:,} ({stats['day_percentage']:.1f}%)")
        print(f"  Night model frames: {stats['night_frames']:,} ({stats['night_percentage']:.1f}%)")
        print(f"  Model switches: {stats['switches']}")
        print(f"  Brightness-triggered switches: {stats['brightness_triggered_switches']}")
