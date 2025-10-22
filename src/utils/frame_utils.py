"""
Frame utilities for video trimming functionality.

This module provides utilities to convert wall-clock time offsets (in seconds)
to frame numbers for processing specific periods of a video.
"""

import logging

# Configure logging with emoji indicators
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def calculate_frame_ranges_from_seconds(trim_periods, fps, total_frames):
    """
    Convert trim periods from seconds offset to frame numbers.

    Args:
        trim_periods: List of dicts with 'start' and 'end' in seconds from video start
                     Example: [{"start": 3600, "end": 10800}, {"start": 54000, "end": 61200}]
        fps: Frames per second of the video (float)
        total_frames: Total number of frames in the video (int)

    Returns:
        List of dicts with 'start_frame' and 'end_frame' (both integers)
        Example: [{"start_frame": 108000, "end_frame": 324000}, ...]

    Notes:
        - Periods that exceed video duration are clamped to actual duration
        - Invalid periods (start >= end after clamping) are filtered out
        - Returns empty list if trim_periods is None or empty
        - Uses int(round(...)) for precise frame number calculation
    """

    if not trim_periods:
        logger.info("📊 No trim periods specified, will process entire video")
        return []

    if fps <= 0:
        logger.warning(f"⚠️ Invalid FPS: {fps}, cannot calculate frame ranges")
        return []

    if total_frames <= 0:
        logger.warning(f"⚠️ Invalid total_frames: {total_frames}, cannot calculate frame ranges")
        return []

    # Calculate video duration in seconds
    video_duration = total_frames / fps
    logger.info(f"📊 Video info: {total_frames} frames @ {fps:.2f} fps = {video_duration:.2f} seconds")

    frame_ranges = []

    for i, period in enumerate(trim_periods):
        try:
            # Extract start and end seconds
            start_seconds = float(period.get('start', 0))
            end_seconds = float(period.get('end', 0))

            logger.info(f"📋 Period {i+1}: {start_seconds}s - {end_seconds}s")

            # Validate period
            if start_seconds < 0:
                logger.warning(f"⚠️ Period {i+1}: start time {start_seconds}s is negative, clamping to 0")
                start_seconds = 0

            if end_seconds <= start_seconds:
                logger.warning(f"⚠️ Period {i+1}: end time {end_seconds}s <= start time {start_seconds}s, skipping period")
                continue

            # Clamp to video duration (don't reject, just clamp)
            original_end = end_seconds
            if end_seconds > video_duration:
                logger.warning(f"⚠️ Period {i+1}: end time {end_seconds}s exceeds video duration {video_duration:.2f}s, clamping to video end")
                end_seconds = video_duration

            if start_seconds >= video_duration:
                logger.warning(f"⚠️ Period {i+1}: start time {start_seconds}s >= video duration {video_duration:.2f}s, skipping period")
                continue

            # Convert to frame numbers using int(round(...)) for precision
            start_frame = int(round(start_seconds * fps))
            end_frame = int(round(end_seconds * fps))

            # Clamp to actual frame count
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))

            # Final validation
            if start_frame >= end_frame:
                logger.warning(f"⚠️ Period {i+1}: start_frame {start_frame} >= end_frame {end_frame} after clamping, skipping")
                continue

            frame_range = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds
            }

            frame_ranges.append(frame_range)

            duration_seconds = end_seconds - start_seconds
            duration_frames = end_frame - start_frame
            logger.info(f"✅ Period {i+1}: frames {start_frame} - {end_frame} ({duration_frames} frames, {duration_seconds:.2f}s)")

            if original_end != end_seconds:
                logger.info(f"   ⚠️ Note: Clamped from {original_end:.2f}s to {end_seconds:.2f}s")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Period {i+1}: Error parsing period {period}: {e}")
            continue

    if not frame_ranges:
        logger.warning("⚠️ No valid trim periods after validation, will process entire video")
        return []

    logger.info(f"✅ Calculated {len(frame_ranges)} valid frame ranges from {len(trim_periods)} trim periods")

    return frame_ranges


def validate_trim_periods(trim_periods):
    """
    Validate trim periods structure without FPS/duration (basic validation).

    Args:
        trim_periods: List of dicts with 'start' and 'end' in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """

    if not trim_periods:
        return (True, None)  # Empty is valid

    if not isinstance(trim_periods, list):
        return (False, "trim_periods must be a list")

    for i, period in enumerate(trim_periods):
        if not isinstance(period, dict):
            return (False, f"Period {i+1} must be a dictionary")

        if 'start' not in period:
            return (False, f"Period {i+1} missing 'start' field")

        if 'end' not in period:
            return (False, f"Period {i+1} missing 'end' field")

        try:
            start = float(period['start'])
            end = float(period['end'])

            if start < 0:
                return (False, f"Period {i+1} start time cannot be negative")

            if end <= start:
                return (False, f"Period {i+1} end time must be greater than start time")

        except (ValueError, TypeError) as e:
            return (False, f"Period {i+1} has invalid start/end values: {e}")

    return (True, None)
