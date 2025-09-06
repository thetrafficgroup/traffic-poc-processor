"""
ATR-specific minute tracking utility for traffic video processing.
Handles lane-based minute-by-minute data collection and batch processing for ATR videos.
"""

import json
import uuid
import logging
from typing import Dict, List, Optional, Callable, Any, Set

# Configure logger for this module
logger = logging.getLogger(__name__)


class ATRMinuteTracker:
    """
    Tracks vehicle detections on a per-minute basis for ATR videos.
    Uses lane-based tracking instead of direction-turn based tracking.
    Manages batching and SQS message preparation with lane-vehicle classification nesting.
    """
    
    def __init__(self, fps: float, video_uuid: str, batch_callback: Optional[Callable] = None, 
                 verbose: bool = False):
        self.fps = fps
        self.video_uuid = video_uuid
        self.batch_callback = batch_callback
        self.verbose = verbose  # Control detailed logging
        
        # Tracking state
        self.minute_data: Dict[int, Dict] = {}
        self.current_minute = 0
        self.batch_counter = 0
        
        # Batch configuration - use same timing as TMC (15-second intervals = 4 batches per minute)
        self.BATCH_SIZE = 5  # Send batch every 5 minutes
        
        # Track processed vehicle IDs per minute to avoid duplicates
        self.processed_vehicles_per_minute: Dict[int, Set[int]] = {}
        
        # Statistics for summary logging
        self.total_vehicles_processed = 0
        self.vehicles_per_minute_count: Dict[int, int] = {}
        
        logger.info(f"🔄 ATRMinuteTracker initialized for video {video_uuid} with FPS={fps}")
    
    def calculate_minute_from_frame(self, frame_number: int) -> int:
        """Calculate which minute a frame belongs to (0-based)."""
        seconds = frame_number / self.fps
        return int(seconds / 60)
    
    def process_vehicle_detection(self, frame_number: int, vehicle_id: int, vehicle_class: str, 
                                 lane_id: str) -> None:
        """
        Process a single vehicle detection with its lane assignment.
        
        Args:
            frame_number: Current frame number
            vehicle_id: Unique vehicle ID
            vehicle_class: Vehicle classification (car, bus, truck, etc.)
            lane_id: Lane identifier where the vehicle was detected
        """
        # Skip if no valid lane assignment
        if lane_id is None:
            if self.verbose:
                logger.warning(f"Skipping vehicle {vehicle_id} - no lane assignment")
            return
        
        # OPTIMIZATION: Convert lane_id to string once at the beginning
        lane_id_str = str(lane_id)
        
        video_minute = self.calculate_minute_from_frame(frame_number)
        
        # Check if we've moved to a new minute
        if video_minute > self.current_minute:
            # Finalize data for completed minutes
            for minute in range(self.current_minute, video_minute):
                self._finalize_minute(minute)
            
            self.current_minute = video_minute
            
            # Check if we should send a batch
            self._check_and_send_batch()
        
        # OPTIMIZATION: Use setdefault for cleaner initialization
        processed_vehicles = self.processed_vehicles_per_minute.setdefault(video_minute, set())
        
        # Skip if vehicle already processed this minute
        if vehicle_id in processed_vehicles:
            return
        
        # Mark vehicle as processed for this minute
        processed_vehicles.add(vehicle_id)
        
        # OPTIMIZATION: Get or create minute data and cache the reference
        minute_data = self.minute_data.setdefault(video_minute, {
            "vehicles": {},
            "lane_counts": {},
            "frame_range": {"start": frame_number, "end": frame_number}
        })
        
        # Update frame range
        minute_data["frame_range"]["end"] = frame_number
        
        # OPTIMIZATION: Cache references to avoid repeated dictionary lookups
        vehicles = minute_data["vehicles"]
        lane_counts = minute_data["lane_counts"]
        
        # Initialize nested structure for vehicle class if not exists
        vehicle_class_lanes = vehicles.setdefault(vehicle_class, {})
        
        # Initialize and update lane count for this class
        vehicle_class_lanes[lane_id_str] = vehicle_class_lanes.get(lane_id_str, 0) + 1
        
        # Update totals for this class
        total_lanes = vehicles.setdefault("total", {})
        total_lanes[lane_id_str] = total_lanes.get(lane_id_str, 0) + 1
        
        # Update lane_counts summary
        lane_counts[lane_id_str] = lane_counts.get(lane_id_str, 0) + 1
        
        # Track statistics for summary logging
        self.total_vehicles_processed += 1
        self.vehicles_per_minute_count[video_minute] = self.vehicles_per_minute_count.get(video_minute, 0) + 1
        
        # Only log individual vehicles in verbose mode (DEBUG level)
        if self.verbose:
            logger.debug(f"Vehicle tracked: ID={vehicle_id}, Class={vehicle_class}, Lane={lane_id}, Minute={video_minute}")
    
    def _finalize_minute(self, minute_number: int) -> None:
        """
        Finalize data for a completed minute by calculating totals and summary data.
        
        Args:
            minute_number: The minute number to finalize
        """
        if minute_number not in self.minute_data:
            # Create empty minute data if no detections occurred
            self.minute_data[minute_number] = {
                "vehicles": {"total": {}},
                "lane_counts": {},
                "frame_range": {"start": 0, "end": 0}
            }
        
        minute_data = self.minute_data[minute_number]
        
        # Calculate additional summary data for ATR format
        total_count = sum(minute_data["lane_counts"].values()) if minute_data["lane_counts"] else 0
        
        # Calculate detected classes summary
        detected_classes = {}
        vehicles = minute_data["vehicles"]
        
        for vehicle_class, lanes in vehicles.items():
            if vehicle_class != "total":  # Skip the total entry
                class_total = sum(lanes.values()) if isinstance(lanes, dict) else 0
                detected_classes[vehicle_class] = class_total
        
        # Add summary fields to match target structure
        minute_data["total_count"] = total_count
        minute_data["detected_classes"] = detected_classes
        minute_data["study_type"] = "ATR"
        
        # Clear processed vehicles for this minute to free memory
        if minute_number in self.processed_vehicles_per_minute:
            vehicle_count = len(self.processed_vehicles_per_minute[minute_number])
            del self.processed_vehicles_per_minute[minute_number]
            # Log summary every 5 minutes or in verbose mode
            if minute_number % 5 == 0 or self.verbose:
                logger.info(f"Minute {minute_number} finalized: {vehicle_count} unique vehicles, total_count={total_count}")
        else:
            if self.verbose:
                logger.debug(f"Minute {minute_number} finalized: 0 vehicles processed")
    
    def _check_and_send_batch(self) -> None:
        """Check if we should send a batch and send it if ready."""
        completed_minutes = [m for m in self.minute_data.keys() if m < self.current_minute]
        
        # Group completed minutes into batches of BATCH_SIZE
        while len(completed_minutes) >= self.BATCH_SIZE:
            batch_minutes = completed_minutes[:self.BATCH_SIZE]
            self._send_batch(batch_minutes)
            
            # Remove sent minutes from tracking
            for minute in batch_minutes:
                completed_minutes.remove(minute)
                del self.minute_data[minute]
    
    def _send_batch(self, minute_numbers: List[int]) -> None:
        """
        Send a batch of minute results via callback.
        
        Args:
            minute_numbers: List of minute numbers to include in batch
        """
        if not self.batch_callback or not minute_numbers:
            return
        
        self.batch_counter += 1
        batch_id = f"{self.video_uuid}-atr-batch-{self.batch_counter:03d}"
        
        minute_results = []
        for minute_num in sorted(minute_numbers):
            minute_data = self.minute_data.get(minute_num, {})
            minute_results.append({
                "minuteNumber": minute_num,
                "results": minute_data
            })
        
        batch_message = {
            "videoUuid": self.video_uuid,
            "status": "minute_batch",
            "batchId": batch_id,
            "startMinute": min(minute_numbers),
            "endMinute": max(minute_numbers),
            "minuteResults": minute_results,
            "studyType": "ATR"
        }
        
        # Calculate total vehicles in this batch for summary
        batch_vehicle_count = sum(
            self.vehicles_per_minute_count.get(m, 0) for m in minute_numbers
        )
        
        logger.info(f"Sending batch {batch_id}: minutes {min(minute_numbers)}-{max(minute_numbers)} ({len(minute_results)} minutes, {batch_vehicle_count} vehicles)")
        
        try:
            self.batch_callback(batch_message)
            logger.info(f"Batch {batch_id} sent successfully")
        except Exception as e:
            logger.error(f"Failed to send batch {batch_id}: {e}")
    
    def finalize_processing(self) -> int:
        """
        Finalize processing and send any remaining batches.
        
        Returns:
            Total video duration in seconds
        """
        # Finalize current minute if it has data
        if self.current_minute in self.minute_data:
            self._finalize_minute(self.current_minute)
        
        # Send any remaining minutes as a final batch
        remaining_minutes = list(self.minute_data.keys())
        if remaining_minutes:
            self._send_batch(remaining_minutes)
        
        # Calculate total duration
        total_duration = (self.current_minute + 1) * 60  # Convert minutes to seconds
        
        # Log final summary
        logger.info(f"ATR Video processing finalized: {self.current_minute + 1} minutes, {self.total_vehicles_processed} total vehicles")
        
        # Log per-minute average if we have data
        if self.vehicles_per_minute_count:
            avg_vehicles = self.total_vehicles_processed / len(self.vehicles_per_minute_count)
            logger.info(f"Average vehicles per minute: {avg_vehicles:.1f}")
        
        return total_duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked data for debugging."""
        return {
            "tracker_type": "ATR",
            "current_minute": self.current_minute,
            "tracked_minutes": list(self.minute_data.keys()),
            "batches_sent": self.batch_counter,
            "fps": self.fps
        }