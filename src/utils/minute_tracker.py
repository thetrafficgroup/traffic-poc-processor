"""
Minute tracking utility for traffic video processing.
Handles minute-by-minute data collection and batch processing for SQS.
"""

import json
import uuid
from typing import Dict, List, Optional, Callable, Any, Set


class MinuteTracker:
    """
    Tracks vehicle detections and movements on a per-minute basis.
    Manages batching and SQS message preparation with proper vehicle-direction-turn nesting.
    """
    
    def __init__(self, fps: float, video_uuid: str, batch_callback: Optional[Callable] = None):
        self.fps = fps
        self.video_uuid = video_uuid
        self.batch_callback = batch_callback
        
        # Tracking state
        self.minute_data: Dict[int, Dict] = {}
        self.current_minute = 0
        self.batch_counter = 0
        
        # Batch configuration
        self.BATCH_SIZE = 5  # Send batch every 5 minutes
        
        # Track processed vehicle IDs per minute to avoid duplicates
        self.processed_vehicles_per_minute: Dict[int, Set[int]] = {}
        
        print(f"🔄 MinuteTracker initialized for video {video_uuid} with FPS={fps}")
    
    def calculate_minute_from_frame(self, frame_number: int) -> int:
        """Calculate which minute a frame belongs to (0-based)."""
        seconds = frame_number / self.fps
        return int(seconds / 60)
    
    def process_vehicle_detection(self, frame_number: int, vehicle_id: int, vehicle_class: str, 
                                 origin_direction: str, turn_type: str) -> None:
        """
        Process a single vehicle detection with its complete movement data.
        
        Args:
            frame_number: Current frame number
            vehicle_id: Unique vehicle ID
            vehicle_class: Vehicle classification (car, bus, etc.)
            origin_direction: Direction vehicle came from (NORTH, SOUTH, EAST, WEST)
            turn_type: Turn type (left, right, straight, u-turn)
        """
        video_minute = self.calculate_minute_from_frame(frame_number)
        
        # Check if we've moved to a new minute
        if video_minute > self.current_minute:
            # Finalize data for completed minutes
            for minute in range(self.current_minute, video_minute):
                self._finalize_minute(minute)
            
            self.current_minute = video_minute
            
            # Check if we should send a batch
            self._check_and_send_batch()
        
        # Initialize minute tracking if not exists
        if video_minute not in self.processed_vehicles_per_minute:
            self.processed_vehicles_per_minute[video_minute] = set()
        
        # Skip if vehicle already processed this minute
        if vehicle_id in self.processed_vehicles_per_minute[video_minute]:
            return
        
        # Mark vehicle as processed for this minute
        self.processed_vehicles_per_minute[video_minute].add(vehicle_id)
        
        # Initialize minute data structure if not exists
        if video_minute not in self.minute_data:
            self.minute_data[video_minute] = {
                "vehicles": {},
                "frame_range": {"start": frame_number, "end": frame_number}
            }
        
        # Update frame range
        self.minute_data[video_minute]["frame_range"]["end"] = frame_number
        
        # Initialize nested structure for vehicle class if not exists
        vehicles = self.minute_data[video_minute]["vehicles"]
        if vehicle_class not in vehicles:
            vehicles[vehicle_class] = self._create_empty_direction_structure()
        
        # Update the count
        if origin_direction in vehicles[vehicle_class] and turn_type in vehicles[vehicle_class][origin_direction]:
            vehicles[vehicle_class][origin_direction][turn_type] += 1
        
        # Update totals
        if "total" not in vehicles:
            vehicles["total"] = self._create_empty_direction_structure()
        
        if origin_direction in vehicles["total"] and turn_type in vehicles["total"][origin_direction]:
            vehicles["total"][origin_direction][turn_type] += 1
    
    def _create_empty_direction_structure(self) -> Dict[str, Dict[str, int]]:
        """Create empty nested structure for directions and turns."""
        return {
            "NORTH": {"left": 0, "right": 0, "straight": 0, "u-turn": 0},
            "SOUTH": {"left": 0, "right": 0, "straight": 0, "u-turn": 0},
            "EAST": {"left": 0, "right": 0, "straight": 0, "u-turn": 0},
            "WEST": {"left": 0, "right": 0, "straight": 0, "u-turn": 0}
        }
    
    def _finalize_minute(self, minute_number: int) -> None:
        """
        Finalize data for a completed minute.
        
        Args:
            minute_number: The minute number to finalize
        """
        if minute_number not in self.minute_data:
            # Create empty minute data if no detections occurred
            self.minute_data[minute_number] = {
                "vehicles": {
                    "total": self._create_empty_direction_structure()
                },
                "frame_range": {"start": 0, "end": 0}
            }
        
        # Clear processed vehicles for this minute to free memory
        if minute_number in self.processed_vehicles_per_minute:
            vehicle_count = len(self.processed_vehicles_per_minute[minute_number])
            del self.processed_vehicles_per_minute[minute_number]
            print(f"📊 Minute {minute_number} finalized: {vehicle_count} unique vehicles processed")
        else:
            print(f"📊 Minute {minute_number} finalized: 0 vehicles processed")
    
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
        batch_id = f"{self.video_uuid}-batch-{self.batch_counter:03d}"
        
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
            "minuteResults": minute_results
        }
        
        print(f"📦 Sending batch {batch_id}: minutes {min(minute_numbers)}-{max(minute_numbers)} ({len(minute_results)} minutes)")
        
        try:
            self.batch_callback(batch_message)
            print(f"✅ Batch {batch_id} sent successfully")
        except Exception as e:
            print(f"❌ Failed to send batch {batch_id}: {e}")
    
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
        print(f"🎬 Video processing finalized. Duration: {total_duration} seconds ({self.current_minute + 1} minutes)")
        
        return total_duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked data for debugging."""
        return {
            "current_minute": self.current_minute,
            "tracked_minutes": list(self.minute_data.keys()),
            "batches_sent": self.batch_counter,
            "fps": self.fps
        }