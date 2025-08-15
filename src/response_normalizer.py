"""
Response normalizer for TMC and ATR video processing results.
Ensures consistent response structure regardless of processor type.
"""

from typing import Dict, Any, Optional


def normalize_response(processor_type: str, raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize response from TMC, ATR, or LOW_RATE_ATR processors to a unified structure.
    
    Args:
        processor_type: Either "TMC", "ATR", or "LOW_RATE_ATR" 
        raw_result: Raw result from the processor
        
    Returns:
        Normalized response with all fields present (null if not applicable)
    """
    
    if processor_type.upper() == "TMC":
        return normalize_tmc_response(raw_result)
    elif processor_type.upper() == "ATR":
        return normalize_atr_response(raw_result)
    elif processor_type.upper() == "LOW_RATE_ATR" or processor_type.upper() == "LOW-RATE-ATR":
        return normalize_low_rate_atr_response(raw_result)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def normalize_tmc_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize TMC response ensuring all fields are present.
    """
    
    # Extract data with defaults
    counts = result.get("counts", {})
    turns = result.get("turns", {})
    total = result.get("total", 0)
    detected_classes = result.get("detected_classes", {})
    validation = result.get("validation", {})
    vehicles = result.get("vehicles", {})  # NEW: Vehicle class analysis
    
    # Ensure all direction counts exist
    direction_counts = {
        "NORTH": counts.get("NORTH", 0),
        "SOUTH": counts.get("SOUTH", 0),
        "EAST": counts.get("EAST", 0),
        "WEST": counts.get("WEST", 0)
    }
    
    # Ensure all turn types exist
    turn_counts = {
        "left": turns.get("left", 0),
        "right": turns.get("right", 0),
        "straight": turns.get("straight", 0),
        "u-turn": turns.get("u-turn", 0)
    }
    
    # Build normalized response
    return {
        "study_type": "TMC",
        "total_count": total,
        "total_vehicles": total,
        
        # Direction-based counts
        "direction_counts": direction_counts,
        "counts": direction_counts,  # Backward compatibility
        
        # Turn analysis
        "turns": turn_counts,
        "turn_summary": {
            "total_turns": sum(turn_counts.values()),
            "left_turns": turn_counts["left"],
            "right_turns": turn_counts["right"],
            "straight": turn_counts["straight"],
            "u_turns": turn_counts["u-turn"]
        },
        
        # Vehicle classification
        "detected_classes": detected_classes,
        "vehicle_types": detected_classes,  # Alias for consistency
        
        # NEW: Vehicle class analysis grouped by vehicle type first
        "vehicles": vehicles,  # Structure: {vehicle_class: {origin: {turn_type: count}}}
        
        # Lane-based counts (null for TMC)
        "lane_counts": None,
        "lanes": None,
        
        # Validation data
        "validation": {
            "total_vehicles": validation.get("total_vehicles", total),
            "total_turns": validation.get("total_turns", sum(turn_counts.values())),
            "validation_passed": validation.get("validation_passed", 
                                               total == sum(turn_counts.values())),
            "entry_vehicles": validation.get("entry_vehicles", total),
            "total_crossings": validation.get("total_crossings", sum(direction_counts.values()))
        },
        
        # Metadata
        "metadata": {
            "processor": "TMC",
            "has_turn_analysis": True,
            "has_lane_analysis": False,
            "has_direction_analysis": True,
            "has_vehicle_class_analysis": bool(vehicles)  # Indicate if new analysis is available
        }
    }


def normalize_atr_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize ATR response ensuring all fields are present.
    """
    
    # Extract data with defaults
    lane_counts = result.get("lane_counts", {})
    total_count = result.get("total_count", 0)
    detected_classes = result.get("detected_classes", {})
    
    # Convert lane counts to list format for better structure
    lanes = []
    for lane_id, count in lane_counts.items():
        lanes.append({
            "id": lane_id,
            "count": count,
            "percentage": round((count / total_count * 100), 2) if total_count > 0 else 0
        })
    
    # Build normalized response
    return {
        "study_type": "ATR",
        "total_count": total_count,
        "total_vehicles": total_count,
        
        # Lane-based counts
        "lane_counts": lane_counts,
        "lanes": lanes,
        
        # Direction-based counts (null for ATR)
        "direction_counts": None,
        "counts": None,
        
        # Turn analysis (null for ATR)
        "turns": None,
        "turn_summary": {
            "total_turns": None,
            "left_turns": None,
            "right_turns": None,
            "straight": None,
            "u_turns": None
        },
        
        # Vehicle classification
        "detected_classes": detected_classes,
        "vehicle_types": detected_classes,  # Alias for consistency
        
        # Validation data
        "validation": {
            "total_vehicles": total_count,
            "total_turns": None,  # Not applicable for ATR
            "validation_passed": True,  # ATR doesn't have turn validation
            "entry_vehicles": total_count,
            "total_crossings": total_count
        },
        
        # Metadata
        "metadata": {
            "processor": "ATR",
            "has_turn_analysis": False,
            "has_lane_analysis": True,
            "has_direction_analysis": False
        }
    }


def normalize_low_rate_atr_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize LOW_RATE_ATR response ensuring all fields are present.
    Similar to ATR but with additional metadata about the zone-based detection.
    """
    
    # Extract data with defaults
    lane_counts = result.get("lane_counts", {})
    total_count = result.get("total_count", 0)
    detected_classes = result.get("detected_classes", {})
    metadata = result.get("metadata", {})
    
    # Convert lane counts to list format for better structure
    lanes = []
    for lane_id, count in lane_counts.items():
        lanes.append({
            "id": lane_id,
            "count": count,
            "percentage": round((count / total_count * 100), 2) if total_count > 0 else 0
        })
    
    # Build normalized response
    return {
        "study_type": "LOW_RATE_ATR",
        "total_count": total_count,
        "total_vehicles": total_count,
        
        # Lane-based counts
        "lane_counts": lane_counts,
        "lanes": lanes,
        
        # Direction-based counts (null for LOW_RATE_ATR)
        "direction_counts": None,
        "counts": None,
        
        # Turn analysis (null for LOW_RATE_ATR)
        "turns": None,
        "turn_summary": {
            "total_turns": None,
            "left_turns": None,
            "right_turns": None,
            "straight": None,
            "u_turns": None
        },
        
        # Vehicle classification
        "detected_classes": detected_classes,
        "vehicle_types": detected_classes,  # Alias for consistency
        
        # Validation data
        "validation": {
            "total_vehicles": total_count,
            "total_turns": None,  # Not applicable for LOW_RATE_ATR
            "validation_passed": True,  # LOW_RATE_ATR doesn't have turn validation
            "entry_vehicles": total_count,
            "total_crossings": total_count,
            "detection_method": metadata.get("detection_method", "zone_based"),
            "total_tracked": metadata.get("total_tracked", 0),
            "total_counted": metadata.get("total_counted", total_count)
        },
        
        # Metadata
        "metadata": {
            "processor": "LOW_RATE_ATR",
            "has_turn_analysis": False,
            "has_lane_analysis": True,
            "has_direction_analysis": False,
            "detection_method": "zone_based",
            "line_a": metadata.get("line_a"),
            "line_b": metadata.get("line_b")
        }
    }


def get_empty_normalized_response(processor_type: str) -> Dict[str, Any]:
    """
    Get an empty normalized response structure with all fields set to null/0.
    Useful for error cases or when no data is available.
    """
    
    # Normalize processor type
    proc_type = processor_type.upper().replace("-", "_")
    
    return {
        "study_type": proc_type,
        "total_count": 0,
        "total_vehicles": 0,
        
        # Direction-based counts
        "direction_counts": {
            "NORTH": 0,
            "SOUTH": 0,
            "EAST": 0,
            "WEST": 0
        } if proc_type == "TMC" else None,
        "counts": {
            "NORTH": 0,
            "SOUTH": 0,
            "EAST": 0,
            "WEST": 0
        } if proc_type == "TMC" else None,
        
        # Turn analysis
        "turns": {
            "left": 0,
            "right": 0,
            "straight": 0,
            "u-turn": 0
        } if proc_type == "TMC" else None,
        "turn_summary": {
            "total_turns": 0 if proc_type == "TMC" else None,
            "left_turns": 0 if proc_type == "TMC" else None,
            "right_turns": 0 if proc_type == "TMC" else None,
            "straight": 0 if proc_type == "TMC" else None,
            "u_turns": 0 if proc_type == "TMC" else None
        },
        
        # Lane-based counts
        "lane_counts": {} if proc_type in ["ATR", "LOW_RATE_ATR"] else None,
        "lanes": [] if proc_type in ["ATR", "LOW_RATE_ATR"] else None,
        
        # Vehicle classification
        "detected_classes": {},
        "vehicle_types": {},
        
        # Validation data
        "validation": {
            "total_vehicles": 0,
            "total_turns": 0 if processor_type.upper() == "TMC" else None,
            "validation_passed": True,
            "entry_vehicles": 0,
            "total_crossings": 0
        },
        
        # Metadata
        "metadata": {
            "processor": proc_type,
            "has_turn_analysis": proc_type == "TMC",
            "has_lane_analysis": proc_type in ["ATR", "LOW_RATE_ATR"],
            "has_direction_analysis": proc_type == "TMC",
            "detection_method": "zone_based" if proc_type == "LOW_RATE_ATR" else None
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Example TMC result
    tmc_result = {
        "counts": {"NORTH": 10, "SOUTH": 8, "EAST": 5},
        "turns": {"left": 3, "right": 2},
        "total": 23,
        "detected_classes": {"car": 20, "truck": 3}
    }
    
    # Example ATR result
    atr_result = {
        "lane_counts": {"1": 15, "2": 10, "3": 5},
        "total_count": 30,
        "detected_classes": {"car": 25, "bus": 5}
    }
    
    # Normalize both
    normalized_tmc = normalize_response("TMC", tmc_result)
    normalized_atr = normalize_response("ATR", atr_result)
    
    print("TMC Normalized:", normalized_tmc)
    print("\nATR Normalized:", normalized_atr)
    
    # Get empty responses
    empty_tmc = get_empty_normalized_response("TMC")
    empty_atr = get_empty_normalized_response("ATR")
    
    print("\nEmpty TMC:", empty_tmc)
    print("\nEmpty ATR:", empty_atr)