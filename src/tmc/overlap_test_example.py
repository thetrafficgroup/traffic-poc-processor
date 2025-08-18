"""
Example script demonstrating overlap detection improvements in TMC processor.
Run this to test the enhanced tracking capabilities.
"""

import cv2
import numpy as np
from tmc_processor import process_video

# Example test configuration
def test_overlap_improvements():
    """
    Test the overlap detection improvements with sample data.
    """
    
    # Sample configuration - adjust paths as needed
    test_config = {
        "video_path": "sample_intersection_video.mp4",
        "model_path": "best.pt",
        "lines_data": {
            "NORTH": {"pt1": [320, 100], "pt2": [380, 100]},
            "SOUTH": {"pt1": [320, 500], "pt2": [380, 500]},
            "EAST": {"pt1": [600, 280], "pt2": [600, 340]},
            "WEST": {"pt1": [50, 280], "pt2": [50, 340]}
        }
    }
    
    def progress_callback(progress_data):
        print(f"Processing: {progress_data['progress']:.1f}% complete, "
              f"ETA: {progress_data.get('estimatedTimeRemaining', 0)} seconds")
    
    # Run enhanced processing
    print("🚀 Starting overlap-enhanced TMC processing...")
    results = process_video(
        VIDEO_PATH=test_config["video_path"],
        LINES_DATA=test_config["lines_data"],
        MODEL_PATH=test_config["model_path"],
        progress_callback=progress_callback,
        generate_video_output=True,
        output_video_path="output_with_overlap_detection.mp4"
    )
    
    # Display overlap analysis results
    print("\n📊 OVERLAP DETECTION ANALYSIS")
    print("=" * 50)
    
    overlap_analysis = results.get("overlap_analysis", {})
    if overlap_analysis:
        print(f"Frames with overlaps: {overlap_analysis['frames_with_overlaps']}")
        print(f"Total overlaps detected: {overlap_analysis['total_overlaps_detected']}")
        print(f"Overlap frame ratio: {overlap_analysis['overlap_frame_ratio']:.3f}")
        
        enhancements = overlap_analysis.get('processing_enhancements', {})
        print(f"\nEnhancements Applied:")
        print(f"✅ Soft-NMS: {enhancements.get('soft_nms_applied', False)}")
        print(f"✅ Track Interpolation: {enhancements.get('track_interpolation', False)}")
        print(f"✅ Confidence Adjustment: {enhancements.get('confidence_adjustment', False)}")
    
    # Display standard results
    print(f"\n📈 DETECTION RESULTS")
    print("=" * 50)
    print(f"Total vehicles detected: {results['total']}")
    print(f"Vehicle classes: {results['detected_classes']}")
    print(f"Turn analysis: {results['turns']}")
    
    # Validation results
    validation = results.get("validation", {})
    print(f"\n✅ VALIDATION")
    print("=" * 50)
    print(f"Validation passed: {validation.get('validation_passed', False)}")
    print(f"Vehicles with movement: {validation.get('vehicles_with_movement', 0)}")
    print(f"Total turns counted: {validation.get('total_turns', 0)}")
    
    return results

if __name__ == "__main__":
    # Run the test
    try:
        results = test_overlap_improvements()
        print("\n🎉 Overlap detection test completed successfully!")
        
        # Save detailed results
        import json
        with open("overlap_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📄 Detailed results saved to overlap_test_results.json")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Check that you have:")
        print("- Valid video file path")
        print("- YOLO model file (best.pt)")
        print("- Correct line coordinates for your video")