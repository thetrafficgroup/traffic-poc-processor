import os
import json
from app import process_video
from aws_utils import download_s3_file, send_sqs_message

import torch

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NOT available, using CPU")


# === Simular input JSON como lo recibe RunPod ===
# Test event for TMC
event_tmc = {
    "input": {
        "s3_bucket": "trafficgroup-ne-data-sb",
        "video_key": "videos/sample.mp4",
        "video_uuid": "test-uuid-123",
        "lines": {
            "NORTE": {
                "pt1": {"x": 100, "y": 200},
                "pt2": {"x": 300, "y": 250}
            },
            "SUR": {
                "pt1": {"x": 150, "y": 400},
                "pt2": {"x": 350, "y": 450}
            }
        },
        "model_key": "models/best.pt",
        "queue_url": "https://sqs.us-east-1.amazonaws.com/test-queue",
        "study_type": "TMC",
        "generate_video_output": False
    }
}

# Test event for ATR
event_atr = {
    "input": {
        "s3_bucket": "trafficgroup-ne-data-sb", 
        "video_key": "videos/sample.mp4",
        "video_uuid": "test-uuid-456",
        "lines": {
            "lanes": [
                {
                    "id": 0,
                    "name": "Lane 1",
                    "points": [
                        {"x": 6.331182795698925, "y": 242.4516129032258},
                        {"x": 628.4387096774194, "y": 68.48172043010753},
                        {"x": 600.9118279569892, "y": 49.763440860215056},
                        {"x": 14.038709677419355, "y": 143.3548387096774}
                    ]
                },
                {
                    "id": 1,
                    "name": "Lane 2", 
                    "points": [
                        {"x": 11.836559139784946, "y": 260.06881720430107},
                        {"x": 23.948387096774194, "y": 458.26236559139784},
                        {"x": 431.3462365591398, "y": 465.9698924731183},
                        {"x": 631.741935483871, "y": 201.71182795698925},
                        {"x": 624.0344086021505, "y": 81.69462365591397}
                    ]
                }
            ],
            "finish_line": [
                {"x": 539.2516129032258, "y": 448.35268817204303},
                {"x": 218.83870967741936, "y": 56.36989247311828}
            ]
        },
        "model_key": "models/best.pt",
        "queue_url": "https://sqs.us-east-1.amazonaws.com/test-queue",
        "study_type": "ATR",
        "generate_video_output": True  # Test video output generation
    }
}

def local_test_handler(event):
    print("🚀 LOCAL TEST STARTED with event:", event)
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    video_uuid = event["input"]["video_uuid"]
    lines_data = event["input"]["lines"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]
    study_type = event["input"].get("study_type", "TMC")
    generate_video_output = event["input"].get("generate_video_output", False)

    print(f"📊 Study Type: {study_type}")
    print(f"🎥 Video Output: {'Enabled' if generate_video_output else 'Disabled'}")

    print("⬇️ Descargando archivos...")
    video_path = download_s3_file(bucket, video_key, "video.mp4")
    model_path = download_s3_file(bucket, model_key, "best.pt")

    def progress_callback(progress_data):
        print(f"📈 Progress: {progress_data['progress']}% | ETA: {progress_data['estimatedTimeRemaining']}s")
        # Simular envío a SQS (comentado para evitar spam en test local)
        # send_sqs_message(queue_url, {
        #     "videoUuid": video_uuid,
        #     "status": "processing",
        #     "progress": progress_data["progress"],
        #     "estimatedTimeRemaining": progress_data["estimatedTimeRemaining"]
        # })

    # Generate output video path if requested
    output_video_path = None
    if generate_video_output:
        base_key = video_key.rsplit('.', 1)[0]
        extension = video_key.rsplit('.', 1)[1] if '.' in video_key else 'mp4'
        output_video_path = f"local_output_{video_uuid}.{extension}"
        print(f"📼 Output video will be saved as: {output_video_path}")

    print("🎬 Procesando video...")
    results = process_video(video_path, lines_data, model_path, study_type, progress_callback,
                           generate_video_output, output_video_path)

    # Handle output video if generated (local testing - don't upload to S3)
    if generate_video_output and output_video_path and os.path.exists(output_video_path):
        print(f"✅ Output video generated locally: {output_video_path}")
        # For local testing, just return the filename (simulating the S3 key format)
        base_key = video_key.rsplit('.', 1)[0]
        extension = video_key.rsplit('.', 1)[1] if '.' in video_key else 'mp4'
        results["videoOutput"] = f"{base_key}_output.{extension}"

    print("📤 Enviando resultado final a SQS...")
    send_sqs_message(queue_url, {
        "videoUuid": video_uuid,
        "status": "completed",
        "video": video_key,
        "results": results
    })

    print("✅ Local test completed successfully.")
    print("\n=== RESULTADO FINAL ===")
    print(json.dumps(results, indent=2))
    return results

def test_both_types():
    """Test both TMC and ATR processors"""
    print("\n" + "="*50)
    print("🧪 TESTING TMC PROCESSOR")
    print("="*50)
    tmc_results = local_test_handler(event_tmc)
    
    print("\n" + "="*50)
    print("🧪 TESTING ATR PROCESSOR") 
    print("="*50)
    atr_results = local_test_handler(event_atr)
    
    return tmc_results, atr_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].upper()
        if test_type == "TMC":
            print("Testing TMC only...")
            local_test_handler(event_tmc)
        elif test_type == "ATR":
            print("Testing ATR only...")
            local_test_handler(event_atr)
        else:
            print("Usage: python local_runpod_test.py [TMC|ATR]")
    else:
        # Test both by default
        test_both_types()