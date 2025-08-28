import os
import runpod
from app import process_video
from aws_utils import download_s3_file, send_sqs_message, upload_s3_file
from response_normalizer import normalize_response

def handler(event):
    print("🚀 HANDLER STARTED with event:", event)  # DEBUG
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    video_uuid = event["input"]["video_uuid"]
    lines_data = event["input"]["lines"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]
    study_type = event["input"].get("study_type", "TMC")  # Default to TMC
    generate_video_output = event["input"].get("generate_video_output", False)  # Default to False

    video_path = download_s3_file(bucket, video_key, "video.mp4")
    model_path = download_s3_file(bucket, model_key, "best.pt")

    def progress_callback(progress_data):
        send_sqs_message(queue_url, {
            "videoUuid": video_uuid,
            "status": "processing",
            "progress": progress_data["progress"],
            "estimatedTimeRemaining": progress_data["estimatedTimeRemaining"]
        })
    
    def minute_batch_callback(batch_data):
        """Callback to send minute batch data via SQS"""
        print(f"📦 Sending minute batch via SQS: {batch_data.get('batchId', 'unknown')}")
        send_sqs_message(queue_url, batch_data)

    # Generate output video path if requested
    output_video_path = None
    if generate_video_output:
        # Create output filename: videos/sample.mp4 -> videos/sample_output.mp4
        base_key = video_key.rsplit('.', 1)[0]  # Remove extension
        extension = video_key.rsplit('.', 1)[1] if '.' in video_key else 'mp4'
        output_video_key = f"{base_key}_output.{extension}"
        output_video_path = f"output_{video_uuid}.{extension}"

    # Process video with optional output generation
    results = process_video(
        video_path, 
        lines_data, 
        model_path, 
        study_type,
        video_uuid=video_uuid,
        progress_callback=progress_callback,
        minute_batch_callback=minute_batch_callback,
        generate_video_output=generate_video_output,
        output_video_path=output_video_path
    )

    # Upload output video to S3 if generated
    if generate_video_output and output_video_path and os.path.exists(output_video_path):
        # Try to optimize video for web streaming using ffmpeg if available
        optimized_path = f"optimized_{output_video_path}"
        try:
            import subprocess
            # Convert to web-friendly format with faststart
            cmd = [
                'ffmpeg', '-i', output_video_path,
                '-c:v', 'libx264',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y', optimized_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(optimized_path):
                print("✅ Video optimized for web streaming")
                os.remove(output_video_path)  # Remove original
                output_video_path = optimized_path  # Use optimized version
            else:
                print(f"⚠️ FFmpeg optimization failed, using original: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Could not optimize video with ffmpeg: {e}")
        
        print(f"📤 Uploading output video to S3: {output_video_key}")
        upload_s3_file(output_video_path, bucket, output_video_key)
        results["videoOutput"] = output_video_key  # Only return the key/location
        
        # Clean up local output video file
        os.remove(output_video_path)
        print(f"🗑️ Cleaned up local output video: {output_video_path}")

    # Normalize results before sending to ensure consistent API structure
    print(f"🔄 Normalizing {study_type} results...")
    print(f"📋 Original results keys: {list(results.keys())}")
    print(f"🎬 Original videoOutput: {results.get('videoOutput', 'NOT_FOUND')}")
    
    normalized_results = normalize_response(study_type, results)
    
    print(f"✅ Results normalized. Original keys: {list(results.keys())}")
    print(f"✅ Normalized keys: {list(normalized_results.keys())}")
    print(f"🎬 Normalized videoOutput: {normalized_results.get('videoOutput', 'NOT_FOUND')}")

    # Prepare completion message with duration if available
    completion_message = {
        "videoUuid": video_uuid,
        "status": "completed",
        "video": video_key,
        "results": normalized_results
    }
    
    # Include video duration from processing results if available
    video_metadata = results.get("video_metadata", {})
    if video_metadata.get("duration_seconds"):
        completion_message["durationSeconds"] = video_metadata["duration_seconds"]
        print(f"📊 Including video duration in completion message: {video_metadata['duration_seconds']} seconds")
    
    send_sqs_message(queue_url, completion_message)

    print("✅ Handler completed successfully.")

runpod.serverless.start({"handler": handler})