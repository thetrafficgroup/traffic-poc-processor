import os
import runpod
from app import process_video
from utils import download_s3_file, send_sqs_message, upload_s3_file

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
        progress_callback,
        generate_video_output=generate_video_output,
        output_video_path=output_video_path
    )

    # Upload output video to S3 if generated
    if generate_video_output and output_video_path and os.path.exists(output_video_path):
        print(f"📤 Uploading output video to S3: {output_video_key}")
        upload_s3_file(output_video_path, bucket, output_video_key)
        results["videoOutput"] = f"s3://{bucket}/{output_video_key}"
        
        # Clean up local output video file
        os.remove(output_video_path)
        print(f"🗑️ Cleaned up local output video: {output_video_path}")

    send_sqs_message(queue_url, {
        "videoUuid": video_uuid,
        "status": "completed",
        "video": video_key,
        "results": results
    })

    print("✅ Handler completed successfully.")

runpod.serverless.start({"handler": handler})