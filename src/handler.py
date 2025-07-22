import os
import runpod
from app import process_video
from utils import download_s3_file, send_sqs_message

def handler(event):
    print("🚀 HANDLER STARTED with event:", event)  # DEBUG
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    video_uuid = event["input"]["video_uuid"]
    lines_data = event["input"]["lines"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]

    video_path = download_s3_file(bucket, video_key, "video.mp4")
    model_path = download_s3_file(bucket, model_key, "best.pt")

    def progress_callback(progress_data):
        send_sqs_message(queue_url, {
            "videoUuid": video_uuid,
            "status": "processing",
            "progress": progress_data["progress"],
            "estimatedTimeRemaining": progress_data["estimatedTimeRemaining"]
        })

    results = process_video(video_path, lines_data, model_path, progress_callback)

    send_sqs_message(queue_url, {
        "videoUuid": video_uuid,
        "status": "completed",
        "video": video_key,
        "results": results
    })

    print("✅ Handler completed successfully.")

runpod.serverless.start({"handler": handler})