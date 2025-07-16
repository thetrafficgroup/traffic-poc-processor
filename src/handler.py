import os
import runpod
from src.app import process_video
from src.utils import download_s3_file, send_sqs_message

def handler(event):
    print("🚀 HANDLER STARTED with event:", event)  # DEBUG
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    lines_key = event["input"]["lines_key"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]

    video_path = download_s3_file(bucket, video_key, "video.mp4")
    lines_path = download_s3_file(bucket, lines_key, "lines.json")
    model_path = download_s3_file(bucket, model_key, "best.pt")

    results = process_video(video_path, lines_path, model_path)

    send_sqs_message(queue_url, {
        "video": video_key,
        "results": results
    })

    print("✅ Handler completed successfully.")

runpod.serverless.start({"handler": handler})