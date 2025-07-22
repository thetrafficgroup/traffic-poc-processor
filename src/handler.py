import os
import runpod
from app import process_video
from utils import download_s3_file, send_sqs_message

def handler(event):
    print("🚀 HANDLER STARTED with event:", event)  # DEBUG
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    lines_data = event["input"]["lines"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]

    video_path = download_s3_file(bucket, video_key, "video.mp4")
    model_path = download_s3_file(bucket, model_key, "best.pt")

    results = process_video(video_path, lines_data, model_path)

    send_sqs_message(queue_url, {
        "video": video_key,
        "results": results
    })

    print("✅ Handler completed successfully.")

runpod.serverless.start({"handler": handler})