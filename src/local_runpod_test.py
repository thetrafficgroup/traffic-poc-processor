import os
from app import process_video
from utils import download_s3_file, send_sqs_message

import torch

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NOT available, using CPU")


# === Simular input JSON como lo recibe RunPod ===
event = {
    "input": {
        "s3_bucket": "dolphintech-traffic-poc",
        "video_key": "videos/tmc3.mp4",
        "lines_key": "configs/lines.json",
        "model_key": "models/best.pt",
        "queue_url": "https://sqs.us-east-1.amazonaws.com/867599252360/dolphintech-traffic-poc-sqs"
    }
}

def local_test_handler(event):
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    lines_key = event["input"]["lines_key"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]

    print("Descargando modelo...")
    model_path = download_s3_file(bucket, model_key, "best.pt")
    print("Descargando líneas...")
    lines_path = download_s3_file(bucket, lines_key, "lines.json")
    print("Descargando video...")
    video_path = download_s3_file(bucket, video_key, "video.mp4")

    print("Procesando video...")
    results = process_video(video_path, lines_path, model_path)

    print("Enviando resultado a SQS...")
    send_sqs_message(queue_url, {
        "video": video_key,
        "results": results
    })

    print("=== RESULTADO FINAL ===")
    print(results)

if __name__ == "__main__":
    local_test_handler(event)