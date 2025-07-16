import boto3
import os
import json

s3 = boto3.client("s3")
sqs = boto3.client("sqs")

def download_s3_file(bucket, key, local_path):
    if os.path.exists(local_path):
        print(f"✓ Ya existe local: {local_path}, no se descarga.")
        return local_path

    print(f"→ Descargando {key} desde bucket {bucket}...")
    s3.download_file(bucket, key, local_path)
    return local_path

def send_sqs_message(queue_url, payload):
    print(f"→ Enviando a SQS: {queue_url}")
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(payload)
    )