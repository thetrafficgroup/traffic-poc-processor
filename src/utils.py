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

def upload_s3_file(local_path, bucket, key):
    """Upload a local file to S3 bucket with proper content type"""
    print(f"→ Subiendo {local_path} a s3://{bucket}/{key}...")
    
    # Determine content type based on file extension
    content_type = "application/octet-stream"  # default
    if key.lower().endswith('.mp4'):
        content_type = "video/mp4"
    elif key.lower().endswith('.avi'):
        content_type = "video/x-msvideo"
    elif key.lower().endswith('.mov'):
        content_type = "video/quicktime"
    elif key.lower().endswith('.mkv'):
        content_type = "video/x-matroska"
    
    s3.upload_file(
        local_path, 
        bucket, 
        key,
        ExtraArgs={
            'ContentType': content_type,
            'Metadata': {
                'uploaded-by': 'traffic-cv-processor'
            }
        }
    )
    print(f"✓ Archivo subido exitosamente con content-type: {content_type}")
    return f"s3://{bucket}/{key}"