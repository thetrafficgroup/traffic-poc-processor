import os
import traceback
import runpod
import boto3
from app import process_video
from aws_utils import download_s3_file, send_sqs_message, upload_s3_file
from response_normalizer import normalize_response
from log_capture import LogCapture, upload_log_to_s3

# S3 client for log uploads
s3_client = boto3.client("s3")


def handler(event):
    # Extract parameters first (before log capture starts)
    bucket = event["input"]["s3_bucket"]
    video_key = event["input"]["video_key"]
    video_uuid = event["input"]["video_uuid"]
    lines_data = event["input"]["lines"]
    model_key = event["input"]["model_key"]
    queue_url = event["input"]["queue_url"]
    study_type = event["input"].get("study_type", "TMC")  # Default to TMC
    generate_video_output = event["input"].get("generate_video_output", False)  # Default to False
    trim_periods = event["input"].get("trim_periods", None)  # Optional trimming periods

    # Start log capture - all print statements will be captured and uploaded to S3
    log_capture = LogCapture(video_uuid, study_type)
    result = None
    caught_error = None

    try:
        with log_capture:
            try:
                result = _process_video_job(
                    event, bucket, video_key, video_uuid, lines_data, model_key,
                    queue_url, study_type, generate_video_output, trim_periods
                )
            except Exception as e:
                # Log the error before it propagates
                print(f"❌ ERROR: {type(e).__name__}: {e}")
                print(f"❌ TRACEBACK:\n{traceback.format_exc()}")
                caught_error = e
    finally:
        # Upload log to S3 AFTER context manager exits (so footer is included)
        # This runs even if an exception occurred
        try:
            log_key = upload_log_to_s3(log_capture, bucket, s3_client)
            if log_key:
                print(f"📋 Log uploaded to S3: s3://{bucket}/{log_key}")
        except Exception as upload_error:
            # Don't let log upload failure mask the original error
            print(f"⚠️ Failed to upload log: {upload_error}")

    # Re-raise original error if one occurred
    if caught_error:
        raise caught_error

    return result


def _process_video_job(event, bucket, video_key, video_uuid, lines_data, model_key,
                       queue_url, study_type, generate_video_output, trim_periods):
    """Main video processing logic, separated for cleaner log capture."""
    print("🚀 HANDLER STARTED with event: ", event)  # DEBUG

    # Use video_uuid to create unique filenames for parallel processing
    # This prevents race conditions where multiple jobs overwrite each other's files
    video_path = download_s3_file(bucket, video_key, f"video_{video_uuid}.mp4")

    # Download the YOLO model
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

    # Log trimming information
    if trim_periods:
        print(f"📊 Video trimming enabled: {len(trim_periods)} period(s)")
        for i, period in enumerate(trim_periods):
            print(f"   Period {i+1}: {period.get('start', 0)}s - {period.get('end', 0)}s")
    else:
        print("📊 No trimming specified, processing entire video")

    # Process video with optional output generation and trimming
    results = process_video(
        video_path,
        lines_data,
        model_path,
        study_type,
        video_uuid=video_uuid,
        progress_callback=progress_callback,
        minute_batch_callback=minute_batch_callback,
        generate_video_output=generate_video_output,
        output_video_path=output_video_path,
        trim_periods=trim_periods
    )

    # CRITICAL: Clean up input video file to prevent disk accumulation
    # RunPod workers are reused, so files accumulate if not deleted
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"🗑️ Cleaned up input video: {video_path}")

    # Upload output video to S3 if generated
    if generate_video_output and output_video_path and os.path.exists(output_video_path):
        print(f"📊 Processing output video for hybrid streaming: {output_video_path}")
        
        # Get original file size for decision making
        original_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"📊 Original video size: {original_size_mb:.2f} MB")
        results["originalSizeMB"] = round(original_size_mb, 2)
        
        # Apply aggressive FFmpeg compression
        compressed_path = f"compressed_{output_video_path}"
        hls_playlist_path = None
        hls_folder = None

        try:
            import subprocess
            import shutil

            # Calculate dynamic timeout based on file size
            # Allow 5 minutes per GB, minimum 10 minutes, maximum 4 hours
            compression_timeout = max(600, min(14400, int(original_size_mb / 1024 * 300)))
            print(f"📊 Applying FFmpeg compression (timeout: {compression_timeout}s for {original_size_mb:.0f}MB)...")

            # First, probe the video to check if it has audio
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', output_video_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            has_audio = 'audio' in probe_result.stdout
            print(f"📊 Video has audio: {has_audio}")

            # Build compression command - handle videos with or without audio
            compression_cmd = [
                'ffmpeg', '-i', output_video_path,
                # Video compression
                '-c:v', 'libx264',
                '-preset', 'fast',  # Good balance of speed and compression (3x faster than veryslow)
                '-crf', '26',  # Good compression with slightly better quality than 28
                '-profile:v', 'baseline',
                '-level', '3.1',
                '-pix_fmt', 'yuv420p',
            ]

            # Only add audio encoding if the source has audio
            if has_audio:
                compression_cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '64k',
                    '-ar', '22050',
                ])
            else:
                compression_cmd.extend(['-an'])  # No audio

            # Add optimization flags
            compression_cmd.extend([
                '-movflags', '+faststart',
                '-tune', 'stillimage',  # Optimize for traffic footage
                '-y', compressed_path
            ])

            print(f"🔄 Running compression: {' '.join(compression_cmd)}")
            result = subprocess.run(compression_cmd, capture_output=True, text=True, timeout=compression_timeout)
            
            if result.returncode == 0 and os.path.exists(compressed_path):
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                compression_ratio = compressed_size_mb / original_size_mb
                print(f"✅ Video compressed: {compressed_size_mb:.2f} MB ({compression_ratio:.2f}x compression)")
                
                # Store compression statistics
                results["compressionRatio"] = round(compression_ratio, 3)
                
                # Remove original and use compressed version
                os.remove(output_video_path)
                output_video_path = compressed_path
                
            else:
                print(f"⚠️ FFmpeg compression failed: {result.stderr}")
                print("⚠️ Using original video without compression")
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
                    
        except Exception as e:
            print(f"⚠️ Compression failed: {e}")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        
        # Check final file size and generate HLS if needed
        final_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"📊 Final video size: {final_size_mb:.2f} MB")
        
        # Generate HLS for videos > 500MB
        if final_size_mb > 500:
            print(f"🎥 Video exceeds 500MB, generating HLS playlist...")

            try:
                # Create HLS output directory
                video_basename = os.path.splitext(output_video_path)[0]
                hls_folder = f"{video_basename}_hls"
                os.makedirs(hls_folder, exist_ok=True)

                hls_playlist_path = os.path.join(hls_folder, "playlist.m3u8")

                # Calculate dynamic timeout for HLS (longer than compression)
                hls_timeout = max(900, int(final_size_mb / 1024 * 600))  # 10 min per GB

                # Check if video has audio (may have been set earlier, or probe again)
                if 'has_audio' not in dir():
                    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', output_video_path]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                    has_audio = 'audio' in probe_result.stdout

                # Build HLS command based on whether video has audio
                if has_audio:
                    # HLS generation command with multiple quality levels (with audio)
                    hls_cmd = [
                        'ffmpeg', '-i', output_video_path,
                        # 720p stream
                        '-map', '0:v:0', '-map', '0:a:0',
                        '-c:v:0', 'libx264', '-c:a:0', 'aac',
                        '-b:v:0', '2500k', '-b:a:0', '128k',
                        '-s:v:0', '1280x720', '-profile:v:0', 'main',
                        # 480p stream
                        '-map', '0:v:0', '-map', '0:a:0',
                        '-c:v:1', 'libx264', '-c:a:1', 'aac',
                        '-b:v:1', '1000k', '-b:a:1', '64k',
                        '-s:v:1', '854x480', '-profile:v:1', 'baseline',
                        # 360p stream
                        '-map', '0:v:0', '-map', '0:a:0',
                        '-c:v:2', 'libx264', '-c:a:2', 'aac',
                        '-b:v:2', '600k', '-b:a:2', '64k',
                        '-s:v:2', '640x360', '-profile:v:2', 'baseline',
                        # HLS settings
                        '-f', 'hls',
                        '-hls_time', '3600',
                        '-hls_playlist_type', 'vod',
                        '-hls_segment_filename', os.path.join(hls_folder, 'segment_%v_%03d.ts'),
                        '-master_pl_name', 'master.m3u8',
                        '-var_stream_map', 'v:0,a:0,name:720p v:1,a:1,name:480p v:2,a:2,name:360p',
                        '-y', hls_playlist_path
                    ]
                else:
                    # HLS generation command for video-only (no audio)
                    hls_cmd = [
                        'ffmpeg', '-i', output_video_path,
                        # 720p stream (video only)
                        '-map', '0:v:0',
                        '-c:v:0', 'libx264',
                        '-b:v:0', '2500k',
                        '-s:v:0', '1280x720', '-profile:v:0', 'main',
                        # 480p stream
                        '-map', '0:v:0',
                        '-c:v:1', 'libx264',
                        '-b:v:1', '1000k',
                        '-s:v:1', '854x480', '-profile:v:1', 'baseline',
                        # 360p stream
                        '-map', '0:v:0',
                        '-c:v:2', 'libx264',
                        '-b:v:2', '600k',
                        '-s:v:2', '640x360', '-profile:v:2', 'baseline',
                        # HLS settings
                        '-f', 'hls',
                        '-hls_time', '3600',
                        '-hls_playlist_type', 'vod',
                        '-hls_segment_filename', os.path.join(hls_folder, 'segment_%v_%03d.ts'),
                        '-master_pl_name', 'master.m3u8',
                        '-var_stream_map', 'v:0,name:720p v:1,name:480p v:2,name:360p',
                        '-y', hls_playlist_path
                    ]

                print(f"🎥 Generating HLS (timeout: {hls_timeout}s, has_audio: {has_audio}): {' '.join(hls_cmd[:10])}... (truncated)")
                hls_result = subprocess.run(hls_cmd, capture_output=True, text=True, timeout=hls_timeout)
                
                if hls_result.returncode == 0 and os.path.exists(hls_playlist_path):
                    print("✅ HLS playlist generated successfully")
                    
                    # Store HLS generation timestamp
                    import datetime
                    results["hlsGeneratedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
                    
                    # Upload HLS files to S3
                    hls_s3_prefix = f"{output_video_key}_hls"
                    
                    # Upload all HLS files
                    for root, dirs, files in os.walk(hls_folder):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, hls_folder)
                            s3_key = f"{hls_s3_prefix}/{relative_path}"
                            upload_s3_file(local_file_path, bucket, s3_key)
                            print(f"📤 Uploaded HLS file: {s3_key}")
                    
                    results["hlsPlaylist"] = f"{hls_s3_prefix}/master.m3u8"
                    results["isHlsEnabled"] = True
                    
                    # Store streaming metadata
                    results["streamingMetadata"] = {
                        "qualities": ["720p", "480p", "360p"],
                        "segmentDuration": 10,
                        "playlistType": "vod",
                        "s3Prefix": hls_s3_prefix,
                        "masterPlaylist": "master.m3u8"
                    }
                    
                    print(f"✅ HLS master playlist available at: {results['hlsPlaylist']}")
                    
                    # Clean up local HLS folder
                    shutil.rmtree(hls_folder)
                    print(f"🗑️ Cleaned up local HLS folder: {hls_folder}")
                    
                else:
                    print(f"⚠️ HLS generation failed: {hls_result.stderr}")
                    results["isHlsEnabled"] = False
                    if os.path.exists(hls_folder):
                        shutil.rmtree(hls_folder)
                        
            except Exception as e:
                print(f"⚠️ HLS generation error: {e}")
                results["isHlsEnabled"] = False
                if hls_folder and os.path.exists(hls_folder):
                    shutil.rmtree(hls_folder)
        else:
            print("📊 Video under 500MB, skipping HLS generation")
            results["isHlsEnabled"] = False
        
        print(f"📤 Uploading final video to S3: {output_video_key}")
        upload_s3_file(output_video_path, bucket, output_video_key)
        results["videoOutput"] = output_video_key  # Only return the key/location
        results["videoSizeMB"] = final_size_mb
        
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
    
    # Include video duration and streaming info from processing results if available
    video_metadata = results.get("video_metadata", {})
    if video_metadata.get("duration_seconds"):
        completion_message["durationSeconds"] = video_metadata["duration_seconds"]
        print(f"📊 Including video duration in completion message: {video_metadata['duration_seconds']} seconds")
    
    # Include HLS streaming information if available
    if results.get("isHlsEnabled"):
        completion_message["isHlsEnabled"] = True
        completion_message["hlsPlaylist"] = results.get("hlsPlaylist")
        print(f"🎥 Including HLS streaming info: {results.get('hlsPlaylist')}")
    else:
        completion_message["isHlsEnabled"] = False
    
    # Include video size information
    if results.get("videoSizeMB"):
        completion_message["videoSizeMB"] = results["videoSizeMB"]
        print(f"📊 Including video size: {results['videoSizeMB']:.2f} MB")
    
    # Include compression statistics
    if results.get("compressionRatio"):
        completion_message["compressionRatio"] = results["compressionRatio"]
        print(f"📊 Including compression ratio: {results['compressionRatio']:.3f}")
    
    if results.get("originalSizeMB"):
        completion_message["originalSizeMB"] = results["originalSizeMB"]
        print(f"📊 Including original size: {results['originalSizeMB']:.2f} MB")
    
    # Include streaming metadata and HLS timestamp
    if results.get("streamingMetadata"):
        completion_message["streamingMetadata"] = results["streamingMetadata"]
        print(f"🎥 Including streaming metadata with {len(results['streamingMetadata'].get('qualities', []))} quality levels")
    
    if results.get("hlsGeneratedAt"):
        completion_message["hlsGeneratedAt"] = results["hlsGeneratedAt"]
        print(f"🎥 Including HLS generation timestamp: {results['hlsGeneratedAt']}")
    
    send_sqs_message(queue_url, completion_message)

    print("✅ Handler completed successfully.")

    # Return results for RunPod
    return {"status": "completed", "video_uuid": video_uuid}


runpod.serverless.start({"handler": handler})