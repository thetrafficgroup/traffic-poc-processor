"""
Log capture utility for S3-based logging.
Captures all stdout during video processing and uploads to S3.
"""
import sys
import io
from datetime import datetime, timezone


class LogCapture:
    """
    Context manager that captures stdout while still printing to console.
    Stores captured output for later upload to S3.
    """

    def __init__(self, video_uuid: str, study_type: str):
        self.video_uuid = video_uuid
        self.study_type = study_type
        self.buffer = io.StringIO()
        self.original_stdout = None
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        self.original_stdout = sys.stdout
        sys.stdout = _TeeOutput(self.original_stdout, self.buffer)

        # Write log header
        print("=" * 80)
        print(f"TRAFFIC PROCESSOR LOG")
        print(f"=" * 80)
        print(f"Video UUID:  {self.video_uuid}")
        print(f"Study Type:  {self.study_type}")
        print(f"Start Time:  {self.start_time.isoformat()}")
        print(f"=" * 80)
        print()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds()

        # Write log footer
        print()
        print("=" * 80)
        print(f"END OF LOG")
        print(f"=" * 80)
        print(f"End Time:    {self.end_time.isoformat()}")
        print(f"Duration:    {duration:.2f} seconds ({duration/60:.2f} minutes)")
        if exc_type:
            print(f"Status:      FAILED")
            print(f"Error Type:  {exc_type.__name__}")
            print(f"Error:       {exc_val}")
        else:
            print(f"Status:      SUCCESS")
        print(f"=" * 80)

        # Restore original stdout
        sys.stdout = self.original_stdout

        # Don't suppress exceptions
        return False

    def get_log_content(self) -> str:
        """Get the captured log content."""
        return self.buffer.getvalue()

    def get_log_key(self, prefix: str = "logs") -> str:
        """
        Generate S3 key for the log file.
        Format: logs/YYYY/MM/DD/{video_uuid}.log
        """
        date_path = self.start_time.strftime("%Y/%m/%d")
        return f"{prefix}/{date_path}/{self.video_uuid}.log"


class _TeeOutput:
    """
    Output stream that writes to multiple destinations.
    Used to capture stdout while still displaying to console.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            # Flush immediately for real-time console output
            if hasattr(stream, 'flush'):
                stream.flush()

    def flush(self):
        for stream in self.streams:
            if hasattr(stream, 'flush'):
                stream.flush()


def upload_log_to_s3(log_capture: LogCapture, bucket: str, s3_client) -> str:
    """
    Upload captured log content to S3.

    Args:
        log_capture: LogCapture instance with captured content
        bucket: S3 bucket name
        s3_client: boto3 S3 client

    Returns:
        S3 key where log was uploaded
    """
    log_key = log_capture.get_log_key()
    log_content = log_capture.get_log_content()

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=log_key,
            Body=log_content.encode('utf-8'),
            ContentType='text/plain; charset=utf-8',
            Metadata={
                'video-uuid': log_capture.video_uuid,
                'study-type': log_capture.study_type,
                'start-time': log_capture.start_time.isoformat() if log_capture.start_time else '',
                'end-time': log_capture.end_time.isoformat() if log_capture.end_time else '',
            }
        )
        return log_key
    except Exception as e:
        # Log upload failure should not break the processing
        # Print to original stdout since we might be outside the context
        print(f"WARNING: Failed to upload log to S3: {e}", file=sys.__stdout__)
        return None
