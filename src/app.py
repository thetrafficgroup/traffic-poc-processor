from tmc.tmc_processor import process_video as tmc_process_video
from atr.atr_processor import process_video as atr_process_video

def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", study_type="TMC", video_uuid=None, progress_callback=None, minute_batch_callback=None, generate_video_output=False, output_video_path=None, trim_periods=None, pedestrian_model_path=None):
    """
    Routes video processing based on study type.

    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Line configuration data
        MODEL_PATH: Path to YOLO model
        study_type: Type of study - "TMC" or "ATR"
        video_uuid: UUID of the video being processed
        progress_callback: Optional callback for progress updates
        minute_batch_callback: Optional callback for minute-by-minute batch data
        generate_video_output: Whether to generate annotated output video
        output_video_path: Path for output video (if generate_video_output=True)
        trim_periods: Optional list of trim periods in seconds [{"start": 3600, "end": 10800}, ...]
        pedestrian_model_path: Optional path to pedestrian/bicycle YOLO model

    Returns:
        Processing results based on study type.
    """

    if study_type.upper() == "TMC":
        return tmc_process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH, video_uuid, progress_callback,
                                minute_batch_callback, generate_video_output, output_video_path, trim_periods,
                                pedestrian_model_path=pedestrian_model_path)
    elif study_type.upper() == "ATR":
        return atr_process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH, progress_callback,
                                generate_video_output, output_video_path, video_uuid, minute_batch_callback, trim_periods)
    else:
        raise ValueError(f"Unknown study_type: {study_type}. Must be 'TMC' or 'ATR'")