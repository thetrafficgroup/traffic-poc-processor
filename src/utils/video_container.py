import json
import os
import subprocess

import cv2


def _probe_duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=120,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def ensure_indexed_mp4(path):
    """Rewrite a fragmented MP4 in place as a regular, fully indexed one.

    Fragmented MP4 (CountCam X 801004, video 1036) carries only the first fragment in its
    moov box, so cv2 reports CAP_PROP_FRAME_COUNT = 10 for a 24 h file. Every consumer of
    that count then misbehaves: the 4-channel background model built one window from the
    first second (midnight IR) and used it all day, trims are clipped to 10 frames, and
    progress stalls. A stream copy rebuilds the index without touching a single frame.
    """
    cap = cv2.VideoCapture(path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    try:
        expected = _probe_duration_s(path) * fps
    except Exception as e:
        print(f"🎞️ container check skipped (ffprobe failed: {e!r})")
        return path
    if expected <= 0 or frames >= 0.9 * expected:
        return path

    print(f"🎞️ frame count {int(frames)} vs ~{int(expected)} from duration: "
          f"re-indexing fragmented MP4 (stream copy)")
    fixed = f"{path}.indexed.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0", "-c", "copy",
         "-movflags", "+faststart", fixed],
        check=True, timeout=3600,
    )
    os.replace(fixed, path)
    cap = cv2.VideoCapture(path)
    print(f"🎞️ re-indexed: frame count now {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    cap.release()
    return path
