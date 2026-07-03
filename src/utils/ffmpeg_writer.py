"""
Drop-in replacement for cv2.VideoWriter that encodes browser-playable H.264 by
piping raw BGR frames to the system ffmpeg CLI (which has libx264).

Needed because opencv-python-headless's bundled ffmpeg has no H.264 encoder, so
cv2.VideoWriter('H264') silently falls back to MPEG-4 Part 2 (mp4v) — unplayable
in browsers. Interface mirrors the slice the processors use: isOpened() / write() /
release(). Frames must be HxWx3 uint8 BGR at the given (width, height).
"""

import shutil
import subprocess
import tempfile

import numpy as np


class FFmpegH264Writer:
    def __init__(self, path, fps, width, height, crf=26, preset="veryfast",
                 profile="baseline"):
        self.path = path
        # libx264 + yuv420p require even dimensions; write() resizes any mismatch.
        self.width = int(width) - int(width) % 2
        self.height = int(height) - int(height) % 2
        self.frames_written = 0
        self._proc = None
        self._ok = False
        # stderr -> temp file (not a PIPE) so it can't fill and deadlock our
        # blocking stdin writes over a long (e.g. 24h) run.
        self._stderr = tempfile.TemporaryFile()

        if not shutil.which("ffmpeg"):
            print("⚠️ FFmpegH264Writer: 'ffmpeg' not found on PATH")
            return
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 0.0
        fps = fps if fps > 0 else 15.0

        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.width}x{self.height}", "-r", f"{fps:g}", "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-profile:v", profile, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", path]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=self._stderr)
            self._ok = True
            print(f"✅ FFmpegH264Writer: libx264 {self.width}x{self.height}@{fps:g}fps -> {path}")
        except Exception as e:
            print(f"⚠️ FFmpegH264Writer: failed to start ffmpeg: {e}")

    def isOpened(self):
        return self._ok and self._proc is not None and self._proc.poll() is None

    def write(self, frame):
        if not self.isOpened():
            return
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            self.frames_written += 1
        except (BrokenPipeError, ValueError, OSError) as e:
            self._ok = False
            print(f"⚠️ FFmpegH264Writer: write failed after {self.frames_written} "
                  f"frames: {e}; ffmpeg stderr: {self._tail_stderr()!r}")

    def release(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            rc = self._proc.wait(timeout=1800)
            if rc != 0:
                print(f"⚠️ FFmpegH264Writer: ffmpeg exited rc={rc}; stderr: {self._tail_stderr()!r}")
        except Exception as e:
            print(f"⚠️ FFmpegH264Writer: ffmpeg did not finalize cleanly: {e}")
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None
            try:
                self._stderr.close()
            except Exception:
                pass

    def _tail_stderr(self):
        try:
            self._stderr.seek(0)
            return (self._stderr.read() or b"")[-800:]
        except Exception:
            return b""
