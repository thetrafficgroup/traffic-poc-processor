"""Runtime background provider for background-conditioned (4-channel) detectors.

A 4-channel model takes RGB + a background-difference channel: per pixel, how
much the current frame differs (max over color channels) from a temporal-median
"empty scene" background. Backgrounds are computed in a cheap sampled pre-pass
over the video (windowed, so day/night lighting is tracked on long recordings).

Safety properties:
  - Only activates when the loaded model actually expects 4 channels
    (model_wants_background), so 3-channel models are completely unaffected.
  - If the background build fails, or BG4CH_DISABLE=1 is set, the 4th channel
    is all zeros — the 4ch models are trained with identity-background
    augmentation and degrade to RGB-only behavior on zero diff.
"""
import os

import cv2
import numpy as np

WINDOW_SECONDS = 1800       # one background per 30 min of video
SAMPLES_PER_WINDOW = 48
MIN_SAMPLES = 12


def model_wants_background(model):
    """True when the loaded ultralytics YOLO model expects 4-channel input."""
    try:
        if model.model.model[0].conv.in_channels == 4:
            return True
    except Exception:
        pass
    try:
        return int(model.model.yaml.get("ch", 3)) == 4
    except Exception:
        return False


class BackgroundProvider:
    """Windowed temporal-median backgrounds with per-frame lookup."""

    def __init__(self, video_path, fps, total_frames):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.windows = []  # [(start_frame, bg_bgr)]
        self._resized_cache = {}
        if os.environ.get("BG4CH_DISABLE", "0") == "1":
            print("🌄 BG4CH_DISABLE=1 — 4th channel will be zero (RGB-only behavior)")
            return
        try:
            self._build(video_path, int(total_frames))
            print(f"🌄 Background model ready: {len(self.windows)} window(s) "
                  f"({WINDOW_SECONDS/60:.0f} min each, {SAMPLES_PER_WINDOW} samples/window)")
        except Exception as e:
            print(f"⚠️ Background build failed ({e!r}) — falling back to zero diff channel")
            self.windows = []

    def _build(self, video_path, total_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video for background pass: {video_path}")
        window_frames = max(1, int(WINDOW_SECONDS * self.fps))
        n_windows = max(1, (total_frames + window_frames - 1) // window_frames)
        for w in range(n_windows):
            f_start = w * window_frames
            f_end = min(total_frames, (w + 1) * window_frames)
            if f_end - f_start < self.fps:  # ignore sub-second tail windows
                continue
            # Short windows (short clips / video tails) need denser sampling for
            # a clean median: a busy scene sampled sparsely leaves vehicle ghosts.
            span_s = (f_end - f_start) / self.fps
            n_samples = int(min(200, max(SAMPLES_PER_WINDOW, span_s)))
            idxs = np.linspace(f_start, f_end - 1, n_samples).astype(int)
            frames = []
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, fr = cap.read()
                if ret:
                    frames.append(fr)
            if len(frames) < MIN_SAMPLES:
                print(f"🌄 window {w:02d}: SKIPPED — only {len(frames)} samples decoded "
                      f"({f_start}-{f_end} frames); previous window's background will cover it")
                continue
            bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
            brightness = float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean())
            # Per-window log: across a 24h video the brightness column should
            # visibly dip at night — a one-glance correctness signal.
            print(f"🌄 window {w:02d}: {f_start/self.fps/60:6.1f}-{f_end/self.fps/60:6.1f} min, "
                  f"{len(frames)} samples, brightness {brightness:.1f}")
            self.windows.append((f_start, bg))
        cap.release()
        if not self.windows:
            raise RuntimeError("no background windows could be computed")
        self._debug_upload()

    def _debug_upload(self):
        """Upload the computed backgrounds for visual inspection when the handler
        provides BG4CH_DEBUG_BUCKET/BG4CH_DEBUG_ID. Never fails the run."""
        bucket = os.environ.get("BG4CH_DEBUG_BUCKET")
        vid = os.environ.get("BG4CH_DEBUG_ID")
        if not bucket or not vid:
            return
        try:
            import boto3
            s3 = boto3.client("s3")
            for w, (f_start, bg) in enumerate(self.windows):
                ok, buf = cv2.imencode(".png", bg)
                if not ok:
                    continue
                key = f"logs/backgrounds/{vid}/w{w:03d}_f{f_start}.png"
                s3.put_object(Bucket=bucket, Key=key, Body=buf.tobytes(),
                              ContentType="image/png")
            print(f"🌄 Uploaded {len(self.windows)} background(s) to "
                  f"s3://{bucket}/logs/backgrounds/{vid}/")
        except Exception as e:
            print(f"🌄 background debug upload skipped ({e!r})")

    def _bg_for(self, frame_idx, shape_hw):
        best = None
        for start, bg in self.windows:
            if start <= frame_idx:
                best = bg
            else:
                break
        if best is None and self.windows:
            best = self.windows[0][1]
        if best is None:
            return None
        if best.shape[:2] != shape_hw:
            key = (id(best), shape_hw)
            if key not in self._resized_cache:
                self._resized_cache[key] = cv2.resize(best, (shape_hw[1], shape_hw[0]))
            best = self._resized_cache[key]
        return best

    DIAG_EVERY_S = 300  # log diff health every ~5 min of video

    def stack(self, frame, frame_idx):
        """Return the 4-channel model input for this frame (frame is untouched)."""
        bg = self._bg_for(frame_idx, frame.shape[:2])
        if bg is None:
            diff = np.zeros(frame.shape[:2], np.uint8)
        else:
            diff = cv2.absdiff(frame, bg).max(axis=2)
        # Periodic health line: healthy scenes idle at mean diff ~2-11 gray
        # levels (Phase 0 measurement); a corrupted/mismatched background shows
        # up as a sustained spike far above that.
        diag_every = max(1, int(self.DIAG_EVERY_S * self.fps))
        if frame_idx % diag_every == 0:
            print(f"🌄 diff health @ {frame_idx/self.fps/60:.1f} min: "
                  f"mean={float(diff.mean()):.1f}, p99={float(np.percentile(diff, 99)):.0f}")
        return np.dstack([frame, diff])
