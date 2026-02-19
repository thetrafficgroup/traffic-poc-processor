#!/usr/bin/env bash
# Split a video into 10-minute (600s) segments using stream copy (no re-encoding).
# Usage: ./split_video.sh <input_video> [output_dir]

set -euo pipefail

INPUT="${1:?Usage: $0 <input_video> [output_dir]}"
OUTPUT_DIR="${2:-.}"

BASENAME="$(basename "${INPUT%.*}")"

mkdir -p "$OUTPUT_DIR"

ffmpeg -i "$INPUT" \
  -c copy \
  -segment_time 600 \
  -f segment \
  -reset_timestamps 1 \
  "${OUTPUT_DIR}/${BASENAME}_%02d.mp4"

echo "Done. Segments written to ${OUTPUT_DIR}/"
