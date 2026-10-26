#!/usr/bin/env bash
# Autonomous low-confidence detection dump for the TMC round-2 experiment.
# Adapted from 09-30-atr-iter/pod/bootstrap.sh, whose guarantees are kept intact:
#   1. results reach S3 even on crash, OOM, SIGTERM or watchdog kill
#   2. the pod terminates itself when done -- it never bills idle
#   3. a failed clip never aborts the remaining clips
#
# NOT `set -e`: a non-zero clip must not kill the run.
set -uo pipefail

W=/workspace
OUT=$W/out
LOG=$W/run.log
RUN_ID="${RUN_ID:?RUN_ID required}"
BASE="experiments/round2-dets"
PREFIX="$BASE/runs/$RUN_ID"
mkdir -p "$OUT" "$W/clips" "$W/models"
exec > >(tee -a "$LOG") 2>&1

echo "=== TMC round-2 detection dump $RUN_ID @ $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?} ==="

PY=$(command -v python3 || command -v python)
S3=$W/s3util.py

# Heartbeat before anything slow: a container that never starts is otherwise
# indistinguishable from a busy one for an hour.
echo "alive $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" > "$OUT/_heartbeat"
"$PY" "$S3" push "$OUT/_heartbeat" "$PREFIX/out/_heartbeat" || echo "HEARTBEAT PUSH FAILED"

flush() {
  "$PY" "$S3" pushdir "$OUT" "$PREFIX/out" || true
  "$PY" "$S3" push "$LOG" "$PREFIX/run.log" || true
}

terminate() {
  echo "=== terminating pod ${RUNPOD_POD_ID:-?} ==="
  "$PY" - <<'EOF' || true
import os, requests
pid = os.environ.get("RUNPOD_POD_ID"); key = os.environ.get("RUNPOD_API_KEY")
if pid and key:
    # urllib gets 403'd by RunPod's edge (requests with an explicit UA works), and
    # podTerminate returns Void -- a selection set like `{ id }` fails validation
    # and the pod silently keeps billing.
    r = requests.post(f"https://api.runpod.io/graphql?api_key={key}",
                      json={"query": 'mutation { podTerminate(input:{podId:"%s"}) }' % pid},
                      headers={"Content-Type": "application/json", "User-Agent": "runpodctl/1.14"},
                      timeout=45)
    print("terminate ->", r.status_code, r.text[:200])
else:
    print("NO POD ID / API KEY -- cannot self-terminate")
EOF
}

cleanup() {
  code=$?
  echo "=== cleanup (exit $code) @ $(date -u +%FT%TZ) ==="
  echo "$code" > "$OUT/_exit_code"
  flush
  terminate > "$OUT/_terminate.txt" 2>&1 || true
  "$PY" "$S3" push "$OUT/_terminate.txt" "$PREFIX/out/_terminate.txt" || true
}
trap cleanup EXIT
trap 'echo "SIGNAL caught"; exit 143' INT TERM

WATCHDOG_SECS="${WATCHDOG_SECS:-21600}"
( sleep "$WATCHDOG_SECS"; echo "!!! WATCHDOG $WATCHDOG_SECS s -- killing run"; kill -TERM $$ ) &
WD=$!

echo "--- deps ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq libgl1 libglib2.0-0 libxcb1 libxrender1 libsm6 libxext6 ffmpeg 2>&1 | tail -3
"$PY" -m pip install --quiet --no-input "ultralytics==8.4.109" boto3 shapely scipy requests 2>&1 | tail -5
# ultralytics pulls the GUI opencv, which shadows a headless install done in the
# same command and dies on libxcb. Swap it AFTER ultralytics.
"$PY" -m pip uninstall -y -q opencv-python opencv-contrib-python 2>/dev/null
"$PY" -m pip install --quiet --no-input opencv-python-headless 2>&1 | tail -3

# nvidia-smi can report a healthy GPU while torch still gets "CUDA unknown error".
# Dying here costs two minutes; falling back to CPU costs hours of billing.
"$PY" - <<'EOF' || { echo "!!! preflight FAILED -- aborting before burning pod time"; exit 3; }
import sys, ultralytics, cv2, torch
print("ultralytics", ultralytics.__version__, "cv2", cv2.__version__,
      "torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("!!! torch cannot see the GPU -- refusing to dump on CPU")
    sys.exit(1)
print("gpu:", torch.cuda.get_device_name(0), flush=True)
EOF

echo "--- processor source + model ---"
"$PY" "$S3" pull "$BASE/code/processor_src.tgz" "$W/src.tgz"
mkdir -p "$W/src" && tar xzf "$W/src.tgz" -C "$W/src" && rm -f "$W/src.tgz"
export PROCESSOR_SRC="$W/src"
ls "$W/src" | head
"$PY" "$S3" pull "$BASE/code/dump_dets.py" "$W/dump_dets.py"
"$PY" "$S3" pull "models/best.pt" "$W/models/best.pt"
"$PY" -c "import hashlib;print('model md5', hashlib.md5(open('$W/models/best.pt','rb').read()).hexdigest())"
flush

echo "--- dumping: sites=[$SITES] conf=${DUMP_CONF:-0.05} ---"
for site in $SITES; do
  dest="$OUT/${site}.dets.csv.gz"
  if [ -f "$dest" ]; then echo "skip $site (exists)"; continue; fi
  echo ">>> $site  $(date -u +%T)"
  "$PY" "$S3" pull "$BASE/clips/${site}.mp4" "$W/clips/${site}.mp4" || { echo "PULL FAILED $site"; continue; }
  # One process per clip: a long single process drifts into a stall after a
  # couple of videos, and a crash must not take the other clips' work with it.
  "$PY" "$W/dump_dets.py" "$W/clips/${site}.mp4" "$W/models/best.pt" "$dest" 2>&1 | tee "$OUT/${site}.log"
  echo "<<< $site rc=${PIPESTATUS[0]}"
  rm -f "$W/clips/${site}.mp4"
  flush
done

echo "=== dump complete @ $(date -u +%FT%TZ) ==="
kill "$WD" 2>/dev/null
exit 0
