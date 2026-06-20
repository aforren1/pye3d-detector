"""Extract grayscale frame clips from the shared sample video to raw uint8.

The sample video now lives at the pupil-pkgs top level (shared by both the
pupil-detectors and pye3d-detector benchmarks):

    pupil-pkgs/sample_data/eye1.mp4   (640x480 MJPEG, 120fps, ~418k frames, 16GB)

Decoding the whole thing is wasteful, so we dump a contiguous subset to a flat
uint8 file (n*H*W bytes) that bench3d.py / bench.py mmap directly.

Usage:
    python bench/extract_frames.py --start 0     --count 2000 --out clean.raw
    python bench/extract_frames.py --start 90000 --count 4000 --out blink.raw
"""
import argparse
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
_OPENCV_BIN = os.environ.get("PUPIL_OPENCV_BIN")
if _OPENCV_BIN and os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKGS_ROOT = os.path.dirname(os.path.dirname(_THIS))  # pupil-pkgs/
DEFAULT_VIDEO = os.path.join(_PKGS_ROOT, "sample_data", "eye1.mp4")
W, H = 640, 480


def main(a):
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    with open(a.out, "wb") as fh:
        written = 0
        while written < a.count:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape != (H, W):
                gray = cv2.resize(gray, (W, H))
            fh.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
            written += 1
    cap.release()
    print(f"wrote {written} frames ({W}x{H}) -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=DEFAULT_VIDEO)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--out", required=True)
    main(p.parse_args())
