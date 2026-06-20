"""End-to-end pipeline budget check at the real operating point.

Target hardware: 400x400 @ 250 fps -> the *combined* 2D+3D per-frame latency
must sit comfortably under 4 ms. This runs the optimized Detector2D (live) +
the candidate Detector3D over a 400x400 center-crop of the sample clip and
reports the combined per-frame distribution against the 4 ms budget, with the
2D and 3D stages broken out.

Needs both the optimized pupil_detectors and the candidate pye3d installed, and
PUPIL_OPENCV_BIN set (the optimized 2D links the system OpenCV DLL):

    PUPIL_OPENCV_BIN=c:/tools/opencv/build/x64/vc16/bin \
        .venv-2dgen/Scripts/python bench/pipeline.py --frames <raw>
"""
import argparse
import os
import time

_OPENCV_BIN = os.environ.get("PUPIL_OPENCV_BIN")
if _OPENCV_BIN and os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)

import numpy as np

W, H = 640, 480           # source clip
CW, CH = 400, 400         # target crop (the real sensor size)
FPS = 250.0
BUDGET_MS = 4.0


def load_crop(path, max_frames=None):
    data = np.fromfile(path, dtype=np.uint8)
    n = data.size // (W * H)
    data = data[: n * W * H].reshape(n, H, W)
    if max_frames:
        data = data[:max_frames]
    y0 = (H - CH) // 2
    x0 = (W - CW) // 2
    return np.ascontiguousarray(data[:, y0:y0 + CH, x0:x0 + CW])


def pct(a, q):
    return np.percentile(a, q)


def main(a):
    from pupil_detectors import Detector2D
    import pupil_detectors
    import pye3d
    from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

    frames = load_crop(a.frames, a.max_frames)
    n = len(frames)
    det2d = Detector2D()
    det3d = Detector3D(camera=CameraModel(focal_length=561.5, resolution=[CW, CH]),
                       long_term_mode=DetectorMode.blocking)
    print(f"pupil_detectors {pupil_detectors.__version__} + pye3d {pye3d.__version__}"
          f"  | {n} frames @ {CW}x{CH}, {FPS}fps, budget {BUDGET_MS}ms", flush=True)

    t2 = np.empty(n); t3 = np.empty(n)
    for i in range(n):
        f = frames[i]
        a0 = time.perf_counter()
        r = det2d.detect(f)
        a1 = time.perf_counter()
        r["timestamp"] = i / FPS
        det3d.update_and_detect(r, f)
        a2 = time.perf_counter()
        t2[i] = (a1 - a0) * 1000.0
        t3[i] = (a2 - a1) * 1000.0

    tot = t2 + t3
    def line(name, x):
        return (f"{name:9} median={np.median(x):.3f} mean={x.mean():.3f} "
                f"p95={pct(x,95):.3f} p99={pct(x,99):.3f} p99.9={pct(x,99.9):.3f} "
                f"max={x.max():.3f}")
    print(line("2D ms", t2))
    print(line("3D ms", t3))
    print(line("TOTAL ms", tot))
    print(f"TOTAL over budget ({BUDGET_MS}ms): {100.0*(tot>BUDGET_MS).mean():.2f}%  "
          f"(>{BUDGET_MS/2}ms: {100.0*(tot>BUDGET_MS/2).mean():.1f}%)  "
          f"headroom at p99.9 = {BUDGET_MS - pct(tot,99.9):.3f}ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    main(p.parse_args())
