"""Per-frame pipeline-latency histogram over a long stretch of real video,
original PyPI pipeline vs the optimized one.

Streams N minutes of the sample video, center-crops to 400x400, and times the
2D+3D processing per frame (decode/cvtColor excluded -- a real camera delivers
grayscale frames directly). Run once per build (different venv), then plot.

    # original baseline venv (PyPI 2.0.2 + pye3d 0.3.2), no PUPIL_OPENCV_BIN:
    .venv-pypi/Scripts/python bench/hist_pipeline.py run --tag pypi --minutes 20
    # optimized venv (dev21 2D + candidate 3D), with PUPIL_OPENCV_BIN:
    PUPIL_OPENCV_BIN=... .venv-2dgen/Scripts/python bench/hist_pipeline.py run --tag new --minutes 20
    # plot (any venv with matplotlib):
    python bench/hist_pipeline.py plot --pypi times_pypi.npy --new times_new.npy
"""
import argparse
import os
import time

_OCV = os.environ.get("PUPIL_OPENCV_BIN")
if _OCV and os.path.isdir(_OCV):
    os.add_dll_directory(_OCV)

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKGS = os.path.dirname(os.path.dirname(_THIS))
VIDEO = os.path.join(_PKGS, "sample_data", "eye1.mp4")
SRC_FPS = 120.0      # native rate of eye1.mp4 (drives the model schedule)
CW, CH = 400, 400    # sensor crop
BUDGET = 4.0


def run(args):
    import cv2
    from pupil_detectors import Detector2D
    import pupil_detectors
    import pye3d
    from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    n = int(args.minutes * 60 * SRC_FPS)
    det2d = Detector2D()
    det3d = Detector3D(camera=CameraModel(focal_length=561.5, resolution=[CW, CH]),
                       long_term_mode=DetectorMode.blocking)
    print(f"[{args.tag}] pupil_detectors {pupil_detectors.__version__} + pye3d "
          f"{pye3d.__version__}: {args.minutes} min = {n} frames @ {CW}x{CH}", flush=True)

    # source is 640x480; center-crop to 400x400
    Y0, X0 = (480 - CH) // 2, (640 - CW) // 2
    t = np.empty(n)
    i = 0
    while i < n:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray[Y0:Y0 + CH, X0:X0 + CW])
        a = time.perf_counter()
        r = det2d.detect(gray)
        r["timestamp"] = i / SRC_FPS
        det3d.update_and_detect(r, gray)
        t[i] = (time.perf_counter() - a) * 1000.0
        i += 1
        if i % 20000 == 0:
            print(f"  [{args.tag}] {i}/{n}  median so far {np.median(t[:i]):.3f}ms", flush=True)
    cap.release()
    t = t[:i]
    out = args.out or os.path.join(_THIS, f"times_{args.tag}.npy")
    np.save(out, t)
    print(f"[{args.tag}] {i} frames: median={np.median(t):.3f} mean={t.mean():.3f} "
          f"p99={np.percentile(t,99):.3f} p99.9={np.percentile(t,99.9):.3f} "
          f"max={t.max():.3f}  >{BUDGET}ms={100.0*(t>BUDGET).mean():.2f}%  -> {out}")


def _lbl(name, t):
    return (f"{name}: median {np.median(t):.2f}ms, p99 {np.percentile(t,99):.2f}ms, "
            f">{BUDGET:.0f}ms {100.0*(t>BUDGET).mean():.2f}%")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.load(args.pypi)
    b = np.load(args.new)
    hi = float(np.percentile(np.concatenate([a, b]), 99.9))
    bins = np.linspace(0, max(hi, BUDGET + 1), 240)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(a, bins=bins, alpha=0.55, color="#c44", label=_lbl("original (PyPI 2.0.2 + pye3d 0.3.2)", a))
    ax.hist(b, bins=bins, alpha=0.55, color="#2a7", label=_lbl("optimized (dev21 2D + opt 3D)", b))
    ax.axvline(BUDGET, color="k", ls="--", lw=1)
    ax.text(BUDGET + 0.05, ax.get_ylim()[1] * 0.6, f"{BUDGET:.0f}ms budget\n(250fps)", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("per-frame 2D+3D pipeline time [ms]  (400x400)")
    ax.set_ylabel("frame count (log)")
    ax.set_title(f"Pipeline latency over {len(a)/SRC_FPS/60:.0f} min of eye video — original vs optimized")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = args.out or os.path.join(_THIS, "hist_pipeline.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")
    print(_lbl("original ", a))
    print(_lbl("optimized", b))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--tag", required=True)
    pr.add_argument("--minutes", type=float, default=20.0)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=run)
    pp = sub.add_parser("plot")
    pp.add_argument("--pypi", required=True)
    pp.add_argument("--new", required=True)
    pp.add_argument("--out", default=None)
    pp.set_defaults(func=plot)
    args = p.parse_args()
    args.func(args)
