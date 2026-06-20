"""Tiny CI benchmark: time the pye3d 3D pipeline over the shared test sequence.

Runs against whatever `pye3d` is importable in the current env, so CI can run it
once with the freshly-built wheel and once with the PyPI release, then diff the
two JSON outputs with ci_compare.py.

    python bench/ci_bench.py --input pye3d_test_input.npz --out built.json

The input is the same `pye3d_test_input.npz` the integration tests use (1000
raytraced 400x400 label images + ground truth), pulled from the project wiki:
https://github.com/pupil-labs/pye3d-detector/wiki/files/pye3d_test_input.npz
Per-update timing on hosted CI runners is noisy in absolute terms, so ci_compare
reports the built/PyPI *ratio* measured back-to-back on the same runner.
"""
import argparse
import json
import math
import os
import statistics
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
DEFAULT_INPUT = os.path.join(
    _REPO, "tests", "integration", "input", "pye3d_test_input.npz"
)


def pupil_datum_from_label_image(img):
    """Fit an ellipse to the raytraced pupil label (==10), mirroring the
    integration test's `pupil_datum_from_raytraced_image` so the 3D detector
    sees the same 2D input. Self-contained (skimage only, no pupil_detectors)."""
    import skimage.measure as skmeas

    if img.ndim == 3:
        img = img[:, :, 0]
    seg = np.zeros(img.shape, np.uint8)
    seg[img == 10] = 10

    datum = {
        "ellipse": {
            "axes": np.array([0.0, 0.0]),
            "angle": -90.0,
            "center": np.array([0.0, 0.0]),
        },
        "confidence": 0.0,
    }
    label_image = skmeas.label(seg, connectivity=1)
    props = skmeas.regionprops(label_image)
    if props:
        p = props[0]
        orientation = p.orientation - np.pi / 2.0
        datum["ellipse"]["axes"] = np.array(
            [p.minor_axis_length, p.major_axis_length]
        )
        datum["ellipse"]["angle"] = 90 - orientation * 180.0 / math.pi
        datum["ellipse"]["center"] = np.array([p.centroid[1], p.centroid[0]])
        datum["confidence"] = 1.0
    return datum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    args = ap.parse_args()

    import pye3d
    from pye3d.detector_3d import CameraModel, DetectorMode
    from pye3d.detector_3d import Detector3D

    data = np.load(args.input)
    images = data["eye_images"][: args.max_frames]

    camera = CameraModel(focal_length=561.5, resolution=np.array([400, 400]))
    detector = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    detector.reset()

    FPS = 200.0
    times_ms = []
    radii = []
    diameters = []
    for i, img in enumerate(images):
        datum = pupil_datum_from_label_image(img)
        datum["timestamp"] = i / FPS
        t0 = time.perf_counter()
        result = detector.update_and_detect(datum, img)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        radii.append(float(result["sphere"]["radius"]))
        diameters.append(float(result["diameter_3d"]))

    times_ms.sort()

    def pct(p):
        return times_ms[min(len(times_ms) - 1, int(p / 100 * len(times_ms)))]

    result = {
        "package": "pye3d",
        "version": pye3d.__version__,
        "n_updates": len(times_ms),
        "latency_ms": {
            "median": statistics.median(times_ms),
            "p90": pct(90),
            "p99": pct(99),
            "mean": statistics.fmean(times_ms),
        },
        # output signature: a model-fit regression shows up here.
        "signature": {
            "mean_sphere_radius": statistics.fmean(radii),
            "mean_diameter_3d": statistics.fmean(diameters),
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
