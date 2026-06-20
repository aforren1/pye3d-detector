"""Absolute 3D-accuracy gold standard for pye3d, using the repo's synthetic
ground-truth fixture (1000 ray-traced eye images + known gaze / eye-center /
pupil-radius), hosted on the pye3d wiki:
    tests/integration/input/pye3d_test_input.npz

This is a *true* ground truth (not a model-vs-model comparison) -- note that
3DeepVOG's "PL" eyeball model wraps pye3d itself, so it cannot serve as an
independent 3D gold standard.

Mirrors tests/integration/test_synthetic_metrics.py (same camera, same
regionprops-based pupil segmentation, same convergence-time logic and EPS
thresholds) but as a single self-contained run so a build can be graded and two
builds compared.

    python bench/gold3d.py            # grade the installed pye3d vs ground truth
    python bench/gold3d.py --out gold_cand.json
"""
import argparse
import json
import math
import os

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(_THIS, "..", "tests", "integration", "input", "pye3d_test_input.npz")
FPS = 200.0
# EPS thresholds from test_synthetic_metrics.py
EPS = dict(pupil_radius=0.057, eye_center_3d=0.5, gaze_angle=1.022)


def pupil_datum_from_raytraced_image(img):
    import skimage.measure as skmeas
    if img.ndim == 3:
        img = img[:, :, 0]
    seg = np.zeros(img.shape, dtype=np.uint8)
    seg[img == 10] = 10  # pupil label
    datum = {"ellipse": {"axes": np.array([0.0, 0.0]), "angle": -90.0,
                         "center": np.array([0.0, 0.0])}, "confidence": 0.0}
    label_image, _ = skmeas.label(seg, return_num=True, connectivity=1)
    props = skmeas.regionprops(label_image)
    if len(props) >= 1:
        p = props[0]
        datum["ellipse"]["axes"] = np.array([p.minor_axis_length, p.major_axis_length])
        datum["ellipse"]["angle"] = 90.0 - (p.orientation - np.pi / 2.0) * 180.0 / math.pi
        datum["ellipse"]["center"] = np.array([p.centroid[1], p.centroid[0]])
        datum["confidence"] = 1.0
    return datum


def main(a):
    import pandas as pd
    import pye3d
    from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

    data = np.load(INPUT)
    images = data["eye_images"]
    gt = pd.DataFrame.from_records(data["ground_truth"])
    n = len(images)

    det = Detector3D(camera=CameraModel(focal_length=561.5, resolution=np.array([400, 400])),
                     long_term_mode=DetectorMode.blocking)
    det.reset()

    sphere = np.empty((n, 3)); normal = np.empty((n, 3))
    radius = np.empty(n); ts = np.empty(n)
    for i in range(n):
        d = pupil_datum_from_raytraced_image(images[i])
        d["timestamp"] = i / FPS
        r = det.update_and_detect(d, images[i], debug=True)
        sphere[i] = r["sphere"]["center"]
        normal[i] = r["circle_3d"]["normal"]
        radius[i] = r["circle_3d"]["radius"]
        ts[i] = r["timestamp"]

    # errors vs ground truth
    gt_sphere = gt[["sphere_center_x", "sphere_center_y", "sphere_center_z"]].values
    gt_normal = gt[["circle_3d_normal_x", "circle_3d_normal_y", "circle_3d_normal_z"]].values
    gt_radius = gt["circle_3d_radius"].values

    eye_center_err = np.linalg.norm(sphere - gt_sphere, axis=1)
    dot = np.clip((normal * gt_normal).sum(1), -1, 1)
    gaze_err = np.rad2deg(np.arccos(dot))
    radius_err = np.abs(radius - gt_radius)

    # convergence time (same logic as the test: last frame the eye-center error
    # exceeds its EPS) then evaluate only after convergence.
    over = eye_center_err > EPS["eye_center_3d"]
    conv_idx = -np.argwhere(over[::-1])[0][0] - 1
    conv_t = float(ts[conv_idx])
    post = ts > conv_t

    res = {
        "pye3d": pye3d.__version__,
        "n": int(n),
        "convergence_time_s": conv_t,
        "eye_center_3d_mm": {"max": float(eye_center_err[post].max()),
                             "mean": float(eye_center_err[post].mean())},
        "gaze_angle_deg": {"max": float(gaze_err[post].max()),
                           "mean": float(gaze_err[post].mean())},
        "pupil_radius_mm": {"max": float(radius_err[post].max()),
                            "mean": float(radius_err[post].mean())},
    }
    res["pass"] = {
        "eye_center_3d": res["eye_center_3d_mm"]["max"] <= EPS["eye_center_3d"],
        "gaze_angle": res["gaze_angle_deg"]["max"] <= EPS["gaze_angle"],
        "pupil_radius": res["pupil_radius_mm"]["max"] <= EPS["pupil_radius"],
        "convergence_time": conv_t <= 2.14,
    }
    print(f"pye3d {res['pye3d']}  ({n} frames, convergence={conv_t:.3f}s)")
    print(f"  eye-center 3d : max={res['eye_center_3d_mm']['max']:.4f} "
          f"mean={res['eye_center_3d_mm']['mean']:.4f} mm  (EPS {EPS['eye_center_3d']}) "
          f"-> {'PASS' if res['pass']['eye_center_3d'] else 'FAIL'}")
    print(f"  gaze angle    : max={res['gaze_angle_deg']['max']:.4f} "
          f"mean={res['gaze_angle_deg']['mean']:.4f} deg (EPS {EPS['gaze_angle']}) "
          f"-> {'PASS' if res['pass']['gaze_angle'] else 'FAIL'}")
    print(f"  pupil radius  : max={res['pupil_radius_mm']['max']:.4f} "
          f"mean={res['pupil_radius_mm']['mean']:.4f} mm  (EPS {EPS['pupil_radius']}) "
          f"-> {'PASS' if res['pass']['pupil_radius'] else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    main(p.parse_args())
