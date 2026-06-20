"""Benchmark + accuracy harness for the pye3d 3D eye-model pipeline.

Mirrors pupil-detectors/bench/bench.py, but for the full 2D->3D pipeline:
the classical Detector2D produces per-frame ellipses which are fed
sequentially into Detector3D.update_and_detect (the 3D model is temporal, so
frames MUST be processed in order). We record the 3D-stage timing separately
from the 2D stage, plus the 3D outputs (sphere center, gaze direction, pupil
diameter, confidences) for accuracy comparison between builds.

Frames are the same pre-extracted grayscale clips used by the 2D bench
(raw uint8, 640x480), sourced from the shared sample video
pupil-pkgs/sample_data/eye1.mp4 (see extract_frames.py).

Usage:
    python bench/bench3d.py run --tag pypi --frames <clean.raw>
    python bench/bench3d.py run --tag pypi-blink --frames <blink.raw>
    python bench/bench3d.py cmp --ref bench/out3d_ref.npz --cand bench/out3d_cand.npz
"""
import argparse
import os
import sys
import time

import numpy as np

# Local-built pye3d/pupil_detectors link the system OpenCV C++ runtime and need
# its DLL dir on the search path before import. PyPI wheels bundle their own, so
# only do this when explicitly pointed at a system OpenCV (env opt-in).
_OPENCV_BIN = os.environ.get("PUPIL_OPENCV_BIN")
if _OPENCV_BIN and os.path.isdir(_OPENCV_BIN):
    os.add_dll_directory(_OPENCV_BIN)

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKGS_ROOT = os.path.dirname(os.path.dirname(_THIS))  # pupil-pkgs/

# Default frames: reuse the 2D bench's pre-extracted clips (sibling repo) so we
# don't duplicate ~1.8 GB of raw frames. Override with --frames.
DEFAULT_FRAMES = os.path.join(
    _PKGS_ROOT, "pupil-detectors", "bench", "frames_2000.raw"
)

W, H = 640, 480
FPS = 120.0  # eye1.mp4 native rate; drives the pye3d model-update schedule

# Benchmark camera. eye1.mp4 is 640x480; focal_length matches the value pye3d's
# own synthetic test/example uses. The 3D pipeline runs the same code paths
# regardless, so this is a fixed, consistent reference for A/B comparison.
FOCAL_LENGTH = float(os.environ.get("PYE3D_FOCAL", "561.5"))


def load_frames(path, max_frames=None):
    data = np.fromfile(path, dtype=np.uint8)
    n = data.size // (W * H)
    data = data[: n * W * H].reshape(n, H, W)
    if max_frames:
        data = data[:max_frames]
    return np.ascontiguousarray(data)


def gen2d(args):
    """Run the (optimized) 2D detector once over a clip and cache the per-frame
    ellipse stream, so the 3D benchmark can feed a fixed, representative 2D
    front-end without re-running / re-installing pupil_detectors. Sequential
    pass (the 2D detector carries a temporal prior)."""
    from pupil_detectors import Detector2D
    import pupil_detectors

    frames = load_frames(args.frames, args.max_frames)
    n = len(frames)
    det = Detector2D()
    cx = np.empty(n); cy = np.empty(n); ax0 = np.empty(n); ax1 = np.empty(n)
    ang = np.empty(n); conf = np.empty(n)
    for i in range(n):
        r = det.detect(frames[i])
        e = r["ellipse"]
        cx[i], cy[i] = e["center"]; ax0[i], ax1[i] = e["axes"]
        ang[i] = e["angle"]; conf[i] = r["confidence"]
    np.savez(args.out, cx=cx, cy=cy, ax0=ax0, ax1=ax1, ang=ang, conf=conf)
    print(f"[gen2d] pupil_detectors {pupil_detectors.__version__}: cached {n} "
          f"frames (mean conf={conf.mean():.4f}, conf<0.7={100.0*(conf<0.7).mean():.1f}%)"
          f" -> {args.out}")


def load_pupil2d(path):
    """Reconstruct per-frame pupil_datum dicts (without timestamp) from a gen2d
    cache, matching what Detector2D.detect() returns for pye3d's consumption."""
    d = np.load(path)
    n = len(d["cx"])
    data = []
    for i in range(n):
        data.append({
            "ellipse": {
                "center": (float(d["cx"][i]), float(d["cy"][i])),
                "axes": (float(d["ax0"][i]), float(d["ax1"][i])),
                "angle": float(d["ang"][i]),
            },
            "confidence": float(d["conf"][i]),
        })
    return data


def run(args):
    import pye3d
    from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

    frames = load_frames(args.frames, args.max_frames)
    n = len(frames)
    # 2D front-end: cached optimized stream (representative) or live Detector2D.
    cached2d = load_pupil2d(args.pupil2d) if args.pupil2d else None
    if cached2d is not None:
        assert len(cached2d) >= n, "cached 2D shorter than frame clip"
        det2d = None
        src = f"cached {os.path.basename(args.pupil2d)}"
    else:
        from pupil_detectors import Detector2D
        det2d = Detector2D()
        src = "live Detector2D"
    print(
        f"[{args.tag}] pye3d {pye3d.__version__}, {n} frames @ {FPS}fps, "
        f"focal={FOCAL_LENGTH}, 2D={src}",
        flush=True,
    )

    camera = CameraModel(focal_length=FOCAL_LENGTH, resolution=[W, H])
    det3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

    # 2D-stage timing
    t2d = np.empty(n)
    # 3D-stage timing
    t3d = np.empty(n)
    # 2D inputs
    conf2d = np.empty(n)
    # 3D outputs
    sx = np.empty(n); sy = np.empty(n); sz = np.empty(n)        # sphere center
    nx = np.empty(n); ny = np.empty(n); nz = np.empty(n)        # gaze normal
    phi = np.empty(n); theta = np.empty(n)
    diam3d = np.empty(n)                                        # 3d pupil diam mm
    conf3d = np.empty(n); mconf = np.empty(n)                   # confidences

    for i in range(n):
        f = frames[i]
        ts = i / FPS

        if det2d is not None:
            a = time.perf_counter()
            r2 = det2d.detect(f)
            t2d[i] = time.perf_counter() - a
        else:
            r2 = cached2d[i]
            t2d[i] = 0.0
        r2["timestamp"] = ts
        conf2d[i] = r2["confidence"]

        a = time.perf_counter()
        r3 = det3d.update_and_detect(r2, f)
        t3d[i] = time.perf_counter() - a

        c = r3["sphere"]["center"]
        sx[i], sy[i], sz[i] = c
        nrm = r3["circle_3d"]["normal"]
        nx[i], ny[i], nz[i] = nrm
        phi[i] = r3.get("phi", np.nan)
        theta[i] = r3.get("theta", np.nan)
        diam3d[i] = r3["diameter_3d"]
        conf3d[i] = r3["confidence"]
        mconf[i] = r3["model_confidence"]

    out = args.out or os.path.join(_THIS, f"out3d_{args.tag}.npz")
    np.savez(
        out,
        t2d=t2d, t3d=t3d, conf2d=conf2d,
        sx=sx, sy=sy, sz=sz, nx=nx, ny=ny, nz=nz,
        phi=phi, theta=theta, diam3d=diam3d, conf3d=conf3d, mconf=mconf,
    )

    m3 = t3d * 1000.0
    m2 = t2d * 1000.0
    tot = (t2d + t3d).sum()
    print(f"[{args.tag}] total {tot:.3f}s  pipeline fps={n/tot:.1f}")
    print(f"[{args.tag}] 2D ms: mean={m2.mean():.3f} median={np.median(m2):.3f} "
          f"p95={np.percentile(m2,95):.3f}")
    print(f"[{args.tag}] 3D ms: mean={m3.mean():.3f} median={np.median(m3):.3f} "
          f"p95={np.percentile(m3,95):.3f} p99={np.percentile(m3,99):.3f} "
          f"max={m3.max():.3f}")
    print(f"[{args.tag}] 3D >1ms: {100.0*(m3>1).mean():.1f}%   "
          f"model_conf>=1: {100.0*(mconf>=1.0).mean():.1f}%   "
          f"mean conf3d={conf3d.mean():.4f}")
    print(f"[{args.tag}] sphere center (median) = "
          f"({np.median(sx):.2f}, {np.median(sy):.2f}, {np.median(sz):.2f}) mm")
    print(f"[{args.tag}] saved -> {out}")


def _ang_between(a, b):
    """Per-row angle (deg) between two (N,3) unit-ish vectors."""
    dot = np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1)
    return np.rad2deg(np.arccos(dot))


def cmp(args):
    a = np.load(args.ref)
    b = np.load(args.cand)
    n = len(a["t3d"])
    print(f"frames={n}")

    # ---- timing ----
    for stage in ("t2d", "t3d"):
        ta, tb = a[stage] * 1000, b[stage] * 1000
        print(f"{stage} ms ref : mean={ta.mean():.3f} median={np.median(ta):.3f} "
              f"p95={np.percentile(ta,95):.3f} p99={np.percentile(ta,99):.3f}")
        print(f"{stage} ms cand: mean={tb.mean():.3f} median={np.median(tb):.3f} "
              f"p95={np.percentile(tb,95):.3f} p99={np.percentile(tb,99):.3f}")
        print(f"{stage} speedup: mean={ta.mean()/tb.mean():.3f}x "
              f"median={np.median(ta)/np.median(tb):.3f}x")

    # ---- accuracy (3D outputs) ----
    print("--- 3D output diffs (cand vs ref) ---")
    sc_a = np.stack([a["sx"], a["sy"], a["sz"]], 1)
    sc_b = np.stack([b["sx"], b["sy"], b["sz"]], 1)
    dsphere = np.linalg.norm(sc_a - sc_b, axis=1)

    nrm_a = np.stack([a["nx"], a["ny"], a["nz"]], 1)
    nrm_b = np.stack([b["nx"], b["ny"], b["nz"]], 1)
    # only where both gaze normals are finite & nonzero
    fin = np.isfinite(nrm_a).all(1) & np.isfinite(nrm_b).all(1)
    fin &= (np.linalg.norm(nrm_a, axis=1) > 0) & (np.linalg.norm(nrm_b, axis=1) > 0)
    dgaze = _ang_between(nrm_a[fin], nrm_b[fin])

    ddiam = np.abs(a["diam3d"] - b["diam3d"])
    dconf = np.abs(a["conf3d"] - b["conf3d"])

    print(f"sphere center |d| mm : max={dsphere.max():.6f} mean={dsphere.mean():.6e}")
    print(f"gaze angle   |d| deg : max={dgaze.max():.6f} mean={dgaze.mean():.6e}  "
          f"(n={fin.sum()})")
    print(f"diameter_3d  |d| mm  : max={ddiam.max():.6f} mean={ddiam.mean():.6e}")
    print(f"confidence   |d|     : max={dconf.max():.6f} mean={dconf.mean():.6e}")
    exact = (dsphere < 1e-9) & (ddiam < 1e-9)
    print(f"frames bit-identical (sphere+diam): {exact.mean()*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--tag", required=True)
    pr.add_argument("--frames", default=DEFAULT_FRAMES)
    pr.add_argument("--pupil2d", default=None,
                    help="gen2d cache npz to use as the 2D front-end (else live)")
    pr.add_argument("--max-frames", type=int, default=None)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=run)
    pg = sub.add_parser("gen2d")
    pg.add_argument("--frames", default=DEFAULT_FRAMES)
    pg.add_argument("--max-frames", type=int, default=None)
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=gen2d)
    pc = sub.add_parser("cmp")
    pc.add_argument("--ref", required=True)
    pc.add_argument("--cand", required=True)
    pc.set_defaults(func=cmp)
    args = p.parse_args()
    args.func(args)
