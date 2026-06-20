"""Interleaved timing for the pye3d 3D stage: alternately run the reference and
candidate venvs many times to control for machine drift (machine variance is
high here -- the same code measured 0.84ms vs 2.2ms median minutes apart).
Reports median-of-run-medians per build.

Each round runs the full 2D->3D pipeline (2D timed only to warm/feed the model)
but reports stats for the 3D stage (update_and_detect) only.

    python bench/interleave3d.py [n_rounds] [frames.raw]

Env: REF_VENV (default .venv-ref) vs CAND_VENV (default .venv).
"""
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pye3d-detector/
PKGS = os.path.dirname(ROOT)
REF_VENV = os.environ.get("REF_VENV", ".venv-ref")
CAND_VENV = os.environ.get("CAND_VENV", ".venv")
REF_PY = os.path.join(ROOT, REF_VENV, "Scripts", "python.exe")
CAND_PY = os.path.join(ROOT, CAND_VENV, "Scripts", "python.exe")
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
FRAMES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    PKGS, "pupil-detectors", "bench", "frames_2000.raw"
)
# Cached optimized-2D ellipse stream (gen2d) -> representative 3D inputs without
# running pupil_detectors in the timing venvs. Default matches the frames clip.
_dflt2d = "pupil2d_blink.npz" if "blink" in os.path.basename(FRAMES) else "pupil2d_clean.npz"
PUPIL2D = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "bench", _dflt2d)

SNIPPET = r'''
import os, time, numpy as np
_b = r"{root}/bench"
import sys; sys.path.insert(0, _b)
from bench3d import load_frames, load_pupil2d, W, H, FPS, FOCAL_LENGTH
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
fr = load_frames(r"{frames}")
n = len(fr)
r2 = load_pupil2d(r"{pupil2d}")[:n]
for i in range(n):
    r2[i]["timestamp"] = i / FPS
det3d = Detector3D(camera=CameraModel(focal_length=FOCAL_LENGTH, resolution=[W, H]),
                   long_term_mode=DetectorMode.blocking)
ts = np.empty(n)
for i in range(n):
    a = time.perf_counter(); det3d.update_and_detect(r2[i], fr[i]); ts[i] = time.perf_counter() - a
ms = ts * 1000
print("STATS %.4f %.4f %.4f %.4f %.4f %.4f" % (
    np.median(ms), ms.mean(), np.percentile(ms, 90), np.percentile(ms, 95),
    np.percentile(ms, 99), 100.0 * (ms > 2).mean()))
'''.format(root=ROOT.replace("\\", "/"), frames=FRAMES.replace("\\", "/"),
           pupil2d=PUPIL2D.replace("\\", "/"))

KEYS = ["med", "mean", "p90", "p95", "p99", "pct_gt2"]


_RUN_CWD = os.path.join(ROOT, "bench")  # no local pye3d/ here -> use installed wheel


def one(py):
    out = subprocess.run([py, "-c", SNIPPET], capture_output=True, text=True,
                         cwd=_RUN_CWD)
    m = re.search(r"STATS " + " ".join([r"([\d.]+)"] * 6), out.stdout)
    if not m:
        print("ERR:", out.stdout[-400:], out.stderr[-1200:]); sys.exit(1)
    return dict(zip(KEYS, map(float, m.groups())))


ref = {k: [] for k in KEYS}
cand = {k: [] for k in KEYS}
for r in range(N_ROUNDS):
    if r % 2 == 0:
        a = one(REF_PY); b = one(CAND_PY)
    else:
        b = one(CAND_PY); a = one(REF_PY)
    for k in KEYS:
        ref[k].append(a[k]); cand[k].append(b[k])
    print(f"round {r}: ref med={a['med']:.3f} p95={a['p95']:.3f} p99={a['p99']:.3f} "
          f"| cand med={b['med']:.3f} p95={b['p95']:.3f} p99={b['p99']:.3f}")

print(f"\n=== summary over {N_ROUNDS} rounds (median of per-run values) ===")
print(f"{'metric':>8}  {'reference':>10}  {'candidate':>10}  {'speedup':>8}")
for k in KEYS:
    rv = statistics.median(ref[k]); cv = statistics.median(cand[k])
    sp = f"{rv/cv:.3f}x" if k != "pct_gt2" and cv else "-"
    print(f"{k:>8}  {rv:>10.4f}  {cv:>10.4f}  {sp:>8}")
