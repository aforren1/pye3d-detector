"""Compare two ci_bench.py JSON outputs (built wheel vs PyPI release).

    python bench/ci_compare.py --built built.json --baseline pypi.json

Prints a GitHub-flavoured-markdown summary (append to $GITHUB_STEP_SUMMARY in
CI). Exit 0 unless --fail-under is given and the median speedup (baseline/built)
falls below it, so the job can optionally guard against a real regression while
tolerating runner noise by default.
"""
import argparse
import json


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fail-under", type=float, default=None)
    args = ap.parse_args()

    built = json.load(open(args.built))
    base = json.load(open(args.baseline))
    bl, pl = built["latency_ms"], base["latency_ms"]
    speedup = pl["median"] / bl["median"] if bl["median"] else float("nan")

    lines = [
        "### pye3d 3D pipeline benchmark — built wheel vs PyPI",
        "",
        f"`{built['version']}` (built) vs `{base['version']}` (PyPI) — "
        f"{built['n_updates']} update_and_detect calls, same runner.",
        "",
        "| latency (ms) | built | PyPI | speedup |",
        "|---|---|---|---|",
    ]
    for k in ("median", "p90", "p99", "mean"):
        s = pl[k] / bl[k] if bl[k] else float("nan")
        lines.append(f"| {k} | {fmt(bl[k])} | {fmt(pl[k])} | {fmt(s)}× |")
    lines += [
        "",
        "| output signature | built | PyPI |",
        "|---|---|---|",
        f"| mean sphere radius | {fmt(built['signature']['mean_sphere_radius'])} | {fmt(base['signature']['mean_sphere_radius'])} |",
        f"| mean diameter_3d | {fmt(built['signature']['mean_diameter_3d'])} | {fmt(base['signature']['mean_diameter_3d'])} |",
        "",
        f"**Median speedup: {fmt(speedup)}×**",
    ]
    report = "\n".join(lines)
    print(report)

    if args.fail_under is not None and speedup < args.fail_under:
        raise SystemExit(
            f"::error::median speedup {speedup:.2f}x < required {args.fail_under}x"
        )


if __name__ == "__main__":
    main()
