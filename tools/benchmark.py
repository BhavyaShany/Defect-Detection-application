"""Measure preprocessing latency on synthetic scans of increasing size.

    python -m tools.benchmark            # sizes 10k .. 1M points, 3 repeats each

Reports the per-stage timings produced by ``run_preprocessing``. Model inference time is
hardware/model specific and is reported by the API per scan (``timings.inference_ms``).
"""
import argparse
import statistics as st
import time

from backend.services.preprocessing import run_preprocessing
from backend.services.scan_io import load_scan
from tests.helpers import ply_binary
from tools.synthetic_parts import generate_part

STAGES = ["parse_ms", "align_ms", "outlier_filter_ms", "surface_analysis_ms", "projection_ms",
          "view_prep_ms", "preprocessing_total_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[10_000, 50_000, 200_000, 1_000_000])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    print(f"{'points':>9} | " + " | ".join(f"{s.replace('_ms',''):>12}" for s in STAGES) + " |  file MB")
    for n in a.sizes:
        pts, _ = generate_part("engine_bracket", "porosity", n=n, seed=1)
        raw = ply_binary(pts)
        runs = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            scan = load_scan(raw, "b.ply", min_points=200)
            parse = (time.perf_counter() - t) * 1000
            r = run_preprocessing(scan)
            runs.append({"parse_ms": parse, **r.timings_ms})
        med = {k: st.median(x[k] for x in runs) for k in STAGES}
        print(f"{n:>9,} | " + " | ".join(f"{med[s]:>12.1f}" for s in STAGES) + f" | {len(raw)/1e6:>7.1f}")


if __name__ == "__main__":
    main()
