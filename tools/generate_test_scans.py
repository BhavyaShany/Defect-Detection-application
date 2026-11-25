"""Write synthetic test scans (.ply) for exercising the pipeline and the upload UI.

    python -m tools.generate_test_scans --out test_scans

Files are named ``<part>__<defect>.ply``. They are geometric stand-ins only: they carry
no ground-truth *classification* for the trained CNN and must not be used to evaluate it.
"""
import argparse
from pathlib import Path

from tools.synthetic_parts import DEFECTS, PARTS, generate_part, write_ply_binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_scans")
    ap.add_argument("--points", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        for defect in DEFECTS:
            pts, _ = generate_part(part, defect, n=args.points, seed=args.seed)
            write_ply_binary(out / f"{part}__{defect}.ply", pts)
    print(f"Wrote {len(PARTS) * len(DEFECTS)} scans to {out.resolve()}")


if __name__ == "__main__":
    main()
