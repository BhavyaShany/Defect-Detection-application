"""Convert labelled point-cloud scans into a depth/normal image dataset for CNN training.

Input layout (class = sub-folder name; any of .ply .pcd .stl .obj):

    scans/
      ok/        part001.ply ...
      porosity/  part101.ply ...
      cracks/    ...

Output layout (ready for ml/train_model.py):

    dataset/
      ok/part001.png ...

It uses exactly the same loading, alignment, filtering and projection code as the live API
(``backend.services.preprocessing``), so training images match inference inputs.

    python -m ml.build_depth_dataset --scans scans --out dataset --mode depth
"""
import argparse
import sys
from pathlib import Path

import cv2

from backend.config import settings
from backend.services.geometry import build_model_input
from backend.services.preprocessing import run_preprocessing
from backend.services.scan_io import ScanError, load_scan

EXTS = {".ply", ".pcd", ".stl", ".obj"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", required=True, help="root folder with one sub-folder per class")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="depth", choices=["depth", "normal", "depth_normal"],
                    help="image representation; must be set as input_mode when training")
    args = ap.parse_args(argv)

    root, out = Path(args.scans), Path(args.out)
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        print("Need at least two class sub-folders.", file=sys.stderr)
        return 2
    ok = bad = 0
    for cdir in classes:
        (out / cdir.name).mkdir(parents=True, exist_ok=True)
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                scan = load_scan(f.read_bytes(), f.name, min_points=settings.MIN_POINTS)
                pre = run_preprocessing(scan)
                img = build_model_input(pre.depth_map, pre.normal_map, args.mode).astype("uint8")
                cv2.imwrite(str(out / cdir.name / (f.stem + ".png")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                ok += 1
            except ScanError as exc:
                bad += 1
                print(f"SKIP {f}: {exc.message}", file=sys.stderr)
    print(f"Converted {ok} scans ({bad} skipped) -> {out}")
    print(f"Train with:  python -m ml.train_model --data-dir {out} --input-mode {args.mode} --pass-classes <ok-class-name>")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
