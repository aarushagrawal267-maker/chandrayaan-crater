"""Command-line interface.

    python -m lunarcrater.cli make-samples          # regenerate the demo images in samples/
    python -m lunarcrater.cli train --epochs 16     # train the U-Net on synthetic scenes
    python -m lunarcrater.cli evaluate              # score U-Net and Hough on held-out scenes
    python -m lunarcrater.cli detect IMAGE --mpp 5  # detect craters in your own image
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .detect import ROOT, CraterDetector

SAMPLES_DIR = ROOT / "samples"

# Demo scenes: (file name, seed, size, metres/pixel, description)
DEMO_SCENES = [
    ("mare_plain_512.png", 31_000_001, 512, 5.0, "Simulated mare plain at TMC-2 resolution (5 m/px)"),
    ("highland_cratered_512.png", 31_000_002, 512, 5.0, "Simulated densely cratered highland terrain (5 m/px)"),
    ("south_polar_lowsun_384.png", 31_000_003, 384, 0.25, "Simulated OHRC-like close-up under grazing south-polar sunlight (0.25 m/px)"),
    ("landing_zone_256.png", 31_000_004, 256, 0.25, "Simulated candidate landing patch (0.25 m/px)"),
]


def make_samples(out_dir: Path = SAMPLES_DIR) -> list[dict]:
    from .synthetic import generate_scene

    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, seed, size, mpp, desc in DEMO_SCENES:
        kwargs = {"n_craters": (int(size * size / 2500), int(size * size / 1200))}
        if "highland" in name:
            kwargs = {"n_craters": (int(size * size / 900), int(size * size / 700))}
        elif "lowsun" in name:
            kwargs["sun_elevation"] = 8.0
        elif "landing" in name:
            kwargs = {"n_craters": (10, 16)}
        sc = generate_scene(size, seed=seed, **kwargs)
        cv2.imwrite(str(out_dir / name), sc.image)
        index.append(
            {
                "file": name,
                "meters_per_pixel": mpp,
                "description": desc,
                "sun_azimuth_deg": round(sc.sun_azimuth, 1),
                "sun_elevation_deg": round(sc.sun_elevation, 1),
                "ground_truth": [[round(c.x, 2), round(c.y, 2), round(c.r, 2)] for c in sc.craters],
            }
        )
    (out_dir / "samples.json").write_text(json.dumps(index, indent=1))
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lunarcrater", description="Chandrayaan crater detection toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("make-samples", help="generate the demo images in samples/")

    t = sub.add_parser("train", help="train the U-Net on synthetic lunar scenes")
    t.add_argument("--epochs", type=int, default=16)
    t.add_argument("--train-size", type=int, default=1600)
    t.add_argument("--val-size", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--out", default=str(ROOT / "models" / "crater_unet.pt"))
    t.add_argument("--init-from", default=None, help="continue training from this checkpoint")
    t.add_argument("--schedule", choices=["onecycle", "cosine"], default="onecycle")

    e = sub.add_parser("evaluate", help="evaluate detectors on held-out synthetic scenes")
    e.add_argument("--images", type=int, default=100)
    e.add_argument("--model", default=None)
    e.add_argument("--out", default=str(ROOT / "models" / "metrics.json"))

    d = sub.add_parser("detect", help="detect craters in an image (PNG/JPG/TIFF or PDS4 .xml label)")
    d.add_argument("image")
    d.add_argument("--mpp", type=float, default=5.0, help="metres per pixel (TMC-2: 5, OHRC: 0.25)")
    d.add_argument("--method", choices=["unet", "hough"], default="unet")
    d.add_argument("--min-confidence", type=float, default=0.6)
    d.add_argument("--footprint", type=float, default=10.0, help="lander footprint in metres for site screening")
    d.add_argument("--csv", help="write crater catalogue CSV here")
    d.add_argument("--json", help="write full analysis JSON here")
    d.add_argument("--annotated", help="write an annotated PNG here")
    d.add_argument("--model", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "make-samples":
        idx = make_samples()
        print(f"wrote {len(idx)} samples to {SAMPLES_DIR}")
        return 0

    if args.cmd == "train":
        from .train import train

        res = train(epochs=args.epochs, train_size=args.train_size, val_size=args.val_size, batch_size=args.batch_size, lr=args.lr, workers=args.workers, out=args.out, init_from=args.init_from, schedule=args.schedule)
        print(f"best val loss {res['best_val_loss']:.4f}; model saved to {res['model_path']}")
        return 0

    if args.cmd == "evaluate":
        from .evaluate import evaluate, write_report

        rep = evaluate(n_images=args.images, detector=CraterDetector(args.model))
        write_report(rep, args.out)
        print(f"report written to {args.out}")
        return 0

    if args.cmd == "detect":
        from .analysis import analyse, to_csv
        from .imageio import ImageLoadError, limit_size, load_image

        try:
            img = load_image(args.image)
        except ImageLoadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        img, scale = limit_size(img)
        mpp = args.mpp / scale
        det = CraterDetector(args.model)
        craters, _ = det.detect(img, method=args.method, min_confidence=args.min_confidence)
        result = analyse(img, craters, mpp, footprint_m=args.footprint)
        s = result["stats"]
        print(f"{s['count']} craters | density {s['density_per_km2']} /km^2 | coverage {s['coverage_percent']}% | SFD slope {result['sfd']['slope']}")
        for site in result["landing"]["sites"]:
            print(f"  candidate site #{site['rank']}: ({site['x_px']}, {site['y_px']}) px, clearance {site['clearance_m']} m")
        if args.csv:
            Path(args.csv).write_text(to_csv(result["catalogue"]))
        if args.json:
            r = dict(result)
            r["landing"] = {k: v for k, v in r["landing"].items() if k != "overlay_png"}
            Path(args.json).write_text(json.dumps(r, indent=2))
        if args.annotated:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for c in craters:
                cv2.circle(vis, (int(round(c["x"])), int(round(c["y"]))), int(round(c["r"])), (0, 200, 255), 1, cv2.LINE_AA)
            for site in result["landing"]["sites"]:
                cv2.drawMarker(vis, (site["x_px"], site["y_px"]), (80, 220, 80), cv2.MARKER_CROSS, 14, 2)
            cv2.imwrite(args.annotated, vis)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
