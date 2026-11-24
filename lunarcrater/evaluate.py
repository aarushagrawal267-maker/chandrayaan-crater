"""Evaluate detectors on held-out synthetic scenes (seeds never used in training)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .detect import CraterDetector, R_MIN
from .metrics import match_craters, summarize
from .synthetic import generate_scene
from .train import TEST_SEED0

SIZE_BINS = [("small (3-8 px)", 3, 8), ("medium (8-20 px)", 8, 20), ("large (>20 px)", 20, 1e9)]


def evaluate(n_images: int = 100, size: int = 256, methods=("unet", "hough"), detector: CraterDetector | None = None, log=print) -> dict:
    det = detector or CraterDetector()
    scenes = [generate_scene(size, seed=TEST_SEED0 + i) for i in range(n_images)]
    report: dict = {
        "test_set": {
            "images": n_images,
            "image_size_px": size,
            "seed_range": [TEST_SEED0, TEST_SEED0 + n_images - 1],
            "ground_truth_craters": int(sum(len(s.craters) for s in scenes)),
            "min_radius_px": R_MIN,
            "match_rule": "centre within 0.5 r_gt and |dr| <= 0.35 r_gt, one-to-one",
        },
        "methods": {},
    }
    for method in methods:
        if method == "unet" and not det.model_available:
            log("skipping unet: no trained model")
            continue
        per_img, times = [], []
        bins = {name: [0, 0] for name, _, _ in SIZE_BINS}  # found, total
        for sc in scenes:
            t = time.perf_counter()
            pred, _ = det.detect(sc.image, method=method)
            times.append(time.perf_counter() - t)
            res = match_craters(pred, sc.craters)
            per_img.append(res)
            matched_gt = {j for _, j in res["matches"]}
            for j, c in enumerate(sc.craters):
                for name, lo, hi in SIZE_BINS:
                    if lo <= c.r < hi:
                        bins[name][1] += 1
                        bins[name][0] += int(j in matched_gt)
        summary = summarize(per_img)
        summary["recall_by_size"] = {k: (round(f / t, 4) if t else None) for k, (f, t) in bins.items()}
        summary["mean_inference_ms"] = round(1000 * float(np.mean(times)), 1)
        report["methods"][method] = summary
        log(f"{method}: {json.dumps(summary)}")
    return report


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
