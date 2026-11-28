"""Evaluate detectors on held-out synthetic scenes (seeds never used in training).

``split="test"`` (default) scores the test seeds that back ``models/metrics.json``.
``split="val"`` scores validation seeds instead; use it to choose thresholds so the test
scenes are never used for tuning.  Precision, recall and F1 come with 95 % bootstrap
confidence intervals (images resampled with replacement), which describe the scatter
between simulated scenes only, not the gap between simulated and real images.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .detect import CraterDetector, R_MIN
from .metrics import match_craters, summarize
from .synthetic import generate_scene
from .train import TEST_SEED0, VAL_SEED0

SIZE_BINS = [("small (3-8 px)", 3, 8), ("medium (8-20 px)", 8, 20), ("large (>20 px)", 20, 1e9)]  # crater radius
SPLITS = {"test": TEST_SEED0, "val": VAL_SEED0}


def bootstrap_ci(per_img: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """95 % percentile bootstrap intervals for pooled precision, recall and F1 (resampling images)."""
    tp = np.array([r["tp"] for r in per_img], float)
    fp = np.array([r["fp"] for r in per_img], float)
    fn = np.array([r["fn"] for r in per_img], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_img), size=(n_boot, len(per_img)))
    T, P, N = tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(T + P > 0, T / (T + P), 0.0)
        rec = np.where(T + N > 0, T / (T + N), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in (("precision", prec), ("recall", rec), ("f1", f1))}


def evaluate(
    n_images: int = 100,
    size: int = 256,
    methods=("unet", "hough"),
    detector: CraterDetector | None = None,
    split: str = "test",
    min_confidence: float = 0.6,
    log=print,
) -> dict:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}")
    seed0 = SPLITS[split]
    det = detector or CraterDetector()
    scenes = [generate_scene(size, seed=seed0 + i) for i in range(n_images)]
    report: dict = {
        "test_set": {
            "split": split,
            "images": n_images,
            "image_size_px": size,
            "seed_range": [seed0, seed0 + n_images - 1],
            "ground_truth_craters": int(sum(len(s.craters) for s in scenes)),
            "min_radius_px": R_MIN,
            "match_rule": "centre within 0.5 r_gt and |dr| <= 0.35 r_gt, one-to-one",
            "min_confidence": min_confidence,
            "ci95": "percentile bootstrap over images, 2000 resamples, seed 0",
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
            pred, _ = det.detect(sc.image, method=method, min_confidence=min_confidence)
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
        summary["ci95"] = bootstrap_ci(per_img)
        summary["recall_by_size"] = {k: (round(f / t, 4) if t else None) for k, (f, t) in bins.items()}
        summary["mean_inference_ms"] = round(1000 * float(np.mean(times)), 1)
        report["methods"][method] = summary
        log(f"{method}: {json.dumps(summary)}")
    return report


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
