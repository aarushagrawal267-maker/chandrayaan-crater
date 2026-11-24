"""Crater-matching metrics (precision, recall, F1, localisation errors).

A predicted crater matches a ground-truth crater when its centre is within
``dist_tol * r_gt`` and its radius differs by less than ``rad_tol * r_gt``.  Matching is
one-to-one and greedy by increasing normalised error, similar to the criteria used in
DeepMoon (Silburt et al., 2019).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _xyr(c) -> tuple[float, float, float]:
    if isinstance(c, dict):
        return float(c["x"]), float(c["y"]), float(c["r"])
    if hasattr(c, "as_tuple"):
        return c.as_tuple()
    return float(c[0]), float(c[1]), float(c[2])


def match_craters(pred: Iterable, gt: Iterable, dist_tol: float = 0.5, rad_tol: float = 0.35) -> dict:
    P = [_xyr(c) for c in pred]
    G = [_xyr(c) for c in gt]
    pairs = []
    for i, (px, py, pr) in enumerate(P):
        for j, (gx, gy, gr) in enumerate(G):
            d = np.hypot(px - gx, py - gy) / gr
            dr = abs(pr - gr) / gr
            if d <= dist_tol and dr <= rad_tol:
                pairs.append((d + dr, i, j))
    pairs.sort()
    used_p, used_g, matches = set(), set(), []
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j))
    tp = len(matches)
    fp = len(P) - tp
    fn = len(G) - tp
    errs_xy = [np.hypot(P[i][0] - G[j][0], P[i][1] - G[j][1]) / G[j][2] for i, j in matches]
    errs_r = [abs(P[i][2] - G[j][2]) / G[j][2] for i, j in matches]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "frac_err_xy": float(np.mean(errs_xy)) if errs_xy else None,
        "frac_err_r": float(np.mean(errs_r)) if errs_r else None,
    }


def summarize(results: list[dict]) -> dict:
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exy = [r["frac_err_xy"] for r in results if r["frac_err_xy"] is not None]
    er = [r["frac_err_r"] for r in results if r["frac_err_r"] is not None]
    return {
        "images": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_frac_err_xy": round(float(np.mean(exy)), 4) if exy else None,
        "mean_frac_err_r": round(float(np.mean(er)), 4) if er else None,
    }


def pixel_scores(prob: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict:
    pred = prob >= threshold
    m = mask.astype(bool)
    inter = float(np.logical_and(pred, m).sum())
    union = float(np.logical_or(pred, m).sum())
    return {"iou": inter / union if union else 1.0, "dice": 2 * inter / (pred.sum() + m.sum()) if (pred.sum() + m.sum()) else 1.0}
