"""Turn a crater-rim probability map into a list of circles.

Ring templates over a geometric range of radii are matched against the probability map
(normalised cross-correlation, as in DeepMoon's template-matching step).  Local maxima
above a threshold become candidates; duplicates are removed with a non-maximum
suppression that compares centre distance and radius.
"""
from __future__ import annotations

import cv2
import numpy as np


def ring_template(r: float) -> np.ndarray:
    thick = max(1, int(round(r * 0.12))) + 1
    half = int(np.ceil(r + thick))
    size = 2 * half + 1
    t = np.zeros((size, size), np.float32)
    cv2.circle(t, (half * 4, half * 4), int(round(r * 4)), 1.0, thick, lineType=cv2.LINE_AA, shift=2)
    return t


def radius_grid(r_min: float, r_max: float, step: float = 1.07) -> list[float]:
    radii = []
    r = r_min
    while r <= r_max:
        radii.append(round(r, 2))
        r = max(r * step, r + 0.5)
    return radii


def ring_support(prob: np.ndarray, x: float, y: float, r: float, n: int = 72, dilated: np.ndarray | None = None) -> float:
    """Mean rim probability sampled around the circle (inside the image only).

    ``dilated`` is ``prob`` after a 3x3 dilation; pass it when scoring many circles so the
    dilation is computed once rather than once per circle."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.round(x + r * np.cos(ang)).astype(int)
    ys = np.round(y + r * np.sin(ang)).astype(int)
    h, w = prob.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if ok.sum() < n * 0.4:
        return 0.0
    # allow +-1 px tolerance: take the max in a 3x3 neighbourhood
    dil = dilated if dilated is not None else cv2.dilate(prob, np.ones((3, 3), np.uint8))
    return float(dil[ys[ok], xs[ok]].mean())


def nms_circles(cands: list[tuple[float, float, float, float]], dist_frac: float = 0.6, rad_frac: float = 0.45):
    """Greedy suppression; ``cands`` are (x, y, r, score). Higher scores win."""
    kept: list[tuple[float, float, float, float]] = []
    kxyr = np.empty((len(cands), 3))  # kept circles as an array: vectorised overlap test
    for c in sorted(cands, key=lambda t: -t[3]):
        x, y, r, _ = c
        m = len(kept)
        if m:
            kx, ky, kr = kxyr[:m, 0], kxyr[:m, 1], kxyr[:m, 2]
            rm = np.minimum(r, kr)
            if np.any(((x - kx) ** 2 + (y - ky) ** 2 < (dist_frac * rm) ** 2) & (np.abs(r - kr) / rm < rad_frac)):
                continue
        kxyr[m] = (x, y, r)
        kept.append(c)
    return kept


def extract_craters(
    prob: np.ndarray,
    r_min: float = 3.0,
    r_max: float | None = None,
    match_threshold: float = 0.5,
    min_support: float = 0.6,
) -> list[dict]:
    prob = prob.astype(np.float32)
    h, w = prob.shape
    r_max = r_max or min(h, w) / 4.0
    cands: list[tuple[float, float, float, float]] = []
    if prob.max() < min_support:
        return []
    for r in radius_grid(r_min, r_max):
        t = ring_template(r)
        pad = t.shape[0] // 2
        padded = cv2.copyMakeBorder(prob, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        res = cv2.matchTemplate(padded, t, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        k = max(3, int(r) | 1)
        peaks = (res >= match_threshold) & (res == cv2.dilate(res, np.ones((k, k), np.uint8)))
        ys, xs = np.nonzero(peaks)
        for y, x in zip(ys, xs):
            if 0 <= x < w and 0 <= y < h:
                cands.append((float(x), float(y), float(r), float(res[y, x])))
    out = []
    dilated = cv2.dilate(prob, np.ones((3, 3), np.uint8))
    for x, y, r, score in nms_circles(cands):
        support = ring_support(prob, x, y, r, dilated=dilated)
        if support >= min_support:
            out.append({"x": x, "y": y, "r": r, "match": round(score, 4), "confidence": round(support, 4)})
    # second pass NMS on confidence to drop concentric duplicates kept by score ordering
    final = nms_circles([(c["x"], c["y"], c["r"], c["confidence"] + c["match"]) for c in out])
    keep = {(x, y, r) for x, y, r, _ in final}
    return [c for c in out if (c["x"], c["y"], c["r"]) in keep]
