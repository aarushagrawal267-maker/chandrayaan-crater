"""Classical computer-vision baseline: contrast enhancement + circular Hough transform.

It needs no training and is used as a reference point for the U-Net in the evaluation.
Under the low sun angles typical of lunar orbital images a crater looks like a crescent of
shadow next to a crescent of light rather than a closed circular edge, which is exactly the
situation where a gradient-based Hough transform struggles - the evaluation report shows it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .extract import nms_circles


def hough_craters(img: np.ndarray, r_min: float = 3.0, r_max: float | None = None, perfectness: float = 0.6) -> list[dict]:
    h, w = img.shape
    r_max = r_max or min(h, w) / 4.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(img)
    blur = cv2.GaussianBlur(enh, (0, 0), 1.0)
    edges = cv2.Canny(blur, 40, 110)
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=max(4, r_min * 1.3),
        param1=100,
        param2=perfectness,
        minRadius=int(np.floor(r_min)),
        maxRadius=int(np.ceil(r_max)),
    )
    if circles is None:
        return []
    out = []
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for x, y, r in circles[0]:
        xs = np.round(x + r * np.cos(ang)).astype(int)
        ys = np.round(y + r * np.sin(ang)).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        support = float(edges_d[ys[ok], xs[ok]].mean()) if ok.any() else 0.0
        out.append((float(x), float(y), float(r), support))
    return [
        {"x": x, "y": y, "r": r, "match": None, "confidence": round(s, 4)}
        for x, y, r, s in nms_circles(out)
    ]
