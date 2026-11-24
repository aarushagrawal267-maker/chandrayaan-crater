"""Scientific summaries of a crater catalogue.

* crater catalogue in metres (needs the image resolution in metres/pixel),
* cumulative size-frequency distribution N(>=D) per km^2 and a power-law slope fit,
* crater density and areal coverage,
* an *educational* landing-site screening map: distance to the nearest detected crater
  hazard combined with a local image-roughness proxy, with the best candidate sites.

A single optical image carries no true elevation, so the screening map is a teaching aid
and not a substitute for the DEM/slope/boulder analysis used for real missions such as
Chandrayaan-3.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np


def catalogue(craters: list[dict], mpp: float) -> list[dict]:
    rows = []
    for i, c in enumerate(sorted(craters, key=lambda c: -c["r"]), start=1):
        rows.append(
            {
                "id": i,
                "x_px": round(c["x"], 1),
                "y_px": round(c["y"], 1),
                "radius_px": round(c["r"], 2),
                "diameter_m": round(2 * c["r"] * mpp, 2),
                "confidence": round(float(c.get("confidence", 0.0)), 3),
            }
        )
    return rows


def size_frequency(diameters_m: list[float], area_km2: float, d_complete_m: float) -> dict:
    """Cumulative SFD and least-squares slope of log N(>=D) vs log D for D >= completeness."""
    d = np.sort(np.asarray(diameters_m, dtype=float))[::-1]
    if len(d) == 0 or area_km2 <= 0:
        return {"points": [], "slope": None, "fit_n": 0, "completeness_m": d_complete_m}
    cum = np.arange(1, len(d) + 1) / area_km2
    points = [{"diameter_m": round(float(x), 3), "n_cum_per_km2": float(f"{y:.6g}")} for x, y in zip(d, cum)]
    sel = d >= d_complete_m
    slope = intercept = None
    if sel.sum() >= 5 and len(np.unique(d[sel])) >= 3:
        slope, intercept = np.polyfit(np.log10(d[sel]), np.log10(cum[sel]), 1)
        slope, intercept = round(float(slope), 3), round(float(intercept), 4)
    return {"points": points, "slope": slope, "intercept": intercept, "fit_n": int(sel.sum()), "completeness_m": round(d_complete_m, 3)}


def coverage_fraction(craters: list[dict], shape: tuple[int, int]) -> float:
    mask = np.zeros(shape, np.uint8)
    for c in craters:
        cv2.circle(mask, (int(round(c["x"])), int(round(c["y"]))), max(1, int(round(c["r"]))), 1, -1)
    return float(mask.mean())


def _colormap_rgba(score: np.ndarray, alpha: int = 150) -> np.ndarray:
    """Red (hazard) -> amber -> green (safe) RGBA image."""
    s = np.clip(score, 0, 1)
    r = np.where(s < 0.5, 220, 220 - (s - 0.5) * 2 * 180)
    g = np.where(s < 0.5, 40 + s * 2 * 150, 190)
    b = np.full_like(s, 60.0)
    a = np.full_like(s, alpha)
    return np.dstack([r, g, b, a]).astype(np.uint8)


def png_base64(rgba_or_gray: np.ndarray) -> str:
    arr = rgba_or_gray
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def landing_safety(
    img: np.ndarray,
    craters: list[dict],
    mpp: float,
    footprint_m: float = 10.0,
    hazard_margin: float = 1.25,
    n_sites: int = 3,
) -> dict:
    h, w = img.shape
    hazard = np.zeros((h, w), np.uint8)
    for c in craters:
        cv2.circle(hazard, (int(round(c["x"])), int(round(c["y"]))), max(1, int(round(c["r"] * hazard_margin))), 1, -1)
    # unknown terrain beyond the frame counts as a hazard
    hazard[0, :] = hazard[-1, :] = 1
    hazard[:, 0] = hazard[:, -1] = 1
    clearance_px = cv2.distanceTransform((1 - hazard).astype(np.uint8), cv2.DIST_L2, 5)
    clearance_m = clearance_px * mpp

    # roughness proxy: local standard deviation of the (slightly smoothed) brightness
    win = max(3, int(round(footprint_m / mpp)) | 1)
    win = min(win, max(3, (min(h, w) // 4) | 1))
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    mean = cv2.blur(f, (win, win))
    sq = cv2.blur(f * f, (win, win))
    rough = np.sqrt(np.clip(sq - mean * mean, 0, None))
    rough_rank = np.argsort(np.argsort(rough.ravel())).reshape(h, w) / max(1, rough.size - 1)

    need = footprint_m / 2.0
    clear_score = np.clip(clearance_m / (2.0 * max(need, 1e-6)), 0, 1)
    score = clear_score * (1.0 - 0.4 * rough_rank)
    score[clearance_m < need] = np.minimum(score[clearance_m < need], 0.2 * clear_score[clearance_m < need])
    safe_fraction = float((clearance_m >= need).mean())
    # Do not propose sites right at the frame edge: the terrain just outside the image is unknown
    # and the roughness window would be cut off there. Keep a band of one footprint (>= 8 px).
    band = min(max(int(np.ceil(footprint_m / mpp)), 8), max(1, min(h, w) // 4))
    edge = np.zeros((h, w), bool)
    edge[:band, :] = edge[-band:, :] = True
    edge[:, :band] = edge[:, -band:] = True
    score[edge] = np.minimum(score[edge], 0.2 * clear_score[edge])

    sites = []
    work = score.copy()
    sep = max(int(round(footprint_m / mpp * 2)), min(h, w) // 8, 4)
    for _ in range(n_sites):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w)
        if work[y, x] <= 0.05:
            break
        sites.append(
            {
                "rank": len(sites) + 1,
                "x_px": int(x),
                "y_px": int(y),
                "clearance_m": round(float(clearance_m[y, x]), 2),
                "roughness_percentile": round(float(rough_rank[y, x] * 100), 1),
                "score": round(float(score[y, x]), 3),
                "fits_footprint": bool(clearance_m[y, x] >= need),
            }
        )
        cv2.circle(work, (int(x), int(y)), sep, 0, -1)
    return {
        "footprint_m": footprint_m,
        "safe_fraction": round(safe_fraction, 4),
        "sites": sites,
        "overlay_png": png_base64(_colormap_rgba(score)),
    }


def analyse(img: np.ndarray, craters: list[dict], mpp: float, footprint_m: float = 10.0, r_min_px: float = 3.0) -> dict:
    h, w = img.shape
    area_km2 = h * w * mpp * mpp / 1e6
    cat = catalogue(craters, mpp)
    diam = [r["diameter_m"] for r in cat]
    d_complete = 2 * r_min_px * mpp * 1.5
    sfd = size_frequency(diam, area_km2, d_complete)
    stats = {
        "count": len(cat),
        "image_width_px": w,
        "image_height_px": h,
        "meters_per_pixel": mpp,
        "area_km2": round(area_km2, 6),
        "density_per_km2": round(len(cat) / area_km2, 3) if area_km2 else None,
        "coverage_percent": round(coverage_fraction(craters, (h, w)) * 100, 2),
        "diameter_min_m": round(min(diam), 2) if diam else None,
        "diameter_median_m": round(float(np.median(diam)), 2) if diam else None,
        "diameter_max_m": round(max(diam), 2) if diam else None,
        "mean_confidence": round(float(np.mean([r["confidence"] for r in cat])), 3) if cat else None,
    }
    return {
        "catalogue": cat,
        "stats": stats,
        "sfd": sfd,
        "landing": landing_safety(img, craters, mpp, footprint_m=footprint_m),
    }


def to_csv(cat: list[dict]) -> str:
    cols = ["id", "x_px", "y_px", "radius_px", "diameter_m", "confidence"]
    lines = [",".join(cols)]
    for r in cat:
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines) + "\n"

