"""High-level crater detection on images of any size.

* ``unet``  - the trained CraterUNet predicts a rim-probability map (tiled with overlap),
  rings are extracted by template matching.  Large images are processed as an image
  pyramid (full, 1/2, 1/4 and, for very large images, 1/8 resolution; a level is used only
  while its shorter side stays >= 128 px) so that big craters still fit the network's
  receptive field; results from all levels are merged with non-maximum suppression.
* ``hough`` - classical baseline (no model needed).
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from .classical import hough_craters
from .extract import extract_craters, nms_circles

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "crater_unet.pt"
METHODS = ("unet", "hough")

TILE = 256
OVERLAP = 32
R_MIN = 3.0
R_MAX = 40.0  # largest radius (px) searched per pyramid level; training craters reach ~32 px


def standardize(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return (x - x.mean()) / (x.std() + 1e-3)


class ModelUnavailable(RuntimeError):
    pass


class CraterDetector:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self.device = device
        self._model = None
        self.model_info: dict = {}
        self.load_error: str | None = None  # set when the model file exists but cannot be loaded
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- model
    @property
    def model_available(self) -> bool:
        """True when the model file exists and has not failed to load."""
        return self.model_path.exists() and self.load_error is None

    def load(self) -> str | None:
        """Load the U-Net now (e.g. at app start-up). Returns None on success, else the error message."""
        try:
            self._get_model()
        except ModelUnavailable as exc:
            return str(exc)
        return None

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if self.load_error:
                        raise ModelUnavailable(self.load_error)
                    if not self.model_path.exists():
                        raise ModelUnavailable(
                            f"No trained model at {self.model_path}. Restore the shipped file "
                            "(git checkout -- models/crater_unet.pt), train one with `python -m lunarcrater train`, "
                            "or use the 'hough' method."
                        )
                    from .model import load_model

                    try:
                        self._model, self.model_info = load_model(self.model_path, self.device)
                    except Exception as exc:  # corrupt / truncated / wrong file (e.g. a Git LFS pointer)
                        self.load_error = (
                            f"The model file {self.model_path} could not be loaded ({type(exc).__name__}: {exc}). "
                            "It is probably incomplete or corrupt: download or restore it again, retrain with "
                            "`python -m lunarcrater train`, or use the 'hough' method."
                        )
                        raise ModelUnavailable(self.load_error) from exc
        return self._model

    def predict_prob(self, img: np.ndarray) -> np.ndarray:
        """Rim probability for an image of any size (tiled inference with overlap)."""
        import torch

        model = self._get_model()
        h, w = img.shape
        stride = 16
        if h <= TILE and w <= TILE:
            ph, pw = -(-h // stride) * stride, -(-w // stride) * stride
            x = np.pad(standardize(img), ((0, ph - h), (0, pw - w)), mode="reflect")
            with torch.no_grad():
                logits = model(torch.from_numpy(x)[None, None].to(self.device))
            return torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()

        prob = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        step = TILE - OVERLAP
        ys = list(range(0, max(1, h - TILE), step)) + [max(0, h - TILE)]
        xs = list(range(0, max(1, w - TILE), step)) + [max(0, w - TILE)]
        win = np.outer(np.hanning(TILE + 2)[1:-1], np.hanning(TILE + 2)[1:-1]).astype(np.float32) + 1e-3
        norm_img = img  # standardise each tile separately: robust to illumination gradients
        batch, coords = [], []

        def flush():
            if not batch:
                return
            with torch.no_grad():
                out = torch.sigmoid(model(torch.from_numpy(np.stack(batch))[:, None].to(self.device)))[:, 0].cpu().numpy()
            for (yy, xx, th, tw), p in zip(coords, out):
                prob[yy : yy + th, xx : xx + tw] += p[:th, :tw] * win[:th, :tw]
                weight[yy : yy + th, xx : xx + tw] += win[:th, :tw]
            batch.clear()
            coords.clear()

        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                tile = norm_img[y : y + TILE, x : x + TILE]
                th, tw = tile.shape
                t = standardize(tile)
                if th < TILE or tw < TILE:
                    t = np.pad(t, ((0, TILE - th), (0, TILE - tw)), mode="reflect")
                batch.append(t)
                coords.append((y, x, th, tw))
                if len(batch) == 8:
                    flush()
        flush()
        return prob / np.maximum(weight, 1e-6)

    # ---------------------------------------------------------------- detect
    def detect(self, img: np.ndarray, method: str = "unet", min_confidence: float = 0.6) -> tuple[list[dict], np.ndarray | None]:
        """Return (craters, probability map or None). Crater coordinates are in pixels of ``img``."""
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}")
        if img.ndim != 2 or img.dtype != np.uint8:
            raise ValueError("detect() expects a single-band uint8 image")
        h, w = img.shape
        if min(h, w) < 32:
            raise ValueError("Image is too small (need at least 32x32 pixels)")

        if method == "hough":
            craters = hough_craters(img, r_min=R_MIN, r_max=min(h, w) / 3.0)
            return [c for c in craters if c["confidence"] >= min_confidence * 0.7], None

        all_cands: list[tuple[float, float, float, float]] = []
        meta: dict[tuple[float, float, float], dict] = {}
        prob_full = None
        scale = 1.0
        level = 0
        while True:
            if scale == 1.0:
                im = img
            else:
                im = cv2.resize(img, (max(32, round(w * scale)), max(32, round(h * scale))), interpolation=cv2.INTER_AREA)
            prob = self.predict_prob(im)
            if prob_full is None:
                prob_full = prob
            r_max = min(R_MAX, min(im.shape) / 3.0)
            r_lo = R_MIN if level == 0 else R_MAX / 2.2  # coarse levels only add the big craters
            for c in extract_craters(prob, r_min=r_lo, r_max=r_max, min_support=min_confidence):
                full = (c["x"] / scale, c["y"] / scale, c["r"] / scale)
                all_cands.append((*full, c["confidence"]))
                meta[full] = c
            level += 1
            scale /= 2.0
            if min(h, w) * scale < 128 or level >= 4:
                break
        merged = nms_circles(all_cands)
        craters = []
        for x, y, r, conf in merged:
            c = meta[(x, y, r)]
            craters.append({"x": round(x, 2), "y": round(y, 2), "r": round(r, 2), "confidence": round(conf, 4), "match": c["match"]})
        return craters, prob_full


_default: CraterDetector | None = None


def get_detector() -> CraterDetector:
    global _default
    if _default is None:
        _default = CraterDetector()
    return _default
