"""Request handling for the browser demo, run by Pyodide (Python compiled to WebAssembly) in the visitor's browser.

It mirrors ``POST /api/detect`` in ``app/main.py`` and uses the same ``lunarcrater`` code for pre-processing,
ring extraction, merging and analysis. Only the U-Net forward pass differs: the detector's batches of tiles are
sent to ``infer`` (onnxruntime-web running an ONNX export of models/crater_unet.pt) instead of PyTorch.
``tests/test_browser_demo.py`` checks that this module gives the same answer as the API.
"""
from __future__ import annotations

import json
import time

import numpy as np

from lunarcrater.analysis import analyse, png_base64
from lunarcrater.detect import METHODS, CraterDetector
from lunarcrater.imageio import ImageLoadError, decode_image_bytes, limit_size
from lunarcrater.metrics import match_craters, summarize

# The browser runs the network on one CPU thread, so large images are analysed at up to 1024 px
# (the full app allows 2048 px). The page reports the downscale factor and the adjusted resolution.
BROWSER_MAX_SIDE = 1024
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class RequestError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def check_params(method: str, meters_per_pixel: float, min_confidence: float, lander_footprint_m: float) -> None:
    """The same parameter checks as the API."""
    if method not in METHODS:
        raise RequestError(422, f"method must be one of {list(METHODS)}")
    if not (0.01 <= meters_per_pixel <= 10_000):
        raise RequestError(422, "meters_per_pixel must be between 0.01 and 10000")
    if not (0.0 <= min_confidence <= 1.0):
        raise RequestError(422, "min_confidence must be between 0 and 1")
    if not (0.1 <= lander_footprint_m <= 100_000):
        raise RequestError(422, "lander_footprint_m must be between 0.1 and 100000")


async def run_detection(data: bytes, source: str, ground_truth, method: str, meters_per_pixel: float,
                        min_confidence: float, lander_footprint_m: float, infer, max_side: int = BROWSER_MAX_SIDE) -> dict:
    """Same steps and result as ``app.main._run_detection``; ``infer`` is an async batch -> probabilities function."""
    check_params(method, meters_per_pixel, min_confidence, lander_footprint_m)
    if len(data) > MAX_UPLOAD_BYTES:
        raise RequestError(413, "File too large (max 40 MB).")
    try:
        img = decode_image_bytes(data)
    except ImageLoadError as exc:
        raise RequestError(422, str(exc)) from exc
    img, scale = limit_size(img, max_side)
    mpp = meters_per_pixel / scale
    if ground_truth is not None and scale != 1.0:
        ground_truth = [[x * scale, y * scale, r * scale] for x, y, r in ground_truth]

    t0 = time.perf_counter()
    steps = CraterDetector().detect_steps(img, method=method, min_confidence=min_confidence)
    try:
        batch = next(steps)
        while True:
            batch = steps.send(await infer(batch))
    except StopIteration as stop:
        craters, _ = stop.value
    except ValueError as exc:  # e.g. image smaller than 32x32 px
        raise RequestError(422, str(exc)) from exc
    detect_ms = (time.perf_counter() - t0) * 1000

    result = analyse(img, craters, mpp, footprint_m=lander_footprint_m)
    result.update(
        {
            "source": source,
            "method": method,
            "downscale_factor": round(scale, 4),
            "effective_meters_per_pixel": round(mpp, 4),
            "detect_ms": round(detect_ms, 1),
            "image_png": png_base64(img),
        }
    )
    if ground_truth is not None:
        m = summarize([match_craters(craters, ground_truth)])
        result["ground_truth"] = {"count": len(ground_truth), "circles": ground_truth, "scores": m}
    return result


async def detect_json(data, source, ground_truth_json, method, meters_per_pixel, min_confidence, lander_footprint_m) -> str:
    """Entry point called from browser-api.js. Returns JSON text: the result, or {"status", "detail"} on error."""
    import crater_js  # registered by browser-api.js
    from pyodide.ffi import to_js

    async def infer(batch: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        out = await crater_js.infer(to_js(batch.tobytes()), to_js(list(batch.shape)))
        return np.frombuffer(out.to_bytes(), dtype=np.float32).reshape(batch.shape)

    raw = data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)
    gt = json.loads(ground_truth_json) if ground_truth_json else None
    try:
        res = await run_detection(raw, source, gt, method, float(meters_per_pixel), float(min_confidence),
                                  float(lander_footprint_m), infer)
    except RequestError as exc:
        return json.dumps({"status": exc.status, "detail": exc.detail})
    return json.dumps(res)
