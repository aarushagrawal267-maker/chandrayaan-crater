"""FastAPI backend + static web UI for Chandrayaan Crater Lens.

Run:  uvicorn app.main:app --port 8000   then open http://localhost:8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lunarcrater import __version__
from lunarcrater.analysis import analyse, png_base64
from lunarcrater.detect import METHODS, ROOT, ModelUnavailable, get_detector
from lunarcrater.imageio import ImageLoadError, decode_image_bytes, limit_size, load_image

STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = ROOT / "samples"
MODELS = ROOT / "models"
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

app = FastAPI(title="Chandrayaan Crater Lens", version=__version__, description="Lunar crater detection and analysis API")


def _sample_index() -> list[dict]:
    f = SAMPLES / "samples.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())


@app.get("/api/health")
def health():
    det = get_detector()
    return {"status": "ok", "version": __version__, "model_available": det.model_available, "methods": list(METHODS)}


@app.get("/api/samples")
def samples():
    return [
        {k: v for k, v in s.items() if k != "ground_truth"} | {"ground_truth_count": len(s.get("ground_truth", []))}
        for s in _sample_index()
    ]


@app.get("/api/samples/{name}")
def sample_file(name: str):
    names = {s["file"] for s in _sample_index()}
    if name not in names:
        raise HTTPException(404, "Unknown sample")
    return FileResponse(SAMPLES / name, media_type="image/png")


@app.get("/api/model")
def model_info():
    out: dict = {"model_available": get_detector().model_available}
    for fname, key in (("metrics.json", "evaluation"), ("training_history.json", "training_history")):
        f = MODELS / fname
        out[key] = json.loads(f.read_text()) if f.exists() else None
    return out


@app.post("/api/detect")
async def detect(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    method: str = Form("unet"),
    meters_per_pixel: float = Form(5.0),
    min_confidence: float = Form(0.6),
    lander_footprint_m: float = Form(10.0),
):
    if method not in METHODS:
        raise HTTPException(422, f"method must be one of {list(METHODS)}")
    if not (0.01 <= meters_per_pixel <= 10_000):
        raise HTTPException(422, "meters_per_pixel must be between 0.01 and 10000")
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(422, "min_confidence must be between 0 and 1")
    if not (0.1 <= lander_footprint_m <= 100_000):
        raise HTTPException(422, "lander_footprint_m must be between 0.1 and 100000")

    ground_truth = None
    try:
        if file is not None and file.filename:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File too large (max 40 MB). Crop or downsample it first, e.g. with scripts/prepare_chandrayaan.py")
            img = decode_image_bytes(data)
            source = file.filename
        elif sample:
            idx = {s["file"]: s for s in _sample_index()}
            if sample not in idx:
                raise HTTPException(404, "Unknown sample")
            img = load_image(SAMPLES / sample)
            source = sample
            ground_truth = idx[sample].get("ground_truth")
        else:
            raise HTTPException(422, "Upload an image or choose a sample")
    except ImageLoadError as exc:
        raise HTTPException(422, str(exc)) from exc

    img, scale = limit_size(img)
    mpp = meters_per_pixel / scale
    if ground_truth is not None and scale != 1.0:
        ground_truth = [[x * scale, y * scale, r * scale] for x, y, r in ground_truth]

    t0 = time.perf_counter()
    try:
        craters, _ = get_detector().detect(img, method=method, min_confidence=min_confidence)
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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
        from lunarcrater.metrics import match_craters, summarize

        m = summarize([match_craters(craters, ground_truth)])
        result["ground_truth"] = {"count": len(ground_truth), "circles": ground_truth, "scores": m}
    return JSONResponse(result)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
