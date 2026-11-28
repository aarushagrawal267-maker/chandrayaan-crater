"""The browser demo (deploy/browser/web_api.py) must give the same answer as the API for the same request.

In the browser the U-Net batches go to onnxruntime-web; here they go to the PyTorch model, so any difference
would come from the demo's own request handling, which must match app/main.py.
"""
import asyncio
import importlib.util
import json
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from lunarcrater.detect import get_detector
from lunarcrater.imageio import MAX_SIDE
from lunarcrater.synthetic import generate_scene

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def _web_api():
    spec = importlib.util.spec_from_file_location("web_api", ROOT / "deploy" / "browser" / "web_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _torch_infer(batch):
    return get_detector()._infer(batch)


def _without_timing(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "detect_ms"}


@pytest.mark.parametrize("method", ["unet", "hough"])
def test_browser_handler_matches_api_on_a_sample(method):
    web = _web_api()
    sample = json.loads((ROOT / "samples" / "samples.json").read_text())[2]
    api = client.post("/api/detect", data={"sample": sample["file"], "method": method, "meters_per_pixel": "0.25",
                                           "min_confidence": "0.6", "lander_footprint_m": "10"})
    assert api.status_code == 200, api.text
    data = (ROOT / "samples" / sample["file"]).read_bytes()
    mine = asyncio.run(web.run_detection(data, sample["file"], sample["ground_truth"], method, 0.25, 0.6, 10.0,
                                         _torch_infer, max_side=MAX_SIDE))
    assert _without_timing(json.loads(json.dumps(mine))) == _without_timing(api.json())


def test_browser_handler_downscales_large_uploads_and_reports_it():
    web = _web_api()
    ok, buf = cv2.imencode(".png", generate_scene(1200, seed=3).image)
    res = asyncio.run(web.run_detection(buf.tobytes(), "big.png", None, "hough", 5.0, 0.6, 10.0, _torch_infer))
    assert res["downscale_factor"] == round(1024 / 1200, 4)
    assert res["stats"]["image_width_px"] == 1024 and res["effective_meters_per_pixel"] > 5.0


def test_browser_handler_rejects_bad_requests_like_the_api():
    web = _web_api()
    with pytest.raises(web.RequestError) as e:
        asyncio.run(web.run_detection(b"not an image", "x.png", None, "unet", 5.0, 0.6, 10.0, _torch_infer))
    assert e.value.status == 422
    with pytest.raises(web.RequestError) as e:
        asyncio.run(web.run_detection(b"", "x.png", None, "sobel", 5.0, 0.6, 10.0, _torch_infer))
    assert e.value.status == 422 and "method" in e.value.detail
