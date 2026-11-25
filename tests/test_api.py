import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from lunarcrater.synthetic import generate_scene

client = TestClient(app)


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health_and_index():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "Chandrayaan Crater Lens" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_samples_listed_and_served():
    samples = client.get("/api/samples").json()
    assert len(samples) >= 3
    assert "ground_truth" not in samples[0]
    r = client.get(f"/api/samples/{samples[0]['file']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/samples/../../etc/passwd").status_code == 404


def test_detect_sample_hough():
    name = client.get("/api/samples").json()[-1]["file"]
    r = client.post("/api/detect", data={"sample": name, "method": "hough", "meters_per_pixel": "0.25"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["stats"]["count"] == len(d["catalogue"])
    assert d["ground_truth"]["count"] > 0 and "f1" in d["ground_truth"]["scores"]
    assert d["image_png"] and d["landing"]["overlay_png"]


def test_detect_upload_unet():
    if not client.get("/api/health").json()["model_available"]:
        pytest.skip("no trained model")
    img = generate_scene(256, seed=99).image
    r = client.post("/api/detect", files={"file": ("moon.png", png_bytes(img), "image/png")}, data={"meters_per_pixel": "5"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["method"] == "unet" and d["stats"]["count"] > 0 and "ground_truth" not in d
    assert d["stats"]["meters_per_pixel"] == 5.0


def test_large_upload_is_downscaled():
    img = np.resize(generate_scene(256, seed=1).image, (2200, 600))
    r = client.post("/api/detect", files={"file": ("big.png", png_bytes(np.ascontiguousarray(img)), "image/png")}, data={"meters_per_pixel": "1", "method": "hough"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["downscale_factor"] < 1 and d["effective_meters_per_pixel"] > 1


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"data": {}}, 422),
        ({"data": {"sample": "nope.png"}}, 404),
        ({"files": {"file": ("x.png", b"garbage", "image/png")}}, 422),
        ({"data": {"sample": "x", "method": "magic"}}, 422),
        ({"data": {"sample": "x", "meters_per_pixel": "-1"}}, 422),
    ],
)
def test_detect_errors(kwargs, code):
    assert client.post("/api/detect", **kwargs).status_code == code


def test_model_endpoint():
    d = client.get("/api/model").json()
    assert "model_available" in d and "evaluation" in d


def test_model_loaded_at_startup():
    import app.main as main_mod

    with TestClient(app) as c:  # runs the lifespan start-up hook
        h = c.get("/api/health").json()
        if main_mod.get_detector().model_path.exists():
            assert main_mod.get_detector()._model is not None
            assert h["model_available"] and h["model_error"] is None


def test_broken_model_reported_not_500(monkeypatch, tmp_path):
    import app.main as main_mod
    from lunarcrater.detect import CraterDetector

    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"\x00not a checkpoint")
    det = CraterDetector(bad)
    monkeypatch.setattr(main_mod, "get_detector", lambda: det)
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["model_available"] is False and "could not be loaded" in h["model_error"]
        name = c.get("/api/samples").json()[-1]["file"]
        r = c.post("/api/detect", data={"sample": name, "method": "unet"})
        assert r.status_code == 503 and "could not be loaded" in r.json()["detail"]
        assert c.post("/api/detect", data={"sample": name, "method": "hough"}).status_code == 200


def test_detections_run_one_at_a_time(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import app.main as main_mod

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake_run(*_args):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.2)
        with lock:
            state["now"] -= 1
        return {"ok": True}

    monkeypatch.setattr(main_mod, "_run_detection", fake_run)
    with TestClient(app) as c:
        name = c.get("/api/samples").json()[-1]["file"]
        with ThreadPoolExecutor(3) as pool:
            codes = list(pool.map(lambda _: c.post("/api/detect", data={"sample": name, "method": "hough"}).status_code, range(3)))
        assert c.get("/api/health").status_code == 200
    assert codes == [200, 200, 200]
    assert state["peak"] == main_mod.MAX_PARALLEL_DETECTIONS
