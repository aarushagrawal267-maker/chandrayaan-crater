import numpy as np
import pytest

from lunarcrater.detect import CraterDetector, ModelUnavailable
from lunarcrater.metrics import match_craters, summarize
from lunarcrater.synthetic import generate_scene
from lunarcrater.train import TEST_SEED0


def test_pipeline_runs_on_odd_sizes(random_detector):
    img = generate_scene(256, seed=5).image
    for shape in [(100, 140), (300, 520)]:  # single-pass and tiled paths
        im = np.ascontiguousarray(np.resize(img, shape))
        prob = random_detector.predict_prob(im)
        assert prob.shape == shape and 0 <= prob.min() and prob.max() <= 1
        craters, _ = random_detector.detect(im)
        assert isinstance(craters, list)


def test_hough_method_needs_no_model():
    det = CraterDetector("does/not/exist.pt")
    assert not det.model_available
    craters, prob = det.detect(generate_scene(128, seed=2).image, method="hough")
    assert prob is None and isinstance(craters, list)
    with pytest.raises(ModelUnavailable):
        det.detect(generate_scene(128, seed=2).image, method="unet")


def test_invalid_inputs(random_detector):
    with pytest.raises(ValueError):
        random_detector.detect(np.zeros((64, 64), np.uint8), method="magic")
    with pytest.raises(ValueError):
        random_detector.detect(np.zeros((16, 16), np.uint8))
    with pytest.raises(ValueError):
        random_detector.detect(np.zeros((64, 64), np.float32))


def test_trained_model_finds_craters(trained_detector):
    """Regression guard for the shipped weights on 8 held-out simulated scenes."""
    res = []
    for i in range(8):
        sc = generate_scene(256, seed=TEST_SEED0 + i)
        pred, _ = trained_detector.detect(sc.image)
        res.append(match_craters(pred, sc.craters))
    s = summarize(res)
    assert s["f1"] > 0.5, s


def test_corrupt_model_file_gives_clear_error(tmp_path):
    bad = tmp_path / "crater_unet.pt"
    bad.write_text("version https://git-lfs.github.com/spec/v1\n")  # e.g. an LFS pointer, not weights
    det = CraterDetector(bad)
    assert det.model_available  # file exists, not yet tried
    err = det.load()
    assert err and "could not be loaded" in err and "hough" in err
    assert not det.model_available and det.load_error == err
    img = generate_scene(128, seed=2).image
    with pytest.raises(ModelUnavailable, match="could not be loaded"):
        det.detect(img, method="unet")
    craters, _ = det.detect(img, method="hough")  # the baseline keeps working
    assert isinstance(craters, list)


def test_load_reports_success(random_model_path):
    det = CraterDetector(random_model_path)
    assert det.load() is None and det.model_available and det.load_error is None
