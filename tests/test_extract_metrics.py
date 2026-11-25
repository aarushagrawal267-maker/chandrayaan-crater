from lunarcrater.extract import extract_craters, nms_circles, radius_grid
from lunarcrater.metrics import match_craters, pixel_scores, summarize

from .conftest import ring_prob

CIRCLES = [(40, 40, 10), (120, 60, 20), (70, 130, 6), (150, 150, 14)]


def test_extract_recovers_perfect_rings():
    prob = ring_prob((200, 200), CIRCLES)
    found = extract_craters(prob, r_min=3, r_max=30)
    res = match_craters(found, CIRCLES)
    assert res["tp"] == len(CIRCLES)
    assert res["fp"] <= 1
    assert res["frac_err_r"] < 0.15


def test_extract_empty_map():
    import numpy as np

    assert extract_craters(np.zeros((64, 64), np.float32)) == []


def test_nms_removes_duplicates():
    kept = nms_circles([(10, 10, 5, 0.9), (10.5, 10, 5.2, 0.8), (40, 40, 5, 0.7)])
    assert len(kept) == 2 and kept[0][3] == 0.9


def test_radius_grid_monotonic():
    g = radius_grid(3, 40)
    assert g[0] == 3 and g[-1] <= 40 and all(b > a for a, b in zip(g, g[1:]))


def test_match_rules():
    gt = [(50, 50, 10)]
    assert match_craters([(52, 50, 11)], gt)["tp"] == 1
    assert match_craters([(60, 50, 10)], gt)["tp"] == 0  # centre off by 1.0 r
    assert match_craters([(50, 50, 16)], gt)["tp"] == 0  # radius off by 60%
    r = match_craters([(50, 50, 10), (50, 50, 10.5)], gt)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)  # one-to-one


def test_summarize():
    s = summarize([{"tp": 8, "fp": 2, "fn": 2, "frac_err_xy": 0.1, "frac_err_r": 0.05}])
    assert s["precision"] == 0.8 and s["recall"] == 0.8 and s["f1"] == 0.8


def test_pixel_scores():
    import numpy as np

    m = np.zeros((10, 10), np.uint8)
    m[2:5, 2:5] = 1
    assert pixel_scores(m.astype(float), m)["iou"] == 1.0


def test_summarize_pools_errors_over_matched_craters():
    # 1 match with 40 % error and 9 matches with 10 %: the crater-level mean is 13 %, not (40 + 10) / 2.
    s = summarize(
        [
            {"tp": 1, "fp": 0, "fn": 0, "frac_err_xy": 0.4, "frac_err_r": 0.2},
            {"tp": 9, "fp": 0, "fn": 0, "frac_err_xy": 0.1, "frac_err_r": 0.0},
            {"tp": 0, "fp": 3, "fn": 1, "frac_err_xy": None, "frac_err_r": None},
        ]
    )
    assert s["mean_frac_err_xy"] == 0.13 and s["mean_frac_err_r"] == 0.02


def test_evaluate_val_split_and_confidence_intervals():
    from lunarcrater.evaluate import bootstrap_ci, evaluate
    from lunarcrater.train import VAL_SEED0

    rep = evaluate(n_images=3, methods=("hough",), split="val", log=lambda *_: None)
    assert rep["test_set"]["split"] == "val" and rep["test_set"]["seed_range"] == [VAL_SEED0, VAL_SEED0 + 2]
    m = rep["methods"]["hough"]
    for k in ("precision", "recall", "f1"):
        lo, hi = m["ci95"][k]
        assert 0.0 <= lo <= hi <= 1.0
    ci = bootstrap_ci([{"tp": 5, "fp": 0, "fn": 0}] * 4)
    assert ci["precision"] == [1.0, 1.0] and ci["f1"] == [1.0, 1.0]
