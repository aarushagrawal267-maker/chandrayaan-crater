import numpy as np

from lunarcrater.analysis import analyse, catalogue, coverage_fraction, landing_safety, size_frequency, to_csv


def test_sfd_slope_recovers_power_law():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=4000)
    d = (1.0 ** -2.0 - u * (1.0 ** -2.0 - 100.0 ** -2.0)) ** (-1 / 2.0)  # cumulative slope -2
    sfd = size_frequency(list(d), area_km2=1.0, d_complete_m=2.0)
    assert -2.3 < sfd["slope"] < -1.7


def test_sfd_too_few():
    assert size_frequency([5.0, 6.0], 1.0, 1.0)["slope"] is None
    assert size_frequency([], 1.0, 1.0)["points"] == []


def test_catalogue_units_and_order():
    cat = catalogue([{"x": 1, "y": 2, "r": 3, "confidence": 0.5}, {"x": 5, "y": 5, "r": 10, "confidence": 0.9}], mpp=2.0)
    assert cat[0]["diameter_m"] == 40.0 and cat[0]["id"] == 1
    assert "diameter_m" in to_csv(cat).splitlines()[0]


def test_coverage():
    assert abs(coverage_fraction([{"x": 50, "y": 50, "r": 20}], (100, 100)) - np.pi * 400 / 10000) < 0.02


def test_landing_sites_avoid_craters():
    img = np.full((200, 200), 120, np.uint8)
    craters = [{"x": 50, "y": 50, "r": 30}, {"x": 150, "y": 60, "r": 25}, {"x": 60, "y": 150, "r": 25}]
    res = landing_safety(img, craters, mpp=1.0, footprint_m=10)
    assert res["sites"], "expected at least one candidate site"
    for s in res["sites"]:
        for c in craters:
            assert np.hypot(s["x_px"] - c["x"], s["y_px"] - c["y"]) > c["r"]
    assert 0 < res["safe_fraction"] < 1 and res["overlay_png"]


def test_analyse_empty():
    out = analyse(np.zeros((64, 64), np.uint8), [], mpp=5.0)
    assert out["stats"]["count"] == 0 and out["stats"]["diameter_median_m"] is None
