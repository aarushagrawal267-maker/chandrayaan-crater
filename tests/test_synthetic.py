import numpy as np

from lunarcrater.synthetic import Crater, generate_scene, rim_mask_for, sample_radii


def test_scene_shapes_and_types():
    sc = generate_scene(128, seed=1)
    assert sc.image.shape == (128, 128) and sc.image.dtype == np.uint8
    assert sc.rim_mask.shape == (128, 128) and set(np.unique(sc.rim_mask)) <= {0, 1}
    assert sc.rim_mask.sum() > 0
    assert 5 <= sc.sun_elevation <= 45


def test_scene_is_deterministic():
    a, b = generate_scene(96, seed=7), generate_scene(96, seed=7)
    assert np.array_equal(a.image, b.image)
    assert [c.as_tuple() for c in a.craters] == [c.as_tuple() for c in b.craters]


def test_labelled_craters_inside_frame_and_above_min_radius():
    sc = generate_scene(160, seed=3, r_min=3.0)
    assert sc.craters, "expected some labelled craters"
    for c in sc.craters:
        assert 0 <= c.x < 160 and 0 <= c.y < 160 and c.r >= 3.0


def test_radii_follow_bounds():
    r = sample_radii(np.random.default_rng(0), 5000, 3, 30, slope=2.0)
    assert r.min() >= 3 and r.max() <= 30
    # power law: many more small than large craters
    assert (r < 6).sum() > 5 * (r > 15).sum()


def test_rim_mask_draws_ring():
    m = rim_mask_for([Crater(50, 50, 20)], (100, 100))
    assert m[50, 50] == 0 and m[50, 70] == 1
