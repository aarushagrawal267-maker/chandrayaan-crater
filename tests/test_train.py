import json

from lunarcrater.model import load_model
from lunarcrater.train import train


def test_tiny_training_and_finetune(tmp_path):
    out = tmp_path / "m.pt"
    res = train(epochs=1, train_size=8, val_size=4, batch_size=4, base=4, size=64, out=out, workers=0, log=lambda *_: None)
    assert out.exists() and len(res["history"]) == 1
    hist = json.loads((tmp_path / "training_history.json").read_text())
    assert hist[0]["epoch"] == 1 and hist[0]["train_loss"] > 0

    out2 = tmp_path / "ft.pt"
    res2 = train(epochs=1, train_size=8, val_size=4, batch_size=4, size=64, out=out2, workers=0, init_from=out, schedule="cosine", lr=1e-4, log=lambda *_: None)
    assert [h["epoch"] for h in res2["history"]] == [1, 2]
    model, extra = load_model(out2)
    assert model.base == 4 and len(extra["history"]) == 2


def test_train_val_test_and_demo_seeds_are_disjoint():
    """No scene seed may be shared between training, validation, test and the demo samples."""
    from lunarcrater.cli import DEMO_SCENES
    from lunarcrater.train import TEST_SEED0, TRAIN_SEED0, VAL_SEED0

    train_seeds = range(TRAIN_SEED0, TRAIN_SEED0 + 1_000_000)  # far above any realistic --train-size
    val_seeds = range(VAL_SEED0, VAL_SEED0 + 1_000_000)
    test_seeds = range(TEST_SEED0, TEST_SEED0 + 1_000_000)
    ranges = [train_seeds, val_seeds, test_seeds]
    for i, a in enumerate(ranges):
        for b in ranges[i + 1 :]:
            assert a.stop <= b.start or b.stop <= a.start
    for _, seed, *_ in DEMO_SCENES:
        assert not any(seed in r for r in ranges)
