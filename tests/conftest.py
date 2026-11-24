import numpy as np
import pytest

from lunarcrater.detect import CraterDetector, DEFAULT_MODEL
from lunarcrater.model import CraterUNet, save_model


@pytest.fixture(scope="session")
def random_model_path(tmp_path_factory):
    """An untrained U-Net checkpoint: exercises the inference pipeline without the real weights."""
    import torch

    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("model") / "random.pt"
    save_model(CraterUNet(base=8), p)
    return p


@pytest.fixture(scope="session")
def random_detector(random_model_path):
    return CraterDetector(random_model_path)


@pytest.fixture(scope="session")
def trained_detector():
    if not DEFAULT_MODEL.exists():
        pytest.skip("trained model models/crater_unet.pt not present")
    return CraterDetector(DEFAULT_MODEL)


def ring_prob(shape, circles, thickness=2):
    import cv2

    prob = np.zeros(shape, np.float32)
    for x, y, r in circles:
        cv2.circle(prob, (int(x), int(y)), int(r), 1.0, thickness)
    return prob
