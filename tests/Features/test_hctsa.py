import os

import numpy as np
import pytest

from ncpi.Features import Features


HCTSA_FOLDER = "/home/pablomc/hctsa-main"


try:
    import matlab.engine  # noqa: F401
except Exception:
    pytest.skip("MATLAB Engine not available", allow_module_level=True)


if not os.path.isdir(HCTSA_FOLDER):
    pytest.skip("hctsa folder not found", allow_module_level=True)


def test_hctsa_basic_smoke():
    rng = np.random.default_rng(123)
    samples = rng.random((2, 1000))

    feat = Features(method="hctsa", params={"hctsa_folder": HCTSA_FOLDER})
    out = feat.hctsa(samples, hctsa_folder=HCTSA_FOLDER, workers=2)

    assert isinstance(out, list)
    assert len(out) == samples.shape[0]
    assert len(out[0]) > 0
