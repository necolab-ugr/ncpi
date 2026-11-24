"""Tests for Analysis.cohend (no R backend required)."""
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ncpi import Analysis


def _frame(groups: dict, sensor="s0") -> pd.DataFrame:
    """Long-format DataFrame from {group: values} for one sensor."""
    rows = [(g, sensor, float(v)) for g, values in groups.items() for v in values]
    return pd.DataFrame(rows, columns=["gr", "sensor", "Y"])


def _cohend(df: pd.DataFrame, **kwargs) -> dict:
    opts: dict[str, Any] = dict(control_group="HC", group_col="gr", sensor_col="sensor", data_col="Y", min_n=2, drop_zeros=False)
    opts.update(kwargs)
    return Analysis(df).cohend(**opts)


def _toy_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {"HC": 0.0, "g1": 4.0, "g2": 2.5}
    rows = [
        (g, sensor, shift + rng.normal())
        for g, shift in groups.items()
        for sensor in range(3)
        for _ in range(10)
    ]
    return pd.DataFrame(rows, columns=["gr", "sensor", "Y"])


def test_cohend_outputs_expected_keys():
    """cohend should compute per-sensor d for each non-control group vs control."""
    toy_df = _toy_df()
    res = _cohend(toy_df)
    assert isinstance(res, dict)
    assert set(res.keys()) == {"g1vsHC", "g2vsHC"}
    for df in res.values():
        assert isinstance(df, pd.DataFrame)
        assert "sensor" in df.columns
        assert "d" in df.columns
        # should have one row per sensor
        assert df["sensor"].nunique() == toy_df["sensor"].nunique()


def test_cohend_known_answer_positive():
    # Means 5 vs 3, both sd = sqrt(2.5) -> d = 2 / sqrt(2.5)
    df = _frame({"HC": [1, 2, 3, 4, 5], "g": [3, 4, 5, 6, 7]})
    d = _cohend(df)["gvsHC"]["d"].iloc[0]
    assert d == pytest.approx(2.0 / np.sqrt(2.5))
    assert d == pytest.approx(1.264911, abs=1e-6)
    assert d > 0


def test_cohend_known_answer_negative_sign():
    # Group mean below control -> negative d of the same magnitude
    df = _frame({"HC": [1, 2, 3, 4, 5], "g": [-1, 0, 1, 2, 3]})
    d = _cohend(df)["gvsHC"]["d"].iloc[0]
    assert d == pytest.approx(-2.0 / np.sqrt(2.5))
    assert d < 0


def test_cohend_known_answer_unequal_sizes_uses_pooled_sd():
    # var(HC)=2.5 (n=5), var(g)=4 (n=3) -> pooled sd = sqrt((4*2.5 + 2*4) / 6) = sqrt(3)
    df = _frame({"HC": [1, 2, 3, 4, 5], "g": [2, 4, 6]})
    d = _cohend(df)["gvsHC"]["d"].iloc[0]
    assert d == pytest.approx(1.0 / np.sqrt(3.0))


def test_cohend_is_computed_per_sensor():
    df = pd.concat(
        [
            _frame({"HC": [1, 2, 3, 4, 5], "g": [3, 4, 5, 6, 7]}, sensor="a"),
            _frame({"HC": [1, 2, 3, 4, 5], "g": [-1, 0, 1, 2, 3]}, sensor="b"),
        ],
        ignore_index=True,
    )
    res = _cohend(df)["gvsHC"].set_index("sensor")["d"]
    assert res["a"] == pytest.approx(2.0 / np.sqrt(2.5))
    assert res["b"] == pytest.approx(-2.0 / np.sqrt(2.5))


def test_cohend_returns_nan_below_min_n():
    df = _frame({"HC": [1, 2], "g": [3, 4, 5]})
    d = _cohend(df, min_n=3)["gvsHC"]["d"].iloc[0]
    assert np.isnan(d)


def test_cohend_returns_nan_for_zero_pooled_variance():
    df = _frame({"HC": [2, 2, 2], "g": [2, 2, 2]})
    d = _cohend(df)["gvsHC"]["d"].iloc[0]
    assert np.isnan(d)


def test_cohend_drop_zeros_excludes_zero_values():
    df = _frame({"HC": [1, 2, 3, 4, 5], "g": [0, 3, 4, 5, 6, 7]})
    d_drop = _cohend(df, drop_zeros=True)["gvsHC"]["d"].iloc[0]
    d_keep = _cohend(df, drop_zeros=False)["gvsHC"]["d"].iloc[0]
    assert d_drop == pytest.approx(2.0 / np.sqrt(2.5))
    assert d_keep != pytest.approx(d_drop)
