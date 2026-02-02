from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from ncpi import tools
from ncpi.EphysDatasetParser import (
    DEFAULT_COLUMNS,
    CanonicalFields,
    EphysDatasetParser,
    ParseConfig,
)

# -------------------------
# Optional deps (checked once, imported at top)
# -------------------------

HAS_PANDAS = tools.ensure_module("pandas")
if HAS_PANDAS:
    import pandas as pd  # type: ignore[import-not-found]

HAS_SCIPY = tools.ensure_module("scipy")
if HAS_SCIPY:
    import scipy.io as sio  # type: ignore[import-not-found]

HAS_MNE = tools.ensure_module("mne")
if HAS_MNE:
    import mne  # type: ignore[import-not-found]


# -------------------------
# Helpers
# -------------------------

def _assert_df_contract(df: Any) -> None:
    """Basic schema expectations for EphysDatasetParser.parse() output."""
    # pandas is required for parse() output
    assert HAS_PANDAS, "pandas must be installed for parse() to return a DataFrame"

    assert isinstance(df, pd.DataFrame)

    # Required columns exist
    for c in DEFAULT_COLUMNS:
        assert c in df.columns

    # `data` column stores numpy arrays (1D)
    assert "data" in df.columns
    assert len(df) >= 1
    x0 = df.iloc[0]["data"]
    assert isinstance(x0, np.ndarray)
    assert x0.ndim == 1


def _assert_row_meta(df: Any, expected: Mapping[str, Any]) -> None:
    """Ensure all expected metadata keys exist and match on every row when provided."""
    for k, v in expected.items():
        assert k in df.columns
        if v is not None:
            # Use pandas-native reduction
            col = df[k]
            arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            assert np.all(arr == v)


# -------------------------
# Dict-like parsing
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_dict_like_time_by_channels() -> None:
    # data shape: (time, channels)
    payload = {
        "data": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        "fs": 2.0,
        "ch_names": ["Fz", "Cz"],
        "meta": {"subject_id": "S01", "recording_type": "EEG", "condition": "rest"},
    }

    cfg = ParseConfig(
        fields=CanonicalFields(
            data="data",
            fs="fs",
            ch_names="ch_names",
            metadata={
                "subject_id": "meta.subject_id",
                "recording_type": "meta.recording_type",
                "condition": "meta.condition",
            },
        )
    )
    parser = EphysDatasetParser(cfg)
    df = parser.parse(payload)

    _assert_df_contract(df)
    assert len(df) == 2  # 2 channels
    assert set(df["sensor"].tolist()) == {"Fz", "Cz"}

    # time bounds inferred from fs and length
    # n=4, fs=2 -> t1=(n-1)/fs = 1.5
    assert all(df["t0"] == 0.0)
    assert all(np.isclose(df["t1"].astype(float), 1.5))

    _assert_row_meta(df, {"subject_id": "S01", "recording_type": "EEG", "condition": "rest"})


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_dict_like_channels_by_time_orientation() -> None:
    # data shape: (channels, time)
    payload = {
        "signal": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        "fs": 1.0,
        "ch_names": ["Fz", "Cz"],
    }

    cfg = ParseConfig(fields=CanonicalFields(data="signal", fs="fs", ch_names="ch_names"))
    df = EphysDatasetParser(cfg).parse(payload)

    _assert_df_contract(df)
    assert len(df) == 2
    # ensure values match expected orientation
    fz = df.loc[df["sensor"] == "Fz", "data"].iloc[0]
    cz = df.loc[df["sensor"] == "Cz", "data"].iloc[0]
    assert np.allclose(fz, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(cz, np.array([10.0, 20.0, 30.0]))


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_dict_like_3d_epochs() -> None:
    # (epochs, time, channels)
    arr = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    payload = {"data": arr, "fs": 10.0, "ch_names": ["A", "B", "C"]}

    cfg = ParseConfig(fields=CanonicalFields(data="data", fs="fs", ch_names="ch_names"))
    df = EphysDatasetParser(cfg).parse(payload)

    _assert_df_contract(df)
    assert len(df) == 2 * 3  # epochs * channels
    assert set(df["epoch"].unique().tolist()) == {0, 1}
    assert set(df["sensor"].unique().tolist()) == {"A", "B", "C"}


# -------------------------
# ndarray parsing
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_ndarray_2d_time_by_channels_default() -> None:
    # For ndarray 2D, parser assumes (time, channels)
    a = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    cfg = ParseConfig(fields=CanonicalFields(data="__self__", fs=100.0, ch_names=["Fz", "Cz"]))
    df = EphysDatasetParser(cfg).parse(a)

    _assert_df_contract(df)
    assert len(df) == 2
    assert set(df["sensor"].tolist()) == {"Fz", "Cz"}

    fz = df.loc[df["sensor"] == "Fz", "data"].iloc[0]
    cz = df.loc[df["sensor"] == "Cz", "data"].iloc[0]
    assert np.allclose(fz, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(cz, np.array([10.0, 20.0, 30.0]))


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_ndarray_3d_epochs_default() -> None:
    # For ndarray 3D, parser assumes (epochs, time, channels)
    a = np.zeros((2, 4, 3), dtype=float)
    a[1, :, 2] = 1.0

    cfg = ParseConfig(fields=CanonicalFields(data="__self__", fs=50.0, ch_names=["X", "Y", "Z"]))
    df = EphysDatasetParser(cfg).parse(a)

    _assert_df_contract(df)
    assert len(df) == 2 * 3
    assert set(df["epoch"].unique().tolist()) == {0, 1}

    # epoch=1 sensor=Z is ones
    z = df[(df["epoch"] == 1) & (df["sensor"] == "Z")]["data"].iloc[0]
    assert np.allclose(z, np.ones(4))


# -------------------------
# pandas parsing (wide + long)
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_pandas_wide_infer_channels() -> None:
    df_in = pd.DataFrame(
        {
            "time": [0.0, 0.01, 0.02],
            "Fz": [1.0, 1.1, 1.2],
            "Cz": [0.9, 0.95, 1.0],
            "group": ["A", "A", "A"],
        }
    )

    cfg = ParseConfig(
        fields=CanonicalFields(
            table_layout="wide",
            time="time",
            fs=100.0,
            # no channel_columns -> infer numeric columns except time + explicit metadata columns
            metadata={"group": "group", "subject_id": "S01"},
        )
    )

    out = EphysDatasetParser(cfg).parse(df_in)
    _assert_df_contract(out)

    assert set(out["sensor"].tolist()) == {"Fz", "Cz"}
    _assert_row_meta(out, {"group": "A", "subject_id": "S01"})


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_pandas_long() -> None:
    df_in = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.01, 0.01],
            "channel": ["Fz", "Cz", "Fz", "Cz"],
            "value": [1.0, 0.9, 1.1, 0.95],
        }
    )

    cfg = ParseConfig(
        fields=CanonicalFields(
            table_layout="long",
            long_time_col="time",
            long_channel_col="channel",
            long_value_col="value",
            fs=100.0,
            metadata={"subject_id": "S01"},
        )
    )

    out = EphysDatasetParser(cfg).parse(df_in)
    _assert_df_contract(out)

    assert set(out["sensor"].tolist()) == {"Fz", "Cz"}
    # each channel should have length 2 after grouping
    assert all(out["data"].apply(lambda x: np.asarray(x).shape == (2,)))


# -------------------------
# File loading (.npy / .json / .mat)
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_from_npy_path(tmp_path: Path) -> None:
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    fpath = tmp_path / "rec.npy"
    np.save(fpath, arr)

    cfg = ParseConfig(fields=CanonicalFields(data="__self__", fs=10.0, ch_names=["a", "b", "c"]))
    out = EphysDatasetParser(cfg).parse(fpath)

    _assert_df_contract(out)
    assert len(out) == 3


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parse_from_json_path(tmp_path: Path) -> None:
    payload = {"data": [[1, 2], [3, 4], [5, 6]], "fs": 2.0, "ch_names": ["Fz", "Cz"]}
    fpath = tmp_path / "rec.json"
    fpath.write_text(json.dumps(payload), encoding="utf-8")

    cfg = ParseConfig(fields=CanonicalFields(data="data", fs="fs", ch_names="ch_names"))
    out = EphysDatasetParser(cfg).parse(fpath)

    _assert_df_contract(out)
    assert len(out) == 2


@pytest.mark.skipif(not HAS_PANDAS or not HAS_SCIPY, reason="requires pandas+scipy")
def test_parse_from_mat_path(tmp_path: Path) -> None:
    mat = {
        "signal": np.random.randn(10, 2).astype(np.float32),
        "fs": np.array([[500.0]]),
        "ch_names": np.array(["Fz", "Cz"], dtype=object),
    }
    fpath = tmp_path / "rec.mat"
    sio.savemat(fpath, mat)

    cfg = ParseConfig(fields=CanonicalFields(data="signal", fs="fs", ch_names="ch_names"))
    out = EphysDatasetParser(cfg).parse(fpath)

    _assert_df_contract(out)
    assert len(out) == 2


# -------------------------
# Post-processing: epoching, zscore, aggregation
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_epoching_splits_rows() -> None:
    # 1 channel, length 10, fs=10Hz; epoch_length=0.2s -> win=2 samples
    payload = {"data": np.arange(10, dtype=float), "fs": 10.0}

    cfg = ParseConfig(
        fields=CanonicalFields(data="data", fs="fs"),
        epoch_length_s=0.2,
        epoch_step_s=0.2,
    )

    out = EphysDatasetParser(cfg).parse(payload)
    _assert_df_contract(out)

    # win=2, step=2, starts: 0,2,4,6,8 => 5 epochs
    assert len(out) == 5
    assert set(out["epoch"].tolist()) == {0, 1, 2, 3, 4}
    assert all(out["data"].apply(lambda x: len(x) == 2))


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_zscore_rowwise_constant_vector() -> None:
    payload = {"data": np.ones((5, 2), dtype=float), "fs": 1.0, "ch_names": ["A", "B"]}

    cfg = ParseConfig(fields=CanonicalFields(data="data", fs="fs", ch_names="ch_names"), zscore=True)
    out = EphysDatasetParser(cfg).parse(payload)

    _assert_df_contract(out)
    # constant -> zeros
    assert all(out["data"].apply(lambda x: np.allclose(x, 0.0)))


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_aggregation_over_sensor_mean() -> None:
    payload = {
        "data": np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ),
        "fs": 1.0,
        "ch_names": ["Fz", "Cz"],
        "meta": {"subject_id": "S01"},
    }

    cfg = ParseConfig(
        fields=CanonicalFields(
            data="data",
            fs="fs",
            ch_names="ch_names",
            metadata={"subject_id": "meta.subject_id"},
        ),
        aggregate_over=("sensor",),
        aggregate_method="mean",
        aggregate_labels={"sensor": "all"},
    )

    out = EphysDatasetParser(cfg).parse(payload)
    _assert_df_contract(out)
    assert len(out) == 1
    assert out.iloc[0]["sensor"] == "all"

    # mean([ [1,2,3], [10,20,30] ]) = [5.5, 11.0, 16.5]
    x = out.iloc[0]["data"]
    assert np.allclose(x, np.array([5.5, 11.0, 16.5]))


# -------------------------
# MNE smoke (if installed)
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS or not HAS_MNE, reason="requires pandas+mne")
def test_parse_mne_rawarray_smoke() -> None:
    sfreq = 250.0
    ch_names = ["Fz", "Cz"]
    data = np.random.randn(2, 100)  # (n_channels, n_times)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info)

    cfg = ParseConfig(
        fields=CanonicalFields(
            data=lambda obj: obj.get_data(),
            fs="info.sfreq",
            ch_names="ch_names",
            metadata={"subject_id": "S01", "recording_type": "EEG"},
        )
    )

    out = EphysDatasetParser(cfg).parse(raw)
    _assert_df_contract(out)
    assert len(out) == 2
    _assert_row_meta(out, {"subject_id": "S01", "recording_type": "EEG"})