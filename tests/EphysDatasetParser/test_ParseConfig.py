from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pytest
from ncpi import tools
from ncpi.EphysDatasetParser import ParseConfig, CanonicalFields

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
# Small helpers (shallow checks)
# -------------------------

def _validate_basic(cfg: ParseConfig) -> None:
    f = cfg.fields
    assert isinstance(f, CanonicalFields)
    assert f.data is not None
    assert isinstance(f.metadata, Mapping)

    if f.table_layout is not None:
        assert f.table_layout in ("wide", "long")

    if f.table_layout == "long":
        assert isinstance(f.long_time_col, str) and f.long_time_col
        assert isinstance(f.long_channel_col, str) and f.long_channel_col
        assert isinstance(f.long_value_col, str) and f.long_value_col

    if cfg.aggregate_over is not None:
        assert isinstance(cfg.aggregate_over, Sequence)
        assert all(isinstance(x, str) and x for x in cfg.aggregate_over)

    assert cfg.aggregate_method in ("sum", "mean", "median")


# -------------------------
# Small helpers (shallow checks)
# -------------------------

def _validate_basic(cfg: ParseConfig) -> None:
    """Lightweight validation that ParseConfig / CanonicalFields are internally consistent."""
    f = cfg.fields
    assert isinstance(f, CanonicalFields)
    assert f.data is not None
    assert isinstance(f.metadata, Mapping)

    if f.table_layout is not None:
        assert f.table_layout in ("wide", "long")

    if f.table_layout == "long":
        assert isinstance(f.long_time_col, str) and f.long_time_col
        assert isinstance(f.long_channel_col, str) and f.long_channel_col
        assert isinstance(f.long_value_col, str) and f.long_value_col

    if cfg.aggregate_over is not None:
        assert isinstance(cfg.aggregate_over, Sequence)
        assert all(isinstance(x, str) and x for x in cfg.aggregate_over)

    assert cfg.aggregate_method in ("sum", "mean", "median")


# -------------------------
# Pandas (wide + long)
# -------------------------

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_pandas_wide_config() -> None:
    df = pd.DataFrame(
        {"time": [0.0, 0.01, 0.02], "Fz": [1.0, 1.1, 1.2], "Cz": [0.9, 0.95, 1.0]}
    )

    cfg = ParseConfig(
        fields=CanonicalFields(
            table_layout="wide",
            time="time",
            fs=250.0,
            channel_columns=["Fz", "Cz"],
            metadata={"subject_id": "S01", "recording_type": "EEG"},
        ),
        aggregate_over=("sensor",),
        aggregate_method="mean",
        aggregate_labels={"sensor": "all"},
    )

    _validate_basic(cfg)
    assert set(cfg.fields.channel_columns or []) == {"Fz", "Cz"}
    assert cfg.fields.time == "time"
    assert cfg.fields.fs == 250.0
    assert "time" in df.columns


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_pandas_long_config() -> None:
    df = pd.DataFrame(
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
            fs=250.0,
            metadata={"subject_id": "S01"},
        )
    )

    _validate_basic(cfg)
    assert cfg.fields.long_time_col in df.columns
    assert cfg.fields.long_channel_col in df.columns
    assert cfg.fields.long_value_col in df.columns


# -------------------------
# NumPy / .npy
# -------------------------

def test_npy_config_and_roundtrip(tmp_path: Path) -> None:
    arr = np.random.randn(100, 3).astype(np.float32)
    fpath = tmp_path / "rec.npy"
    np.save(fpath, arr)

    loaded = np.load(fpath)
    assert loaded.shape == (100, 3)

    cfg = ParseConfig(
        fields=CanonicalFields(
            data="__self__",         # convention: the array itself is the data
            fs=1000.0,               # no metadata in .npy: provide literal
            ch_names=["Fz", "Cz", "Pz"],
            metadata={"subject_id": "S01", "recording_type": "EEG"},
            data_domain="time",
        )
    )

    _validate_basic(cfg)
    assert cfg.fields.fs == 1000.0
    assert isinstance(cfg.fields.ch_names, Sequence)


# -------------------------
# JSON (dict-like)
# -------------------------

def test_json_dict_config(tmp_path: Path) -> None:
    payload = {
        "data": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
        "fs": 250.0,
        "ch_names": ["Fz", "Cz"],
        "meta": {"subject_id": "S01", "condition": "rest"},
    }
    fpath = tmp_path / "rec.json"
    fpath.write_text(json.dumps(payload))

    loaded = json.loads(fpath.read_text())
    assert "data" in loaded

    cfg = ParseConfig(
        fields=CanonicalFields(
            data="data",
            fs="fs",
            ch_names="ch_names",
            metadata={"subject_id": "meta.subject_id", "condition": "meta.condition"},
        )
    )

    _validate_basic(cfg)
    assert cfg.fields.data == "data"
    assert cfg.fields.fs == "fs"


# -------------------------
# MATLAB .mat (scipy.io)
# -------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_mat_config_and_roundtrip(tmp_path: Path) -> None:
    mat = {
        "signal": np.random.randn(200, 2).astype(np.float32),
        "fs": np.array([[500.0]]),  # common MATLAB scalar shape
        "ch_names": np.array(["Fz", "Cz"], dtype=object),
        "meta": {"subject_id": "S01"},
    }
    fpath = tmp_path / "rec.mat"
    sio.savemat(fpath, mat)

    loaded = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    assert "signal" in loaded

    cfg = ParseConfig(
        fields=CanonicalFields(
            data="signal",
            fs="fs",
            ch_names="ch_names",
            metadata={"subject_id": "meta.subject_id"},
        )
    )

    _validate_basic(cfg)


# -------------------------
# MNE (RawArray)
# -------------------------

@pytest.mark.skipif(not HAS_MNE, reason="mne not installed")
def test_mne_raw_config_smoke() -> None:
    sfreq = 250.0
    ch_names = ["Fz", "Cz"]
    data = np.random.randn(2, 1000)  # (n_channels, n_times) for MNE

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info)

    cfg = ParseConfig(
        fields=CanonicalFields(
            # Using lambda keeps this explicit (no reliance on special-casing method names).
            data=lambda obj: obj.get_data(),
            fs="info.sfreq",
            ch_names="ch_names",
            metadata={"recording_type": "EEG", "subject_id": "S01"},
            data_domain="time",
        ),
        drop_bads=True,
        preload=True,
    )

    _validate_basic(cfg)
    assert callable(cfg.fields.data)
    assert cfg.fields.fs == "info.sfreq"
    assert cfg.fields.ch_names == "ch_names"
    assert raw.info["sfreq"] == sfreq


# -------------------------
# Parquet-like tabular config (NO parquet I/O)
# -------------------------
# Rationale:
# - This suite is meant to test ParseConfig/CannonicalFields, not the pandas<->pyarrow stack.
# - Parquet loads into pandas DataFrames; so testing config against a DataFrame is sufficient.

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_parquet_like_tabular_config_smoke() -> None:
    df = pd.DataFrame(
        {"time": [0.0, 0.01], "Fz": [1.0, 1.1], "Cz": [0.9, 0.95], "group": ["A", "A"]}
    )

    cfg = ParseConfig(
        fields=CanonicalFields(
            table_layout="wide",
            time="time",
            fs=250.0,
            channel_columns=["Fz", "Cz"],
            metadata={"group": "group"},
        ),
        aggregate_over=("group",),
        aggregate_method="mean",
        aggregate_labels={"group": "all_groups"},
    )

    _validate_basic(cfg)

    # Shallow but meaningful: config refers to columns that exist
    assert cfg.fields.time in df.columns
    for c in cfg.fields.channel_columns or []:
        assert c in df.columns
    # Metadata locator points to a column (in tabular case)
    assert cfg.fields.metadata["group"] in df.columns
