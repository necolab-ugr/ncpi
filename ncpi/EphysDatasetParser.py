from __future__ import annotations
from dataclasses import dataclass, field
import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union, Literal, Tuple
from collections.abc import Mapping as MappingABC
import numpy as np
from ncpi import tools
import scipy.io as sio 
import h5py  
from functools import lru_cache


# ---- Type aliases ---------------------------------------------------------
# Type of neural recording modality
RecordingType = Literal["EEG", "MEG", "ECoG", "LFP", "Unknown"]

# Representation/state of the data being parsed
# - raw: continuous unsegmented data (e.g., raw MNE objects, time series or power spectra)
# - epochs: segmented trials or events
DataKind = Literal["raw", "epochs"]

# Method used when aggregating data across dimensions
# (e.g., across trials, channels, or time windows)
AggregateMethod = Literal["sum", "mean", "median"]

# Domain/representation of the stored signal in `data`.
# - time: samples over time (typical raw/epochs time series)
# - frequency: spectral representation (e.g., power/PSD over frequency)
DataDomain = Literal["time", "frequency"]

# If `data_domain` is frequency, you may also provide explicit frequency axis info.
# This is intentionally flexible because different sources may store spectra differently
# (e.g., linear power, log-power, complex FFT, etc.).
SpectralKind = Literal["psd", "amplitude", "fft", "unknown"]


# ---- Default column schema -----------------------------------------------
# Canonical column names expected in the parsed output.
# This acts as a contract between the parser and downstream consumers
# (e.g., analysis code, DataFrames, or storage layers).
DEFAULT_COLUMNS = [
    "subject_id",      # Unique identifier for the subject/animal
    "species",         # Species of the subject (e.g., human, mouse)
    "group",           # Experimental or cohort grouping (e.g., control, treatment)
    "condition",       # Experimental condition (e.g., rest, task)
    "epoch",           # Epoch or trial identifier
    "sensor",          # Sensor / channel name or index
    "recording_type",  # Modality of recording (see RecordingType)
    "fs",              # Sampling frequency (Hz)
    "data",            # The actual signal data (array-like or serialized)
    "t0",              # Start time of the data segment
    "t1",              # End time of the data segment

    # Optional frequency-domain metadata
    "data_domain",     # "time" for time-series; "frequency" for spectral data
    "f0",              # Start frequency (Hz) if data_domain == "frequency"
    "f1",              # End frequency (Hz) if data_domain == "frequency"
    "spectral_kind",   # Kind of spectral data (psd/amplitude/fft/unknown) if data_domain == "frequency"

    "source_file",     # Originating file path or identifier
]


def _require_mne(context: str = "") -> None:
    if not tools.ensure_module("mne"):
        msg = "mne is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)

@lru_cache(maxsize=None)
def _require_scipy(context: str = "") -> None:
    if not tools.ensure_module("scipy"):
        msg = "scipy is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)

@lru_cache(maxsize=None)
def _require_h5py(context: str = "") -> None:
    if not tools.ensure_module("h5py"):
        msg = "h5py is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


def _require_pyedflib(context: str = "") -> None:
    if not tools.ensure_module("pyedflib"):
        msg = "pyEDFlib is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


@lru_cache(maxsize=None)
def _require_pynwb(context: str = "") -> None:
    if not tools.ensure_module("pynwb"):
        msg = "pynwb is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


class _HDF5MatLazyMapping(MappingABC):
    __lazy_hdf5__ = True

    def __init__(self, file_path: str, group_path: str = "/") -> None:
        self.file_path = str(file_path)
        self.group_path = str(group_path or "/")

    def _keys(self) -> list[str]:
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            return [str(k) for k in grp.keys() if not str(k).startswith("#")]

    def __iter__(self):
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def __getitem__(self, key: str) -> Any:
        key_str = str(key)
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            if key_str not in grp:
                raise KeyError(key_str)
            return _hdf5_mat_to_python_node(grp[key_str], h5f, self.file_path)

    def __contains__(self, key: object) -> bool:
        key_str = str(key)
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            return key_str in grp

    def keys(self):
        return self._keys()

    def items(self):
        for k in self._keys():
            yield k, self[k]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except Exception:
            return default


def _hdf5_mat_to_python_value(value: Any, h5file: Any, file_path: Optional[str] = None) -> Any:
    if isinstance(value, h5py.Reference):
        if not value:
            return None
        return _hdf5_mat_to_python_node(h5file[value], h5file, file_path)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            out = np.empty(value.shape, dtype=object)
            for idx, item in np.ndenumerate(value):
                out[idx] = _hdf5_mat_to_python_value(item, h5file, file_path)
            if out.ndim == 0:
                try:
                    return out.item()
                except Exception:
                    return out
            return out
        if value.ndim == 0:
            try:
                return value.item()
            except Exception:
                return value
        return value

    return value


def _hdf5_mat_to_python_node(node: Any, h5file: Any, file_path: Optional[str] = None) -> Any:

    if isinstance(node, h5py.Group):
        if file_path:
            return _HDF5MatLazyMapping(file_path, node.name)
        out: Dict[str, Any] = {}
        for key in node.keys():
            if str(key).startswith("#"):
                continue
            out[str(key)] = _hdf5_mat_to_python_node(node[key], h5file, file_path)
        return out

    if isinstance(node, h5py.Dataset):
        return _hdf5_mat_to_python_value(node[()], h5file, file_path)

    return node


def _load_mat_with_fallback(path: Path) -> Any:
    # Fast file version detection
    if h5py.is_hdf5(str(path)):
        _require_h5py("MATLAB v7.3 .mat loading")
        try:
            return _HDF5MatLazyMapping(str(path), "/")
        except Exception as exc:
            raise ValueError(f"Failed to load v7.3 MATLAB file '{path}' with h5py: {exc}")
    else:
        _require_scipy("MATLAB .mat loading")
        try:
            return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            raise ValueError(f"Failed to load legacy MATLAB file '{path}' with scipy: {exc}")


def _load_edf_with_pyedflib(path: Path) -> Dict[str, Any]:
    _require_pyedflib("EDF .edf loading")
    import pyedflib  # type: ignore

    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip()
        return str(value).strip()

    def _safe_reader_call(reader: Any, method_name: str) -> Any:
        fn = getattr(reader, method_name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        return None

    reader = pyedflib.EdfReader(str(path))
    try:
        n_signals = int(getattr(reader, "signals_in_file", 0) or 0)
        if n_signals <= 0:
            raise ValueError(f"EDF file '{path}' does not contain readable signals.")

        labels_raw = _safe_reader_call(reader, "getSignalLabels") or []
        ch_names = [
            (_as_text(labels_raw[i]) if i < len(labels_raw) else "") or f"ch{i}"
            for i in range(n_signals)
        ]

        sample_freqs = np.asarray(_safe_reader_call(reader, "getSampleFrequencies"), dtype=float).reshape(-1)
        if sample_freqs.size != n_signals:
            raise ValueError(
                f"EDF file '{path}' returned {sample_freqs.size} sample-frequency values for {n_signals} channels."
            )
        if not np.isfinite(sample_freqs).all() or float(sample_freqs[0]) <= 0:
            raise ValueError(f"EDF file '{path}' contains invalid sample-frequency values.")

        fs = float(sample_freqs[0])
        if np.any(np.abs(sample_freqs - fs) > 1e-9):
            unique_freqs = sorted({float(v) for v in sample_freqs.tolist()})
            raise ValueError(
                f"EDF file '{path}' contains mixed sampling frequencies across channels: {unique_freqs}. "
                "A single shared frequency is required."
            )

        n_samples_arr = np.asarray(_safe_reader_call(reader, "getNSamples"), dtype=int).reshape(-1)
        if n_samples_arr.size != n_signals:
            raise ValueError(
                f"EDF file '{path}' returned {n_samples_arr.size} sample-count values for {n_signals} channels."
            )
        if np.any(n_samples_arr <= 0):
            raise ValueError(f"EDF file '{path}' contains non-positive sample counts.")
        if np.any(n_samples_arr != int(n_samples_arr[0])):
            unique_counts = sorted({int(v) for v in n_samples_arr.tolist()})
            raise ValueError(
                f"EDF file '{path}' contains inconsistent sample counts across channels: {unique_counts}."
            )

        n_samples = int(n_samples_arr[0])
        data = np.empty((n_signals, n_samples), dtype=float)
        for ch_idx in range(n_signals):
            sig = np.asarray(reader.readSignal(ch_idx), dtype=float).reshape(-1)
            if sig.size != n_samples:
                raise ValueError(
                    f"EDF file '{path}' channel {ch_idx} has {sig.size} samples, expected {n_samples}."
                )
            data[ch_idx, :] = sig

        start_dt = _safe_reader_call(reader, "getStartdatetime")
        recording_start = start_dt.isoformat() if start_dt is not None and hasattr(start_dt, "isoformat") else None

        annotations: list[Dict[str, Any]] = []
        raw_annotations = _safe_reader_call(reader, "readAnnotations")
        if isinstance(raw_annotations, tuple) and len(raw_annotations) >= 3:
            onsets, durations, descriptions = raw_annotations[0], raw_annotations[1], raw_annotations[2]
            for onset, duration, description in zip(onsets, durations, descriptions):
                try:
                    onset_val = float(onset)
                except Exception:
                    onset_val = None
                try:
                    duration_val = float(duration)
                except Exception:
                    duration_val = None
                annotations.append(
                    {
                        "onset_s": onset_val,
                        "duration_s": duration_val,
                        "description": _as_text(description),
                    }
                )

        header = {}
        header_fields = {
            "patient_code": "getPatientCode",
            "patient_name": "getPatientName",
            "admin_code": "getAdmincode",
            "technician": "getTechnician",
            "equipment": "getEquipment",
            "recording_additional": "getRecordingAdditional",
            "patient_additional": "getPatientAdditional",
            "gender": "getGender",
            "birthdate": "getBirthdate",
        }
        for key, method_name in header_fields.items():
            value = _safe_reader_call(reader, method_name)
            value_text = _as_text(value)
            if value_text:
                header[key] = value_text

        subject_id = header.get("patient_code") or header.get("patient_name")
        duration_s = float(n_samples) / fs
        out: Dict[str, Any] = {
            "__source_format__": "edf",
            "data": data,
            "fs": fs,
            "ch_names": ch_names,
            "time": np.arange(n_samples, dtype=float) / fs,
            "duration_s": duration_s,
            "recording_start": recording_start,
            "n_channels": int(n_signals),
            "n_samples": int(n_samples),
            "annotations": annotations,
            "header": header,
        }
        if subject_id:
            out["subject_id"] = subject_id
        return out
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _nwb_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, (list, tuple, set)):
        parts = [_nwb_text(item) for item in value]
        text = ", ".join(part for part in parts if part)
        return text or None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _nwb_text(value.reshape(-1)[0])
        if value.size <= 8:
            parts = [_nwb_text(item) for item in value.reshape(-1)]
            text = ", ".join(part for part in parts if part)
            return text or None
        return None
    text = str(value).strip()
    return text or None


def _nwb_safe_key(value: Any, fallback: str = "series") -> str:
    text = _nwb_text(value) or fallback
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"s_{text}"
    return text


def _nwb_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        out = float(arr.reshape(-1)[0])
    except Exception:
        try:
            out = float(value)
        except Exception:
            return None
    return out if np.isfinite(out) else None


def _nwb_array(value: Any) -> np.ndarray:
    last_exc: Optional[Exception] = None
    loaders = (
        lambda: np.asarray(value),
        lambda: np.asarray(value[:]),
        lambda: np.asarray(value[()]),
    )
    for load in loaders:
        try:
            arr = np.asarray(load())
        except Exception as exc:
            last_exc = exc
            continue
        if arr.size == 0:
            last_exc = ValueError("empty data")
            continue
        if arr.dtype == np.object_:
            try:
                arr = arr.astype(float)
            except Exception as exc:
                last_exc = exc
                continue
        if np.issubdtype(arr.dtype, np.number):
            return arr
        last_exc = ValueError(f"non-numeric data dtype {arr.dtype}")
    detail = f": {last_exc}" if last_exc is not None else ""
    raise ValueError(f"NWB data could not be converted to a numeric NumPy array{detail}")


def _nwb_timestamps(series: Any) -> Optional[np.ndarray]:
    try:
        timestamps = getattr(series, "timestamps", None)
    except Exception:
        timestamps = None
    if timestamps is None:
        return None
    try:
        arr = _nwb_array(timestamps).reshape(-1)
    except Exception:
        return None
    return arr if arr.size > 0 else None


def _nwb_sampling_rate(series: Any, *, allow_timestamps: bool = True) -> Optional[float]:
    for attr in ("rate", "sampling_rate"):
        try:
            value = getattr(series, attr, None)
        except Exception:
            value = None
        fs = _nwb_float(value)
        if fs is not None and fs > 0:
            return fs

    if not allow_timestamps:
        return None

    timestamps = _nwb_timestamps(series)
    if timestamps is None or timestamps.size < 2:
        return None
    diffs = np.diff(timestamps.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median_dt = float(np.median(diffs))
    return (1.0 / median_dt) if median_dt > 0 else None


def _nwb_electrode_labels(series: Any, n_channels: int) -> Optional[list[str]]:
    try:
        electrodes = getattr(series, "electrodes", None)
    except Exception:
        electrodes = None
    if electrodes is None:
        return None

    try:
        frame = electrodes.to_dataframe()
    except Exception:
        return None

    label_columns = ("label", "channel_name", "channel", "name", "electrode_label", "id")
    for column in label_columns:
        if column not in getattr(frame, "columns", []):
            continue
        labels = [_nwb_text(value) for value in frame[column].tolist()]
        labels = [label for label in labels if label]
        if len(labels) >= n_channels:
            return labels[:n_channels]

    try:
        labels = [_nwb_text(value) for value in frame.index.tolist()]
        labels = [label for label in labels if label]
        if len(labels) >= n_channels:
            return labels[:n_channels]
    except Exception:
        pass
    return None


def _nwb_data_shape(value: Any, *, allow_array: bool = True) -> tuple[int, ...]:
    try:
        shape = getattr(value, "shape", None)
    except Exception:
        shape = None
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except Exception:
            pass
    if allow_array:
        try:
            return tuple(int(dim) for dim in np.asarray(value).shape)
        except Exception:
            pass
    return ()


def _nwb_data_dtype(value: Any, *, allow_array: bool = True) -> Optional[str]:
    try:
        dtype = getattr(value, "dtype", None)
    except Exception:
        dtype = None
    if dtype is not None:
        return str(dtype)
    if allow_array:
        try:
            return str(np.asarray(value).dtype)
        except Exception:
            pass
    return None


def _nwb_channel_names(series: Any, n_channels: int, ndim: int) -> list[str]:
    if ndim <= 1:
        return [_nwb_text(getattr(series, "name", None)) or "ch0"]

    labels = _nwb_electrode_labels(series, n_channels)
    if labels is not None:
        return labels
    return [f"ch{i}" for i in range(n_channels)]


def _nwb_recording_type(series: Any, path: str = "") -> RecordingType:
    name = " ".join(
        item for item in (
            _nwb_text(getattr(series, "name", None)),
            type(series).__name__,
            path,
        )
        if item
    ).lower()
    if "ecog" in name:
        return "ECoG"
    if "eeg" in name:
        return "EEG"
    if "meg" in name:
        return "MEG"
    if "lfp" in name or "electrical" in name or "voltage" in name:
        return "LFP"
    return "Unknown"


def _nwb_iter_mapping(container: Any):
    if container is None:
        return
    items_fn = getattr(container, "items", None)
    if callable(items_fn):
        try:
            for key, value in items_fn():
                yield str(key), value
            return
        except Exception:
            pass
    if isinstance(container, MappingABC):
        for key, value in container.items():
            yield str(key), value


def _nwb_walk_timeseries(root_name: str, container: Any):
    visited: set[int] = set()

    def _is_terminal(value: Any) -> bool:
        return value is None or isinstance(value, (str, bytes, int, float, bool, np.ndarray))

    def _walk(path: str, obj: Any, depth: int):
        if obj is None or depth > 8:
            return
        marker = id(obj)
        if marker in visited:
            return
        visited.add(marker)

        try:
            data = getattr(obj, "data", None)
        except Exception:
            data = None
        if data is not None and not isinstance(obj, np.ndarray):
            yield path, obj

        for key, child in _nwb_iter_mapping(obj) or ():
            if _is_terminal(child):
                continue
            child_path = f"{path}.{_nwb_safe_key(key)}" if path else _nwb_safe_key(key)
            yield from _walk(child_path, child, depth + 1)

        for attr in ("data_interfaces", "electrical_series", "time_series", "roi_response_series"):
            try:
                nested = getattr(obj, attr, None)
            except Exception:
                nested = None
            for key, child in _nwb_iter_mapping(nested) or ():
                if _is_terminal(child):
                    continue
                child_path = f"{path}.{_nwb_safe_key(key)}" if path else _nwb_safe_key(key)
                yield from _walk(child_path, child, depth + 1)

        try:
            fields = getattr(obj, "fields", None)
        except Exception:
            fields = None
        for key, child in _nwb_iter_mapping(fields) or ():
            if str(key) in {"data", "timestamps", "starting_time", "electrodes"} or _is_terminal(child):
                continue
            child_path = f"{path}.{_nwb_safe_key(key)}" if path else _nwb_safe_key(key)
            yield from _walk(child_path, child, depth + 1)

    yield from _walk(_nwb_safe_key(root_name), container, 0)


def _nwb_session_metadata(nwbfile: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    subject = getattr(nwbfile, "subject", None)
    if subject is not None:
        subject_fields = {
            "subject_id": "subject_id",
            "species": "species",
            "subject_sex": "sex",
            "subject_age": "age",
            "subject_description": "description",
            "subject_genotype": "genotype",
            "subject_strain": "strain",
        }
        for out_key, attr in subject_fields.items():
            value = _nwb_text(getattr(subject, attr, None))
            if value:
                metadata[out_key] = value

    file_fields = (
        "identifier",
        "session_id",
        "session_description",
        "experiment_description",
        "institution",
        "lab",
        "experimenter",
    )
    for attr in file_fields:
        value = _nwb_text(getattr(nwbfile, attr, None))
        if value:
            metadata[attr] = value
    return metadata


def _nwb_series_payload(
    series: Any,
    path: str,
    metadata: Mapping[str, Any],
    *,
    load_data: bool = True,
) -> Dict[str, Any]:
    data_obj = getattr(series, "data")
    data_shape = _nwb_data_shape(data_obj, allow_array=load_data)
    data_dtype = _nwb_data_dtype(data_obj, allow_array=load_data)
    if not load_data and data_dtype:
        try:
            if not np.issubdtype(np.dtype(data_dtype), np.number):
                raise ValueError(f"non-numeric data dtype {data_dtype}")
        except TypeError:
            pass
    data = _nwb_array(data_obj) if load_data else None
    if data is not None:
        data_shape = tuple(int(dim) for dim in data.shape)
        data_dtype = str(data.dtype)
    fs = _nwb_sampling_rate(series, allow_timestamps=load_data)
    timestamps = _nwb_timestamps(series) if load_data else None
    ndim = len(data_shape)
    n_samples = int(data_shape[0]) if ndim >= 1 else (int(data.size) if data is not None else 0)
    n_channels = int(data_shape[1]) if ndim >= 2 else 1
    ch_names = _nwb_channel_names(series, n_channels, ndim)

    duration_s = None
    if fs is not None and fs > 0 and n_samples > 0:
        duration_s = float(n_samples) / float(fs)
    elif timestamps is not None and timestamps.size >= 2:
        duration_s = float(timestamps[-1] - timestamps[0])

    payload: Dict[str, Any] = dict(metadata)
    payload.update(
        {
            "fs": fs,
            "ch_names": ch_names,
            "recording_type": _nwb_recording_type(series, path),
            "series_name": _nwb_text(getattr(series, "name", None)) or _nwb_safe_key(path),
            "series_path": path,
            "series_type": type(series).__name__,
            "data_shape": list(data_shape),
            "data_dtype": data_dtype,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "duration_s": duration_s,
            "starting_time": _nwb_float(getattr(series, "starting_time", None)),
            "unit": _nwb_text(getattr(series, "unit", None)),
            "conversion": _nwb_float(getattr(series, "conversion", None)),
            "resolution": _nwb_float(getattr(series, "resolution", None)),
        }
    )
    if data is not None:
        payload["data"] = data
    if timestamps is not None:
        payload["timestamps"] = timestamps
    return payload


def _nwb_series_priority(entry: Mapping[str, Any]) -> tuple:
    payload = entry.get("payload") or {}
    path = str(payload.get("series_path") or entry.get("path") or "").lower()
    name = str(payload.get("series_name") or "").lower()
    data = payload.get("data")
    data_shape = payload.get("data_shape") or ()
    score = 0
    if path.startswith("acquisition"):
        score -= 40
    if any(token in f"{path} {name}" for token in ("lfp", "electrical", "eeg", "ecog", "meg", "voltage")):
        score -= 25
    try:
        if data is not None and np.asarray(data).ndim >= 2:
            score -= 10
        elif len(data_shape) >= 2:
            score -= 10
    except Exception:
        pass
    n_channels = payload.get("n_channels")
    try:
        score -= min(int(n_channels), 64)
    except Exception:
        pass
    return (score, str(payload.get("series_path") or ""))


def _load_nwb_with_pynwb(path: Path, *, load_data: bool = True) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"NWB file does not exist or is not a regular file: '{path}'. "
            "Verify the dataset path and that the file is available on disk."
        )

    _require_pynwb("NWB .nwb loading")
    from pynwb import NWBHDF5IO  # type: ignore

    failures: list[str] = []
    try:
        try:
            io_obj = NWBHDF5IO(str(path), "r", load_namespaces=True)
        except TypeError:
            io_obj = NWBHDF5IO(str(path), "r")

        with io_obj as io_file:
            nwbfile = io_file.read()
            metadata = _nwb_session_metadata(nwbfile)
            roots: list[tuple[str, Any]] = [
                ("acquisition", getattr(nwbfile, "acquisition", None)),
                ("stimulus", getattr(nwbfile, "stimulus", None)),
            ]
            for module_name, module in _nwb_iter_mapping(getattr(nwbfile, "processing", None)) or ():
                roots.append((f"processing.{_nwb_safe_key(module_name)}", module))

            entries: list[Dict[str, Any]] = []
            used_keys: set[str] = set()
            seen_paths: set[str] = set()
            for root_name, root in roots:
                for series_path, series in _nwb_walk_timeseries(root_name, root):
                    if series_path in seen_paths:
                        continue
                    seen_paths.add(series_path)
                    try:
                        payload = _nwb_series_payload(series, series_path, metadata, load_data=load_data)
                    except Exception as exc:
                        failures.append(f"{series_path}: {exc}")
                        continue

                    base_key = _nwb_safe_key(payload.get("series_name") or series_path, "series")
                    key = base_key
                    counter = 2
                    while key in used_keys:
                        key = f"{base_key}_{counter}"
                        counter += 1
                    used_keys.add(key)
                    entries.append({"key": key, "root": root_name, "path": series_path, "payload": payload})

            if not entries:
                detail = f" Skipped series: {'; '.join(failures[:5])}" if failures else ""
                raise ValueError(f"NWB file '{path}' does not contain readable numeric TimeSeries data.{detail}")

            entries.sort(key=_nwb_series_priority)
            selected = entries[0]
            series_map = {str(entry["key"]): entry["payload"] for entry in entries}
            grouped: Dict[str, Dict[str, Any]] = {}
            for entry in entries:
                group_key = str(entry["root"]).split(".", 1)[0]
                grouped.setdefault(group_key, {})[str(entry["key"])] = entry["payload"]

            result: Dict[str, Any] = dict(selected["payload"])
            result.update(
                {
                    "__source_format__": "nwb",
                    "metadata": metadata,
                    "selected_series": selected["key"],
                    "series_count": len(entries),
                    "series": series_map,
                    "timeseries": series_map,
                    "acquisition": grouped.get("acquisition", {}),
                    "processing": grouped.get("processing", {}),
                    "stimulus": grouped.get("stimulus", {}),
                }
            )
            return result
    except Exception as exc:
        if isinstance(exc, (ImportError, ValueError)):
            raise
        raise ValueError(f"Failed to load NWB file '{path}' with pynwb: {exc}") from exc


# ---- Locators -------------------------------------------------------------
# Locator grammar (recommended):
# - "__self__" -> use the input object itself
# - "a.b.c"    -> nested dict keys OR nested object attributes
# - "get_data" -> if resolves to a callable attribute, the parser may call it (implementation choice)
# - callable   -> escape hatch for complex sources (e.g., JSON lists, HDF5 trees)
# - scalar     -> literal value (e.g., fs=1000.0)
# - sequence   -> either literal (e.g., ch_names=["Fz","Cz"]) OR tokenized path (e.g., ("a", 0, "b"))
Scalar = Union[str, int, float, bool]
Locator = Union[
    str,
    Callable[[Any], Any],
    Scalar,
    Sequence[Any],
]


@dataclass(frozen=True)
class CanonicalFields:
    """
    User-specified mapping that tells the parser where to find each canonical piece.

    The parser should interpret locators depending on the input type:
      - MNE objects: dotted attribute paths (e.g., "info.sfreq", "ch_names", "get_data")
      - dict-like: keys (including nested via dotted path)
      - pandas/parquet/csv: column names for time, and channel columns / long columns
      - numpy arrays: usually data is the array itself ("__self__"), fs is literal or sidecar/callable
      - JSON/HDF5/etc: often best handled via dotted keys or a callable locator for complex layouts
    """

    # Required (in practice data is always required)
    # Common patterns:
    #   - dict-like: "data" or "signal"
    #   - numpy array: "__self__"
    #   - MNE raw/epochs: "get_data" (if your resolver calls it) or callable lambda obj: obj.get_data()
    data: Locator = "data"

    # Optional but strongly recommended
    fs: Optional[Locator] = None

    # Optional channel information
    ch_names: Optional[Locator] = None

    # Optional time information (if present explicitly)
    time: Optional[Locator] = None

    # Optional: declare the domain of `data` ("time" vs "frequency").
    # If not provided, parsers should assume time-domain signals.
    data_domain: Optional[Locator] = None  # resolves to DataDomain

    # Optional spectral axis information (only relevant when data_domain == "frequency")
    freqs: Optional[Locator] = None          # array of frequencies (Hz), if available
    spectral_kind: Optional[Locator] = None  # resolves to SpectralKind

    # Optional epoching hints / fields (if the input already contains epochs)
    epoch: Optional[Locator] = None

    # Optional metadata mapping (subject_id, condition, etc.)
    # Each entry is a locator like above.
    metadata: Mapping[str, Locator] = field(default_factory=dict)

    # Optional: for tabular containers (pandas/parquet/csv) specify how to interpret columns
    # - wide: channels are columns (one row per time sample)
    # - long: columns include [time, channel, value] (one row per channel-time observation)
    table_layout: Optional[Literal["wide", "long"]] = None

    # If wide layout, specify which columns are channels (else infer by exclusion)
    # Note: in wide layout, channel column names often *are* the channel names, but you may also
    # set ch_names separately if you want different output labels.
    channel_columns: Optional[Sequence[str]] = None

    # If long layout, define column names
    long_channel_col: Optional[str] = None
    long_value_col: Optional[str] = None
    long_time_col: Optional[str] = None

    # Optional: explicit axis mapping for multi-dimensional arrays.
    # Dict mapping "channels", "samples", and optionally "epochs"/"ids" to axis indices.
    # When provided, overrides heuristic axis detection.
    # Example: {"ids": 0, "channels": 1, "samples": 2, "epochs": 3}
    array_axes: Optional[Mapping[str, int]] = None


@dataclass
class ParseConfig:
    """Configuration for parsing neural data from MNE and non-MNE sources.

    This configuration is explicit: the user specifies where each canonical field
    (data, fs, channel names, time, etc.) can be found via `fields`.

    Epoching, MNE load options, and post-processing are controlled here as well.
    """

    # How to extract canonical fields (data, fs, ch_names, time, domain, etc.)
    fields: CanonicalFields = field(default_factory=CanonicalFields)

    # If the user wants to epoch continuous data after extraction
    epoch_length_s: Optional[float] = None
    epoch_step_s: Optional[float] = None

    # Optional temporal segmentation (applied before epoching), in seconds.
    # Example:
    #   segment_t0_s=10.0, segment_t1_s=60.0 -> keep only [10, 60]s from each row.
    segment_t0_s: Optional[float] = None
    segment_t1_s: Optional[float] = None

    # MNE-specific operational options (only used if input is MNE or you load with MNE)
    preload: bool = True
    pick_types: Optional[Dict[str, bool]] = None
    max_seconds: Optional[float] = None
    drop_bads: bool = True

    # ---------------------------
    # Post-processing
    # ---------------------------
    zscore: bool = False
    zscore_after_epoch: bool = False  # If True, apply z-score AFTER epoching (per-epoch)
    exclude_last_epoch: bool = False  # If True, exclude the last epoch from each time series
    # If True, trim each grouped epoch series to the minimum count observed across groups.
    # Disable this for faster parsing when variable epoch counts are acceptable downstream.
    align_epoch_count_to_minimum: bool = True

    # Aggregation: aggregate over one or more categorical columns.
    # Examples:
    #   aggregate_over=("sensor",)            -> collapse sensors
    #   aggregate_over=("group",)             -> collapse groups
    #   aggregate_over=("sensor", "group")    -> collapse both
    aggregate_over: Optional[Sequence[str]] = None
    aggregate_method: AggregateMethod = "sum"

    # Replacement labels for aggregated columns, e.g. {"sensor": "aggregate", "group": "all_groups"}
    aggregate_labels: Mapping[str, str] = field(default_factory=lambda: {"sensor": "aggregate"})

    # Behavior
    warn_unimplemented: bool = True


# ---------------------------
# Parser implementation
# ---------------------------

class EphysDatasetParser:
    """
    Minimal-but-functional parser that complements ParseConfig/CanonicalFields.

    Supported inputs:
      - MNE Raw/Epochs objects (if mne installed)
      - numpy arrays (ndarray)
      - dict-like (including scipy.io.loadmat output, json-loaded dict)
      - pandas DataFrame (wide/long), and file paths to csv/parquet if pandas installed
      - .npy, .json, .mat, .nwb, .set, .fif, .edf, .ds, .tsv paths

    Output:
      - pandas DataFrame with DEFAULT_COLUMNS
      - one row per (epoch, sensor) with `data` stored as a 1D numpy array
        (time-domain or frequency-domain)
    """

    def __init__(self, config: Optional[ParseConfig] = None):
        self.config = config or ParseConfig()

    # -------------------------
    # Public API
    # -------------------------

    def parse(self, source: Union[str, Path, Any]) -> "Any":
        """
        Parse a source (path or in-memory object) into a pandas DataFrame
        with DEFAULT_COLUMNS.
        """
        if (
            self.config.epoch_length_s is not None
            and self.config.segment_t0_s is not None
            and self.config.segment_t1_s is not None
        ):
            seg_len = float(self.config.segment_t1_s) - float(self.config.segment_t0_s)
            if seg_len < float(self.config.epoch_length_s):
                raise ValueError(
                    "Temporal segmentation window must be at least as long as epoch_length_s when epoching is enabled."
                )

        obj, source_file = self._load_source(source)

        # 1) Parse raw rows (one row per sensor / epoch)
        rows = self._parse_object(obj, source_file=source_file)

        # 1b) Optional temporal segmentation BEFORE z-score/aggregation/epoching
        rows = self._apply_time_segment_rows(rows)

        df = self._rows_to_df(rows)

        # 2) Optional z-scoring BEFORE epoching (default behavior)
        if self.config.zscore and not self.config.zscore_after_epoch:
            df = self._apply_zscore(df)

        # 3) Aggregate FIRST (e.g. collapse sensors)
        if self.config.aggregate_over:
            df = self._apply_aggregation(df)

        # 4) Epoch LAST (temporal operation)
        if self.config.epoch_length_s is not None:
            rows = df.to_dict("records")
            rows = self._apply_epoching_rows(rows)
            df = self._rows_to_df(rows)

        # 4b) Keep all parsed series aligned when parser-driven epoching produces
        # different counts. Do not alter pre-labeled epochs parsed from source data.
        if (
            self.config.epoch_length_s is not None
            and bool(getattr(self.config, "align_epoch_count_to_minimum", True))
        ):
            df = self._limit_epoch_count_to_minimum(df)

        # 4c) Exclude final epochs explicitly or when an incomplete final epoch is detected.
        if self.config.exclude_last_epoch:
            df = self._exclude_last_epoch(df)
        else:
            df = self._exclude_incomplete_final_epoch(df)

        # 5) Optional z-scoring AFTER epoching (per-epoch normalization)
        if self.config.zscore and self.config.zscore_after_epoch:
            df = self._apply_zscore(df)

        return df

    # -------------------------
    # Loading
    # -------------------------

    def _load_source(self, source: Union[str, Path, Any]) -> Tuple[Any, Optional[str]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            suffix = path.suffix.lower()
            source_file = str(path)

            if suffix == ".npy":
                return np.load(path), source_file

            if suffix == ".json":
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f), source_file

            if suffix == ".mat":
                # Supports both classic MAT files and MATLAB v7.3 (HDF5-backed).
                return _load_mat_with_fallback(path), source_file

            if suffix == ".nwb":
                return _load_nwb_with_pynwb(path), source_file

            if suffix == ".set":
                _require_mne("EEGLAB .set loading")
                import mne  # type: ignore

                return mne.io.read_raw_eeglab(str(path), preload=self.config.preload), source_file

            if suffix == ".fif":
                if not path.is_file():
                    raise ValueError(
                        f"FIF file does not exist or is not a regular file: '{path}'. "
                        "Verify the dataset path and that the file is available on disk."
                    )
                _require_mne("MNE FIF .fif loading")
                import mne  # type: ignore

                return mne.io.read_raw_fif(str(path), preload=self.config.preload, verbose=False), source_file

            if suffix == ".edf":
                if not path.is_file():
                    raise ValueError(
                        f"EDF file does not exist or is not a regular file: '{path}'. "
                        "Verify the dataset path and that the file is available on disk."
                    )
                return _load_edf_with_pyedflib(path), source_file

            if suffix == ".ds":
                if not path.is_dir():
                    raise ValueError(
                        f"CTF dataset does not exist or is not a directory: '{path}'. "
                        "CTF recordings must be provided as the .ds folder."
                    )
                _require_mne("CTF .ds loading")
                import mne  # type: ignore

                return mne.io.read_raw_ctf(str(path), preload=self.config.preload, verbose=False), source_file

            if suffix in {".vhdr", ".dat"}:
                def _read_vhdr_refs(vhdr_path: Path) -> tuple[Optional[str], Optional[str]]:
                    data_file_val = None
                    marker_file_val = None
                    with vhdr_path.open("r", encoding="utf-8", errors="ignore") as f:
                        for raw_line in f:
                            line = raw_line.strip()
                            if not line or line.startswith(";") or "=" not in line:
                                continue
                            key_part, value_part = line.split("=", 1)
                            key = key_part.strip().lower()
                            value = value_part.strip()
                            if key == "datafile":
                                data_file_val = value
                            elif key == "markerfile":
                                marker_file_val = value
                    return data_file_val, marker_file_val

                def _validate_vhdr_and_companions(vhdr_path: Path) -> tuple[Path, Path]:
                    if not vhdr_path.is_file():
                        raise ValueError(
                            f"BrainVision header file does not exist or is not a regular file: '{vhdr_path}'."
                        )
                    data_file_val, marker_file_val = _read_vhdr_refs(vhdr_path)
                    missing = []
                    if not data_file_val:
                        missing.append("DataFile entry in .vhdr")
                    if not marker_file_val:
                        missing.append("MarkerFile entry in .vhdr")
                    if missing:
                        raise ValueError(
                            f"Invalid .vhdr file '{vhdr_path}': missing {', '.join(missing)}."
                        )

                    data_path = (vhdr_path.parent / data_file_val).resolve()
                    vmrk_path = (vhdr_path.parent / marker_file_val).resolve()
                    if not data_path.is_file():
                        raise ValueError(
                            f"Missing BrainVision data file referenced by .vhdr: '{data_path}'"
                        )
                    if not vmrk_path.is_file():
                        raise ValueError(
                            f"Missing BrainVision .vmrk file referenced by .vhdr: '{vmrk_path}'"
                        )
                    return data_path, vmrk_path

                vhdr_path = path
                if suffix == ".dat":
                    if not path.is_file():
                        raise ValueError(
                            f"BrainVision data file does not exist or is not a regular file: '{path}'."
                        )

                    candidates = sorted(path.parent.glob("*.vhdr"))
                    if not candidates:
                        raise ValueError(
                            f"Could not find a companion .vhdr file next to BrainVision data file '{path}'."
                        )

                    selected_vhdr = None
                    for candidate in candidates:
                        data_file_val, _ = _read_vhdr_refs(candidate)
                        if not data_file_val:
                            continue
                        candidate_data_path = (candidate.parent / data_file_val).resolve()
                        if candidate_data_path == path.resolve():
                            selected_vhdr = candidate
                            break

                    if selected_vhdr is None and len(candidates) == 1:
                        selected_vhdr = candidates[0]

                    if selected_vhdr is None:
                        raise ValueError(
                            "Found multiple .vhdr files and could not uniquely match one to "
                            f"the BrainVision data file '{path}'."
                        )
                    vhdr_path = selected_vhdr

                _validate_vhdr_and_companions(vhdr_path)
                _require_mne("BrainVision loading (.vhdr/.dat)")
                import mne
                return mne.io.read_raw_brainvision(str(vhdr_path), preload=self.config.preload, verbose=False), source_file

            if suffix in (".csv", ".parquet", ".tsv"):
                if not tools.ensure_module("pandas"):
                    raise ImportError("pandas is required to load tabular files (.csv/.parquet/.tsv).")
                import pandas as pd  # type: ignore

                if suffix == ".csv":
                    return pd.read_csv(path), source_file
                if suffix == ".tsv":
                    return pd.read_csv(path, sep="\t"), source_file
                else:
                    return pd.read_parquet(path), source_file

            raise ValueError(f"Unsupported file type: {suffix}")

        # in-memory object
        return source, None

    # -------------------------
    # Locators / resolving
    # -------------------------

    def _resolve(self, obj: Any, locator: Optional[Locator]) -> Any:
        if locator is None:
            return None

        # Callable locator: escape hatch for complex sources
        if callable(locator):
            return locator(obj)

        # Tokenized path: ("a", 0, "b") -> obj["a"][0]["b"] / getattr / index
        # Treat sequences as literal values (e.g., ch_names=["Fz","Cz"], freqs=[...]).
        # If you want tokenized paths, use a callable locator.
        if isinstance(locator, (list, tuple)):
            return locator

        # Scalar literal
        if isinstance(locator, (int, float, bool)):
            return locator

        # String locator
        if isinstance(locator, str):
            if locator == "__self__":
                return obj

            # Divide by ., but considering possible "[" / "]"
            # E.g.: "OptoRampsLFP[0,0].LFP" -> ["OptoRampsLFP[0,0]", "LFP"]
            parts = []
            for part in locator.split('.'):
                parts.append(part)
            
            cur = obj
            for part in parts:
                # Extract base name and indexes between []
                base = part
                indices = []
                if '[' in part and part.endswith(']'):
                    base, rest = part.split('[', 1)
                    rest = rest.rstrip(']')
                    # Allow multiple indexes separated by a comma: "0,0"
                    for idx_str in rest.split(','):
                        idx_str = idx_str.strip()
                        try:
                            indices.append(int(idx_str))
                        except ValueError:
                            # Si no es entero, lo ignoramos (podría ser slice, etc.)
                            pass
                
                cur = self._step(cur, base)
                
                # Apply indexes sequentially
                for idx in indices:
                    if isinstance(cur, (list, tuple, np.ndarray)):
                        cur = cur[idx]
                    else:
                        raise KeyError(f"Cannot index into {type(cur)} with {idx}")
            return cur

        return locator

    def _step(self, cur: Any, key: Union[str, int]) -> Any:
        # pandas DataFrame: treat string keys as column names if present
        if tools.ensure_module("pandas"):
            import pandas as pd  # type: ignore
            if isinstance(cur, pd.DataFrame) and isinstance(key, str) and key in cur.columns:
                return cur[key]

        # dict-like
        if isinstance(cur, MappingABC) and key in cur:
            return cur[key]

        # list/tuple indexing
        if isinstance(key, int) and isinstance(cur, (list, tuple)):
            return cur[key]

        # attribute access
        if isinstance(key, str) and hasattr(cur, key):
            return getattr(cur, key)

        # dict-like but key might be present as string for MATLAB objects
        if isinstance(cur, MappingABC) and isinstance(key, str):
            # try common variations
            if key in cur:
                return cur[key]
            if key.encode() in cur:  # rare
                return cur[key.encode()]

        # Object-arrays handling (MATLAB structures 1xN)
        if isinstance(cur, np.ndarray) and cur.dtype == np.object_:
            if cur.size == 1:
                return self._step(cur.item(), key)
            results = []
            for item in cur.flat:
                try:
                    results.append(self._step(item, key))
                except KeyError as e:
                    raise KeyError(
                        f"Cannot resolve '{key}' on element of type {type(item)!r} "
                        f"inside object array of shape {cur.shape}"
                    ) from e
            return results

        raise KeyError(f"Cannot resolve step '{key}' on object of type {type(cur)!r}")

    # -------------------------
    # Parsing for different object types
    # -------------------------

    def _parse_object(self, obj: Any, source_file: Optional[str]) -> list[dict]:
        # pandas DataFrame
        if tools.ensure_module("pandas"):
            import pandas as pd  # type: ignore

            if isinstance(obj, pd.DataFrame):
                return self._parse_dataframe(obj, source_file)

        # MNE objects
        if tools.ensure_module("mne"):
            import mne  # type: ignore

            if isinstance(obj, mne.io.BaseRaw):
                return self._parse_mne_raw(obj, source_file)
            if isinstance(obj, mne.Epochs):
                return self._parse_mne_epochs(obj, source_file)

        # dict-like
        if isinstance(obj, MappingABC):
            return self._parse_dict_like(obj, source_file)

        # numpy array
        if isinstance(obj, np.ndarray):
            return self._parse_ndarray(obj, source_file)

        raise TypeError(f"Unsupported input object type: {type(obj)!r}")

    # ---- DataFrame (wide/long) ----

    def _parse_dataframe(self, df: "Any", source_file: Optional[str]) -> list[dict]:
        f = self.config.fields

        layout = f.table_layout
        if layout not in ("wide", "long"):
            raise ValueError("For pandas inputs, CanonicalFields.table_layout must be 'wide' or 'long'.")

        common_meta = self._resolve_metadata(df)

        fs = self._resolve(df, f.fs)
        fs = float(fs) if fs is not None else None

        data_domain = self._resolve(df, f.data_domain) or "time"
        spectral_kind = self._resolve(df, f.spectral_kind)
        freqs = self._resolve(df, f.freqs)

        if layout == "wide":
            time_col = f.time if isinstance(f.time, str) else None
            times = df[time_col].to_numpy() if time_col and time_col in df.columns else None

            channel_cols = list(f.channel_columns) if f.channel_columns is not None else None
            if channel_cols is None:
                # infer: all numeric columns excluding time + any metadata columns explicitly referenced
                exclude = set()
                if time_col:
                    exclude.add(time_col)
                for loc in self.config.fields.metadata.values():
                    if isinstance(loc, str) and loc in df.columns:
                        exclude.add(loc)

                channel_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

            # channel labels
            ch_names = self._normalize_channel_names(self._resolve(df, f.ch_names))
            if isinstance(ch_names, list):
                labels = list(ch_names)
                if len(labels) != len(channel_cols):
                    raise ValueError("Length of ch_names does not match number of channel_columns.")
            else:
                labels = list(channel_cols)

            # Build one row per channel, store whole vector in `data`
            rows: list[dict] = []
            for col, ch in zip(channel_cols, labels):
                y = df[col].to_numpy()
                t0, t1 = self._infer_t0_t1(times=times, n=len(y), fs=fs)
                r = self._base_row(common_meta, source_file)
                r.update(
                    sensor=str(ch),
                    epoch=None,
                    fs=fs,
                    data=np.asarray(y),
                    t0=t0,
                    t1=t1,
                    data_domain=data_domain,
                    spectral_kind=spectral_kind,
                )
                if data_domain == "frequency":
                    f0, f1 = self._infer_f0_f1(freqs=freqs, n=len(y))
                    r.update(f0=f0, f1=f1)
                rows.append(r)
            return rows

        # long layout
        tcol = f.long_time_col
        ccol = f.long_channel_col
        vcol = f.long_value_col
        if not (tcol and ccol and vcol):
            raise ValueError("For long layout, long_time_col/long_channel_col/long_value_col must be set.")

        if tcol not in df.columns or ccol not in df.columns or vcol not in df.columns:
            raise ValueError("Long layout columns not found in DataFrame.")

        rows = []
        for ch, g in df.groupby(ccol):
            g2 = g.sort_values(tcol)
            y = g2[vcol].to_numpy()
            times = g2[tcol].to_numpy()
            t0, t1 = self._infer_t0_t1(times=times, n=len(y), fs=fs)
            r = self._base_row(common_meta, source_file)
            r.update(
                sensor=str(ch),
                epoch=None,
                fs=fs,
                data=np.asarray(y),
                t0=t0,
                t1=t1,
                data_domain=data_domain,
                spectral_kind=spectral_kind,
            )
            if data_domain == "frequency":
                f0, f1 = self._infer_f0_f1(freqs=freqs, n=len(y))
                r.update(f0=f0, f1=f1)
            rows.append(r)
        return rows


    def _extract_first_if_list(self, value):
        """If the value is not an empty list, return its first element; if not, the original value."""
        if isinstance(value, list) and value:
            return value[0]
        return value

    # ---- dict-like (.mat/.json) ----

    def _parse_dict_like(self, d: dict, source_file: Optional[str]) -> list[dict]:
        f = self.config.fields

        data = self._resolve(d, f.data)

        fs_raw = self._resolve(d, f.fs)
        # Handle a possible list that comes from a struct array
        fs = self._extract_first_if_list(fs_raw)
        fs = float(np.asarray(fs).item()) if fs is not None and np.asarray(fs).size == 1 else (float(fs) if fs is not None else None)

        data_domain = self._resolve(d, f.data_domain) or "time"
        spectral_kind = self._resolve(d, f.spectral_kind)
        freqs = self._resolve(d, f.freqs)

        # epoch label (optional)
        epoch_val = self._resolve(d, f.epoch)

        # channel names
        ch_names = self._normalize_channel_names(self._resolve(d, f.ch_names))

        meta = self._resolve_metadata(d)

        def _subject_ids_for_axis(base_meta: Mapping[str, Any], n_ids: int) -> list[Any]:
            return self._normalize_subject_id_list(base_meta.get("subject_id", None), n_ids)

        condition_rows = self._parse_condition_sequence_data(
            data,
            meta,
            ch_names,
            epoch_val,
            fs_raw,
            data_domain,
            freqs,
            spectral_kind,
            source_file,
        )
        if condition_rows is not None:
            return condition_rows

        # data can be:
        # - 2D: (time, channels) or (channels, time)
        # - 3D: (epochs, channels, time) or (epochs, time, channels)
        try:
            arr = self._as_array_excluding_incomplete_final_epoch(data)
        except ValueError:
            rows = self._parse_ragged_dict_data(
                data,
                meta,
                ch_names,
                epoch_val,
                fs,
                data_domain,
                freqs,
                spectral_kind,
                source_file,
            )
            if rows is not None:
                return rows
            raise
        if arr.dtype == object and arr.ndim == 1 and arr.size > 0:
            try:
                stacked = np.stack([np.asarray(a).squeeze() for a in arr.tolist()], axis=0)
                if stacked.ndim >= 2:
                    arr = stacked
            except Exception:
                pass
        arr, array_axes = self._squeeze_unmapped_singleton_axes(arr, f.array_axes)
        array_axes = self._normalize_array_axes_for_data(arr, array_axes, ch_names)

        if arr.ndim == 1:
            # single channel
            labels = [ch_names[0]] if isinstance(ch_names, list) and ch_names else ["ch0"]
            return [self._row_from_series(meta, labels[0], None, arr, fs, data_domain, freqs, spectral_kind, source_file)]

        if arr.ndim == 2:
            ax = array_axes
            if ax and "samples" in ax:
                sa_ax = int(ax["samples"])
                if "channels" in ax:
                    # Explicit axis mapping: transpose to (channels, samples)
                    ch_ax = int(ax["channels"])
                    arr2 = arr.transpose(ch_ax, sa_ax)
                    n_ch, n_time = arr2.shape
                    labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                    return [self._row_from_series(meta, labels[i], epoch_val, arr2[i, :], fs, data_domain, freqs, spectral_kind, source_file) for i in range(n_ch)]
                if "epochs" in ax or "ids" in ax:
                    outer_role = "epochs" if "epochs" in ax else "ids"
                    outer_ax = int(ax[outer_role])
                    arr2 = arr.transpose(outer_ax, sa_ax)
                    n_outer, _ = arr2.shape
                    sensor_name = str(ch_names[0]) if isinstance(ch_names, list) and ch_names else "ch0"
                    subject_ids = _subject_ids_for_axis(meta, n_outer) if outer_role == "ids" else None
                    rows = []
                    for idx in range(n_outer):
                        meta_i = {
                            key: self._metadata_row_value(str(key), value, idx, n_outer)
                            for key, value in meta.items()
                        }
                        if subject_ids is not None:
                            meta_i["subject_id"] = subject_ids[idx]
                        rows.append(
                            self._row_from_series(
                                meta_i,
                                sensor_name,
                                idx if outer_role == "epochs" else epoch_val,
                                arr2[idx, :],
                                fs,
                                data_domain,
                                freqs,
                                spectral_kind,
                                source_file,
                            )
                        )
                    return rows

            # Infer orientation:
            # - If ch_names provided, align to whichever axis matches len(ch_names)
            # - Else, assume (channels, time) when first dim is smaller than second dim
            if isinstance(ch_names, list):
                if len(ch_names) == arr.shape[0]:
                    time_by_ch = False  # (channels, time)
                elif len(ch_names) == arr.shape[1]:
                    time_by_ch = True  # (time, channels)
                else:
                    time_by_ch = arr.shape[0] >= arr.shape[1]
            else:
                # Common electrophys case: (channels, time)
                time_by_ch = arr.shape[0] >= arr.shape[1]

            if time_by_ch:
                n_time, n_ch = arr.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in
                                                                                                range(n_ch)]
                rows = []
                for i, ch in enumerate(labels):
                    rows.append(
                        self._row_from_series(meta, ch, epoch_val, arr[:, i], fs, data_domain, freqs, spectral_kind,
                                              source_file))
                return rows
            else:
                n_ch, n_time = arr.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in
                                                                                                range(n_ch)]
                rows = []
                for i, ch in enumerate(labels):
                    rows.append(
                        self._row_from_series(meta, ch, epoch_val, arr[i, :], fs, data_domain, freqs, spectral_kind,
                                              source_file))
                return rows

        if arr.ndim == 3:
            ax = array_axes
            if ax and "channels" in ax and "samples" in ax and "epochs" in ax:
                # Explicit axis mapping: transpose to (epochs, channels, samples)
                ep_ax = int(ax["epochs"])
                ch_ax = int(ax["channels"])
                sa_ax = int(ax["samples"])
                arr3 = arr.transpose(ep_ax, ch_ax, sa_ax)
                n_ep, n_ch, n_time = arr3.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                rows = []
                for e in range(n_ep):
                    for i, ch in enumerate(labels):
                        rows.append(self._row_from_series(meta, ch, e, arr3[e, i, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows
            if ax and "channels" in ax and "samples" in ax and "ids" in ax:
                id_ax = int(ax["ids"])
                ch_ax = int(ax["channels"])
                sa_ax = int(ax["samples"])
                arr3 = arr.transpose(id_ax, ch_ax, sa_ax)
                n_ids, n_ch, _ = arr3.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = {
                        key: self._metadata_row_value(str(key), value, sid, n_ids)
                        for key, value in meta.items()
                    }
                    meta_i["subject_id"] = subject_ids[sid]
                    for i, ch in enumerate(labels):
                        rows.append(self._row_from_series(meta_i, ch, epoch_val, arr3[sid, i, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows
            if ax and "epochs" in ax and "samples" in ax and "ids" in ax:
                id_ax = int(ax["ids"])
                ep_ax = int(ax["epochs"])
                sa_ax = int(ax["samples"])
                arr3 = arr.transpose(id_ax, ep_ax, sa_ax)
                n_ids, n_ep, _ = arr3.shape
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                sensor_name = str(ch_names[0]) if isinstance(ch_names, list) and ch_names else "ch0"
                rows = []
                for sid in range(n_ids):
                    meta_i = {
                        key: self._metadata_row_value(str(key), value, sid, n_ids)
                        for key, value in meta.items()
                    }
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_ep):
                        rows.append(self._row_from_series(meta_i, sensor_name, e, arr3[sid, e, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows

            # heuristics: treat first axis as epochs
            n_ep = arr.shape[0]
            # remaining could be (channels, time) or (time, channels)
            rest = arr.shape[1:]
            # attempt to align ch_names
            time_by_ch = True
            if isinstance(ch_names, list):
                if len(ch_names) == rest[0]:
                    time_by_ch = False
                elif len(ch_names) == rest[1]:
                    time_by_ch = True

            rows = []
            for e in range(n_ep):
                ep_arr = arr[e]
                if time_by_ch:
                    n_time, n_ch = ep_arr.shape
                    labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                    for i, ch in enumerate(labels):
                        rows.append(self._row_from_series(meta, ch, e, ep_arr[:, i], fs, data_domain, freqs, spectral_kind, source_file))
                else:
                    n_ch, n_time = ep_arr.shape
                    labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                    for i, ch in enumerate(labels):
                        rows.append(self._row_from_series(meta, ch, e, ep_arr[i, :], fs, data_domain, freqs, spectral_kind, source_file))
            return rows

        if arr.ndim == 4:
            ax = array_axes
            if not (ax and "channels" in ax and "samples" in ax):
                raise ValueError(
                    "4D dict-like arrays require explicit array_axes mapping with at least "
                    "'channels' and 'samples' (and typically 'ids' and/or 'epochs')."
                )

            ch_ax = int(ax["channels"])
            sa_ax = int(ax["samples"])
            id_ax = int(ax["ids"]) if "ids" in ax else -1
            ep_ax = int(ax["epochs"]) if "epochs" in ax else -1

            used = [ch_ax, sa_ax] + ([id_ax] if id_ax >= 0 else []) + ([ep_ax] if ep_ax >= 0 else [])
            if len(set(used)) != len(used):
                raise ValueError("array_axes contains duplicated axis indices.")
            if any(idx < 0 or idx >= arr.ndim for idx in used):
                raise ValueError(f"array_axes indices must be in [0, {arr.ndim - 1}] for shape {arr.shape}.")

            remaining = [i for i in range(arr.ndim) if i not in used]
            available_roles = []
            if ep_ax < 0:
                available_roles.append("epochs")
            if id_ax < 0:
                available_roles.append("ids")
            if len(remaining) > len(available_roles):
                raise ValueError(
                    "4D dict-like arrays must map all non-sample dimensions via array_axes "
                    "(use ids/epochs/channels/samples)."
                )
            for role, axis_idx in zip(available_roles, remaining):
                if role == "epochs":
                    ep_ax = axis_idx
                elif role == "ids":
                    id_ax = axis_idx

            if id_ax >= 0 and ep_ax >= 0:
                arr4 = arr.transpose(id_ax, ep_ax, ch_ax, sa_ax)
                n_ids, n_ep, n_ch, _ = arr4.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = dict(meta)
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_ep):
                        for i, ch in enumerate(labels):
                            rows.append(
                                self._row_from_series(
                                    meta_i, ch, e, arr4[sid, e, i, :], fs, data_domain, freqs, spectral_kind, source_file
                                )
                            )
                return rows

            if id_ax >= 0:
                arr4 = arr.transpose(id_ax, ch_ax, sa_ax, ep_ax) if ep_ax >= 0 else arr.transpose(id_ax, ch_ax, sa_ax, [i for i in range(arr.ndim) if i not in {id_ax, ch_ax, sa_ax}][0])
                # Normalize to (ids, channels, samples, epochs_like)
                if arr4.ndim != 4:
                    raise ValueError(f"Unexpected 4D transpose result shape: {arr4.shape}")
                n_ids, n_ch, _, n_extra = arr4.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = dict(meta)
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_extra):
                        for i, ch in enumerate(labels):
                            rows.append(
                                self._row_from_series(
                                    meta_i, ch, e, arr4[sid, i, :, e], fs, data_domain, freqs, spectral_kind, source_file
                                )
                            )
                return rows

            if ep_ax >= 0:
                arr4 = arr.transpose(ep_ax, ch_ax, sa_ax, [i for i in range(arr.ndim) if i not in {ep_ax, ch_ax, sa_ax}][0])
                n_ep, n_ch, _, n_extra = arr4.shape
                labels = ch_names if isinstance(ch_names, list) and len(ch_names) == n_ch else [f"ch{i}" for i in range(n_ch)]
                rows = []
                for e in range(n_ep):
                    for sid in range(n_extra):
                        meta_i = dict(meta)
                        if "subject_id" not in meta_i:
                            meta_i["subject_id"] = sid
                        for i, ch in enumerate(labels):
                            rows.append(
                                self._row_from_series(
                                    meta_i, ch, e, arr4[e, i, :, sid], fs, data_domain, freqs, spectral_kind, source_file
                                )
                            )
                return rows

        raise ValueError(f"Unsupported data ndim for dict-like source: {arr.ndim}")

    # ---- ndarray (.npy in-memory) ----

    def _parse_ndarray(self, arr: np.ndarray, source_file: Optional[str]) -> list[dict]:
        f = self.config.fields
        # Resolve from the ndarray itself: allow user to set fs/ch_names as literals
        fs = self._resolve(arr, f.fs)
        fs = float(fs) if fs is not None else None

        data_domain = self._resolve(arr, f.data_domain) or "time"
        spectral_kind = self._resolve(arr, f.spectral_kind)
        freqs = self._resolve(arr, f.freqs)

        ch_names = self._normalize_channel_names(self._resolve(arr, f.ch_names))
        labels = ch_names if isinstance(ch_names, list) else None

        meta = self._resolve_metadata(arr)

        def _subject_ids_for_axis(base_meta: Mapping[str, Any], n_ids: int) -> list[Any]:
            return self._normalize_subject_id_list(base_meta.get("subject_id", None), n_ids)

        a = np.asarray(arr)
        if a.dtype == object and a.ndim == 1 and a.size > 0:
            try:
                stacked = np.stack([np.asarray(item).squeeze() for item in a.tolist()], axis=0)
                if stacked.ndim >= 2:
                    a = stacked
            except Exception:
                pass
        if a.ndim == 1:
            ch = labels[0] if labels else "ch0"
            return [self._row_from_series(meta, ch, None, a, fs, data_domain, freqs, spectral_kind, source_file)]

        if a.ndim == 2:
            ax = f.array_axes
            if ax and "samples" in ax:
                sa_ax = int(ax["samples"])
                if "channels" in ax:
                    ch_ax = int(ax["channels"])
                    a2 = a.transpose(ch_ax, sa_ax)
                    n_ch = a2.shape[0]
                    labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                    return [self._row_from_series(meta, labels2[i], None, a2[i, :], fs, data_domain, freqs, spectral_kind, source_file) for i in range(n_ch)]
                if "epochs" in ax or "ids" in ax:
                    outer_role = "epochs" if "epochs" in ax else "ids"
                    outer_ax = int(ax[outer_role])
                    a2 = a.transpose(outer_ax, sa_ax)
                    n_outer, _ = a2.shape
                    sensor_name = labels[0] if labels else "ch0"
                    subject_ids = _subject_ids_for_axis(meta, n_outer) if outer_role == "ids" else None
                    rows = []
                    for idx in range(n_outer):
                        meta_i = {
                            key: self._metadata_row_value(str(key), value, idx, n_outer)
                            for key, value in meta.items()
                        }
                        if subject_ids is not None:
                            meta_i["subject_id"] = subject_ids[idx]
                        rows.append(
                            self._row_from_series(
                                meta_i,
                                sensor_name,
                                idx if outer_role == "epochs" else None,
                                a2[idx, :],
                                fs,
                                data_domain,
                                freqs,
                                spectral_kind,
                                source_file,
                            )
                        )
                    return rows
            # assume (time, channels) by default for non-MNE numpy
            n_time, n_ch = a.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            return [self._row_from_series(meta, labels2[i], None, a[:, i], fs, data_domain, freqs, spectral_kind, source_file) for i in range(n_ch)]

        if a.ndim == 3:
            ax = f.array_axes
            if ax and "channels" in ax and "samples" in ax and "epochs" in ax:
                ep_ax = int(ax["epochs"])
                ch_ax = int(ax["channels"])
                sa_ax = int(ax["samples"])
                a3 = a.transpose(ep_ax, ch_ax, sa_ax)
                n_ep, n_ch = a3.shape[0], a3.shape[1]
                labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                rows = []
                for e in range(n_ep):
                    for i in range(n_ch):
                        rows.append(self._row_from_series(meta, labels2[i], e, a3[e, i, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows
            if ax and "channels" in ax and "samples" in ax and "ids" in ax:
                id_ax = int(ax["ids"])
                ch_ax = int(ax["channels"])
                sa_ax = int(ax["samples"])
                a3 = a.transpose(id_ax, ch_ax, sa_ax)
                n_ids, n_ch = a3.shape[0], a3.shape[1]
                labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = {
                        key: self._metadata_row_value(str(key), value, sid, n_ids)
                        for key, value in meta.items()
                    }
                    meta_i["subject_id"] = subject_ids[sid]
                    for i in range(n_ch):
                        rows.append(self._row_from_series(meta_i, labels2[i], None, a3[sid, i, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows
            if ax and "epochs" in ax and "samples" in ax and "ids" in ax:
                id_ax = int(ax["ids"])
                ep_ax = int(ax["epochs"])
                sa_ax = int(ax["samples"])
                a3 = a.transpose(id_ax, ep_ax, sa_ax)
                n_ids, n_ep = a3.shape[0], a3.shape[1]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                sensor_name = labels[0] if labels else "ch0"
                rows = []
                for sid in range(n_ids):
                    meta_i = {
                        key: self._metadata_row_value(str(key), value, sid, n_ids)
                        for key, value in meta.items()
                    }
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_ep):
                        rows.append(self._row_from_series(meta_i, sensor_name, e, a3[sid, e, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows
            # assume (epochs, time, channels)
            n_ep, n_time, n_ch = a.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            rows = []
            for e in range(n_ep):
                for i in range(n_ch):
                    rows.append(self._row_from_series(meta, labels2[i], e, a[e, :, i], fs, data_domain, freqs, spectral_kind, source_file))
            return rows

        if a.ndim == 4:
            ax = f.array_axes
            if not (ax and "channels" in ax and "samples" in ax):
                raise ValueError(
                    "4D ndarray inputs require explicit array_axes mapping with at least "
                    "'channels' and 'samples' (and typically 'ids' and/or 'epochs')."
                )

            ch_ax = int(ax["channels"])
            sa_ax = int(ax["samples"])
            id_ax = int(ax["ids"]) if "ids" in ax else -1
            ep_ax = int(ax["epochs"]) if "epochs" in ax else -1

            used = [ch_ax, sa_ax] + ([id_ax] if id_ax >= 0 else []) + ([ep_ax] if ep_ax >= 0 else [])
            if len(set(used)) != len(used):
                raise ValueError("array_axes contains duplicated axis indices.")
            if any(idx < 0 or idx >= a.ndim for idx in used):
                raise ValueError(f"array_axes indices must be in [0, {a.ndim - 1}] for shape {a.shape}.")

            remaining = [i for i in range(a.ndim) if i not in used]
            if len(remaining) > 0:
                raise ValueError(
                    "4D ndarray inputs must map all dimensions via array_axes "
                    "(use ids/epochs/channels/samples)."
                )

            if id_ax >= 0 and ep_ax >= 0:
                a4 = a.transpose(id_ax, ep_ax, ch_ax, sa_ax)
                n_ids, n_ep, n_ch, _ = a4.shape
                labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = dict(meta)
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_ep):
                        for i in range(n_ch):
                            rows.append(self._row_from_series(meta_i, labels2[i], e, a4[sid, e, i, :], fs, data_domain, freqs, spectral_kind, source_file))
                return rows

            if id_ax >= 0:
                extra_ax = [i for i in range(a.ndim) if i not in {id_ax, ch_ax, sa_ax}][0]
                a4 = a.transpose(id_ax, ch_ax, sa_ax, extra_ax)
                n_ids, n_ch, _, n_extra = a4.shape
                labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                subject_ids = _subject_ids_for_axis(meta, n_ids)
                rows = []
                for sid in range(n_ids):
                    meta_i = dict(meta)
                    meta_i["subject_id"] = subject_ids[sid]
                    for e in range(n_extra):
                        for i in range(n_ch):
                            rows.append(self._row_from_series(meta_i, labels2[i], e, a4[sid, i, :, e], fs, data_domain, freqs, spectral_kind, source_file))
                return rows

            if ep_ax >= 0:
                extra_ax = [i for i in range(a.ndim) if i not in {ep_ax, ch_ax, sa_ax}][0]
                a4 = a.transpose(ep_ax, ch_ax, sa_ax, extra_ax)
                n_ep, n_ch, _, n_extra = a4.shape
                labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
                rows = []
                for e in range(n_ep):
                    for sid in range(n_extra):
                        meta_i = dict(meta)
                        if "subject_id" not in meta_i:
                            meta_i["subject_id"] = sid
                        for i in range(n_ch):
                            rows.append(self._row_from_series(meta_i, labels2[i], e, a4[e, i, :, sid], fs, data_domain, freqs, spectral_kind, source_file))
                return rows

        raise ValueError(f"Unsupported ndarray ndim: {a.ndim}")

    # ---- MNE Raw / Epochs ----

    def _parse_mne_raw(self, raw: Any, source_file: Optional[str]) -> list[dict]:
        _require_mne("MNE parsing")
        f = self.config.fields

        # optional MNE operations
        if self.config.drop_bads and hasattr(raw, "drop_channels") and getattr(raw, "info", None) is not None:
            bads = list(getattr(raw.info, "bads", [])) if hasattr(raw.info, "bads") else raw.info.get("bads", [])
            if bads:
                raw = raw.copy().drop_channels(bads)

        if self.config.pick_types:
            # best-effort pick_types
            try:
                raw = raw.copy().pick_types(**self.config.pick_types)
            except Exception:
                if self.config.warn_unimplemented:
                    import warnings
                    warnings.warn("pick_types could not be applied to this Raw object.")

        if self.config.max_seconds is not None:
            try:
                raw = raw.copy().crop(tmin=0.0, tmax=float(self.config.max_seconds))
            except Exception:
                if self.config.warn_unimplemented:
                    import warnings
                    warnings.warn("max_seconds crop could not be applied to this Raw object.")

        data = self._resolve(raw, f.data)
        if data is None:
            # MNE Raw: auto-extract signal array when no locator is configured
            data = raw.get_data()
        elif callable(data):
            data = data()

        arr = np.asarray(data)  # expected (n_channels, n_times)

        fs = self._resolve(raw, f.fs)
        fs = float(fs) if fs is not None else float(getattr(raw.info, "sfreq", 0))

        ch_names = self._normalize_channel_names(self._resolve(raw, f.ch_names))
        if isinstance(ch_names, list):
            labels = ch_names
        else:
            labels = list(getattr(raw, "ch_names", []))

        data_domain = self._resolve(raw, f.data_domain) or "time"
        spectral_kind = self._resolve(raw, f.spectral_kind)
        freqs = self._resolve(raw, f.freqs)

        meta = self._resolve_metadata(raw)

        rows = []
        n_ch, n_time = arr.shape
        for i, ch in enumerate(labels[:n_ch]):
            y = arr[i, :]
            t0, t1 = self._infer_t0_t1(times=None, n=len(y), fs=fs)
            r = self._base_row(meta, source_file)
            r.update(
                sensor=str(ch),
                epoch=None,
                fs=fs,
                data=np.asarray(y),
                t0=t0,
                t1=t1,
                data_domain=data_domain,
                spectral_kind=spectral_kind,
            )
            if data_domain == "frequency":
                f0, f1 = self._infer_f0_f1(freqs=freqs, n=len(y))
                r.update(f0=f0, f1=f1)
            rows.append(r)
        return rows

    def _parse_mne_epochs(self, epochs: Any, source_file: Optional[str]) -> list[dict]:
        _require_mne("MNE parsing")
        f = self.config.fields

        data = self._resolve(epochs, f.data)
        if callable(data):
            # For MNE Epochs, `get_data` is typically a zero-arg bound method.
            # If the resolved object is callable, try calling it with no args first.
            try:
                data = data()
            except TypeError:
                # Fall back to calling with the epochs object for user-supplied callables
                data = data(epochs)

        arr = np.asarray(data)  # expected (n_epochs, n_channels, n_times)

        fs = self._resolve(epochs, f.fs)
        fs = float(fs) if fs is not None else float(getattr(epochs.info, "sfreq", epochs.info.get("sfreq")))

        ch_names = self._normalize_channel_names(self._resolve(epochs, f.ch_names))
        labels = ch_names if isinstance(ch_names, list) else list(getattr(epochs, "ch_names", []))

        data_domain = self._resolve(epochs, f.data_domain) or "time"
        spectral_kind = self._resolve(epochs, f.spectral_kind)
        freqs = self._resolve(epochs, f.freqs)

        meta = self._resolve_metadata(epochs)

        n_ep, n_ch, n_time = arr.shape
        rows = []
        for e in range(n_ep):
            for i, ch in enumerate(labels[:n_ch]):
                y = arr[e, i, :]
                t0, t1 = self._infer_t0_t1(times=None, n=len(y), fs=fs)
                r = self._base_row(meta, source_file)
                r.update(
                    sensor=str(ch),
                    epoch=e,
                    fs=fs,
                    data=np.asarray(y),
                    t0=t0,
                    t1=t1,
                    data_domain=data_domain,
                    spectral_kind=spectral_kind,
                )
                if data_domain == "frequency":
                    f0, f1 = self._infer_f0_f1(freqs=freqs, n=len(y))
                    r.update(f0=f0, f1=f1)
                rows.append(r)
        return rows


    # -------------------------
    # Row construction utilities
    # -------------------------

    def _try_ascii_decode(self, value: Any) -> Optional[str]:
        nums: list[int] = []

        def _collect(x: Any) -> bool:
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    try:
                        return _collect(x.item())
                    except Exception:
                        return False
                for item in x.tolist():
                    if not _collect(item):
                        return False
                return True
            if isinstance(x, (list, tuple)):
                for item in x:
                    if not _collect(item):
                        return False
                return True
            if isinstance(x, (np.integer, int)):
                nums.append(int(x))
                return True
            if isinstance(x, (np.floating, float)):
                if not np.isfinite(float(x)):
                    return False
                xi = int(float(x))
                if abs(float(x) - float(xi)) > 1e-9:
                    return False
                nums.append(xi)
                return True
            return False

        ok = _collect(value)
        if not ok or not nums:
            return None
        printable = [i for i in nums if i != 0]
        if len(printable) < 3:
            return None
        if not all(32 <= i <= 126 for i in printable):
            return None
        return "".join(chr(i) for i in printable).strip()

    def _coerce_channel_label_token(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, np.generic):
            try:
                value = value.item()
            except Exception:
                return None
        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8", errors="ignore").strip()
            except Exception:
                text = str(value).strip()
            return text if text else None
        if isinstance(value, str):
            text = value.strip()
            return text if text else None
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if not np.isfinite(float(value)):
                return None
            fv = float(value)
            return str(int(fv)) if fv.is_integer() else str(fv)
        if isinstance(value, (list, tuple, set, dict, np.ndarray)):
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text if text else None

    def _normalize_channel_names(self, raw: Any) -> Optional[list[str]]:
        if raw is None:
            return None
        if isinstance(raw, str):
            label = self._coerce_channel_label_token(raw)
            return [label] if label is not None else None

        if isinstance(raw, np.ndarray):
            arr = raw
            if arr.ndim == 0:
                label = self._coerce_channel_label_token(arr.item() if hasattr(arr, "item") else raw)
                return [label] if label is not None else None
            if arr.dtype.kind in {"U", "S"} and arr.ndim == 2:
                labels = []
                for row in arr.tolist():
                    token = "".join(str(ch) for ch in row).strip()
                    if token:
                        labels.append(token)
                return labels or None
            items = arr.reshape(-1).tolist()
        elif isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            label = self._coerce_channel_label_token(raw)
            return [label] if label is not None else None

        labels = []
        for item in items:
            label = self._coerce_channel_label_token(item)
            if label is not None:
                labels.append(label)
        return labels or None

    def _coerce_subject_id_token(self, value: Any) -> Any:
        """Normalize subject_id payloads from MATLAB/HDF5 into readable values."""
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore").strip()
            except Exception:
                return str(value)
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    decoded = self._try_ascii_decode(parsed)
                    if decoded:
                        return decoded
                except Exception:
                    pass
            return text

        decoded_direct = self._try_ascii_decode(value)
        if decoded_direct:
            return decoded_direct

        try:
            if np.isscalar(value):
                if isinstance(value, (np.integer, int)):
                    return int(value)
                if isinstance(value, (np.floating, float)):
                    if float(value).is_integer():
                        return int(value)
                    return float(value)
                return value
        except Exception:
            pass

        try:
            arr = np.asarray(value)
        except Exception:
            return str(value)

        if arr.size == 0:
            return None

        # Character arrays
        if arr.dtype.kind in {"U", "S"}:
            flat = [str(x) for x in arr.reshape(-1).tolist()]
            if all(len(x) == 1 for x in flat):
                return "".join(flat).strip()
            if len(flat) == 1:
                return flat[0].strip()
            return [x.strip() for x in flat]

        # Numeric arrays that encode ASCII chars
        try:
            nums = np.asarray(value, dtype=float).reshape(-1)
            ints = []
            ok = True
            for x in nums.tolist():
                if not np.isfinite(x):
                    ok = False
                    break
                xi = int(x)
                if abs(float(x) - float(xi)) > 1e-9:
                    ok = False
                    break
                ints.append(xi)
            if ok and ints:
                printable = [i for i in ints if i != 0]
                if printable and all(32 <= i <= 126 for i in printable):
                    return "".join(chr(i) for i in printable).strip()
                if len(ints) == 1:
                    return ints[0]
        except Exception:
            pass

        # Object arrays / nested cells
        try:
            obj_arr = np.asarray(value, dtype=object).reshape(-1)
            flat_tokens = [self._coerce_subject_id_token(item) for item in obj_arr.tolist()]
            if flat_tokens and all(isinstance(t, str) and len(t) == 1 for t in flat_tokens):
                return "".join(flat_tokens).strip()
            if len(flat_tokens) == 1:
                return flat_tokens[0]
            return flat_tokens
        except Exception:
            return str(value)

    def _normalize_subject_id_list(self, raw: Any, n_ids: int) -> list[Any]:
        if n_ids <= 0:
            return []
        if raw is None:
            return list(range(n_ids))
        arr = np.asarray(raw, dtype=object)
        if arr.ndim == 0:
            token = self._coerce_subject_id_token(arr.item() if hasattr(arr, "item") else raw)
            return [token for _ in range(n_ids)]
        flat = arr.reshape(-1).tolist()
        norm_flat = [self._coerce_subject_id_token(item) for item in flat]
        if len(norm_flat) == n_ids:
            return norm_flat
        if len(norm_flat) > 0:
            return [norm_flat[0] for _ in range(n_ids)]
        return list(range(n_ids))

    def _coerce_subject_id_metadata_value(self, value: Any) -> Any:
        """Keep multi-id containers as containers; decode element-wise."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(value, dtype=object)
            except Exception:
                return self._coerce_subject_id_token(value)
            if arr.ndim == 0:
                try:
                    return self._coerce_subject_id_token(arr.item())
                except Exception:
                    return self._coerce_subject_id_token(value)
            return [self._coerce_subject_id_token(item) for item in arr.reshape(-1).tolist()]
        return self._coerce_subject_id_token(value)

    def _metadata_row_value(self, key: str, value: Any, index: int, total: int) -> Any:
        if key == "subject_id":
            values = self._normalize_subject_id_list(value, total)
            return values[index] if index < len(values) else None

        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(value, dtype=object)
            except Exception:
                return value
            if arr.ndim == 0:
                return arr.item() if hasattr(arr, "item") else value
            flat = arr.reshape(-1).tolist()
            if len(flat) == total:
                return flat[index]
            if len(flat) > 0:
                return flat[0]
        return value

    def _resolve_metadata(self, obj: Any) -> dict:
        out: dict = {}
        HAS_PANDAS = tools.ensure_module("pandas")
        if HAS_PANDAS:
            import pandas as pd  # type: ignore

        for k, loc in self.config.fields.metadata.items():
            try:
                v = self._resolve(obj, loc)

                if str(k) != "subject_id" and isinstance(v, list) and len(v) == 1:
                    # Collapse singleton list metadata but preserve aligned per-row sequences.
                    v = v[0]

                # If metadata resolves to a pandas Series (e.g., df["group"]),
                # reduce it to a scalar if it's constant across rows.
                if HAS_PANDAS:
                    import pandas as pd  # type: ignore
                    if isinstance(v, pd.Series):
                        if str(k) == "subject_id":
                            v = v.to_list()
                        else:
                            # If constant (including all-NaN), take the first value
                            if v.nunique(dropna=False) <= 1:
                                v = v.iloc[0] if len(v) > 0 else None
                            else:
                                # Non-constant metadata column: keep as list (or raise, your choice)
                                v = v.to_list()
                if str(k) == "subject_id":
                    v = self._coerce_subject_id_metadata_value(v)

                out[k] = v
            except Exception:
                out[k] = loc if not callable(loc) else None
        return out


    def _base_row(self, meta: Mapping[str, Any], source_file: Optional[str]) -> dict:
        row = {c: None for c in DEFAULT_COLUMNS}
        for k, v in meta.items():
            if k in row:
                row[k] = v
            else:
                # allow extra metadata without breaking
                row[k] = v
        if "subject_id" in row:
            sid = row["subject_id"]
            if isinstance(sid, (list, tuple, np.ndarray)):
                sid_list = self._normalize_subject_id_list(sid, 1)
                row["subject_id"] = sid_list[0] if sid_list else None
            else:
                row["subject_id"] = self._coerce_subject_id_token(sid)
        row["source_file"] = source_file
        return row

    def _row_from_series(
        self,
        meta: Mapping[str, Any],
        sensor: str,
        epoch: Optional[Any],
        y: np.ndarray,
        fs: Optional[float],
        data_domain: Any,
        freqs: Any,
        spectral_kind: Any,
        source_file: Optional[str],
    ) -> dict:
        t0, t1 = self._infer_t0_t1(times=None, n=len(y), fs=fs if data_domain == "time" else None)
        r = self._base_row(meta, source_file)
        r.update(
            sensor=str(sensor),
            epoch=epoch,
            fs=fs,
            data=np.asarray(y),
            t0=t0 if data_domain == "time" else None,
            t1=t1 if data_domain == "time" else None,
            data_domain=data_domain or "time",
            spectral_kind=spectral_kind,
        )
        if (data_domain or "time") == "frequency":
            f0, f1 = self._infer_f0_f1(freqs=freqs, n=len(y))
            r.update(f0=f0, f1=f1)
        return r

    def _infer_t0_t1(self, times: Optional[np.ndarray], n: int, fs: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if times is not None and len(times) > 0:
            return float(times[0]), float(times[-1])
        if fs is not None and fs > 0 and n > 0:
            return 0.0, float((n - 1) / fs)
        return None, None

    def _infer_f0_f1(self, freqs: Any, n: int) -> Tuple[Optional[float], Optional[float]]:
        if freqs is None:
            return None, None
        a = np.asarray(freqs).astype(float)
        if a.size == 0:
            return None, None
        return float(a.min()), float(a.max())

    # -------------------------
    # Output DF + post-processing
    # -------------------------

    def _rows_to_df(self, rows: list[dict]) -> "Any":
        if not tools.ensure_module("pandas"):
            raise ImportError("pandas is required to return a DataFrame output.")
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)

        # Ensure all default columns exist
        for c in DEFAULT_COLUMNS:
            if c not in df.columns:
                df[c] = None

        # Keep default columns first; allow extras after
        cols = [c for c in DEFAULT_COLUMNS if c in df.columns] + [c for c in df.columns if c not in DEFAULT_COLUMNS]
        return df[cols]


    def _apply_epoching_rows(self, rows: list[dict]) -> list[dict]:
        """Split each time-domain row into fixed-length epochs. Simple sliding window."""
        L = self.config.epoch_length_s
        if L is None:
            return rows

        out: list[dict] = []
        for row in rows:
            data_domain = row.get("data_domain") or "time"
            if data_domain != "time":
                out.append(row)
                continue

            fs = row.get("fs")
            x = row.get("data")

            # Need fs and a 1D array to epoch
            if fs is None or x is None:
                out.append(row)
                continue

            x = np.asarray(x)
            if x.ndim != 1:
                # your schema assumes each row is a single 1D series; keep as-is
                out.append(row)
                continue

            win = int(round(float(L) * float(fs)))
            if win <= 0 or win > x.shape[0]:
                out.append(row)
                continue

            step_s = self.config.epoch_step_s if self.config.epoch_step_s is not None else L
            step = int(round(float(step_s) * float(fs)))
            if step <= 0:
                step = win

            base_epoch = row.get("epoch")
            is_missing = (
                    base_epoch is None
                    or (isinstance(base_epoch, float) and np.isnan(base_epoch))
            )

            # Use a string epoch id if base_epoch already exists
            base_t0 = row.get("t0")
            try:
                base_t0 = float(base_t0) if base_t0 is not None else 0.0
            except Exception:
                base_t0 = 0.0
            for k, start in enumerate(range(0, x.shape[0] - win + 1, step)):
                seg = x[start: start + win]

                r = dict(row)  # shallow copy
                r["data"] = seg

                # epoch id
                if is_missing:
                    r["epoch"] = k
                else:
                    r["epoch"] = f"{base_epoch}:{k}"

                # time bounds for this epoch
                r["t0"] = float(base_t0 + (start / fs))
                r["t1"] = float(base_t0 + ((start + win - 1) / fs))

                out.append(r)

        return out

    def _apply_time_segment_rows(self, rows: list[dict]) -> list[dict]:
        """Crop each time-domain row to [segment_t0_s, segment_t1_s] before epoching."""
        t0_cfg = self.config.segment_t0_s
        t1_cfg = self.config.segment_t1_s
        if t0_cfg is None and t1_cfg is None:
            return rows

        if t0_cfg is not None:
            t0_cfg = float(t0_cfg)
        if t1_cfg is not None:
            t1_cfg = float(t1_cfg)
        if t0_cfg is not None and t0_cfg < 0:
            raise ValueError("Temporal segmentation requires non-negative segment_t0_s.")
        if t1_cfg is not None and t1_cfg < 0:
            raise ValueError("Temporal segmentation requires non-negative segment_t1_s.")
        if t0_cfg is not None and t1_cfg is not None and t1_cfg <= t0_cfg:
            raise ValueError("Temporal segmentation requires segment_t1_s > segment_t0_s.")

        out: list[dict] = []
        has_segmentable_time_row = False
        has_overlapping_time_row = False
        eps = 1e-12
        for row in rows:
            data_domain = row.get("data_domain") or "time"
            if data_domain != "time":
                out.append(row)
                continue

            x = row.get("data")
            if x is None:
                out.append(row)
                continue
            x = np.asarray(x)
            if x.ndim != 1 or x.size == 0:
                out.append(row)
                continue

            n = int(x.shape[0])
            fs = row.get("fs", None)
            fs_val: Optional[float] = None
            try:
                fs_float = float(fs)
                if np.isfinite(fs_float) and fs_float > 0:
                    fs_val = fs_float
            except Exception:
                fs_val = None

            row_t0 = row.get("t0", None)
            row_t1 = row.get("t1", None)
            try:
                row_t0_val = float(row_t0) if row_t0 is not None else 0.0
            except Exception:
                row_t0_val = 0.0
            try:
                row_t1_val = float(row_t1) if row_t1 is not None else None
            except Exception:
                row_t1_val = None

            if row_t1_val is None:
                if fs_val is not None and n > 0:
                    row_t1_val = float(row_t0_val + (n - 1) / fs_val)
                else:
                    # Cannot infer temporal support: keep as-is.
                    out.append(row)
                    continue

            has_segmentable_time_row = True
            seg_t0 = t0_cfg if t0_cfg is not None else row_t0_val
            seg_t1 = t1_cfg if t1_cfg is not None else row_t1_val
            # Clip requested interval to this row's support.
            clip_t0 = max(float(seg_t0), float(row_t0_val))
            clip_t1 = min(float(seg_t1), float(row_t1_val))
            if clip_t1 <= clip_t0:
                # No overlap for this row: drop it.
                continue

            if fs_val is not None:
                i0 = int(np.ceil(((clip_t0 - row_t0_val) * fs_val) - eps))
                i1 = int(np.floor(((clip_t1 - row_t0_val) * fs_val) + eps))
                i0 = max(0, min(i0, n - 1))
                i1 = max(0, min(i1, n - 1))
                if i1 < i0:
                    continue
                y = x[i0:i1 + 1]
                r = dict(row)
                r["data"] = np.asarray(y)
                r["t0"] = float(row_t0_val + (i0 / fs_val))
                r["t1"] = float(row_t0_val + (i1 / fs_val))
                out.append(r)
                has_overlapping_time_row = True
                continue

            # Fallback when fs is missing: infer dt from row t0/t1 and array length.
            if n <= 1:
                if clip_t0 <= row_t0_val <= clip_t1:
                    r = dict(row)
                    r["data"] = np.asarray(x[:1])
                    r["t0"] = float(row_t0_val)
                    r["t1"] = float(row_t0_val)
                    out.append(r)
                    has_overlapping_time_row = True
                continue

            dt = (row_t1_val - row_t0_val) / float(n - 1)
            if not np.isfinite(dt) or dt <= 0:
                out.append(row)
                continue

            i0 = int(np.ceil(((clip_t0 - row_t0_val) / dt) - eps))
            i1 = int(np.floor(((clip_t1 - row_t0_val) / dt) + eps))
            i0 = max(0, min(i0, n - 1))
            i1 = max(0, min(i1, n - 1))
            if i1 < i0:
                continue
            y = x[i0:i1 + 1]
            r = dict(row)
            r["data"] = np.asarray(y)
            r["t0"] = float(row_t0_val + i0 * dt)
            r["t1"] = float(row_t0_val + i1 * dt)
            out.append(r)
            has_overlapping_time_row = True

        if has_segmentable_time_row and not has_overlapping_time_row:
            raise ValueError(
                "Temporal segmentation window does not overlap with any time-domain samples in the loaded data."
            )

        return out


    def _apply_zscore(self, df: "Any") -> "Any":
        # zscore each row's `data` vector independently
        def z(x):
            a = np.asarray(x, dtype=float)
            if a.size == 0:
                return a
            mu = float(a.mean())
            sd = float(a.std(ddof=0))
            if sd == 0.0:
                return a * 0.0
            return (a - mu) / sd

        df = df.copy()
        df["data"] = df["data"].apply(z)
        return df

    def _exclude_last_epoch(self, df: "Any") -> "Any":
        """Exclude the last epoch for each unique combination of grouping columns.

        This is useful for discarding the last (potentially incomplete) epoch
        from each time series when the signal length is not perfectly divisible
        by the epoch length.
        """
        if "epoch" not in df.columns:
            return df

        # Group by columns that identify unique time series (e.g., subject_id, sensor)
        group_cols = [c for c in ["subject_id", "sensor"] if c in df.columns]

        if not group_cols:
            # No grouping columns, just exclude the global max epoch
            max_epoch = df["epoch"].max()
            return df[df["epoch"] < max_epoch].reset_index(drop=True)

        # For each group, find the max epoch and exclude it
        max_epochs = df.groupby(group_cols)["epoch"].transform("max")
        return df[df["epoch"] < max_epochs].reset_index(drop=True)

    def _exclude_incomplete_final_epoch(self, df: "Any") -> "Any":
        """Exclude final epochs retrospectively when any final epoch is shorter."""
        if "epoch" not in df.columns or "data" not in df.columns or df.empty:
            return df
        # When epoching is performed by this parser, windows are emitted only when complete
        # (`range(0, n - win + 1, step)`), so no incomplete final epoch can be produced.
        if self.config.epoch_length_s is not None:
            return df

        def data_length(value):
            if value is None:
                return None
            try:
                if isinstance(value, float) and np.isnan(value):
                    return None
            except Exception:
                pass
            try:
                arr = np.asarray(value)
            except Exception:
                return None
            if arr.ndim == 0:
                return 1
            return int(arr.size)

        def ordered_group(frame):
            if "t0" in frame.columns:
                try:
                    t0_values = frame["t0"].astype(float)
                    if np.isfinite(t0_values).any():
                        return frame.assign(_epoch_sort_t0=t0_values).sort_values("_epoch_sort_t0").drop(columns=["_epoch_sort_t0"])
                except Exception:
                    pass
            return frame.sort_index()

        def is_incomplete(frame):
            ordered = ordered_group(frame)
            lengths = [data_length(value) for value in ordered["data"].tolist()]
            if len(lengths) < 2:
                return False
            last_len = lengths[-1]
            previous = [length for length in lengths[:-1] if length is not None]
            if last_len is None or not previous:
                return False
            counts = {}
            for length in previous:
                counts[int(length)] = counts.get(int(length), 0) + 1
            typical_len = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            return int(last_len) < int(typical_len)

        group_cols = [
            col for col in [
                "source_file",
                "subject_id",
                "species",
                "group",
                "condition",
                "sensor",
                "recording_type",
                "fs",
                "data_domain",
                "spectral_kind",
            ]
            if col in df.columns and not df[col].isna().all()
        ]
        grouped_df, group_cols = self._with_hashable_group_columns(df, group_cols)
        grouped = grouped_df.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, df)]
        any_incomplete_group = False
        drop_indices = []
        for _, frame in grouped:
            if is_incomplete(frame):
                any_incomplete_group = True
            ordered = ordered_group(frame)
            if ordered.empty:
                continue
            last_epoch = ordered.iloc[-1]["epoch"]
            last_epoch_missing = last_epoch is None
            try:
                if isinstance(last_epoch, float) and np.isnan(last_epoch):
                    last_epoch_missing = True
            except Exception:
                pass
            if last_epoch_missing:
                drop_indices.append(ordered.index[-1])
                continue
            drop_indices.extend(list(frame.index[frame["epoch"].map(lambda value: value == last_epoch)]))

        if not any_incomplete_group or not drop_indices:
            return df
        return df.drop(index=list(set(drop_indices))).reset_index(drop=True)

    def _limit_epoch_count_to_minimum(self, df: "Any") -> "Any":
        """Limit all epoch series to the smallest epoch count produced."""
        if "epoch" not in df.columns or df.empty:
            return df

        valid_epoch = df["epoch"].map(lambda value: not self._is_missing_value(value))
        if not bool(valid_epoch.any()):
            return df

        group_cols = [
            col for col in [
                "source_file",
                "subject_id",
                "species",
                "group",
                "condition",
                "sensor",
                "recording_type",
                "fs",
                "data_domain",
                "spectral_kind",
            ]
            if col in df.columns and not df[col].isna().all()
        ]
        grouped_df, group_cols = self._with_hashable_group_columns(df, group_cols)

        # Fast numeric path for common epoch labels produced by parser epoching.
        # This avoids per-group Python loops on large non-aggregated datasets.
        if not tools.ensure_module("pandas"):
            return df
        import pandas as pd  # type: ignore
        try:
            epoch_num = pd.to_numeric(df["epoch"], errors="coerce")
            has_numeric_epochs = bool(epoch_num.notna().all())
        except Exception:
            epoch_num = None
            has_numeric_epochs = False

        if has_numeric_epochs:
            df_fast = df.assign(_epoch_num=epoch_num)
            # Very common case for parser-created epoch indices: per-group epochs are
            # contiguous and start at 0. In that case, aligning epoch counts is just
            # clipping by the smallest per-group max epoch index.
            if group_cols:
                per_group_max = (
                    df_fast.assign(**grouped_df[group_cols])
                    .groupby(group_cols, sort=False, dropna=False)["_epoch_num"]
                    .max()
                )
            else:
                per_group_max = pd.Series([float(df_fast["_epoch_num"].max())])

            if per_group_max.empty or int(per_group_max.shape[0]) < 2:
                return df

            min_max = float(per_group_max.min())
            max_max = float(per_group_max.max())
            if not np.isfinite(min_max) or not np.isfinite(max_max):
                return df
            if min_max < 0 or min_max == max_max:
                return df

            # Keep epochs up to smallest available max index across groups.
            keep_mask = df_fast["_epoch_num"] <= min_max

            import warnings
            warnings.warn(
                "Epoch count was limited to the smallest available count "
                f"({int(min_max + 1)}; max was {int(max_max + 1)}) because series produced different "
                "numbers of epochs after epoching.",
                RuntimeWarning,
            )
            return df_fast.loc[keep_mask].drop(columns=["_epoch_num"]).reset_index(drop=True)

        def epoch_sort_key(value):
            if self._is_missing_value(value):
                return (1, 0, "")
            try:
                return (0, float(value), str(value))
            except Exception:
                text = str(value)
                tail = text.rsplit(":", 1)[-1]
                try:
                    return (0, float(tail), text)
                except Exception:
                    return (0, 0, text)

        def ordered_epoch_values(frame):
            seen = set()
            values = []
            for value in frame["epoch"].tolist():
                if self._is_missing_value(value):
                    continue
                marker = str(value)
                if marker in seen:
                    continue
                seen.add(marker)
                values.append(value)
            return sorted(values, key=epoch_sort_key)

        grouped = grouped_df.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, df)]
        num_groups_with_epochs = 0
        min_count = None
        max_count = 0
        for _, frame in grouped:
            values = ordered_epoch_values(frame)
            if not values:
                continue
            num_groups_with_epochs += 1
            count = len(values)
            if min_count is None or count < min_count:
                min_count = count
            if count > max_count:
                max_count = count

        if num_groups_with_epochs < 2 or min_count is None:
            return df

        if min_count <= 0 or min_count == max_count:
            return df

        drop_indices = []
        grouped = grouped_df.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, df)]
        for _, frame in grouped:
            values = ordered_epoch_values(frame)
            if not values:
                continue
            keep_markers = {str(value) for value in values[:min_count]}
            drop_indices.extend(
                list(frame.index[frame["epoch"].map(lambda value: str(value) not in keep_markers)])
            )

        if not drop_indices:
            return df

        import warnings
        warnings.warn(
            "Epoch count was limited to the smallest available count "
            f"({min_count}; max was {max_count}) because series produced different "
            "numbers of epochs after epoching.",
            RuntimeWarning,
        )
        return df.drop(index=list(set(drop_indices))).reset_index(drop=True)

    def _with_hashable_group_columns(self, df: "Any", group_cols: list[str]) -> tuple["Any", list[str]]:
        """Return a view-like copy whose group columns can be used by pandas.groupby."""
        if not group_cols:
            return df, group_cols

        def make_hashable(value):
            if self._is_missing_value(value):
                return value
            if isinstance(value, np.ndarray):
                return tuple(make_hashable(v) for v in value.tolist())
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, tuple):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, dict):
                return tuple(
                    (make_hashable(k), make_hashable(v))
                    for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                )
            try:
                hash(value)
            except TypeError:
                return repr(value)
            return value

        out = df.copy()
        for col in group_cols:
            out[col] = out[col].map(make_hashable)
        return out, group_cols

    def _is_missing_value(self, value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(isinstance(value, float) and np.isnan(value))
        except Exception:
            return False

    def _as_array_excluding_incomplete_final_epoch(self, data: Any) -> np.ndarray:
        """Convert data to an array, dropping a ragged final epoch when needed."""
        try:
            return np.asarray(data)
        except ValueError as exc:
            trimmed, changed = self._drop_ragged_incomplete_final_epoch(data)
            if not changed:
                raise exc
            try:
                return np.asarray(trimmed)
            except ValueError:
                raise exc

    def _squeeze_unmapped_singleton_axes(
        self,
        arr: np.ndarray,
        axes: Optional[Mapping[str, int]],
    ) -> tuple[np.ndarray, Optional[Mapping[str, int]]]:
        """Drop MATLAB-style singleton dimensions that are not part of array_axes."""
        if arr.ndim <= 1:
            return arr, axes

        mapped_axes: set[int] = set()
        if axes:
            for axis in axes.values():
                try:
                    axis_idx = int(axis)
                except Exception:
                    continue
                if axis_idx >= 0:
                    mapped_axes.add(axis_idx)

        squeeze_axes = [
            idx for idx, size in enumerate(arr.shape)
            if int(size) == 1 and idx not in mapped_axes
        ]
        if not squeeze_axes:
            return arr, axes

        squeezed = np.squeeze(arr, axis=tuple(squeeze_axes))
        if not axes:
            return squeezed, axes

        adjusted: dict[str, int] = {}
        for role, axis in axes.items():
            try:
                axis_idx = int(axis)
            except Exception:
                continue
            if axis_idx < 0:
                adjusted[str(role)] = axis_idx
                continue
            adjusted[str(role)] = axis_idx - sum(1 for removed in squeeze_axes if removed < axis_idx)
        return squeezed, adjusted

    def _normalize_array_axes_for_data(
        self,
        arr: np.ndarray,
        axes: Optional[Mapping[str, int]],
        ch_names: Any,
    ) -> Optional[Mapping[str, int]]:
        """Correct obvious channel/sample axis swaps using resolved channel names."""
        if not axes:
            return axes

        adjusted: dict[str, int] = {}
        for role, axis in axes.items():
            try:
                adjusted[str(role)] = int(axis)
            except Exception:
                continue

        labels = ch_names if isinstance(ch_names, list) else None
        if labels:
            ch_count = len(labels)
            channel_matches = [
                idx for idx, size in enumerate(arr.shape)
                if int(size) == int(ch_count)
            ]
            current_ch = adjusted.get("channels", -1)
            current_ch_ok = (
                isinstance(current_ch, int)
                and 0 <= current_ch < arr.ndim
                and int(arr.shape[current_ch]) == int(ch_count)
            )
            if channel_matches and not current_ch_ok:
                adjusted["channels"] = channel_matches[0]

        used_non_sample = {
            axis for role, axis in adjusted.items()
            if role != "samples" and axis >= 0
        }
        sample_axis = adjusted.get("samples", -1)
        sample_axis_ok = (
            isinstance(sample_axis, int)
            and 0 <= sample_axis < arr.ndim
            and sample_axis not in used_non_sample
        )
        if not sample_axis_ok:
            candidates = [idx for idx in range(arr.ndim) if idx not in used_non_sample]
            non_singleton = [idx for idx in candidates if int(arr.shape[idx]) > 1]
            candidates = non_singleton or candidates
            if candidates:
                adjusted["samples"] = max(candidates, key=lambda idx: int(arr.shape[idx]))

        return adjusted

    def _drop_ragged_incomplete_final_epoch(self, data: Any) -> tuple[Any, bool]:
        def sequence_items(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return None
                return list(value)
            if isinstance(value, (list, tuple)):
                return list(value)
            return None

        def max_sequence_depth(value, depth=0):
            items = sequence_items(value)
            if items is None or not items:
                return depth
            return max(max_sequence_depth(item, depth + 1) for item in items)

        def try_drop_at_epoch_axis(epoch_axis, sample_axis=None):
            try:
                epoch_axis = int(epoch_axis)
            except Exception:
                return data, False
            if epoch_axis < 0:
                return data, False

            def sample_length(epoch_value):
                try:
                    arr = np.asarray(epoch_value)
                    if sample_axis is not None:
                        local_axis = int(sample_axis) - epoch_axis - 1
                        if 0 <= local_axis < arr.ndim:
                            return int(arr.shape[local_axis])
                    if arr.ndim == 0:
                        return 1
                    return int(arr.size)
                except Exception:
                    items = sequence_items(epoch_value)
                    return len(items) if items is not None else None

            epoch_sequences = []

            def collect(node, depth):
                items = sequence_items(node)
                if items is None:
                    return
                if depth == epoch_axis:
                    epoch_sequences.append(items)
                    return
                for item in items:
                    collect(item, depth + 1)

            collect(data, 0)
            if not epoch_sequences:
                return data, False

            should_drop = False
            for seq in epoch_sequences:
                if len(seq) < 2:
                    continue
                lengths = [sample_length(item) for item in seq]
                last_len = lengths[-1]
                previous = [length for length in lengths[:-1] if length is not None]
                if last_len is None or not previous:
                    continue
                counts = {}
                for length in previous:
                    counts[int(length)] = counts.get(int(length), 0) + 1
                typical_len = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
                if int(last_len) < int(typical_len):
                    should_drop = True
                    break

            if not should_drop:
                return data, False

            def trim(node, depth):
                items = sequence_items(node)
                if items is None:
                    return node
                if depth == epoch_axis:
                    return items[:-1]
                return [trim(item, depth + 1) for item in items]

            return trim(data, 0), True

        axes = self.config.fields.array_axes or {}
        if "epochs" in axes:
            sample_axis = None
            try:
                sample_axis = int(axes["samples"]) if "samples" in axes else None
            except Exception:
                sample_axis = None
            return try_drop_at_epoch_axis(axes["epochs"], sample_axis=sample_axis)

        try:
            depth = max_sequence_depth(data)
        except Exception:
            depth = 0

        for candidate_axis in range(max(0, depth)):
            trimmed, changed = try_drop_at_epoch_axis(candidate_axis)
            if not changed:
                continue
            try:
                np.asarray(trimmed)
            except ValueError:
                continue
            return trimmed, True

        return data, False

    def _parse_ragged_dict_data(
        self,
        data: Any,
        meta: Mapping[str, Any],
        ch_names: Any,
        epoch_val: Any,
        fs: Optional[float],
        data_domain: Any,
        freqs: Any,
        spectral_kind: Any,
        source_file: Optional[str],
    ) -> Optional[list[dict]]:
        """Parse nested list/cell-array data whose sample vectors have variable length."""

        def sequence_items(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return None
                return list(value)
            if isinstance(value, (list, tuple)):
                return list(value)
            return None

        def axis_value(name):
            axes = self.config.fields.array_axes or {}
            if name not in axes:
                return None
            try:
                axis = int(axes[name])
            except Exception:
                return None
            return axis if axis >= 0 else None

        labels = ch_names if isinstance(ch_names, list) else None

        def channel_label(index: Optional[int]) -> str:
            if index is None:
                return str(labels[0]) if labels else "ch0"
            if labels and 0 <= int(index) < len(labels):
                return str(labels[int(index)])
            return f"ch{int(index)}"

        def series_from_node(node):
            try:
                series = np.asarray(node)
            except ValueError:
                try:
                    series = np.asarray(node, dtype=object)
                except Exception:
                    return None
            series = np.squeeze(series)
            if series.ndim == 0:
                return None
            if series.dtype == object:
                flat = np.ravel(series)
                if any(sequence_items(item) is not None for item in flat):
                    return None
            if series.ndim != 1:
                return None
            return np.asarray(series)

        def parse_with_axes(axis_map: Mapping[str, Any]) -> Optional[list[dict]]:
            sample_axis = axis_map.get("samples")
            if sample_axis is None:
                return None
            sample_axis = int(sample_axis)

            roles_by_axis: dict[int, str] = {}
            for role in ("ids", "channels", "epochs"):
                axis = axis_map.get(role)
                if axis is None:
                    continue
                axis = int(axis)
                if axis == sample_axis:
                    return None
                roles_by_axis[axis] = role

            for axis in range(sample_axis):
                if axis in roles_by_axis:
                    continue
                if "epochs" not in roles_by_axis.values():
                    roles_by_axis[axis] = "epochs"
                elif "ids" not in roles_by_axis.values():
                    roles_by_axis[axis] = "ids"

            rows: list[dict] = []

            def visit(node, depth: int, context: dict[str, Any], current_meta: Mapping[str, Any]):
                if depth == sample_axis:
                    series = series_from_node(node)
                    if series is None:
                        return
                    rows.append(
                        self._row_from_series(
                            current_meta,
                            channel_label(context.get("channels")),
                            context.get("epochs", epoch_val),
                            series,
                            fs,
                            data_domain,
                            freqs,
                            spectral_kind,
                            source_file,
                        )
                    )
                    return

                items = sequence_items(node)
                if items is None:
                    return

                role = roles_by_axis.get(depth)
                subject_ids = None
                if role == "ids":
                    subject_ids = self._normalize_subject_id_list(current_meta.get("subject_id", None), len(items))

                for index, item in enumerate(items):
                    next_context = dict(context)
                    next_meta = current_meta
                    if role == "ids":
                        next_meta = {
                            key: self._metadata_row_value(str(key), value, index, len(items))
                            for key, value in current_meta.items()
                        }
                        if subject_ids is not None:
                            next_meta["subject_id"] = subject_ids[index]
                        next_context["ids"] = index
                    elif role == "channels":
                        next_context["channels"] = index
                    elif role == "epochs":
                        next_context["epochs"] = index
                    visit(item, depth + 1, next_context, next_meta)

            visit(data, 0, {}, meta)
            return rows if rows else None

        candidates: list[dict[str, int]] = []
        explicit_axes = {
            name: axis
            for name, axis in {
                "samples": axis_value("samples"),
                "ids": axis_value("ids"),
                "channels": axis_value("channels"),
                "epochs": axis_value("epochs"),
            }.items()
            if axis is not None
        }
        if "samples" in explicit_axes:
            candidates.append(explicit_axes)

        inferred_axes = self._infer_ragged_axes(data, ch_names, meta)
        if inferred_axes and inferred_axes not in candidates:
            candidates.append(inferred_axes)

        for axis_map in candidates:
            rows = parse_with_axes(axis_map)
            if rows:
                return rows

        return None

    def _parse_condition_sequence_data(
        self,
        data: Any,
        meta: Mapping[str, Any],
        ch_names: Any,
        epoch_val: Any,
        fs_value: Any,
        data_domain: Any,
        freqs: Any,
        spectral_kind: Any,
        source_file: Optional[str],
    ) -> Optional[list[dict]]:
        """Parse MATLAB struct-array fields resolved as one LFP array per condition."""

        def sequence_items(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.dtype != object:
                    return None
                return list(value.flat)
            if isinstance(value, (list, tuple)):
                return list(value)
            return None

        items = sequence_items(data)
        if not items:
            return None

        condition_arrays = []
        for item in items:
            try:
                arr = np.asarray(item)
            except Exception:
                return None
            arr = np.squeeze(arr)
            if arr.ndim < 2:
                return None
            condition_arrays.append(arr)

        # A plain list of 1D channels is not a struct-array condition list.
        if not any(arr.ndim >= 3 for arr in condition_arrays):
            return None

        rows: list[dict] = []
        total = len(condition_arrays)
        for condition_idx, arr in enumerate(condition_arrays):
            meta_i = {
                key: self._metadata_row_value(str(key), value, condition_idx, total)
                for key, value in meta.items()
            }
            fs_i = self._value_at_index(fs_value, condition_idx, total)
            fs_i = self._coerce_optional_float(fs_i)
            epoch_i = self._value_at_index(epoch_val, condition_idx, total)
            rows.extend(
                self._rows_from_condition_array(
                    arr,
                    meta_i,
                    ch_names,
                    epoch_i,
                    fs_i,
                    data_domain,
                    freqs,
                    spectral_kind,
                    source_file,
                )
            )

        return rows if rows else None

    def _rows_from_condition_array(
        self,
        arr: np.ndarray,
        meta: Mapping[str, Any],
        ch_names: Any,
        epoch_val: Any,
        fs: Optional[float],
        data_domain: Any,
        freqs: Any,
        spectral_kind: Any,
        source_file: Optional[str],
    ) -> list[dict]:
        arr = np.asarray(arr).squeeze()
        labels = ch_names if isinstance(ch_names, list) else None

        def choose_axes(shape: tuple[int, ...]) -> tuple[Optional[int], Optional[int], Optional[int]]:
            axes = self.config.fields.array_axes or {}
            ch_ax = self._valid_axis(axes.get("channels"), len(shape))
            ep_ax = self._valid_axis(axes.get("epochs"), len(shape))
            sa_ax = self._valid_axis(axes.get("samples"), len(shape))

            if labels:
                matches = [idx for idx, size in enumerate(shape) if int(size) == len(labels)]
                if matches and (ch_ax is None or int(shape[ch_ax]) != len(labels)):
                    ch_ax = matches[0]

            if sa_ax is None or sa_ax == ch_ax:
                candidates = [idx for idx in range(len(shape)) if idx != ch_ax]
                if candidates:
                    sa_ax = max(candidates, key=lambda idx: int(shape[idx]))

            if ch_ax is None:
                candidates = [idx for idx, size in enumerate(shape) if int(size) > 1 and idx != sa_ax]
                if candidates:
                    ch_ax = min(candidates, key=lambda idx: int(shape[idx]))
            elif not labels:
                candidates = [idx for idx, size in enumerate(shape) if int(size) > 1 and idx != sa_ax]
                if candidates:
                    inferred_ch_ax = min(candidates, key=lambda idx: int(shape[idx]))
                    if int(shape[inferred_ch_ax]) < int(shape[ch_ax]):
                        ch_ax = inferred_ch_ax

            used = {idx for idx in (ch_ax, sa_ax) if idx is not None}
            if ep_ax is None or ep_ax in used:
                candidates = [idx for idx in range(len(shape)) if idx not in used]
                ep_ax = candidates[0] if candidates else None

            return ch_ax, ep_ax, sa_ax

        if arr.ndim == 2:
            ch_ax, _, sa_ax = choose_axes(arr.shape)
            if ch_ax is None or sa_ax is None or ch_ax == sa_ax:
                return []
            arr2 = arr.transpose(ch_ax, sa_ax)
            n_ch, _ = arr2.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            return [
                self._row_from_series(meta, labels2[i], epoch_val, arr2[i, :], fs, data_domain, freqs, spectral_kind, source_file)
                for i in range(n_ch)
            ]

        if arr.ndim == 3:
            ch_ax, ep_ax, sa_ax = choose_axes(arr.shape)
            if ch_ax is None or ep_ax is None or sa_ax is None:
                return []
            if len({ch_ax, ep_ax, sa_ax}) != 3:
                return []
            arr3 = arr.transpose(ep_ax, ch_ax, sa_ax)
            n_ep, n_ch, _ = arr3.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            rows = []
            for e in range(n_ep):
                row_epoch = e if self._is_missing_value(epoch_val) else f"{epoch_val}:{e}"
                for i, ch in enumerate(labels2):
                    rows.append(
                        self._row_from_series(
                            meta,
                            ch,
                            row_epoch,
                            arr3[e, i, :],
                            fs,
                            data_domain,
                            freqs,
                            spectral_kind,
                            source_file,
                        )
                    )
            return rows

        return []

    def _valid_axis(self, axis: Any, ndim: int) -> Optional[int]:
        try:
            axis = int(axis)
        except Exception:
            return None
        if axis < 0 or axis >= ndim:
            return None
        return axis

    def _value_at_index(self, value: Any, index: int, total: int) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(value, dtype=object)
            except Exception:
                return value
            if arr.ndim == 0:
                return arr.item() if hasattr(arr, "item") else value
            flat = arr.reshape(-1).tolist()
            if len(flat) == total and index < len(flat):
                return flat[index]
            if flat:
                return flat[0]
        return value

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                return float(arr.item())
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def _infer_ragged_axes(
        self,
        data: Any,
        ch_names: Any,
        meta: Mapping[str, Any],
    ) -> Optional[dict[str, int]]:
        """Infer common MATLAB-cell shapes such as epochs/channels/samples."""

        def sequence_items(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return None
                return list(value)
            if isinstance(value, (list, tuple)):
                return list(value)
            return None

        outer = sequence_items(data)
        if not outer:
            return None
        inner = sequence_items(outer[0])
        if not inner:
            return {"samples": 1}

        cell_items = sequence_items(inner[0]) if inner else None
        first_cell_item = sequence_items(cell_items[0]) if cell_items else None
        inferred = {"samples": 3, "epochs": 2} if first_cell_item is not None else {"samples": 2}
        labels = ch_names if isinstance(ch_names, list) else None
        subject_id = meta.get("subject_id") if isinstance(meta, MappingABC) else None
        subject_items = sequence_items(subject_id)

        def outer_axis_role() -> str:
            if subject_items and len(subject_items) == len(outer):
                return "ids"
            return "epochs"

        def inner_axis_role() -> str:
            if subject_items and len(subject_items) == len(inner):
                return "ids"
            return "epochs"

        if labels and len(labels) == len(inner):
            inferred["channels"] = 1
            inferred[outer_axis_role()] = 0
            return inferred
        if labels and len(labels) == len(outer):
            inferred["channels"] = 0
            inferred[inner_axis_role()] = 1
            return inferred

        if subject_items and len(subject_items) == len(outer):
            inferred["ids"] = 0
            inferred["channels"] = 1
            return inferred

        inferred["epochs"] = 0
        inferred["channels"] = 1
        return inferred

    def _apply_aggregation(self, df):
        over = list(self.config.aggregate_over or [])
        method = self.config.aggregate_method

        if not over:
            return df

        # Grouping keys: preserve epoch explicitly
        group_keys = [
            c for c in df.columns
            if c != "data" and c not in over
        ]

        if "epoch" in df.columns and "epoch" not in group_keys:
            group_keys.append("epoch")

        def reduce_arrays(arrs):
            stack = np.stack([np.asarray(a) for a in arrs], axis=0)
            if method == "sum":
                return stack.sum(axis=0)
            if method == "mean":
                return stack.mean(axis=0)
            if method == "median":
                return np.median(stack, axis=0)
            raise ValueError(f"Unknown aggregate_method: {method}")

        g = df.groupby(group_keys, dropna=False, sort=False)
        out = g["data"].apply(lambda s: reduce_arrays(list(s))).reset_index()

        for dim in over:
            label = self.config.aggregate_labels.get(dim, "aggregate")
            out[dim] = label

        return out
