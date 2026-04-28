from __future__ import annotations
from dataclasses import dataclass, field
import ast
import json
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


@lru_cache(maxsize=128)
def _is_hdf5_mat(file_path: str) -> bool:
    """Return True if file .mat is v7.3 (HDF5)."""
    with open(file_path, "rb") as f:
        return f.read(8) == b'\x89HDF\r\n\x1a\n'


def _load_mat_with_fallback(path: Path) -> Any:
    # Fast file version detection
    if _is_hdf5_mat(str(path)):
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
      - .npy, .json, .mat, .set, .tsv paths

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

            # 4b) Exclude last epoch if requested
            if self.config.exclude_last_epoch:
                df = self._exclude_last_epoch(df)

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

            if suffix == ".set":
                _require_mne("EEGLAB .set loading")
                import mne  # type: ignore

                return mne.io.read_raw_eeglab(str(path), preload=self.config.preload), source_file

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
            ch_names = self._resolve(df, f.ch_names)
            if isinstance(ch_names, (list, tuple)) and not isinstance(ch_names, str):
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

        fs = self._resolve(d, f.fs)
        # Handle a possible list that comes from a struct array
        fs = self._extract_first_if_list(fs)
        fs = float(np.asarray(fs).item()) if fs is not None and np.asarray(fs).size == 1 else (float(fs) if fs is not None else None)

        data_domain = self._resolve(d, f.data_domain) or "time"
        spectral_kind = self._resolve(d, f.spectral_kind)
        freqs = self._resolve(d, f.freqs)

        # epoch label (optional)
        epoch_val = self._resolve(d, f.epoch)

        # channel names
        ch_names = self._resolve(d, f.ch_names)
        if ch_names is not None and not isinstance(ch_names, str):
            # MATLAB sometimes returns numpy arrays of dtype object
            if isinstance(ch_names, np.ndarray):
                ch_names = [str(x) for x in ch_names.tolist()]
            elif isinstance(ch_names, (list, tuple)):
                ch_names = [str(x) for x in ch_names]

        meta = self._resolve_metadata(d)

        def _subject_ids_for_axis(base_meta: Mapping[str, Any], n_ids: int) -> list[Any]:
            return self._normalize_subject_id_list(base_meta.get("subject_id", None), n_ids)

        # data can be:
        # - 2D: (time, channels) or (channels, time)
        # - 3D: (epochs, channels, time) or (epochs, time, channels)
        arr = np.asarray(data)
        if arr.dtype == object and arr.ndim == 1 and arr.size > 0:
            try:
                stacked = np.stack([np.asarray(a).squeeze() for a in arr.tolist()], axis=0)
                if stacked.ndim >= 2:
                    arr = stacked
            except Exception:
                pass

        if arr.ndim == 1:
            # single channel
            labels = [ch_names[0]] if isinstance(ch_names, list) and ch_names else ["ch0"]
            return [self._row_from_series(meta, labels[0], None, arr, fs, data_domain, freqs, spectral_kind, source_file)]

        if arr.ndim == 2:
            ax = f.array_axes
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
            ax = f.array_axes
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
            ax = f.array_axes
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
            if len(remaining) > 0:
                raise ValueError(
                    "4D dict-like arrays must map all dimensions via array_axes "
                    "(use ids/epochs/channels/samples)."
                )

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

        ch_names = self._resolve(arr, f.ch_names)
        if isinstance(ch_names, (list, tuple)) and not isinstance(ch_names, str):
            labels = [str(x) for x in ch_names]
        else:
            labels = None

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

        ch_names = self._resolve(raw, f.ch_names)
        if isinstance(ch_names, (list, tuple)):
            labels = [str(x) for x in ch_names]
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

        ch_names = self._resolve(epochs, f.ch_names)
        labels = [str(x) for x in ch_names] if isinstance(ch_names, (list, tuple)) else list(getattr(epochs, "ch_names", []))

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
