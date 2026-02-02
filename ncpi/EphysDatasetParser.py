from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union, Literal, Tuple
import numpy as np
from ncpi import tools


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


def _require_scipy(context: str = "") -> None:
    if not tools.ensure_module("scipy"):
        msg = "scipy is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


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

    # MNE-specific operational options (only used if input is MNE or you load with MNE)
    preload: bool = True
    pick_types: Optional[Dict[str, bool]] = None
    max_seconds: Optional[float] = None
    drop_bads: bool = True

    # ---------------------------
    # Post-processing
    # ---------------------------
    zscore: bool = False

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
      - .npy, .json, .mat paths

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
        obj, source_file = self._load_source(source)

        # 1) Parse raw rows (one row per sensor / epoch)
        rows = self._parse_object(obj, source_file=source_file)
        df = self._rows_to_df(rows)

        # 2) Optional z-scoring (sensor-wise, pre-aggregation)
        if self.config.zscore:
            df = self._apply_zscore(df)

        # 3) Aggregate FIRST (e.g. collapse sensors)
        if self.config.aggregate_over:
            df = self._apply_aggregation(df)

        # 4) Epoch LAST (temporal operation)
        if self.config.epoch_length_s is not None:
            rows = df.to_dict("records")
            rows = self._apply_epoching_rows(rows)
            df = self._rows_to_df(rows)

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
                _require_scipy("MATLAB .mat loading")
                import scipy.io as sio  # type: ignore

                # squeeze_me helps reduce MATLAB scalars/arrays to python scalars/1D arrays
                return sio.loadmat(path, squeeze_me=True, struct_as_record=False), source_file

            if suffix in (".csv", ".parquet"):
                if not tools.ensure_module("pandas"):
                    raise ImportError("pandas is required to load tabular files (.csv/.parquet).")
                import pandas as pd  # type: ignore

                if suffix == ".csv":
                    return pd.read_csv(path), source_file
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

            # dotted path
            parts = locator.split(".")
            cur = obj
            for p in parts:
                cur = self._step(cur, p)

            # If final resolves to a zero-arg callable attribute and locator looked like a method name,
            # we DO NOT auto-call here by default; user can provide lambda if desired.
            return cur

        return locator

    def _step(self, cur: Any, key: Union[str, int]) -> Any:
        # pandas DataFrame: treat string keys as column names if present
        if tools.ensure_module("pandas"):
            import pandas as pd  # type: ignore
            if isinstance(cur, pd.DataFrame) and isinstance(key, str) and key in cur.columns:
                return cur[key]

        # dict-like
        if isinstance(cur, dict) and key in cur:
            return cur[key]

        # list/tuple indexing
        if isinstance(key, int) and isinstance(cur, (list, tuple)):
            return cur[key]

        # attribute access
        if isinstance(key, str) and hasattr(cur, key):
            return getattr(cur, key)

        # dict-like but key might be present as string for MATLAB objects
        if isinstance(cur, dict) and isinstance(key, str):
            # try common variations
            if key in cur:
                return cur[key]
            if key.encode() in cur:  # rare
                return cur[key.encode()]

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
        if isinstance(obj, dict):
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

    # ---- dict-like (.mat/.json) ----

    def _parse_dict_like(self, d: dict, source_file: Optional[str]) -> list[dict]:
        f = self.config.fields

        data = self._resolve(d, f.data)
        fs = self._resolve(d, f.fs)
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

        # data can be:
        # - 2D: (time, channels) or (channels, time)
        # - 3D: (epochs, channels, time) or (epochs, time, channels)
        arr = np.asarray(data)

        if arr.ndim == 1:
            # single channel
            labels = [ch_names[0]] if isinstance(ch_names, list) and ch_names else ["ch0"]
            return [self._row_from_series(meta, labels[0], None, arr, fs, data_domain, freqs, spectral_kind, source_file)]

        if arr.ndim == 2:
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

        a = np.asarray(arr)
        if a.ndim == 1:
            ch = labels[0] if labels else "ch0"
            return [self._row_from_series(meta, ch, None, a, fs, data_domain, freqs, spectral_kind, source_file)]

        if a.ndim == 2:
            # assume (time, channels) by default for non-MNE numpy
            n_time, n_ch = a.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            return [self._row_from_series(meta, labels2[i], None, a[:, i], fs, data_domain, freqs, spectral_kind, source_file) for i in range(n_ch)]

        if a.ndim == 3:
            # assume (epochs, time, channels)
            n_ep, n_time, n_ch = a.shape
            labels2 = labels if labels and len(labels) == n_ch else [f"ch{i}" for i in range(n_ch)]
            rows = []
            for e in range(n_ep):
                for i in range(n_ch):
                    rows.append(self._row_from_series(meta, labels2[i], e, a[e, :, i], fs, data_domain, freqs, spectral_kind, source_file))
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
        if callable(data):
            data = data()

        arr = np.asarray(data)  # expected (n_channels, n_times)

        fs = self._resolve(raw, f.fs)
        fs = float(fs) if fs is not None else float(getattr(raw.info, "sfreq", raw.info.get("sfreq")))

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

    def _resolve_metadata(self, obj: Any) -> dict:
        out: dict = {}
        HAS_PANDAS = tools.ensure_module("pandas")
        if HAS_PANDAS:
            import pandas as pd  # type: ignore

        for k, loc in self.config.fields.metadata.items():
            try:
                v = self._resolve(obj, loc)

                # If metadata resolves to a pandas Series (e.g., df["group"]),
                # reduce it to a scalar if it's constant across rows.
                if HAS_PANDAS:
                    import pandas as pd  # type: ignore
                    if isinstance(v, pd.Series):
                        # If constant (including all-NaN), take the first value
                        if v.nunique(dropna=False) <= 1:
                            v = v.iloc[0] if len(v) > 0 else None
                        else:
                            # Non-constant metadata column: keep as list (or raise, your choice)
                            v = v.to_list()

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
                r["t0"] = float(start / fs)
                r["t1"] = float((start + win - 1) / fs)

                out.append(r)

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
