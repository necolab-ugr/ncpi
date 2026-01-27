# Copy/paste from the attached file if you prefer.
# File: EphysDatasetParser_fixed.py

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from ncpi import tools

# Optional deps (used if installed)
try:
    import mne  # type: ignore
except Exception:
    mne = None

try:
    import scipy.io  # type: ignore
except Exception:
    scipy = None


RecordingType = Literal["EEG", "MEG", "ECoG", "LFP", "Unknown"]
DataKind = Literal["raw", "epochs", "time_series"]

# How to aggregate multi-sensor recordings into a single trace.
AggregateMethod = Literal["sum", "mean", "median"]

DEFAULT_COLUMNS = [
    "subject_id",
    "species",
    "group",
    "condition",
    "epoch",
    "sensor",
    "recording_type",
    "fs",
    "data",
    "t0",
    "t1",
    "source_file",
]


def _require_mne(context: str = "") -> None:
    if not tools.ensure_module("mne"):
        msg = "mne is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


def _require_scipy(context: str = "") -> None:
    if scipy is None:
        msg = "scipy is required"
        if context:
            msg += f" for {context}"
        msg += " but is not installed."
        raise ImportError(msg)


@dataclass
class ParseConfig:
    # Epoching
    epoch_length_s: Optional[float] = None
    epoch_step_s: Optional[float] = None

    # MNE options
    preload: bool = True
    pick_types: Optional[Dict[str, bool]] = None
    max_seconds: Optional[float] = None
    drop_bads: bool = True

    # Warnings
    warn_unimplemented: bool = True

    # .mat heuristics
    mat_signal_key_patterns: Sequence[str] = (
        r"\bsignal\b",
        r"\bdata\b",
        r"\beeg\b",
        r"\blfp\b",
        r"\bmeg\b",
        r"\becog\b",
        r"\bseeg\b",
        r"\bts\b",
        r"time[_\- ]?series",
    )
    mat_fs_key_patterns: Sequence[str] = (
        r"\bfs\b",
        r"\bsfreq\b",
        r"\bsrate\b",
        r"sampling[_\- ]?rate",
        r"\bhz\b",
    )
    mat_channel_key_patterns: Sequence[str] = (
        r"\bchannels?\b",
        r"\bch_names\b",
        r"\belectrodes?\b",
        r"\bsensors?\b",
        r"\blabels?\b",
    )
    mat_prefer_times_by_channels: bool = True

    # .csv heuristics
    csv_time_column_candidates: Sequence[str] = ("time", "t", "timestamp", "seconds", "sec")

    # Fallback fs
    default_fs: Optional[float] = None

    # ---------------------------
    # Post-processing
    # ---------------------------
    zscore: bool = False

    aggregate_sensors: bool = False
    aggregate_method: AggregateMethod = "sum"
    aggregate_sensor_label: str = "aggregate"


class EphysDatasetParser:
    def __init__(
        self,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        config: Optional[ParseConfig] = None,
    ) -> None:
        self.columns = list(columns)
        self.config = config or ParseConfig()

        self._mne_raw_exts = {
            ".fif",
            ".edf",
            ".bdf",
            ".vhdr",
            ".set",
            ".cnt",
            ".mff",
            ".ds",
            ".sqd", ".con",
        }
        self._mne_epochs_exts = {".fif"}

        self._mat_exts = {".mat"}
        self._csv_exts = {".csv", ".tsv", ".txt"}

    def parse(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        subject_id: Union[str, int] = "unknown",
        species: str = "unknown",
        group: str = "unknown",
        condition: str = "unknown",
        recording_type: RecordingType = "Unknown",
        data_kind: DataKind = "raw",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        files = self._gather_files(paths)
        rows: List[Dict[str, Any]] = []

        base_meta = dict(
            subject_id=subject_id,
            species=species,
            group=group,
            condition=condition,
            recording_type=recording_type,
        )
        if metadata:
            base_meta.update(metadata)

        for f in files:
            try:
                suf = f.suffix.lower()
                if data_kind == "epochs" and suf in self._mne_epochs_exts:
                    rows.extend(self._parse_epochs_file(f, base_meta))
                    continue

                if suf in self._mne_raw_exts or (f.is_dir() and suf in {".ds", ".mff"}):
                    if data_kind in ("raw", "time_series"):
                        rows.extend(self._parse_raw_file(f, base_meta))
                    else:
                        self._warn(f"Unsupported data_kind={data_kind!r} for MNE file {str(f)}")
                    continue

                if suf in self._mat_exts:
                    rows.extend(self._parse_mat_file(f, base_meta))
                    continue
                if suf in self._csv_exts:
                    rows.extend(self._parse_csv_file(f, base_meta))
                    continue

                self._warn_unimplemented(f)

            except Exception as e:
                self._warn(f"Failed parsing {str(f)}: {e}")

        df = pd.DataFrame(rows)

        # ---- Post-processing (optional) ----
        if not df.empty and self.config.zscore:
            df = self._zscore_df(df)

        if not df.empty and self.config.aggregate_sensors:
            df = self._aggregate_sensors_df(df)

        for c in self.columns:
            if c not in df.columns:
                df[c] = None
        df = df[self.columns]
        return df

    # ---------------------------
    # Post-processing
    # ---------------------------
    def _zscore_df(self, df: pd.DataFrame) -> pd.DataFrame:
        def _z(x: Any) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return arr
            mu = float(np.nanmean(arr))
            sd = float(np.nanstd(arr))
            if not np.isfinite(sd) or sd == 0.0:
                return np.zeros_like(arr, dtype=float)
            return (arr - mu) / sd

        out = df.copy()
        if "data" in out.columns:
            out["data"] = out["data"].apply(_z)
        return out

    def _aggregate_sensors_df(self, df: pd.DataFrame) -> pd.DataFrame:
        method: AggregateMethod = self.config.aggregate_method

        def _agg_stack(series: pd.Series) -> np.ndarray:
            arrs = [np.asarray(a, dtype=float).ravel() for a in series.to_list()]
            if not arrs:
                return np.asarray([], dtype=float)
            min_len = min(a.size for a in arrs)
            if min_len == 0:
                return np.asarray([], dtype=float)
            stack = np.vstack([a[:min_len] for a in arrs])
            if method == "sum":
                return np.nansum(stack, axis=0)
            if method == "mean":
                return np.nanmean(stack, axis=0)
            if method == "median":
                return np.nanmedian(stack, axis=0)
            raise ValueError(f"Unknown aggregate_method: {method!r}")

        group_cols = [c for c in df.columns if c not in ("sensor", "data")]
        grouped = df.groupby(group_cols, dropna=False, sort=False, as_index=False)
        out = grouped.agg(data=("data", _agg_stack))
        out["sensor"] = self.config.aggregate_sensor_label
        return out

    # ---------------------------
    # Core parsers (structured via MNE)
    # ---------------------------
    def _parse_raw_file(self, file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_mne("parsing structured MNE electrophysiology files")
        import mne  # type: ignore

        if not file_path.exists():
            self._warn(f"Path does not exist: {str(file_path)}")
            return []

        if not self._is_mne_candidate(file_path):
            self._warn_unimplemented(file_path)
            return []

        raw = self._read_raw_mne(file_path, preload=self.config.preload)

        if self.config.drop_bads and raw.info.get("bads"):
            raw = raw.copy().drop_channels(raw.info["bads"])

        if self.config.pick_types:
            picks = mne.pick_types(raw.info, **self.config.pick_types)
            raw = raw.copy().pick(picks)

        fs = float(raw.info["sfreq"])
        if self.config.max_seconds is not None:
            max_samp = int(round(self.config.max_seconds * fs))
            raw = raw.copy().crop(tmin=0.0, tmax=(max_samp - 1) / fs)

        rec_type = base_meta.get("recording_type", "Unknown")
        if rec_type == "Unknown":
            rec_type = self._infer_recording_type_from_mne(raw)

        if self.config.epoch_length_s is None:
            return self._raw_to_rows(raw, fs=fs, base_meta=base_meta, recording_type=rec_type, epoch_index=0)
        return self._raw_to_fixed_epochs_rows(raw, fs=fs, base_meta=base_meta, recording_type=rec_type)

    def _parse_epochs_file(self, file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_mne("parsing epochs files")
        import mne  # type: ignore

        if not file_path.exists():
            self._warn(f"Path does not exist: {str(file_path)}")
            return []

        if file_path.suffix.lower() not in self._mne_epochs_exts:
            self._warn_unimplemented(file_path)
            return []

        if not re.search(r"(epo|epochs)", file_path.name, flags=re.IGNORECASE):
            self._warn(f"Attempting read_epochs on FIF that doesn't look like epochs: {file_path.name}")

        epochs = mne.read_epochs(str(file_path), preload=self.config.preload, verbose="ERROR")

        if self.config.drop_bads and epochs.info.get("bads"):
            epochs = epochs.copy().drop_channels(epochs.info["bads"])

        if self.config.pick_types:
            picks = mne.pick_types(epochs.info, **self.config.pick_types)
            epochs = epochs.copy().pick(picks)

        fs = float(epochs.info["sfreq"])
        rec_type = base_meta.get("recording_type", "Unknown")
        if rec_type == "Unknown":
            rec_type = self._infer_recording_type_from_mne(epochs)

        data = epochs.get_data()
        n_epochs, n_ch, _ = data.shape
        t0 = float(epochs.times[0])
        t1 = float(epochs.times[-1])

        rows: List[Dict[str, Any]] = []
        ch_names = list(epochs.ch_names)

        for ei in range(n_epochs):
            for ci in range(n_ch):
                rows.append(
                    dict(
                        subject_id=base_meta.get("subject_id"),
                        species=base_meta.get("species"),
                        group=base_meta.get("group"),
                        condition=base_meta.get("condition"),
                        epoch=int(ei),
                        sensor=ch_names[ci],
                        recording_type=rec_type,
                        fs=fs,
                        data=np.asarray(data[ei, ci, :], dtype=float),
                        t0=t0,
                        t1=t1,
                        source_file=str(file_path),
                    )
                )
        return rows

    # ---------------------------
    # Core parsers (unstructured: .mat / .csv)
    # ---------------------------
    def _parse_mat_file(self, file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_scipy("reading MATLAB .mat files")
        assert scipy is not None

        if not file_path.exists():
            self._warn(f"Path does not exist: {str(file_path)}")
            return []

        mat = scipy.io.loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
        mat = {k: v for k, v in mat.items() if not k.startswith("__")}

        signal, _sig_path = self._mat_find_best_signal(mat)
        fs, _fs_path = self._mat_find_best_fs(mat)
        ch_names, _ch_path = self._mat_find_best_channels(mat, n_channels=self._infer_n_channels(signal))

        if fs is None:
            fs = self.config.default_fs
        if fs is None:
            self._warn(
                f"Could not infer sampling frequency from {file_path.name}; "
                f"set ParseConfig.default_fs to avoid this warning."
            )
            fs = np.nan

        signal = self._normalize_signal_array(signal)

        if signal.ndim == 3:
            n_epochs, n_ch, _ = signal.shape
            rows: List[Dict[str, Any]] = []
            for ei in range(n_epochs):
                for ci in range(n_ch):
                    rows.append(
                        dict(
                            subject_id=base_meta.get("subject_id"),
                            species=base_meta.get("species"),
                            group=base_meta.get("group"),
                            condition=base_meta.get("condition"),
                            epoch=int(ei),
                            sensor=ch_names[ci] if ci < len(ch_names) else f"ch_{ci}",
                            recording_type=base_meta.get("recording_type", "Unknown"),
                            fs=float(fs),
                            data=np.asarray(signal[ei, ci, :], dtype=float),
                            t0=None,
                            t1=None,
                            source_file=str(file_path),
                        )
                    )
            return rows

        if self.config.epoch_length_s is None:
            return self._array2d_to_rows(
                signal,
                fs=float(fs),
                base_meta=base_meta,
                source_file=str(file_path),
                epoch_index=0,
                t0=0.0,
                t1=(signal.shape[1] - 1) / float(fs) if np.isfinite(fs) and signal.shape[1] > 0 else None,
                ch_names=ch_names,
            )
        return self._array2d_to_fixed_epochs_rows(
            signal,
            fs=float(fs),
            base_meta=base_meta,
            source_file=str(file_path),
            ch_names=ch_names,
        )

    def _parse_csv_file(self, file_path: Path, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not file_path.exists():
            self._warn(f"Path does not exist: {str(file_path)}")
            return []

        sep = "," if file_path.suffix.lower() == ".csv" else "\t"
        try:
            df = pd.read_csv(file_path, sep=sep)
        except Exception:
            df = pd.read_csv(file_path, delim_whitespace=True)

        time_col = None
        lower_cols = {c.lower(): c for c in df.columns}
        for cand in self.config.csv_time_column_candidates:
            if cand.lower() in lower_cols:
                time_col = lower_cols[cand.lower()]
                break

        fs: Optional[float] = None
        if time_col is not None:
            t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
            t = t[np.isfinite(t)]
            if t.size >= 3:
                dt = np.diff(t)
                dt = dt[np.isfinite(dt) & (dt > 0)]
                if dt.size:
                    fs = float(1.0 / np.median(dt))

        if fs is None:
            fs = self.config.default_fs

        if fs is None:
            self._warn(
                f"Could not infer sampling frequency from {file_path.name}; "
                f"set ParseConfig.default_fs or include a time column to avoid this warning."
            )
            fs = np.nan

        num_df = df.select_dtypes(include=[np.number]).copy()
        if time_col is not None and time_col in num_df.columns:
            num_df = num_df.drop(columns=[time_col])

        if num_df.shape[1] == 0:
            coerced = df.apply(pd.to_numeric, errors="coerce")
            if time_col is not None and time_col in coerced.columns:
                coerced = coerced.drop(columns=[time_col])
            num_df = coerced.dropna(axis=1, how="all")

        data = num_df.to_numpy()
        if data.ndim != 2 or data.size == 0:
            self._warn(f"No numeric signal data found in {file_path.name}")
            return []

        data = np.asarray(data, dtype=float)
        data_2d = data.T
        ch_names = list(num_df.columns) if num_df.shape[1] > 0 else [f"ch_{i}" for i in range(data_2d.shape[0])]

        if self.config.epoch_length_s is None:
            return self._array2d_to_rows(
                data_2d,
                fs=float(fs),
                base_meta=base_meta,
                source_file=str(file_path),
                epoch_index=0,
                t0=0.0,
                t1=(data_2d.shape[1] - 1) / float(fs) if np.isfinite(fs) and data_2d.shape[1] > 0 else None,
                ch_names=ch_names,
            )
        return self._array2d_to_fixed_epochs_rows(
            data_2d,
            fs=float(fs),
            base_meta=base_meta,
            source_file=str(file_path),
            ch_names=ch_names,
        )

    # ---------------------------
    # Helpers: unstructured to tidy rows
    # ---------------------------
    def _array2d_to_rows(
        self,
        data_2d: np.ndarray,
        *,
        fs: float,
        base_meta: Dict[str, Any],
        source_file: str,
        epoch_index: int,
        t0: Optional[float],
        t1: Optional[float],
        ch_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if data_2d.ndim != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_times), got shape {data_2d.shape}")

        n_ch, _ = data_2d.shape
        ch_names = list(ch_names)
        if len(ch_names) != n_ch:
            ch_names = (ch_names[:n_ch] + [f"ch_{i}" for i in range(len(ch_names), n_ch)])

        rows: List[Dict[str, Any]] = []
        for ci in range(n_ch):
            rows.append(
                dict(
                    subject_id=base_meta.get("subject_id"),
                    species=base_meta.get("species"),
                    group=base_meta.get("group"),
                    condition=base_meta.get("condition"),
                    epoch=int(epoch_index),
                    sensor=ch_names[ci],
                    recording_type=base_meta.get("recording_type", "Unknown"),
                    fs=float(fs),
                    data=np.asarray(data_2d[ci, :], dtype=float),
                    t0=t0,
                    t1=t1,
                    source_file=source_file,
                )
            )
        return rows

    def _array2d_to_fixed_epochs_rows(
        self,
        data_2d: np.ndarray,
        *,
        fs: float,
        base_meta: Dict[str, Any],
        source_file: str,
        ch_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        assert self.config.epoch_length_s is not None
        epoch_len_s = float(self.config.epoch_length_s)
        step_s = float(self.config.epoch_step_s) if self.config.epoch_step_s is not None else epoch_len_s

        win = int(round(epoch_len_s * fs)) if np.isfinite(fs) else None
        step = int(round(step_s * fs)) if np.isfinite(fs) else None

        if win is None or step is None or win <= 0 or step <= 0:
            raise ValueError("Cannot epoch without a finite fs. Provide default_fs or a time column / fs in .mat.")

        _, n_times = data_2d.shape
        rows: List[Dict[str, Any]] = []
        start = 0
        epoch_idx = 0
        while start + win <= n_times:
            stop = start + win
            t0 = start / fs
            t1 = (stop - 1) / fs
            snippet = data_2d[:, start:stop]
            rows.extend(
                self._array2d_to_rows(
                    snippet,
                    fs=fs,
                    base_meta=base_meta,
                    source_file=source_file,
                    epoch_index=epoch_idx,
                    t0=t0,
                    t1=t1,
                    ch_names=ch_names,
                )
            )
            start += step
            epoch_idx += 1
        return rows

    # ---------------------------
    # Raw -> tidy rows helpers (MNE)
    # ---------------------------
    def _raw_to_fixed_epochs_rows(
        self,
        raw: "mne.io.BaseRaw",
        *,
        fs: float,
        base_meta: Dict[str, Any],
        recording_type: RecordingType,
    ) -> List[Dict[str, Any]]:
        assert self.config.epoch_length_s is not None
        epoch_len_s = float(self.config.epoch_length_s)
        step_s = float(self.config.epoch_step_s) if self.config.epoch_step_s is not None else epoch_len_s

        n_total = raw.n_times
        win = int(round(epoch_len_s * fs))
        step = int(round(step_s * fs))

        if win <= 0 or step <= 0:
            raise ValueError("epoch_length_s and epoch_step_s must be > 0")

        rows: List[Dict[str, Any]] = []
        start = 0
        epoch_idx = 0

        while start + win <= n_total:
            stop = start + win
            t0 = start / fs
            t1 = (stop - 1) / fs
            snippet = raw.get_data(start=start, stop=stop)
            rows.extend(
                self._array_to_rows(
                    snippet,
                    fs=fs,
                    base_meta=base_meta,
                    recording_type=recording_type,
                    epoch_index=epoch_idx,
                    t0=t0,
                    t1=t1,
                    ch_names=list(raw.ch_names),
                    source_file=str(raw.filenames[0]) if raw.filenames else base_meta.get("source_file", "unknown"),
                )
            )
            start += step
            epoch_idx += 1

        return rows

    def _raw_to_rows(
        self,
        raw: "mne.io.BaseRaw",
        *,
        fs: float,
        base_meta: Dict[str, Any],
        recording_type: RecordingType,
        epoch_index: int,
    ) -> List[Dict[str, Any]]:
        data = raw.get_data()
        t0 = 0.0
        t1 = (raw.n_times - 1) / fs if raw.n_times else None
        source = str(raw.filenames[0]) if raw.filenames else base_meta.get("source_file", "unknown")
        return self._array_to_rows(
            data,
            fs=fs,
            base_meta=base_meta,
            recording_type=recording_type,
            epoch_index=epoch_index,
            t0=t0,
            t1=t1,
            ch_names=list(raw.ch_names),
            source_file=source,
        )

    def _array_to_rows(
        self,
        data_2d: np.ndarray,
        *,
        fs: float,
        base_meta: Dict[str, Any],
        recording_type: RecordingType,
        epoch_index: int,
        t0: Optional[float],
        t1: Optional[float],
        ch_names: List[str],
        source_file: str,
    ) -> List[Dict[str, Any]]:
        if data_2d.ndim != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_times), got shape {data_2d.shape}")

        n_ch, _ = data_2d.shape
        if n_ch != len(ch_names):
            ch_names = ch_names[:n_ch] + [f"ch_{i}" for i in range(len(ch_names), n_ch)]

        rows: List[Dict[str, Any]] = []
        for ci in range(n_ch):
            rows.append(
                dict(
                    subject_id=base_meta.get("subject_id"),
                    species=base_meta.get("species"),
                    group=base_meta.get("group"),
                    condition=base_meta.get("condition"),
                    epoch=int(epoch_index),
                    sensor=ch_names[ci],
                    recording_type=recording_type,
                    fs=float(fs),
                    data=np.asarray(data_2d[ci, :], dtype=float),
                    t0=t0,
                    t1=t1,
                    source_file=str(source_file),
                )
            )
        return rows

    # ---------------------------
    # MNE reading + inference
    # ---------------------------
    def _read_raw_mne(self, file_path: Path, *, preload: bool) -> "mne.io.BaseRaw":
        import mne  # type: ignore
        fp = str(file_path)

        if file_path.suffix.lower() == ".fif":
            return mne.io.read_raw_fif(fp, preload=preload, verbose="ERROR")
        if file_path.suffix.lower() == ".vhdr":
            return mne.io.read_raw_brainvision(fp, preload=preload, verbose="ERROR")
        if file_path.suffix.lower() in {".edf", ".bdf"}:
            return mne.io.read_raw_edf(fp, preload=preload, verbose="ERROR")
        if file_path.suffix.lower() == ".set":
            return mne.io.read_raw_eeglab(fp, preload=preload, verbose="ERROR")

        if hasattr(mne.io, "read_raw"):
            return mne.io.read_raw(fp, preload=preload, verbose="ERROR")  # type: ignore[attr-defined]

        raise RuntimeError(f"No suitable MNE raw reader route for: {fp}")

    def _infer_recording_type_from_mne(self, inst: Any) -> RecordingType:
        if mne is None:
            return "Unknown"
        import mne  # type: ignore

        info = inst.info
        if len(mne.pick_types(info, meg=True, eeg=False, ecog=False, seeg=False, stim=False, misc=False)) > 0:
            return "MEG"
        if len(mne.pick_types(info, eeg=True, meg=False, ecog=False, seeg=False, stim=False, misc=False)) > 0:
            return "EEG"
        if len(mne.pick_types(info, ecog=True, meg=False, eeg=False, seeg=False, stim=False, misc=False)) > 0:
            return "ECoG"
        if len(mne.pick_types(info, seeg=True, meg=False, eeg=False, ecog=False, stim=False, misc=False)) > 0:
            return "LFP"
        return "Unknown"

    # ---------------------------
    # .mat heuristics
    # ---------------------------
    def _mat_walk(self, obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                yield p, v
                yield from self._mat_walk(v, p)
            return

        if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, np.ndarray)):
            for k, v in vars(obj).items():
                p = f"{prefix}.{k}" if prefix else str(k)
                yield p, v
                yield from self._mat_walk(v, p)
            return

        if isinstance(obj, np.ndarray):
            yield prefix, obj
            if obj.dtype == object:
                flat = obj.ravel()
                for i in range(min(flat.size, 50)):
                    v = flat[i]
                    p = f"{prefix}[{i}]"
                    yield p, v
                    yield from self._mat_walk(v, p)
            return

        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                p = f"{prefix}[{i}]"
                yield p, v
                yield from self._mat_walk(v, p)
            return

        yield prefix, obj

    def _mat_key_matches(self, path: str, patterns: Sequence[str]) -> bool:
        return any(re.search(pat, path, flags=re.IGNORECASE) for pat in patterns)

    def _mat_find_best_signal(self, mat: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        candidates: List[Tuple[float, str, np.ndarray]] = []
        for path, v in self._mat_walk(mat):
            if not isinstance(v, np.ndarray):
                continue
            if v.size < 50:
                continue
            if not np.issubdtype(v.dtype, np.number):
                continue
            if v.ndim not in (1, 2, 3):
                continue

            score = float(np.log10(v.size + 1))
            if self._mat_key_matches(path, self.config.mat_signal_key_patterns):
                score += 2.0
            if v.ndim == 2:
                score += 0.5
            if v.ndim == 3:
                score += 1.0
            candidates.append((score, path, v))

        if not candidates:
            raise ValueError("No numeric signal-like arrays found in .mat file.")

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_path, best_arr = candidates[0]
        return np.asarray(best_arr), best_path

    def _mat_find_best_fs(self, mat: Dict[str, Any]) -> Tuple[Optional[float], str]:
        best: Tuple[float, str, float] | None = None
        for path, v in self._mat_walk(mat):
            if not self._mat_key_matches(path, self.config.mat_fs_key_patterns):
                continue

            fs_val: Optional[float] = None
            if isinstance(v, (int, float, np.number)):
                fs_val = float(v)
            elif isinstance(v, np.ndarray) and v.size == 1 and np.issubdtype(v.dtype, np.number):
                fs_val = float(np.asarray(v).reshape(-1)[0])

            if fs_val is None:
                continue
            if not (0.1 <= fs_val <= 50000):
                continue

            score = 0.0
            if re.search(r"\b(fs|sfreq|srate)\b", path, flags=re.IGNORECASE):
                score += 2.0
            if 10 <= fs_val <= 5000:
                score += 0.5

            if best is None or score > best[0]:
                best = (score, path, fs_val)

        if best is None:
            return None, ""
        return best[2], best[1]

    def _mat_find_best_channels(self, mat: Dict[str, Any], n_channels: int) -> Tuple[List[str], str]:
        for path, v in self._mat_walk(mat):
            if not self._mat_key_matches(path, self.config.mat_channel_key_patterns):
                continue

            names: List[str] = []
            if isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, bytes)) for x in v):
                names = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            elif isinstance(v, np.ndarray):
                if v.dtype.kind in ("U", "S"):
                    flat = v.ravel().tolist()
                    names = [x.decode() if isinstance(x, bytes) else str(x) for x in flat]
                elif v.dtype == object:
                    flat = v.ravel().tolist()
                    for x in flat:
                        if isinstance(x, (str, bytes)):
                            names.append(x.decode() if isinstance(x, bytes) else str(x))

            if names and (len(names) == n_channels or len(names) >= 1):
                return names, path

        return [f"ch_{i}" for i in range(n_channels)], ""

    def _infer_n_channels(self, signal: np.ndarray) -> int:
        if signal.ndim == 1:
            return 1
        if signal.ndim == 2:
            a, b = signal.shape
            if a <= 512 and b > a:
                return a
            if b <= 512 and a > b:
                return b
            return b if self.config.mat_prefer_times_by_channels else a
        if signal.ndim == 3:
            dims = list(signal.shape)
            small = [d for d in dims if d <= 512]
            return int(min(small)) if small else int(min(dims))
        return 1

    def _normalize_signal_array(self, signal: np.ndarray) -> np.ndarray:
        sig = np.asarray(signal)
        if sig.ndim == 1:
            return sig.reshape(1, -1)

        if sig.ndim == 2:
            a, b = sig.shape
            if self.config.mat_prefer_times_by_channels:
                if b <= 512 and a > b:
                    return sig.T
                if a <= 512 and b > a:
                    return sig
                return sig.T
            else:
                if a <= 512 and b > a:
                    return sig
                if b <= 512 and a > b:
                    return sig.T
                return sig

        if sig.ndim == 3:
            dims = list(sig.shape)
            ch_dim = int(np.argmin([d if d <= 512 else 1e9 for d in dims]))
            time_dim = int(np.argmax(dims))
            ep_dim = ({0, 1, 2} - {ch_dim, time_dim}).pop()
            perm = (ep_dim, ch_dim, time_dim)
            return np.transpose(sig, perm)

        raise ValueError(f"Unsupported signal ndim={sig.ndim} in .mat")

    # ---------------------------
    # File discovery
    # ---------------------------
    def _gather_files(self, paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
        if isinstance(paths, (str, Path)):
            paths_list = [Path(paths)]
        else:
            paths_list = [Path(p) for p in paths]

        files: List[Path] = []
        for p in paths_list:
            if p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_dir():
                        if fp.suffix.lower() in {".ds", ".mff"}:
                            files.append(fp)
                        continue
                    files.append(fp)
            else:
                files.append(p)

        seen = set()
        uniq: List[Path] = []
        for f in files:
            s = str(f.resolve()) if f.exists() else str(f)
            if s not in seen:
                seen.add(s)
                uniq.append(f)
        return uniq

    def _is_mne_candidate(self, p: Path) -> bool:
        if p.is_dir():
            return p.suffix.lower() in {".ds", ".mff"}
        return p.suffix.lower() in self._mne_raw_exts

    # ---------------------------
    # Warnings
    # ---------------------------
    def _warn_unimplemented(self, file_path: Path) -> None:
        if self.config.warn_unimplemented:
            warnings.warn(
                f"Format not implemented or not recognized. Skipping: {str(file_path)}",
                category=UserWarning,
                stacklevel=2,
            )

    def _warn(self, msg: str) -> None:
        warnings.warn(msg, category=UserWarning, stacklevel=2)
