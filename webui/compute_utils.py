import os
import glob
import pandas as pd
import pickle
import numpy as np
import sys

# Prefer local repository package over globally installed ncpi.
_webui_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_webui_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import ncpi
import shutil
import ast
import importlib.util
import tempfile
import io
import json
import inspect
import re
import contextlib
import itertools

sim_data_path = 'zenodo_sim_files/data/'
model_scaler_path = 'zenodo_sim_files/ML_models/4_param/MLP'
DEFAULT_SIM_DATA_DIR = '/tmp/simulation_data'
MAX_OUTPUT_LINES = 200

# Dataframe file upload format check
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'parquet', 'feather', 'pkl', 'pickle'}

# Check if the dataframe has an allowed extension
def allowed_file(filename):
    if not filename or '.' not in filename:
        return False

    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension[1:] in ALLOWED_EXTENSIONS # file_extension without the dot


def read_file_preprocessing(file_path):
     # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if file has allowed extension
    if not allowed_file(file_path):
        raise ValueError(
            f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower() # .csv

    return file_extension


def read_df_file(file_path):
    """ Read file as pandas dataframe """
    file_extension = read_file_preprocessing(file_path)

    try:
        # Read file as pandas dataframe based on extension
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_extension == '.feather':
            df = pd.read_feather(file_path)
        elif file_extension in ['.pkl', '.pickle']:
            df = pd.read_pickle(file_path)
        else:
            # This shouldn't happen if allowed_file() works correctly
            raise ValueError(f"Unsupported file format: {file_extension}")

        return df

    except Exception as e:
        # Re-raise with more context
        print(f"Error occurred: {type(e).__name__}: {e}")
        raise Exception(f"Failed to read file {file_path}: {type(e).__name__}: {str(e)}")


def read_file(file_path):
    """ Read file as file object """
    file_extension = read_file_preprocessing(file_path)

    try:
        # Read file as file object based on extension
        if file_extension in ['.pkl', '.pickle']:
            with open(os.path.join(file_path), 'rb') as file:
                file_object = pickle.load(file)
        elif file_extension == '.csv':
            # Load as numpy array
            file_object = np.loadtxt(os.path.join(file_path),  delimiter=',', skiprows=1)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return file_object

    except Exception as e:
        # Re-raise with more context
        print(f"Error occurred: {type(e).__name__}: {e}")
        raise Exception(f"Failed to read file {file_path}: {type(e).__name__}: {str(e)}")


def save_df(job_id, output_df, temp_uploaded_files):
    """ Saves the output dataframe to a pickle file and returns its name """
    output_df_name = f"{job_id}_output.pkl"
    output_path = f"{temp_uploaded_files}/{output_df_name}"
    output_df.to_pickle(output_path)

    return output_path


def _features_data_dir():
    path = os.path.realpath("/tmp/features_data")
    os.makedirs(path, exist_ok=True)
    return path


def _persist_features_dataframe(output_df, method):
    features_dir = _features_data_dir()
    safe_method = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(method or "features"))
    # Cleanup legacy per-job files for this method so only one canonical file remains.
    legacy_pattern = os.path.join(features_dir, f"features_computed_{safe_method}_*.pkl")
    for legacy_path in glob.glob(legacy_pattern):
        try:
            if os.path.isfile(legacy_path):
                os.remove(legacy_path)
        except OSError:
            pass

    output_path = os.path.join(features_dir, f"{safe_method}_features.pkl")
    output_df.to_pickle(output_path)
    return output_path


def cleanup_temp_files(file_paths, keep_paths=None):
    """Delete temporary files in file_paths (params['file_paths']) silently."""
    keep = set()
    if keep_paths:
        for path in keep_paths:
            if path:
                keep.add(os.path.realpath(path))

    for file_path in file_paths.values():
        try:
            if not file_path:
                continue
            if os.path.realpath(file_path) in keep:
                continue
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Silently ignore errors


def _resolve_sim_file(file_paths, key, default_name, required=True):
    uploaded_path = file_paths.get(key)
    if uploaded_path and allowed_file(uploaded_path) and os.path.exists(uploaded_path):
        return uploaded_path

    if default_name:
        default_path = os.path.join(DEFAULT_SIM_DATA_DIR, default_name)
        if os.path.exists(default_path):
            return default_path

    if required:
        raise FileNotFoundError(
            f"Missing required input '{key}'. Upload a file or place {default_name} in {DEFAULT_SIM_DATA_DIR}."
        )
    return None


def _append_job_output(job_status, job_id, message):
    if job_id not in job_status:
        return
    if not message.endswith("\n"):
        message += "\n"
    current = job_status[job_id].get("output", "")
    if current and not current.endswith("\n"):
        current += "\n"
    combined = current + message
    lines = combined.splitlines()[-MAX_OUTPUT_LINES:]
    job_status[job_id]["output"] = "\n".join(lines)


_PROGRESS_PERCENT_RE = re.compile(r"(?:^|\s)(\d{1,3})%")
_FOLD_PROGRESS_RE = re.compile(r"Fold\s+(\d+)\s*/\s*(\d+)")


class _JobOutputCapture(io.TextIOBase):
    def __init__(self, job_status, job_id, progress_base=0, progress_span=0, line_callback=None):
        self.job_status = job_status
        self.job_id = job_id
        self.progress_base = int(progress_base)
        self.progress_span = int(progress_span)
        self.line_callback = line_callback
        self._buf = ""

    def write(self, text):
        if text is None:
            return 0
        chunk = str(text)
        if not chunk:
            return 0
        normalized = chunk.replace("\r", "\n")

        if self.job_id in self.job_status and self.progress_span > 0:
            for match in _PROGRESS_PERCENT_RE.finditer(normalized):
                try:
                    pct = int(match.group(1))
                except Exception:
                    continue
                if pct < 0 or pct > 100:
                    continue
                mapped = self.progress_base + int((self.progress_span * pct) / 100.0)
                self.job_status[self.job_id]["progress"] = max(
                    self.job_status[self.job_id].get("progress", 0),
                    mapped,
                )

        self._buf += normalized
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                if callable(self.line_callback):
                    try:
                        self.line_callback(line, self)
                    except Exception:
                        pass
                _append_job_output(self.job_status, self.job_id, line)
        return len(chunk)

    def flush(self):
        if self._buf.strip():
            _append_job_output(self.job_status, self.job_id, self._buf.strip())
        self._buf = ""


def _compute_features_with_compat(features_obj, samples, exec_opts, progress_callback, log_callback):
    """
    Call Features.compute_features() across ncpi versions.
    Older installed versions do not accept progress/log callbacks.
    """
    fn = features_obj.compute_features
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {
        "n_jobs": exec_opts["n_jobs"],
        "chunksize": exec_opts["chunksize"],
        "start_method": exec_opts["start_method"],
    }
    has_progress = "progress_callback" in params
    has_log = "log_callback" in params
    if has_progress:
        kwargs["progress_callback"] = progress_callback
    if has_log:
        kwargs["log_callback"] = log_callback

    if log_callback and (not has_progress or not has_log):
        missing = []
        if not has_progress:
            missing.append("progress_callback")
        if not has_log:
            missing.append("log_callback")
        log_callback(
            "Installed ncpi.Features.compute_features does not support "
            f"{', '.join(missing)}; running without those hooks."
        )

    return fn(samples, **kwargs)


def _predict_inference_with_compat(inference_obj, features, base_kwargs, exec_kwargs, log_callback=None):
    """
    Call Inference.predict() across ncpi versions.
    Older versions may not accept n_jobs/chunksize/start_method kwargs.
    """
    fn = inference_obj.predict
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = dict(base_kwargs)
    unsupported = []
    for key, value in (exec_kwargs or {}).items():
        if value is None:
            continue
        if key in params:
            kwargs[key] = value
        else:
            unsupported.append(key)

    if unsupported and log_callback:
        log_callback(
            "Installed ncpi.Inference.predict does not support "
            f"{', '.join(sorted(unsupported))}; using default execution behavior."
        )

    return fn(features, **kwargs)


def _load_uploaded_source_bytes(name, ext, content):
    safe_name = str(name or "uploaded_file")
    ext = str(ext or os.path.splitext(safe_name)[1]).lower()
    raw = content
    if raw is None:
        raise ValueError(f"Uploaded file '{safe_name}' is empty.")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if not isinstance(raw, (bytes,)):
        raise ValueError(f"Invalid uploaded content type for '{safe_name}'.")

    if ext in {".pkl", ".pickle"}:
        bio = io.BytesIO(raw)
        try:
            return pd.read_pickle(bio)
        except Exception:
            bio.seek(0)
            return pickle.load(bio)

    if ext == ".json":
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON file '{safe_name}': {exc}")

    if ext == ".npy":
        return np.load(io.BytesIO(raw), allow_pickle=True)

    if ext == ".csv":
        return pd.read_csv(io.BytesIO(raw))

    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))

    if ext == ".feather":
        return pd.read_feather(io.BytesIO(raw))

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))

    if ext == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:
            raise ValueError(f"scipy is required to parse .mat files: {exc}")
        return sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)

    raise ValueError(f"Unsupported empirical file extension '{ext}' for '{safe_name}'.")


def _load_uploaded_source_path(path, name=None, ext=None):
    safe_path = str(path or "")
    if not safe_path or not os.path.exists(safe_path):
        raise ValueError(f"Uploaded file path does not exist: '{safe_path}'.")
    safe_name = str(name or os.path.basename(safe_path) or "uploaded_file")
    ext = str(ext or os.path.splitext(safe_name)[1]).lower()

    if ext in {".pkl", ".pickle"}:
        try:
            return pd.read_pickle(safe_path)
        except Exception:
            with open(safe_path, "rb") as handle:
                return pickle.load(handle)

    if ext == ".json":
        try:
            with open(safe_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON file '{safe_name}': {exc}")

    if ext == ".npy":
        return np.load(safe_path, allow_pickle=True)

    if ext == ".csv":
        return pd.read_csv(safe_path)

    if ext == ".parquet":
        return pd.read_parquet(safe_path)

    if ext == ".feather":
        return pd.read_feather(safe_path)

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(safe_path)

    if ext == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:
            raise ValueError(f"scipy is required to parse .mat files: {exc}")
        return sio.loadmat(safe_path, squeeze_me=True, struct_as_record=False)

    raise ValueError(f"Unsupported empirical file extension '{ext}' for '{safe_name}'.")


def _load_module_from_path(path, name="kernel_params"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_literal_value(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list, tuple, float, int, bool)):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default if default is not None else value


def _parse_bool(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _compute_mean_firing_rate(times, gids, bin_size_ms):
    times = _flatten_spike_data(times)
    gids = _flatten_spike_data(gids)
    if times.size == 0:
        return np.zeros((1, 0))
    if bin_size_ms <= 0:
        raise ValueError("bin_size must be a positive number (ms).")

    n_units = max(int(np.unique(gids).size), 1) if gids.size > 0 else 1
    t_min = float(np.min(times))
    t_max = float(np.max(times))
    bins = np.arange(t_min, t_max + bin_size_ms, bin_size_ms)
    hist, _ = np.histogram(times, bins=bins)
    mean_rate = hist.astype(float) / float(n_units)
    return mean_rate.reshape(1, -1)


def _flatten_spike_data(data):
    if data is None:
        return np.asarray([])
    if isinstance(data, dict):
        values = [np.asarray(v).ravel() for v in data.values()]
        return np.concatenate(values) if values else np.asarray([])
    if isinstance(data, (list, tuple)):
        if not data:
            return np.asarray([])
        if all(np.isscalar(x) for x in data):
            return np.asarray(data).ravel()
        values = [np.asarray(v).ravel() for v in data]
        return np.concatenate(values) if values else np.asarray([])
    return np.asarray(data).ravel()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_dt_ms(value):
    if isinstance(value, pd.DataFrame):
        dt_ms = _first_numeric_from_column(value, "dt_ms")
        if dt_ms is None:
            dt_ms = _first_numeric_from_column(value, "dt")
        if dt_ms is not None:
            return dt_ms
        if "metadata" in value.columns and not value.empty:
            meta_value = value["metadata"].dropna()
            if not meta_value.empty:
                meta = meta_value.iloc[0]
                if isinstance(meta, str):
                    try:
                        meta = ast.literal_eval(meta)
                    except Exception:
                        meta = None
                if isinstance(meta, dict):
                    dt_ms = _safe_float(meta.get("dt_ms"))
                    if dt_ms is None:
                        dt_ms = _safe_float(meta.get("dt"))
                    if dt_ms is not None:
                        return dt_ms
        return None

    if isinstance(value, dict):
        dt_ms = _safe_float(value.get("dt_ms"))
        if dt_ms is None:
            dt_ms = _safe_float(value.get("dt"))
        return dt_ms

    return _safe_float(value)


def _load_simulation_dt_ms(file_paths):
    dt_path = _resolve_sim_file(file_paths, "dt_file", "dt.pkl", required=False)
    if not dt_path:
        return None, None
    try:
        dt_obj = read_file(dt_path)
    except Exception:
        return None, dt_path
    return _coerce_dt_ms(dt_obj), dt_path


def _first_numeric_from_column(df, column_name):
    if column_name not in df.columns:
        return None
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _extract_signal_and_meta_from_source(obj):
    meta = {"dt_ms": None, "decimation_factor": 1, "fs_hz": None}

    if isinstance(obj, pd.DataFrame):
        dt_ms = _first_numeric_from_column(obj, "dt_ms")
        decimation_factor = _first_numeric_from_column(obj, "decimation_factor")
        fs_hz = _first_numeric_from_column(obj, "fs_hz")

        if "metadata" in obj.columns and (dt_ms is None or fs_hz is None or decimation_factor is None):
            meta_series = obj["metadata"].dropna()
            if not meta_series.empty:
                meta_value = meta_series.iloc[0]
                if isinstance(meta_value, str):
                    try:
                        meta_value = ast.literal_eval(meta_value)
                    except Exception:
                        meta_value = None
                if isinstance(meta_value, dict):
                    if dt_ms is None:
                        dt_ms = _safe_float(meta_value.get("dt_ms"))
                    if decimation_factor is None:
                        decimation_factor = _safe_float(meta_value.get("decimation_factor"))
                    if fs_hz is None:
                        fs_hz = _safe_float(meta_value.get("fs_hz"))

        if decimation_factor is None or decimation_factor <= 0:
            decimation_factor = 1
        meta["dt_ms"] = dt_ms
        meta["decimation_factor"] = int(decimation_factor)
        meta["fs_hz"] = fs_hz
        if "data" in obj.columns and not obj.empty:
            return obj["data"].iloc[0], meta
        if not obj.empty:
            return obj.iloc[0, 0], meta
        return None, meta

    return obj, meta


def _sum_signal_dict(signal_dict):
    total = None
    for value in signal_dict.values():
        arr = np.asarray(value)
        if total is None:
            total = np.array(arr, copy=True)
        else:
            total = total + arr
    return total


def _get_param(params, key, default=None):
    value = params.get(key, default)
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return default
    return value


def _parse_float_param(params, key, default=None):
    value = _get_param(params, key, None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid float for '{key}': {value}")


def _parse_int_param(params, key, default=None):
    value = _get_param(params, key, None)
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer for '{key}': {value}")


def _parse_bool_param(params, key, default=False):
    raw = _get_param(params, key, None)
    parsed = _parse_bool(raw, default=None)
    if parsed is None:
        return default
    return bool(parsed)


def _parse_dict_param(params, key):
    raw = _get_param(params, key, None)
    if raw is None:
        return None
    value = _parse_literal_value(raw, None)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a JSON/dict mapping.")
    return dict(value)


def _parse_idx_list_param(params, key):
    raw = _get_param(params, key, None)
    if raw is None:
        return None
    parsed = _parse_literal_value(raw, None)
    if isinstance(parsed, (list, tuple, np.ndarray)):
        out = [int(x) for x in np.asarray(parsed).tolist()]
        return out if out else None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _parse_range_pair(params, min_key, max_key):
    lo = _parse_float_param(params, min_key, default=None)
    hi = _parse_float_param(params, max_key, default=None)
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise ValueError(f"Both '{min_key}' and '{max_key}' must be provided.")
    if lo >= hi:
        raise ValueError(f"Invalid range: {min_key} must be less than {max_key}.")
    return [float(lo), float(hi)]


def _resolve_fs_from_df(df):
    if "fs" not in df.columns:
        return None
    series = pd.to_numeric(df["fs"], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _extract_feature_samples(df):
    if "data" not in df.columns:
        raise ValueError("Input dataframe must contain a 'data' column from EphysDatasetParser.")
    samples = []
    for idx, value in enumerate(df["data"].tolist()):
        arr = np.asarray(value).squeeze()
        if arr.ndim == 0:
            arr = np.asarray([arr])
        if arr.ndim != 1:
            raise ValueError(f"Row {idx} in 'data' is not a 1D signal.")
        samples.append(arr)
    if not samples:
        raise ValueError("No signals found in dataframe 'data' column.")
    return samples


def _normalize_feature_method_name(raw_method):
    mapping = {
        "power_spectrum_parameterization": "specparam",
        "fei": "fEI",
    }
    method = (raw_method or "").strip()
    return mapping.get(method, method)


def _resolve_sampling_frequency(params, key, df, label):
    fs = _parse_float_param(params, key, default=None)
    if fs is None:
        fs = _resolve_fs_from_df(df)
    if fs is None:
        raise ValueError(f"Sampling frequency is required for {label}.")
    return float(fs)


def _build_feature_method_params(method, params, df):
    method_params = {}
    n_jobs = _parse_int_param(params, "features_n_jobs", default=None)
    chunksize = _parse_int_param(params, "features_chunksize", default=None)
    start_method = _get_param(params, "features_start_method", "spawn")
    if start_method not in {"spawn", "fork", "forkserver"}:
        start_method = "spawn"

    if method == "catch22":
        method_params["normalize"] = _parse_bool_param(params, "catch22_normalize", default=False)

    elif method == "specparam":
        method_params["normalize"] = _parse_bool_param(params, "specparam_normalize", default=False)
        method_params["fs"] = _resolve_sampling_frequency(params, "specparam_fs", df, "specparam")
        freq_range = _parse_range_pair(params, "specparam_freq_min", "specparam_freq_max")
        if freq_range is not None:
            method_params["freq_range"] = tuple(freq_range)

        welch_kwargs = _parse_dict_param(params, "specparam_welch_kwargs") or {}
        nperseg = _parse_int_param(params, "specparam_welch_nperseg", default=None)
        noverlap = _parse_int_param(params, "specparam_welch_noverlap", default=None)
        if nperseg is not None:
            welch_kwargs["nperseg"] = int(nperseg)
        if noverlap is not None:
            welch_kwargs["noverlap"] = int(noverlap)
        if welch_kwargs:
            method_params["welch_kwargs"] = welch_kwargs

        model_kwargs = _parse_dict_param(params, "specparam_model_kwargs")
        if model_kwargs:
            method_params["model_kwargs"] = model_kwargs

        method_params["select_peak"] = _get_param(params, "specparam_select_peak", "max_pw")
        method_params["metric_policy"] = _get_param(params, "specparam_metric_policy", "reject")
        method_params["debug"] = _parse_bool_param(params, "specparam_debug", default=False)

        thresholds = _parse_dict_param(params, "specparam_metric_thresholds") or {}
        gof_r2 = _parse_float_param(params, "specparam_threshold_gof_rsquared", default=None)
        if gof_r2 is not None:
            thresholds["gof_rsquared"] = float(gof_r2)
        if thresholds:
            method_params["metric_thresholds"] = thresholds

    elif method == "dfa":
        method_params["normalize"] = _parse_bool_param(params, "dfa_normalize", default=False)
        method_params["sampling_frequency"] = _resolve_sampling_frequency(params, "dfa_sampling_frequency", df, "DFA")
        fit_interval = _parse_range_pair(params, "dfa_fit_min", "dfa_fit_max")
        compute_interval = _parse_range_pair(params, "dfa_compute_min", "dfa_compute_max")
        if fit_interval is not None:
            method_params["fit_interval"] = fit_interval
        if compute_interval is not None:
            method_params["compute_interval"] = compute_interval
        method_params["overlap"] = _parse_bool_param(params, "dfa_overlap", default=True)

        dfa_mode = _get_param(params, "dfa_analysis_mode", "envelope")
        if dfa_mode == "frequency_range":
            method_params["frequency_range"] = _parse_range_pair(params, "dfa_frequency_min", "dfa_frequency_max")
            if method_params["frequency_range"] is None:
                raise ValueError("DFA frequency_range mode requires min/max frequency values.")
        elif dfa_mode == "spectrum_range":
            method_params["spectrum_range"] = _parse_range_pair(params, "dfa_spectrum_min", "dfa_spectrum_max")
            if method_params["spectrum_range"] is None:
                raise ValueError("DFA spectrum_range mode requires min/max frequency values.")
        else:
            if fit_interval is None or compute_interval is None:
                raise ValueError("DFA envelope mode requires fit_interval and compute_interval.")

        bad_idxes = _parse_idx_list_param(params, "dfa_bad_idxes")
        if bad_idxes is not None:
            method_params["bad_idxes"] = bad_idxes

        method_params["input_is_envelope"] = _parse_bool_param(params, "dfa_input_is_envelope", default=True)
        filter_kwargs = _parse_dict_param(params, "dfa_filter_kwargs")
        if filter_kwargs:
            method_params["filter_kwargs"] = filter_kwargs
        trim_seconds = _parse_float_param(params, "dfa_trim_seconds", default=None)
        if trim_seconds is not None:
            method_params["trim_seconds"] = float(trim_seconds)
        hilbert_n_fft = _parse_int_param(params, "dfa_hilbert_n_fft", default=None)
        if hilbert_n_fft is not None:
            method_params["hilbert_n_fft"] = int(hilbert_n_fft)

    elif method == "fEI":
        method_params["normalize"] = _parse_bool_param(params, "fei_normalize", default=False)
        method_params["sampling_frequency"] = _resolve_sampling_frequency(params, "fei_sampling_frequency", df, "fEI")
        window_size_sec = _parse_float_param(params, "fei_window_size_sec", default=None)
        if window_size_sec is None:
            raise ValueError("fEI requires window_size_sec.")
        method_params["window_size_sec"] = float(window_size_sec)
        method_params["window_overlap"] = _parse_float_param(params, "fei_window_overlap", default=0.0)
        dfa_value = _parse_float_param(params, "fei_dfa_value", default=None)
        if dfa_value is not None:
            method_params["dfa_value"] = float(dfa_value)
        dfa_threshold = _parse_float_param(params, "fei_dfa_threshold", default=0.6)
        method_params["dfa_threshold"] = float(dfa_threshold)

        dfa_fit_interval = _parse_range_pair(params, "fei_dfa_fit_min", "fei_dfa_fit_max")
        dfa_compute_interval = _parse_range_pair(params, "fei_dfa_compute_min", "fei_dfa_compute_max")
        if dfa_fit_interval is not None:
            method_params["dfa_fit_interval"] = dfa_fit_interval
        if dfa_compute_interval is not None:
            method_params["dfa_compute_interval"] = dfa_compute_interval
        method_params["dfa_overlap"] = _parse_bool_param(params, "fei_dfa_overlap", default=True)

        fei_mode = _get_param(params, "fei_analysis_mode", "envelope")
        if fei_mode == "frequency_range":
            method_params["frequency_range"] = _parse_range_pair(params, "fei_frequency_min", "fei_frequency_max")
            if method_params["frequency_range"] is None:
                raise ValueError("fEI frequency_range mode requires min/max frequency values.")
        elif fei_mode == "spectrum_range":
            method_params["spectrum_range"] = _parse_range_pair(params, "fei_spectrum_min", "fei_spectrum_max")
            if method_params["spectrum_range"] is None:
                raise ValueError("fEI spectrum_range mode requires min/max frequency values.")
        else:
            if dfa_value is None and (dfa_fit_interval is None or dfa_compute_interval is None):
                raise ValueError("fEI envelope mode requires dfa intervals unless dfa_value is provided.")

        bad_idxes = _parse_idx_list_param(params, "fei_bad_idxes")
        if bad_idxes is not None:
            method_params["bad_idxes"] = bad_idxes
        method_params["input_is_envelope"] = _parse_bool_param(params, "fei_input_is_envelope", default=True)
        filter_kwargs = _parse_dict_param(params, "fei_filter_kwargs")
        if filter_kwargs:
            method_params["filter_kwargs"] = filter_kwargs
        trim_seconds = _parse_float_param(params, "fei_trim_seconds", default=None)
        if trim_seconds is not None:
            method_params["trim_seconds"] = float(trim_seconds)
        hilbert_n_fft = _parse_int_param(params, "fei_hilbert_n_fft", default=None)
        if hilbert_n_fft is not None:
            method_params["hilbert_n_fft"] = int(hilbert_n_fft)

    elif method == "hctsa":
        hctsa_folder = _get_param(params, "hctsa_folder", None)
        if not hctsa_folder:
            raise ValueError("hctsa_folder is required for hctsa.")
        method_params["hctsa_folder"] = hctsa_folder

    else:
        raise ValueError(f"Unsupported features method '{method}'.")

    return method_params, {
        "n_jobs": n_jobs,
        "chunksize": chunksize,
        "start_method": start_method,
        "hctsa_return_meta": _parse_bool_param(params, "hctsa_return_meta", default=False),
    }




#############################################################
##########        COMPUTATION FUNCTIONS           ###########
#############################################################


def features_computation(job_id, job_status, params, temp_uploaded_files):
    output_df_path = None
    try:
        _append_job_output(job_status, job_id, "Starting features computation.")
        if job_id in job_status:
            # Keep progress at 0 during data loading/preparation.
            job_status[job_id]["progress"] = 0
        _append_job_output(job_status, job_id, "Loading data for features...")

        prepared_df = params.get("prepared_features_df")
        if isinstance(prepared_df, pd.DataFrame):
            df = prepared_df
            _append_job_output(job_status, job_id, "Using in-memory parsed empirical dataframe (no additional load).")
        elif params.get("empirical_upload_paths"):
            parse_cfg = params.get("parser_config_obj")
            if parse_cfg is None:
                raise ValueError("Missing parser configuration for empirical uploads.")
            from ncpi.EphysDatasetParser import EphysDatasetParser
            parser = EphysDatasetParser(parse_cfg)
            subject_id_cfg = ((parse_cfg.fields.metadata or {}).get("subject_id")
                              if getattr(parse_cfg, "fields", None) is not None else None)
            use_file_subject_id = isinstance(subject_id_cfg, str) and subject_id_cfg == "file_ID"

            empirical_uploads = list(params.get("empirical_upload_paths") or [])
            total_uploads = len(empirical_uploads)
            if total_uploads == 0:
                raise ValueError("No empirical uploads were provided.")

            _append_job_output(job_status, job_id, f"Parsing {total_uploads} empirical file(s)...")
            parsed_frames = []
            log_every = max(1, total_uploads // 20)
            for idx, payload in enumerate(empirical_uploads, start=1):
                name = payload.get("name") or f"file_{idx}"
                source_path = payload.get("path")
                size_mb = 0.0
                if source_path and os.path.exists(source_path):
                    try:
                        size_mb = os.path.getsize(source_path) / (1024 * 1024)
                    except OSError:
                        size_mb = 0.0
                _append_job_output(
                    job_status,
                    job_id,
                    f"Loading empirical file {idx}/{total_uploads}: {name} ({size_mb:.2f} MB)."
                )
                source_obj = _load_uploaded_source_path(
                    source_path,
                    payload.get("name"),
                    payload.get("ext"),
                )
                parsed = parser.parse(source_obj)
                if not isinstance(parsed, pd.DataFrame):
                    raise ValueError(f"Parser output for '{name}' is not a dataframe.")
                if use_file_subject_id:
                    parsed["subject_id"] = str(name)
                parsed_frames.append(parsed)

                # Do not advance global progress bar during data loading.
                if idx == 1 or idx == total_uploads or (idx % log_every == 0):
                    _append_job_output(job_status, job_id, f"Parsed empirical file {idx}/{total_uploads}.")

            df = pd.concat(parsed_frames, ignore_index=True)
            _append_job_output(job_status, job_id, f"Merged empirical parsed dataframe shape: {df.shape}.")
        else:
            input_path = params.get("file_paths", {}).get("data_file")
            if not input_path:
                raise ValueError("No parsed dataframe input was provided for feature computation.")
            source_label = os.path.basename(str(input_path))
            _append_job_output(job_status, job_id, f"Reading dataframe from {source_label}...")
            df = read_df_file(input_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Prepared parser output is not a pandas dataframe.")
        _append_job_output(job_status, job_id, f"Loaded parsed dataframe with shape {df.shape}.")

        samples = _extract_feature_samples(df)
        _append_job_output(job_status, job_id, f"Extracted {len(samples)} signal sample(s) from dataframe.")

        method = _normalize_feature_method_name(params.get('select-method'))
        method_params, exec_opts = _build_feature_method_params(method, params, df)
        _append_job_output(
            job_status,
            job_id,
            f"Method: {method} | n_jobs={exec_opts['n_jobs']} | chunksize={exec_opts['chunksize']} | start_method={exec_opts['start_method']}"
        )
        features = ncpi.Features(method=method, params=method_params)
        _append_job_output(job_status, job_id, "Starting feature extraction...")

        if method == "hctsa" and exec_opts["hctsa_return_meta"]:
            _append_job_output(job_status, job_id, "Running hctsa feature extraction...")
            hctsa_result = features.hctsa(
                samples,
                hctsa_folder=method_params["hctsa_folder"],
                workers=exec_opts["n_jobs"],
                return_meta=True,
            )
            if job_id in job_status:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 99)
            output_df = df.copy()
            output_df["Features"] = hctsa_result["features"]
            output_df["hctsa_num_valid_features"] = int(hctsa_result["num_valid_features"])
            output_df["hctsa_num_total_features"] = int(hctsa_result["num_total_features"])
        else:
            sig = inspect.signature(features.compute_features)
            supports_progress = "progress_callback" in sig.parameters

            def _on_feature_progress(completed, total, pct):
                if total <= 0:
                    return
                mapped = int(max(0, min(99, float(pct))))
                if job_id in job_status:
                    # Follow ncpi compute_features progress directly.
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), mapped)

            def _on_feature_log(message):
                _append_job_output(job_status, job_id, str(message))

            if not supports_progress:
                _append_job_output(
                    job_status,
                    job_id,
                    "Warning: compute_features progress callbacks are unavailable in this ncpi version."
                )

            computed = _compute_features_with_compat(
                features_obj=features,
                samples=samples,
                exec_opts=exec_opts,
                progress_callback=_on_feature_progress,
                log_callback=_on_feature_log,
            )
            if job_id in job_status:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 99)
            output_df = df.copy()
            output_df["Features"] = computed

        _append_job_output(job_status, job_id, "Attaching computed features to dataframe.")
        output_df["FeatureMethod"] = method

        # By default, persist features into the same parsed dataframe pickle.
        input_df_path = params['file_paths'].get('data_file')
        input_ext = os.path.splitext(str(input_df_path or ""))[1].lower()
        if input_df_path and input_ext in {'.pkl', '.pickle'}:
            output_df.to_pickle(input_df_path)
            output_df_path = input_df_path
            _append_job_output(job_status, job_id, f"Updated input dataframe in-place: {input_df_path}")
        else:
            output_df_path = save_df(job_id, output_df, temp_uploaded_files)
            _append_job_output(job_status, job_id, f"Saved computed dataframe: {output_df_path}")
        persisted_dashboard_path = _persist_features_dataframe(output_df, method)
        _append_job_output(job_status, job_id, f"Persisted dashboard features file: {persisted_dashboard_path}")

        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": output_df_path, # Return to the client the output filepath
                "dashboard_features_path": persisted_dashboard_path,
                "error": False
            })

    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    # Remove temporary inputs but keep result file available for download.
    cleanup_temp_files(params['file_paths'], keep_paths=[output_df_path])



def _first_existing_file(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _copy_artifact_if_present(src_path, dst_path):
    if not src_path:
        return False
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Artifact file not found: {src_path}")
    shutil.copy2(src_path, dst_path)
    return True


def _load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _features_series_to_matrix(features_series):
    rows = []
    expected_dim = None
    for idx, value in enumerate(features_series):
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"Features row {idx} is a string that cannot be parsed as numeric data.") from exc
        try:
            row = np.asarray(value, dtype=float)
        except Exception as exc:
            raise ValueError(f"Features row {idx} cannot be converted to a numeric vector.") from exc
        if row.ndim == 0:
            row = row.reshape(1)
        else:
            row = row.reshape(-1)
        if expected_dim is None:
            expected_dim = int(row.shape[0])
        elif int(row.shape[0]) != expected_dim:
            raise ValueError(
                f"Features row {idx} has length {row.shape[0]}, expected {expected_dim}. "
                "All feature rows must have the same length."
            )
        rows.append(row)
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _generic_series_to_matrix(series, label):
    rows = []
    expected_dim = None
    for idx, value in enumerate(series):
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"{label} row {idx} is a string that cannot be parsed as numeric data.") from exc
        try:
            row = np.asarray(value, dtype=float)
        except Exception as exc:
            raise ValueError(f"{label} row {idx} cannot be converted to a numeric vector.") from exc
        if row.ndim == 0:
            row = row.reshape(1)
        else:
            row = row.reshape(-1)
        if expected_dim is None:
            expected_dim = int(row.shape[0])
        elif int(row.shape[0]) != expected_dim:
            raise ValueError(
                f"{label} row {idx} has length {row.shape[0]}, expected {expected_dim}. "
                f"All {label.lower()} rows must have the same length."
            )
        rows.append(row)
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _coerce_training_matrix(obj, label):
    if isinstance(obj, pd.DataFrame):
        if label == "Features":
            if "Features" in obj.columns:
                return _features_series_to_matrix(obj["Features"])
            if obj.shape[1] == 1 and str(obj.columns[0]).lower() in {"features", "feature", "x"}:
                return _features_series_to_matrix(obj.iloc[:, 0])
        if label == "Parameters":
            if "Parameters" in obj.columns:
                return _generic_series_to_matrix(obj["Parameters"], label)
            if obj.shape[1] == 1 and str(obj.columns[0]).lower() in {"parameters", "parameter", "theta", "y"}:
                return _generic_series_to_matrix(obj.iloc[:, 0], label)

        # Generic dataframe path: all numeric columns.
        numeric_df = obj.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 0:
            arr = numeric_df.to_numpy(dtype=float, copy=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr

        # If dataframe is object-heavy, attempt row-wise vector coercion.
        if obj.shape[1] == 1:
            return _generic_series_to_matrix(obj.iloc[:, 0], label)
        raise ValueError(f"{label} dataframe does not contain numeric columns.")

    if isinstance(obj, pd.Series):
        return _generic_series_to_matrix(obj, label)

    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj, dtype=object if isinstance(obj, (list, tuple)) else None)
        if arr.ndim == 1 and arr.dtype == object:
            return _generic_series_to_matrix(arr, label)
        try:
            out = np.asarray(arr, dtype=float)
        except Exception as exc:
            raise ValueError(f"{label} array cannot be converted to numeric values.") from exc
        if out.ndim == 0:
            out = out.reshape(1, 1)
        elif out.ndim == 1:
            out = out.reshape(-1, 1)
        elif out.ndim > 2:
            raise ValueError(f"{label} array must be 1D or 2D.")
        return out

    raise ValueError(f"Unsupported {label.lower()} input type: {type(obj).__name__}.")


def _read_training_input_any(file_path):
    """
    Load training inputs with permissive format handling.
    This is intentionally broader than read_df_file()/allowed_file checks so
    training uploads can accept arbitrary extensions as long as content is parseable.
    """
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

    ext = os.path.splitext(str(file_path))[1].lower()
    loaders = []

    # Extension-prioritized attempts first.
    def _pickle_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if ext in {".pkl", ".pickle", ".bin", ".dat"}:
        loaders.extend([
            ("pandas.read_pickle", lambda p: pd.read_pickle(p)),
            ("pickle.load", _pickle_load),
        ])
    elif ext == ".csv":
        loaders.append(("pandas.read_csv", lambda p: pd.read_csv(p)))
    elif ext == ".parquet":
        loaders.append(("pandas.read_parquet", lambda p: pd.read_parquet(p)))
    elif ext == ".feather":
        loaders.append(("pandas.read_feather", lambda p: pd.read_feather(p)))
    elif ext in {".xlsx", ".xls"}:
        loaders.append(("pandas.read_excel", lambda p: pd.read_excel(p)))
    elif ext in {".npy", ".npz"}:
        loaders.append(("numpy.load", lambda p: np.load(p, allow_pickle=True)))
    elif ext == ".mat":
        try:
            import scipy.io as sio
            loaders.append(("scipy.io.loadmat", lambda p: sio.loadmat(p, squeeze_me=True, struct_as_record=False)))
        except Exception:
            pass

    # Generic fallbacks (for unknown extensions or when extension is misleading).
    generic_fallbacks = [
        ("pandas.read_pickle", lambda p: pd.read_pickle(p)),
        ("pickle.load", _pickle_load),
        ("pandas.read_csv", lambda p: pd.read_csv(p)),
        ("pandas.read_parquet", lambda p: pd.read_parquet(p)),
        ("pandas.read_feather", lambda p: pd.read_feather(p)),
        ("pandas.read_excel", lambda p: pd.read_excel(p)),
        ("numpy.load", lambda p: np.load(p, allow_pickle=True)),
    ]
    if ext != ".mat":
        try:
            import scipy.io as sio
            generic_fallbacks.append(("scipy.io.loadmat", lambda p: sio.loadmat(p, squeeze_me=True, struct_as_record=False)))
        except Exception:
            pass
    loaders.extend(generic_fallbacks)

    seen = set()
    errors = []
    for name, loader in loaders:
        if name in seen:
            continue
        seen.add(name)
        try:
            obj = loader(file_path)
            return obj
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            continue

    raise ValueError(
        "Could not parse uploaded training file with available loaders. "
        f"Tried: {', '.join(errors[:8])}."
    )


def _load_training_with_example_loader(features_path, parameters_path):
    """
    Try loading X/theta using examples/tools.py::load_model_features by creating
    a temporary Zenodo-like folder structure around uploaded files.
    """
    tools_path = os.path.join(_repo_root, "examples", "tools.py")
    if not os.path.isfile(tools_path):
        raise FileNotFoundError("examples/tools.py not found.")

    module = _load_module_from_path(tools_path, name=f"examples_tools_{os.getpid()}")
    loader = getattr(module, "load_model_features", None)
    if not callable(loader):
        raise AttributeError("load_model_features not found in examples/tools.py.")

    temp_root = tempfile.mkdtemp(prefix="ncpi_train_loader_")
    method_name = "uploaded"
    method_dir = os.path.join(temp_root, "data", method_name)
    os.makedirs(method_dir, exist_ok=True)
    sim_x = os.path.join(method_dir, "sim_X")
    sim_theta = os.path.join(method_dir, "sim_theta")
    shutil.copy2(features_path, sim_x)
    shutil.copy2(parameters_path, sim_theta)

    try:
        X, theta = loader(method_name, temp_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if not isinstance(theta, dict) or "data" not in theta:
        raise ValueError("examples.tools.load_model_features returned theta without key 'data'.")

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 0:
        X_arr = X_arr.reshape(1, 1)
    elif X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim > 2:
        raise ValueError("X loaded from examples loader must be 1D or 2D.")

    Y_arr = np.asarray(theta["data"], dtype=float)
    if Y_arr.ndim == 0:
        Y_arr = Y_arr.reshape(1, 1)
    elif Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    elif Y_arr.ndim > 2:
        raise ValueError("theta['data'] loaded from examples loader must be 1D or 2D.")

    return X_arr, Y_arr


def _parse_param_grid(raw_value):
    parsed = _parse_literal_value(raw_value, None)
    if parsed is None:
        return None

    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError("param_grid must be a list of dicts.")
        return [dict(item) for item in parsed]

    if isinstance(parsed, dict):
        keys = list(parsed.keys())
        value_lists = []
        for key in keys:
            value = parsed[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                values = list(np.asarray(value, dtype=object).tolist())
            else:
                values = [value]
            if not values:
                raise ValueError(f"param_grid entry '{key}' has no values.")
            value_lists.append(values)

        combos = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
        if len(combos) > 256:
            raise ValueError("param_grid expands to more than 256 combinations; reduce the search space.")
        return combos

    raise ValueError("param_grid must be a dict or a list of dicts.")


def _parse_numeric_vector(raw_value, default_vector):
    if raw_value is None or str(raw_value).strip() == "":
        return np.asarray(default_vector, dtype=float)
    parsed = _parse_literal_value(raw_value, None)
    if parsed is None:
        parsed = raw_value
    if isinstance(parsed, str):
        parts = [p.strip() for p in parsed.split(",") if p.strip()]
        if not parts:
            return np.asarray(default_vector, dtype=float)
        parsed = [float(p) for p in parts]
    arr = np.asarray(parsed, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray(default_vector, dtype=float)
    return arr


def _is_posterior_like(obj):
    return hasattr(obj, "sample")


def _is_inference_like(obj):
    return hasattr(obj, "build_posterior")


def _sample_sbi_posterior(posterior, torch_mod, x_valid, num_samples, batch_size, show_progress=False):
    x_local = np.asarray(x_valid, dtype=np.float32)
    x_t = torch_mod.from_numpy(x_local)
    total = int(x_t.shape[0])
    if total == 0:
        return np.empty((num_samples, 0, 0), dtype=float)

    chunks = []
    for start in range(0, total, batch_size):
        xb = x_t[start:start + batch_size]
        b = int(xb.shape[0])
        if b == 1:
            sb = posterior.sample((num_samples,), x=xb, show_progress_bars=bool(show_progress))
            if sb.ndim == 2:
                sb = sb.unsqueeze(1)
        else:
            if hasattr(posterior, "sample_batched"):
                sb = posterior.sample_batched((num_samples,), x=xb, show_progress_bars=bool(show_progress))
            else:
                single_chunks = []
                for one_idx in range(b):
                    one = posterior.sample(
                        (num_samples,),
                        x=xb[one_idx:one_idx + 1],
                        show_progress_bars=bool(show_progress),
                    )
                    if one.ndim == 2:
                        one = one.unsqueeze(1)
                    single_chunks.append(one)
                sb = torch_mod.cat(single_chunks, dim=1)
        chunks.append(sb)

    sample_tensor = torch_mod.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    samples = sample_tensor.detach().cpu().numpy()
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    return samples


def _summarize_sbi_samples(samples, mode):
    if mode == "median":
        summary = np.median(samples, axis=0)
    else:
        summary = np.mean(samples, axis=0)
    if summary.ndim == 1:
        summary = summary.reshape(1, -1)
    return summary


def _resolve_sbi_posteriors(model_obj, density_obj, build_kwargs):
    source = model_obj
    if source is None:
        raise ValueError(
            "Missing SBI model artifact. Provide a model/posterior file."
        )

    if _is_posterior_like(source):
        return [source]

    if isinstance(source, list):
        if source and all(_is_posterior_like(item) for item in source):
            return list(source)
        if source and all(_is_inference_like(item) for item in source):
            if density_obj is None:
                raise ValueError(
                    "SBI model list contains inference objects. Upload density_estimator.pkl "
                    "to build posteriors for prediction."
                )
            if isinstance(density_obj, list):
                if len(density_obj) != len(source):
                    raise ValueError(
                        "density_estimator list length does not match number of inference objects."
                    )
                densities = density_obj
            else:
                densities = [density_obj] * len(source)
            return [
                inf_obj.build_posterior(de_obj, **build_kwargs)
                for inf_obj, de_obj in zip(source, densities)
            ]

    if _is_inference_like(source):
        if density_obj is None:
            raise ValueError(
                "SBI inference object requires density_estimator.pkl to build posterior for prediction."
            )
        return [source.build_posterior(density_obj, **build_kwargs)]

    raise ValueError(
        f"Unsupported SBI artifact type: {type(source).__name__}. "
        "Expected posterior object(s) or inference object(s)."
    )


def _normalize_prediction_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) if np.isscalar(v) else v for v in value]
    if value is None:
        return np.nan
    if np.isscalar(value):
        try:
            return float(value)
        except Exception:
            return value
    return value


def _infer_artifact_backend(model_obj):
    if model_obj is None:
        return None, "None"

    if _is_posterior_like(model_obj) or _is_inference_like(model_obj):
        return "sbi", type(model_obj).__name__

    if isinstance(model_obj, list):
        if not model_obj:
            return None, "list[empty]"
        first_name = type(model_obj[0]).__name__
        if all(_is_posterior_like(item) for item in model_obj):
            return "sbi", f"list[{first_name}]"
        if all(_is_inference_like(item) for item in model_obj):
            return "sbi", f"list[{first_name}]"
        if all(hasattr(item, "predict") for item in model_obj):
            return "sklearn", f"list[{first_name}]"
        return None, f"list[{first_name}]"

    if hasattr(model_obj, "predict"):
        return "sklearn", type(model_obj).__name__

    return None, type(model_obj).__name__


def _initialize_inference_from_artifact(model_obj, requested_model_name):
    inferred_backend, inferred_type = _infer_artifact_backend(model_obj)
    if inferred_backend is None:
        raise ValueError(
            f"Could not infer backend from uploaded model artifact type '{inferred_type}'."
        )

    requested = (requested_model_name or "").strip()
    if requested in {"", "__auto__", "auto"}:
        requested = ""

    if inferred_backend == "sklearn":
        candidates = []
        if requested:
            candidates.append(requested)
        if isinstance(model_obj, list) and model_obj:
            candidates.append(type(model_obj[0]).__name__)
        elif model_obj is not None:
            candidates.append(type(model_obj).__name__)
        # Safe fallback that guarantees sklearn backend initialization.
        candidates.append("MLPRegressor")

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                obj = ncpi.Inference(model=candidate)
                if obj.backend == "sklearn":
                    return obj, candidate, inferred_backend, inferred_type
            except Exception:
                continue
        raise ValueError(
            "Unable to initialize sklearn inference backend from uploaded model artifact."
        )

    sbi_candidates = []
    if requested in {"NPE", "NLE", "NRE"}:
        sbi_candidates.append(requested)
    sbi_candidates.extend(["NPE", "NLE", "NRE"])

    seen = set()
    for candidate in sbi_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = ncpi.Inference(model=candidate)
            if obj.backend == "sbi":
                return obj, candidate, inferred_backend, inferred_type
        except Exception:
            continue
    raise ValueError("Unable to initialize SBI inference backend.")


def inference_computation(job_id, job_status, params, temp_uploaded_files):
    file_paths = params.get("file_paths", {})
    artifacts_dir = os.path.join(temp_uploaded_files, f"inference_assets_{job_id}")
    output_df_path = None

    try:
        _append_job_output(job_status, job_id, "Starting predictions computation.")

        features_predict_path = file_paths.get("features_predict")
        if not features_predict_path:
            raise ValueError("Missing features prediction input. Upload features data or select existing features data.")

        _append_job_output(job_status, job_id, "Loading features dataframe...")
        df_features_predict = read_df_file(features_predict_path)
        if not isinstance(df_features_predict, pd.DataFrame):
            raise ValueError("Features prediction input must be a pandas dataframe.")
        if "Features" not in df_features_predict.columns:
            raise ValueError("Input dataframe must contain a 'Features' column.")

        feature_matrix = _features_series_to_matrix(df_features_predict["Features"])
        if feature_matrix.shape[0] == 0:
            raise ValueError("Input features dataframe is empty.")
        _append_job_output(
            job_status,
            job_id,
            f"Loaded features dataframe shape={df_features_predict.shape}, rows for prediction={feature_matrix.shape[0]}."
        )

        model_assets_source = (params.get("model_assets_source") or "upload").strip().lower()
        if model_assets_source != "upload":
            raise ValueError("Only uploaded prediction artifacts are supported.")

        model_uploaded = _first_existing_file([file_paths.get("model_file"), file_paths.get("model-file")])
        scaler_uploaded = _first_existing_file([file_paths.get("scaler_file"), file_paths.get("scaler-file")])
        density_uploaded = _first_existing_file([
            file_paths.get("density_estimator_file"),
            file_paths.get("density-estimator-file"),
        ])

        model_source = model_uploaded
        scaler_source = scaler_uploaded
        density_source = density_uploaded

        os.makedirs(artifacts_dir, exist_ok=True)
        model_dst = os.path.join(artifacts_dir, "model.pkl")
        scaler_dst = os.path.join(artifacts_dir, "scaler.pkl")
        density_dst = os.path.join(artifacts_dir, "density_estimator.pkl")

        model_present = _copy_artifact_if_present(model_source, model_dst)
        scaler_present = _copy_artifact_if_present(scaler_source, scaler_dst)
        density_present = _copy_artifact_if_present(density_source, density_dst)
        _append_job_output(
            job_status,
            job_id,
            f"Prepared artifacts: model={'yes' if model_present else 'no'}, scaler={'yes' if scaler_present else 'no'}, density_estimator={'yes' if density_present else 'no'}."
        )
        if not model_present:
            raise ValueError("Model artifact is required for prediction.")

        requested_model_name = (params.get("inference_model_name") or params.get("model") or "").strip()
        if requested_model_name == "__custom__":
            requested_model_name = (params.get("inference_model_name_custom") or "").strip()

        model_obj = _load_pickle_file(model_dst)
        inference_obj, init_model_name, inferred_backend, inferred_type = _initialize_inference_from_artifact(
            model_obj,
            requested_model_name,
        )
        _append_job_output(
            job_status,
            job_id,
            f"Inferred model artifact: type={inferred_type}, backend={inferred_backend}. "
            f"Initialized ncpi.Inference(model='{init_model_name}').",
        )

        output_df = df_features_predict.copy()

        use_scaler = str(params.get("use_scaler", "")).lower() in {"1", "true", "on", "yes"}
        _append_job_output(job_status, job_id, f"Scaler enabled={'yes' if use_scaler else 'no'}.")
        inference_n_jobs = _parse_int_param(params, "inference_n_jobs", default=None)
        inference_chunksize = _parse_int_param(params, "inference_chunksize", default=None)
        inference_start_method = _get_param(params, "inference_start_method", "spawn")
        if inference_start_method not in {"spawn", "fork", "forkserver"}:
            inference_start_method = "spawn"
        _append_job_output(
            job_status,
            job_id,
            "Execution settings: "
            f"n_jobs={inference_n_jobs if inference_n_jobs is not None else 'auto'}, "
            f"chunksize={inference_chunksize if inference_chunksize is not None else 'auto'}, "
            f"start_method={inference_start_method}.",
        )

        if inference_obj.backend == "sklearn":
            if not model_present:
                raise ValueError("Sklearn prediction requires model.pkl.")
            if use_scaler and not scaler_present:
                raise ValueError("Scaler usage is enabled, but scaler.pkl was not provided.")

            _append_job_output(job_status, job_id, "Running ncpi.Inference.predict() for sklearn model(s)...")
            capture = _JobOutputCapture(job_status, job_id, progress_base=0, progress_span=95)
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                predictions = _predict_inference_with_compat(
                    inference_obj,
                    feature_matrix,
                    base_kwargs={
                        "result_dir": artifacts_dir,
                        "scaler": True if (use_scaler and scaler_present) else None,
                    },
                    exec_kwargs={
                        "n_jobs": inference_n_jobs,
                        "chunksize": inference_chunksize,
                        "start_method": inference_start_method,
                    },
                    log_callback=lambda msg: _append_job_output(job_status, job_id, msg),
                )
            capture.flush()

            output_df["Predictions"] = [_normalize_prediction_value(pred) for pred in predictions]
            if job_id in job_status:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 92)

        else:
            # model_obj already loaded above for automatic backend/type inference.
            density_obj = _load_pickle_file(density_dst) if density_present else None
            _append_job_output(job_status, job_id, "Preparing SBI posterior sampling...")

            build_kwargs_raw = (params.get("sbi_build_posterior_kwargs") or "").strip()
            build_kwargs = {}
            if build_kwargs_raw:
                try:
                    build_kwargs = json.loads(build_kwargs_raw)
                except json.JSONDecodeError as exc:
                    raise ValueError("SBI build posterior kwargs must be valid JSON.") from exc
                if not isinstance(build_kwargs, dict):
                    raise ValueError("SBI build posterior kwargs JSON must decode to an object/dict.")

            sbi_summary_mode = (params.get("sbi_summary_mode") or "mean").strip().lower()
            if sbi_summary_mode not in {"mean", "median"}:
                raise ValueError("SBI summary mode must be 'mean' or 'median'.")

            num_samples_raw = (
                params.get("sbi_num_posterior_samples")
                or params.get("posterior-samples")
                or "500"
            )
            sbi_batch_raw = (
                params.get("sbi_batch_size")
                or params.get("sbi-batch-size")
                or "256"
            )
            try:
                num_samples = int(num_samples_raw)
            except Exception as exc:
                raise ValueError("SBI posterior samples must be a valid integer.") from exc
            try:
                sbi_batch_size = int(sbi_batch_raw)
            except Exception as exc:
                raise ValueError("SBI batch size must be a valid integer.") from exc
            if num_samples <= 0:
                raise ValueError("SBI posterior samples must be > 0.")
            if sbi_batch_size <= 0:
                raise ValueError("SBI batch size must be > 0.")

            scaler_obj = None
            if use_scaler:
                if not scaler_present:
                    raise ValueError("Scaler usage is enabled, but scaler.pkl was not provided.")
                scaler_obj = _load_pickle_file(scaler_dst)

            posteriors = _resolve_sbi_posteriors(model_obj, density_obj, build_kwargs)
            _append_job_output(job_status, job_id, f"Resolved {len(posteriors)} SBI posterior object(s).")

            x_local = np.asarray(feature_matrix, dtype=float)
            finite_mask = np.isfinite(x_local).all(axis=1)
            preds = [np.nan] * int(x_local.shape[0])
            if np.any(finite_mask):
                x_valid = x_local[finite_mask]
                if scaler_obj is not None:
                    x_valid = scaler_obj.transform(x_valid)
                    valid_after_scale = np.isfinite(x_valid).all(axis=1)
                    valid_indices = np.where(finite_mask)[0]
                    finite_mask[:] = False
                    finite_mask[valid_indices[valid_after_scale]] = True
                    x_valid = x_valid[valid_after_scale]

                if x_valid.shape[0] > 0:
                    per_model_summaries = []
                    posterior_count = max(1, len(posteriors))
                    for idx, posterior in enumerate(posteriors, start=1):
                        _append_job_output(job_status, job_id, f"Sampling posterior {idx}/{len(posteriors)}...")
                        segment_start = int(((idx - 1) * 95) / posterior_count)
                        segment_end = int((idx * 95) / posterior_count)
                        capture = _JobOutputCapture(
                            job_status,
                            job_id,
                            progress_base=segment_start,
                            progress_span=max(1, segment_end - segment_start),
                        )
                        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                            samples = _sample_sbi_posterior(
                                posterior,
                                inference_obj.torch,
                                x_valid,
                                num_samples,
                                sbi_batch_size,
                                show_progress=True,
                            )
                        capture.flush()
                        per_model_summaries.append(_summarize_sbi_samples(samples, sbi_summary_mode))

                    stacked = np.stack(per_model_summaries, axis=0)
                    combined = np.mean(stacked, axis=0)
                    valid_rows = np.where(finite_mask)[0]
                    for local_idx, global_idx in enumerate(valid_rows):
                        preds[int(global_idx)] = _normalize_prediction_value(combined[local_idx])

            output_df["Predictions"] = preds
            if job_id in job_status:
                job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 92)

        _append_job_output(job_status, job_id, "Attaching predictions and saving output dataframe.")
        output_df_path = save_df(job_id, output_df, temp_uploaded_files)
        persisted_predictions_path = None
        try:
            predictions_dir = os.path.realpath("/tmp/predictions_data")
            os.makedirs(predictions_dir, exist_ok=True)
            persisted_name = "predictions.pkl"
            persisted_predictions_path = os.path.join(predictions_dir, persisted_name)
            output_df.to_pickle(persisted_predictions_path)
            _append_job_output(job_status, job_id, f"Persisted dashboard predictions file: {persisted_predictions_path}")
        except Exception as persist_exc:
            _append_job_output(job_status, job_id, f"Warning: could not persist dashboard predictions file: {persist_exc}")
        _append_job_output(job_status, job_id, f"Saved predictions dataframe: {output_df_path}")
        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "estimated_time_remaining": 0,
            "results": output_df_path,
            "error": False,
            "dashboard_predictions_path": persisted_predictions_path,
        })

    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": job_status[job_id].get("progress", 0),
        })
    finally:
        cleanup_temp_files(file_paths)
        if os.path.isdir(artifacts_dir):
            shutil.rmtree(artifacts_dir, ignore_errors=True)
def inference_training_computation(job_id, job_status, params, temp_uploaded_files):
    file_paths = params.get("file_paths", {})
    artifacts_dir = os.path.join(temp_uploaded_files, f"inference_training_artifacts_{job_id}")
    zip_path = None

    try:
        _append_job_output(job_status, job_id, "Starting inference model training.")

        features_path = _first_existing_file([
            file_paths.get("training_features_file"),
            file_paths.get("file-upload-x"),
            file_paths.get("features_train_file"),
        ])
        parameters_path = _first_existing_file([
            file_paths.get("training_parameters_file"),
            file_paths.get("file-upload-y"),
            file_paths.get("parameters_train_file"),
        ])
        if not features_path or not parameters_path:
            raise ValueError("Both features and parameters training files are required.")

        try:
            X, Y = _load_training_with_example_loader(features_path, parameters_path)
            _append_job_output(job_status, job_id, "Loaded X/theta using examples.tools.load_model_features.")
        except Exception as loader_exc:
            _append_job_output(
                job_status,
                job_id,
                f"examples.tools.load_model_features unavailable for these files ({loader_exc}); using generic loaders.",
            )
            features_obj = _read_training_input_any(features_path)
            parameters_obj = _read_training_input_any(parameters_path)
            X = _coerce_training_matrix(features_obj, "Features")
            Y = _coerce_training_matrix(parameters_obj, "Parameters")
        if X.shape[0] == 0 or Y.shape[0] == 0:
            raise ValueError("Training inputs are empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Features/parameters row mismatch: {X.shape[0]} vs {Y.shape[0]}.")

        subsample_percent = _parse_float_param(params, "training_subsample_percent", default=100.0)
        if subsample_percent is None:
            subsample_percent = 100.0
        if subsample_percent <= 0 or subsample_percent > 100:
            raise ValueError("Random Subsample (%) must be > 0 and <= 100.")

        original_rows = int(X.shape[0])
        if subsample_percent < 100.0 and original_rows > 1:
            seed_for_sample = _parse_int_param(params, "training_seed", default=0)
            rng = np.random.default_rng(seed_for_sample if seed_for_sample is not None else 0)
            n_select = max(1, int(np.floor((subsample_percent / 100.0) * original_rows)))
            if n_select < original_rows:
                indices = rng.choice(original_rows, size=n_select, replace=False)
                indices.sort()
                X = X[indices]
                Y = Y[indices]
                _append_job_output(
                    job_status,
                    job_id,
                    f"Applied random subsample: {subsample_percent:.2f}% ({original_rows} -> {n_select} rows).",
                )
        else:
            _append_job_output(job_status, job_id, f"Subsample disabled (using {original_rows} rows).")

        _append_job_output(job_status, job_id, f"Loaded training data: X={X.shape}, Y={Y.shape}.")
        if job_id in job_status:
            job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 5)

        model_name = (_get_param(params, "training_model_name", "MLPRegressor") or "").strip()
        if model_name == "__custom__":
            model_name = (_get_param(params, "training_model_name_custom", "") or "").strip()
        if not model_name:
            raise ValueError("Training model is required.")

        base_hyperparams = _parse_dict_param(params, "training_hyperparams_json") or {}
        inference_obj = ncpi.Inference(
            model=model_name,
            hyperparams=base_hyperparams if model_name not in {"NPE", "NLE", "NRE"} else None,
        )

        use_scaler = _parse_bool_param(params, "training_use_scaler", default=False)
        seed = _parse_int_param(params, "training_seed", default=0)
        enable_cv = _parse_bool_param(params, "training_enable_cv", default=False)
        n_splits = _parse_int_param(params, "training_n_splits", default=10)
        n_repeats = _parse_int_param(params, "training_n_repeats", default=10)
        if enable_cv:
            if n_splits is None or n_splits <= 1:
                raise ValueError("n_splits must be an integer > 1 when CV is enabled.")
            if n_repeats is None or n_repeats <= 0:
                raise ValueError("n_repeats must be an integer > 0 when CV is enabled.")
        else:
            n_splits = 2
            n_repeats = 1

        param_grid = None
        if enable_cv and _parse_bool_param(params, "training_enable_param_grid", default=False):
            raw_grid = _get_param(params, "training_param_grid_json", None)
            if raw_grid is None:
                raise ValueError("Parameter grid is enabled, but no grid was provided.")
            param_grid = _parse_param_grid(raw_grid)
            _append_job_output(job_status, job_id, f"Parameter grid enabled with {len(param_grid)} combination(s).")
        elif (not enable_cv) and _parse_bool_param(params, "training_enable_param_grid", default=False):
            _append_job_output(job_status, job_id, "CV is disabled: parameter grid will be ignored.")

        train_params = {}
        sbi_eval_num = 2000
        sbi_eval_batch = 256
        if inference_obj.backend == "sbi":
            theta_dim = int(Y.shape[1]) if Y.ndim == 2 else 1
            low_vec = _parse_numeric_vector(_get_param(params, "training_sbi_prior_low", None), np.zeros(theta_dim, dtype=float))
            high_vec = _parse_numeric_vector(_get_param(params, "training_sbi_prior_high", None), np.ones(theta_dim, dtype=float))
            if low_vec.size == 1:
                low_vec = np.repeat(low_vec[0], theta_dim)
            if high_vec.size == 1:
                high_vec = np.repeat(high_vec[0], theta_dim)
            if low_vec.size != theta_dim or high_vec.size != theta_dim:
                raise ValueError(f"SBI prior bounds must have length 1 or match theta dimension ({theta_dim}).")
            if np.any(low_vec >= high_vec):
                raise ValueError("SBI prior bounds require low < high for every dimension.")

            estimator_name = _get_param(params, "training_sbi_estimator", "nsf")
            estimator_kwargs = _parse_dict_param(params, "training_sbi_estimator_kwargs_json") or {}
            estimator_kwargs["estimator"] = estimator_name
            inference_kwargs = _parse_dict_param(params, "training_sbi_inference_kwargs_json") or {}
            build_posterior_kwargs = _parse_dict_param(params, "training_sbi_build_posterior_kwargs_json") or {}

            BoxUniform = ncpi.tools.dynamic_import("sbi.utils", "BoxUniform")
            torch_mod = inference_obj.torch
            prior = BoxUniform(
                low=torch_mod.tensor(low_vec, dtype=torch_mod.float32),
                high=torch_mod.tensor(high_vec, dtype=torch_mod.float32),
            )
            sbi_hyperparams = dict(base_hyperparams or {})
            sbi_hyperparams.update({
                "prior": prior,
                "estimator_kwargs": estimator_kwargs,
                "inference_kwargs": inference_kwargs,
                "build_posterior_kwargs": build_posterior_kwargs,
            })
            inference_obj.hyperparams = sbi_hyperparams
            train_params = _parse_dict_param(params, "training_sbi_train_params_json") or {}
            sbi_eval_num = _parse_int_param(params, "training_sbi_eval_num_posterior_samples", default=2000)
            sbi_eval_batch = _parse_int_param(params, "training_sbi_eval_batch_size", default=256)
            if sbi_eval_num is None or sbi_eval_num <= 0:
                raise ValueError("SBI eval posterior samples must be > 0.")
            if sbi_eval_batch is None or sbi_eval_batch <= 0:
                raise ValueError("SBI eval batch size must be > 0.")
            _append_job_output(
                job_status,
                job_id,
                f"SBI prior configured with theta_dim={theta_dim}; estimator={estimator_name}.",
            )

        scaler_obj = None
        if use_scaler:
            StandardScaler = ncpi.tools.dynamic_import("sklearn.preprocessing", "StandardScaler")
            scaler_obj = StandardScaler()

        inference_obj.add_simulation_data(X, Y)
        _append_job_output(job_status, job_id, f"Initialized model='{model_name}' backend='{inference_obj.backend}'.")
        _append_job_output(
            job_status,
            job_id,
            f"Training config: seed={seed}, n_splits={n_splits}, n_repeats={n_repeats}, scaler={'on' if use_scaler else 'off'}.",
        )
        total_fold_count = int(max(1, n_splits * n_repeats)) if enable_cv and param_grid else 1
        total_candidates = int(len(param_grid)) if (enable_cv and param_grid) else 1
        _append_job_output(
            job_status,
            job_id,
            (
                f"Planned training work: {total_candidates} configuration(s), {total_fold_count} fold(s) each."
                if (enable_cv and param_grid)
                else "Planned training work: direct training (no CV)."
            ),
        )

        os.makedirs(artifacts_dir, exist_ok=True)
        progress_state = {
            "param_idx": 0,
            "total_candidates": total_candidates,
            "folds_per_candidate": total_fold_count,
        }

        def _training_line_progress(line, _capture_obj):
            text = str(line or "")
            if not text:
                return

            if "Starting hyperparameter search with cross-validation" in text:
                if job_id in job_status:
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 12)
                return

            if "Evaluating params:" in text:
                progress_state["param_idx"] = min(
                    progress_state["total_candidates"],
                    progress_state["param_idx"] + 1,
                )
                if job_id in job_status:
                    candidate_done = max(0, progress_state["param_idx"] - 1)
                    coarse = 12 + int(70 * candidate_done / max(1, progress_state["total_candidates"]))
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), coarse)
                return

            fold_match = _FOLD_PROGRESS_RE.search(text)
            if fold_match:
                fold_idx = int(fold_match.group(1))
                fold_total = max(1, int(fold_match.group(2)))
                param_idx = max(1, progress_state["param_idx"]) if progress_state["total_candidates"] > 1 else 1
                completed_units = (param_idx - 1) * fold_total + min(fold_idx, fold_total)
                total_units = max(1, progress_state["total_candidates"] * fold_total)
                mapped = 12 + int(75 * completed_units / total_units)
                if job_id in job_status:
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), mapped)
                return

            if "Training single sklearn model on full data" in text:
                if job_id in job_status:
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 30)
                return

            if "Model saved at" in text:
                if job_id in job_status:
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 92)
                return

            if "Density estimator saved at" in text:
                if job_id in job_status:
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 94)
                return

        capture = _JobOutputCapture(
            job_status,
            job_id,
            progress_base=12,
            progress_span=80,
            line_callback=_training_line_progress,
        )
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            inference_obj.train(
                param_grid=param_grid,
                n_splits=n_splits,
                n_repeats=n_repeats,
                train_params=train_params,
                result_dir=artifacts_dir,
                scaler=scaler_obj,
                seed=seed,
                sbi_eval_num_posterior_samples=sbi_eval_num,
                sbi_eval_batch_size=sbi_eval_batch,
            )
        capture.flush()
        if job_id in job_status:
            job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 95)

        produced = sorted(
            name for name in os.listdir(artifacts_dir)
            if os.path.isfile(os.path.join(artifacts_dir, name))
        )
        if not produced:
            raise ValueError("Training completed but no artifacts were produced.")
        _append_job_output(job_status, job_id, f"Generated artifacts: {', '.join(produced)}")

        archive_base = os.path.join(temp_uploaded_files, f"inference_training_{job_id}")
        zip_path = shutil.make_archive(archive_base, "zip", root_dir=artifacts_dir)
        _append_job_output(job_status, job_id, f"Saved training artifacts archive: {zip_path}")
        persisted_zip_path = None
        try:
            persisted_dir = os.path.realpath("/tmp/inference_training_data")
            os.makedirs(persisted_dir, exist_ok=True)
            persisted_zip_path = os.path.join(persisted_dir, "training_artifacts.zip")
            shutil.copy2(zip_path, persisted_zip_path)
            _append_job_output(job_status, job_id, f"Persisted training artifacts to: {persisted_zip_path}")
        except Exception as persist_exc:
            _append_job_output(job_status, job_id, f"Warning: could not persist training artifacts to /tmp: {persist_exc}")

        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "estimated_time_remaining": 0,
            "results": persisted_zip_path or zip_path,
            "error": False,
        })

    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": job_status[job_id].get("progress", 0),
        })
    finally:
        cleanup_temp_files(file_paths)
        if os.path.isdir(artifacts_dir):
            shutil.rmtree(artifacts_dir, ignore_errors=True)


def analysis_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        # Save the image in temp_uploaded_files/LFP_predictions.png
        # LFP_predictions_webversion.run_full_pipeline([params['method-plot']], params['method'])
        
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": f'{temp_uploaded_files}/LFP_predictions.png', # Return to the client the output filepath
                "error": False
            })

    except Exception as e:
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    # Remove the file after using it
    cleanup_temp_files(params['file_paths'])


def field_potential_proxy_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting field potential proxy computation.")
        method = params.get('proxy_method', 'FR')
        _append_job_output(job_status, job_id, f"Proxy method: {method}")
        sim_step_value = params.get('sim_step')
        sim_step = float(sim_step_value) if sim_step_value not in (None, '') else None
        bin_size_value = params.get('bin_size')
        bin_size = float(bin_size_value) if bin_size_value not in (None, '') else 1.0

        excitatory_only_value = params.get('excitatory_only', 'default').lower()
        if excitatory_only_value == 'true':
            excitatory_only = True
        elif excitatory_only_value == 'false':
            excitatory_only = False
        else:
            excitatory_only = None

        file_paths = params.get('file_paths', {})
        sim_data = {}

        dt_from_stage, dt_path = _load_simulation_dt_ms(file_paths)
        proxy_sim_step = None
        dt_source = None

        if method == 'FR':
            _append_job_output(job_status, job_id, "Loading spike times and gids...")
            times_path = _resolve_sim_file(file_paths, 'times_file', 'times.pkl', required=True)
            gids_path = _resolve_sim_file(file_paths, 'gids_file', 'gids.pkl', required=True)
            times = read_file(times_path)
            gids = read_file(gids_path)
            sim_data['FR'] = _compute_mean_firing_rate(times, gids, bin_size)
            proxy_sim_step = float(bin_size)
            dt_source = "bin_size_ms"

        elif method == 'AMPA':
            _append_job_output(job_status, job_id, "Loading AMPA currents...")
            ampa_path = _resolve_sim_file(file_paths, 'ampa_file', 'ampa.pkl', required=True)
            sim_data['AMPA'] = read_file(ampa_path)

        elif method == 'GABA':
            _append_job_output(job_status, job_id, "Loading GABA currents...")
            gaba_path = _resolve_sim_file(file_paths, 'gaba_file', 'gaba.pkl', required=True)
            sim_data['GABA'] = read_file(gaba_path)

        elif method == 'Vm':
            _append_job_output(job_status, job_id, "Loading membrane potentials...")
            vm_path = _resolve_sim_file(file_paths, 'vm_file', 'vm.pkl', required=True)
            sim_data['Vm'] = read_file(vm_path)

        elif method in {'I', 'I_abs', 'LRWS', 'ERWS1', 'ERWS2'}:
            _append_job_output(job_status, job_id, "Loading AMPA and GABA currents...")
            ampa_path = _resolve_sim_file(file_paths, 'ampa_file', 'ampa.pkl', required=True)
            gaba_path = _resolve_sim_file(file_paths, 'gaba_file', 'gaba.pkl', required=True)
            sim_data['AMPA'] = read_file(ampa_path)
            sim_data['GABA'] = read_file(gaba_path)

            if method == 'ERWS2':
                _append_job_output(job_status, job_id, "Loading nu_ext...")
                nu_ext_path = _resolve_sim_file(file_paths, 'nu_ext_file', 'nu_ext.pkl', required=False)
                nu_ext_value = params.get('nu_ext_value')
                if nu_ext_path:
                    sim_data['nu_ext'] = read_file(nu_ext_path)
                elif nu_ext_value not in (None, ''):
                    sim_data['nu_ext'] = float(nu_ext_value)
                else:
                    raise FileNotFoundError(
                        f"Missing nu_ext. Upload a file or provide a value, or place nu_ext.pkl in {DEFAULT_SIM_DATA_DIR}."
                    )
        else:
            raise ValueError(f"Unknown proxy method '{method}'.")

        if method != "FR":
            if dt_from_stage is not None and dt_from_stage > 0:
                proxy_sim_step = float(dt_from_stage)
                dt_source = "simulation_dt_file"
                if sim_step is not None and abs(sim_step - proxy_sim_step) > 1e-12:
                    _append_job_output(
                        job_status,
                        job_id,
                        f"Using simulation dt from {dt_path} ({proxy_sim_step:g} ms) instead of form sim_step ({sim_step:g} ms).",
                    )
            elif sim_step is not None and sim_step > 0:
                proxy_sim_step = float(sim_step)
                dt_source = "form_sim_step"
            else:
                raise ValueError(
                    "Simulation step could not be determined. Provide sim_step or ensure dt.pkl is available in simulation outputs."
                )

        _append_job_output(job_status, job_id, f"Effective proxy sampling step: {proxy_sim_step:g} ms ({dt_source}).")

        _append_job_output(job_status, job_id, "Computing proxy with ncpi.FieldPotential.compute_proxy...")
        potential = ncpi.FieldPotential()
        proxy = potential.compute_proxy(method, sim_data, proxy_sim_step, excitatory_only=excitatory_only)

        output_root = '/tmp/field_potential_proxy'
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)

        sim_data_path = os.path.join(run_dir, 'sim_data.pkl')
        proxy_path = os.path.join(run_dir, 'proxy.pkl')

        with open(sim_data_path, 'wb') as f:
            pickle.dump(sim_data, f)
        # Save proxy outputs in a dataframe so downstream parsing can use "data" as locator.
        dt_ms = _safe_float(proxy_sim_step)
        proxy_df = pd.DataFrame([{
            "data": proxy,
            "proxy_method": method,
            "dt_ms": dt_ms,
            "decimation_factor": 1,
            "fs_hz": None,
            "metadata": {"dt_ms": dt_ms, "decimation_factor": 1, "fs_hz": None, "dt_source": dt_source},
        }])
        dt_val = proxy_df["dt_ms"].iloc[0]
        if dt_val is not None and dt_val > 0:
            fs_hz = 1000.0 / float(dt_val)
            proxy_df["fs_hz"] = fs_hz
            proxy_df.at[proxy_df.index[0], "metadata"] = {
                "dt_ms": float(dt_val),
                "decimation_factor": 1,
                "fs_hz": fs_hz,
                "dt_source": dt_source,
            }
        proxy_df.to_pickle(proxy_path)

        _append_job_output(job_status, job_id, f"Saved sim_data.pkl and proxy.pkl to {run_dir}")
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": proxy_path,
                "error": False
            })

    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_kernel_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting kernel computation.")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_mc_folder = os.path.expandvars(
            os.path.expanduser(os.path.join("$HOME", "multicompartment_neuron_network"))
        )
        default_output_sim = os.path.join(
            default_mc_folder, "output", "adb947bfb931a5a8d09ad078a6d256b0"
        )
        default_params_path = os.path.join(
            repo_root,
            "examples",
            "simulation",
            "Hagen_model",
            "simulation",
            "params",
            "analysis_params.py",
        )

        mc_folder = params.get("mc_folder") or default_mc_folder
        output_sim_path = params.get("output_sim_path") or default_output_sim
        params_path = params.get("kernel_params_module") or default_params_path

        _append_job_output(job_status, job_id, f"MC folder: {mc_folder}")
        _append_job_output(job_status, job_id, f"Params module: {params_path}")
        _append_job_output(job_status, job_id, f"Output simulation path: {output_sim_path}")

        module = _load_module_from_path(params_path, name="kernel_params")
        KernelParams = getattr(module, "KernelParams", None)
        if KernelParams is None:
            raise AttributeError("KernelParams not found in params module.")
        pop_names = getattr(KernelParams, "population_names", None)
        pop_count = len(pop_names) if pop_names is not None else None

        dt = params.get("dt")
        dt = float(dt) if dt not in (None, "") else KernelParams.networkParameters.get("dt", 0.0625)
        tstop = params.get("tstop")
        tstop = float(tstop) if tstop not in (None, "") else KernelParams.networkParameters.get("tstop", 12000.0)
        t_X = params.get("t_x")
        t_X = float(t_X) if t_X not in (None, "") else getattr(KernelParams, "transient", None)
        tau = params.get("tau")
        tau = float(tau) if tau not in (None, "") else getattr(KernelParams, "tau", None)

        biophys = _parse_literal_value(params.get("biophys"), None)
        if biophys is None:
            biophys = ["set_Ih_linearized_hay2011", "make_cell_uniform"]

        g_eff = _parse_bool(params.get("g_eff"), KernelParams.MC_params.get("g_eff", None))
        n_ext = _parse_literal_value(params.get("n_ext"), KernelParams.MC_params.get("n_ext", None))
        if isinstance(n_ext, (int, float, np.integer, np.floating)) and pop_count:
            n_ext = [int(n_ext)] * pop_count

        weights = _parse_literal_value(params.get("weights"), None)
        if weights is None:
            weight_vals = [
                params.get("weight_ee"),
                params.get("weight_ie"),
                params.get("weight_ei"),
                params.get("weight_ii"),
            ]
            if all(v not in (None, "") for v in weight_vals):
                weights = [
                    [float(weight_vals[0]), float(weight_vals[1])],
                    [float(weight_vals[2]), float(weight_vals[3])],
                ]

        file_paths = params.get("file_paths", {})
        probe_selection_present = params.get("probe_selection_present") not in (None, "")
        probe_kernel_approx = _parse_bool(params.get("probe_kernel_approx"), False)
        probe_current_dipole = _parse_bool(params.get("probe_current_dipole"), False)
        probe_gauss_cylinder = _parse_bool(params.get("probe_gauss_cylinder"), False)
        if probe_kernel_approx and probe_current_dipole:
            raise ValueError("Select only one of KernelApproxCurrentDipoleMoment or CurrentDipoleMoment.")

        electrodeParameters = None
        electrode_path = file_paths.get("electrode_parameters_file")
        if electrode_path:
            electrodeParameters = read_file(electrode_path)
        else:
            electrode_value = params.get("electrode_parameters")
            if electrode_value not in (None, ""):
                if electrode_value.strip() == "KernelParams.electrodeParameters":
                    electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
                else:
                    electrodeParameters = _parse_literal_value(electrode_value, None)
            else:
                electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
        if probe_selection_present and not probe_gauss_cylinder:
            electrodeParameters = None
        if probe_selection_present and probe_gauss_cylinder and electrodeParameters is None:
            electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
            if electrodeParameters is None:
                raise ValueError("electrodeParameters must be provided when GaussCylinderPotential is selected.")

        if probe_selection_present:
            cdm = probe_kernel_approx or probe_current_dipole
            if probe_kernel_approx:
                cdm_probe = "KernelApproxCurrentDipoleMoment"
            elif probe_current_dipole:
                cdm_probe = "CurrentDipoleMoment"
            else:
                cdm_probe = None
        else:
            cdm_probe = params.get("cdm_probe") or "KernelApproxCurrentDipoleMoment"
            cdm = _parse_bool(params.get("cdm"), True)
        probes = _parse_literal_value(params.get("probes"), None)
        selected_probe_names = []
        if probe_selection_present:
            if probe_current_dipole:
                selected_probe_names.append("CurrentDipoleMoment")
            if probe_kernel_approx:
                selected_probe_names.append("KernelApproxCurrentDipoleMoment")
            if probe_gauss_cylinder:
                selected_probe_names.append("GaussCylinderPotential")
            if not selected_probe_names:
                raise ValueError("Select at least one probe for CDM/LFP computation.")
        else:
            fallback_probe = params.get("cdm_probe_name") or params.get("cdm_probe")
            if fallback_probe:
                selected_probe_names = [fallback_probe]

        mean_nu_x = None
        vrest = None
        mean_nu_value = params.get("mean_nu_x")
        if mean_nu_value in (None, ""):
            mean_nu_value = params.get("mean_nu_x_value")
        if mean_nu_value not in (None, ""):
            parsed = _parse_literal_value(mean_nu_value, None)
            if isinstance(parsed, dict):
                mean_nu_x = parsed
            elif isinstance(parsed, (list, tuple, np.ndarray)):
                if pop_names and len(parsed) == len(pop_names):
                    mean_nu_x = {name: float(parsed[i]) for i, name in enumerate(pop_names)}
                else:
                    mean_nu_x = parsed
            elif isinstance(parsed, (int, float, np.integer, np.floating)):
                if pop_names:
                    mean_nu_x = {name: float(parsed) for name in pop_names}
                else:
                    mean_nu_x = float(parsed)
        else:
            mean_nu_path = params.get("mean_nu_x_path")
            if mean_nu_path:
                mean_nu_x = read_file(mean_nu_path)
        vrest_path = params.get("vrest_path")
        if vrest_path:
            vrest = read_file(vrest_path)
        vrest_value = params.get("vrest_value")
        if vrest is None and vrest_value not in (None, ""):
            vrest = float(vrest_value)
        if mean_nu_x is None and vrest is None:
            mean_nu_x = None
            vrest = None

        if not output_sim_path or not os.path.exists(output_sim_path):
            if mean_nu_x is None or vrest is None:
                raise FileNotFoundError(
                    "Output simulation path not found. Provide mean_nu_X and Vrest or a valid output_sim_path."
                )
            output_sim_path = None

        _append_job_output(job_status, job_id, "Computing kernels with ncpi.FieldPotential.create_kernel...")
        potential = ncpi.FieldPotential()
        kernels = potential.create_kernel(
            mc_folder,
            KernelParams,
            biophys,
            dt,
            tstop,
            output_sim_path=output_sim_path,
            electrodeParameters=electrodeParameters,
            CDM=cdm,
            probes=probes,
            cdm_probe=cdm_probe,
            mean_nu_X=mean_nu_x,
            Vrest=vrest,
            t_X=t_X,
            tau=tau,
            g_eff=g_eff,
            n_ext=n_ext,
            weights=weights,
        )

        output_root = "/tmp/field_potential_kernel"
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)
        kernels_path = os.path.join(run_dir, "kernels.pkl")
        with open(kernels_path, "wb") as f:
            pickle.dump(kernels, f)

        _append_job_output(job_status, job_id, f"Saved kernels to {kernels_path}")

        _append_job_output(job_status, job_id, "Computing CDM/LFP from kernels...")
        spike_times_path = _resolve_sim_file(
            file_paths, "kernel_spike_times_file", "times.pkl", required=True
        )
        spike_times = read_file(spike_times_path)

        population_sizes = None
        pop_path = _resolve_sim_file(
            file_paths, "kernel_population_sizes_file", "population_sizes.pkl", required=False
        )
        if pop_path:
            population_sizes = read_file(pop_path)

        cdm_dt = params.get("cdm_dt")
        cdm_dt = float(cdm_dt) if cdm_dt not in (None, "") else dt
        cdm_tstop = params.get("cdm_tstop")
        cdm_tstop = float(cdm_tstop) if cdm_tstop not in (None, "") else tstop
        cdm_transient = params.get("cdm_transient")
        if cdm_transient not in (None, ""):
            transient = float(cdm_transient)
        elif t_X not in (None, ""):
            transient = float(t_X)
        else:
            transient = 0.0
        probe_names = selected_probe_names or ["KernelApproxCurrentDipoleMoment"]
        component_val = params.get("cdm_component")
        component = None if component_val in (None, "", "None") else int(float(component_val))
        mode = params.get("cdm_mode") or "same"
        scale_val = params.get("cdm_scale")
        scale = float(scale_val) if scale_val not in (None, "") else 1.0
        aggregate_val = params.get("cdm_aggregate")
        aggregate = aggregate_val if aggregate_val else None
        probe_output_map = {
            "KernelApproxCurrentDipoleMoment": "kernel_approx_cdm.pkl",
            "CurrentDipoleMoment": "current_dipole_moment.pkl",
            "GaussCylinderPotential": "gauss_cylinder_potential.pkl",
        }

        probe_outputs = {}
        output_paths = {}
        fs_hz = (1000.0 / float(cdm_dt)) if float(cdm_dt) > 0 else None
        for probe_name in probe_names:
            cdm_signals = potential.compute_cdm_lfp_from_kernels(
                kernels,
                spike_times,
                cdm_dt,
                cdm_tstop,
                population_sizes=population_sizes,
                transient=transient,
                probe=probe_name,
                component=component,
                mode=mode,
                scale=scale,
                aggregate=aggregate,
            )
            probe_outputs[probe_name] = cdm_signals

            output_name = probe_output_map.get(probe_name)
            if not output_name:
                safe_probe = "".join(ch.lower() if ch.isalnum() else "_" for ch in probe_name).strip("_")
                output_name = f"{safe_probe or 'probe_output'}.pkl"
            output_path = os.path.join(run_dir, output_name)
            combined_signal = _sum_signal_dict(cdm_signals)
            payload_df = pd.DataFrame([{
                "data": combined_signal,
                "raw_signals": cdm_signals,
                "probe": probe_name,
                "dt_ms": float(cdm_dt),
                "decimation_factor": 1,
                "fs_hz": fs_hz,
                "metadata": {"dt_ms": float(cdm_dt), "decimation_factor": 1, "fs_hz": fs_hz},
            }])
            payload_df.to_pickle(output_path)
            output_paths[probe_name] = output_path
            _append_job_output(job_status, job_id, f"Saved probe output to {output_path}")

        results_path = None
        if len(probe_outputs) == 1:
            results_path = next(iter(output_paths.values()))
        else:
            combined_path = os.path.join(run_dir, "probe_outputs.pkl")
            rows = []
            for probe_name, probe_dict in probe_outputs.items():
                rows.append({
                    "data": _sum_signal_dict(probe_dict),
                    "raw_signals": probe_dict,
                    "probe": probe_name,
                    "dt_ms": float(cdm_dt),
                    "decimation_factor": 1,
                    "fs_hz": fs_hz,
                    "metadata": {"dt_ms": float(cdm_dt), "decimation_factor": 1, "fs_hz": fs_hz},
                })
            pd.DataFrame(rows).to_pickle(combined_path)
            results_path = combined_path
            _append_job_output(job_status, job_id, f"Saved combined probe outputs to {combined_path}")

        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": results_path,
                "error": False
            })
    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })
    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_meeg_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting M/EEG computation.")
        job_status[job_id]["progress"] = 5
        file_paths = params.get("file_paths", {})

        cdm_path = file_paths.get("meeg_cdm_file")
        if not cdm_path or not os.path.exists(cdm_path):
            preferred = glob.glob(os.path.join("/tmp/field_potential_kernel", "*", "current_dipole_moment.pkl"))
            if preferred:
                cdm_path = max(preferred, key=os.path.getmtime)
            else:
                fallback = glob.glob(os.path.join("/tmp/field_potential_kernel", "*", "kernel_approx_cdm.pkl"))
                if fallback:
                    cdm_path = max(fallback, key=os.path.getmtime)
        if not cdm_path or not os.path.exists(cdm_path):
            raise FileNotFoundError("CDM input is required (upload .pkl or compute kernels first).")
        CDM_obj = read_file(cdm_path)
        CDM, cdm_meta = _extract_signal_and_meta_from_source(CDM_obj)
        job_status[job_id]["progress"] = 20
        if isinstance(CDM, dict):
            if "sum" in CDM:
                _append_job_output(job_status, job_id, "CDM input is a dict; using 'sum' entry.")
                CDM = CDM["sum"]
            elif len(CDM) == 1:
                key = next(iter(CDM.keys()))
                _append_job_output(job_status, job_id, f"CDM input is a dict; using '{key}' entry.")
                CDM = CDM[key]
            else:
                _append_job_output(job_status, job_id, "CDM input has multiple entries; summing all keys for M/EEG.")
                total = None
                for value in CDM.values():
                    arr = np.asarray(value)
                    if total is None:
                        total = np.array(arr, copy=True)
                    else:
                        total = total + arr
                CDM = total
        CDM = np.asarray(CDM)
        if CDM.ndim == 2 and CDM.shape[0] != 3 and CDM.shape[1] == 3:
            CDM = CDM.T
        if CDM.ndim == 3 and CDM.shape[1] != 3 and CDM.shape[2] == 3:
            CDM = np.transpose(CDM, (0, 2, 1))
        if CDM.ndim == 1:
            _append_job_output(job_status, job_id, "CDM is 1D; assuming z-axis dipole (x=y=0).")
            CDM = np.vstack([np.zeros_like(CDM), np.zeros_like(CDM), CDM])
        elif CDM.ndim == 2 and CDM.shape[0] != 3:
            _append_job_output(job_status, job_id, "CDM has 1 component per dipole; assuming z-axis dipoles (x=y=0).")
            zeros = np.zeros_like(CDM)
            CDM = np.stack([zeros, zeros, CDM], axis=1)
        if not ((CDM.ndim == 2 and CDM.shape[0] == 3) or (CDM.ndim == 3 and CDM.shape[1] == 3)):
            raise ValueError(
                "CDM must have 3 components. Ensure you computed a dipole-moment probe (e.g., "
                "KernelApproxCurrentDipoleMoment or CurrentDipoleMoment), not a scalar potential."
            )

        dipole_locations = None
        dipole_path = file_paths.get("meeg_dipole_file")
        if dipole_path and os.path.exists(dipole_path):
            dipole_locations = read_file(dipole_path)
        else:
            dipole_text = params.get("meeg_dipole_locations")
            dipole_locations = _parse_literal_value(dipole_text, None)

        sensor_locations = None
        sensor_path = file_paths.get("meeg_sensor_file")
        if sensor_path and os.path.exists(sensor_path):
            sensor_locations = read_file(sensor_path)
        else:
            sensor_text = params.get("meeg_sensor_locations")
            sensor_locations = _parse_literal_value(sensor_text, None)

        model = params.get("meeg_model") or "NYHeadModel"
        model_kwargs = _parse_literal_value(params.get("meeg_model_kwargs"), None)
        align_to_surface = _parse_bool(params.get("meeg_align_to_surface"), True)
        auto_1020 = _parse_bool(params.get("meeg_auto_1020"), False)

        _append_job_output(job_status, job_id, f"Model: {model}")
        job_status[job_id]["progress"] = 45
        potential = ncpi.FieldPotential()
        if auto_1020 and model == "NYHeadModel":
            dipole_locations, _ = potential._get_eeg_1020_locations()
            dipole_locations = np.asarray(dipole_locations, dtype=float)
            n_dip = 1 if CDM.ndim == 2 else int(CDM.shape[0])
            if n_dip == 1:
                dipole_locations = dipole_locations[0]
                _append_job_output(job_status, job_id, "Using first EEG 10–20 dipole location for NYHeadModel.")
            else:
                if dipole_locations.shape[0] < n_dip:
                    needed = n_dip - dipole_locations.shape[0]
                    tail = np.repeat(dipole_locations[-1:], needed, axis=0)
                    dipole_locations = np.vstack([dipole_locations, tail])
                    _append_job_output(
                        job_status,
                        job_id,
                        "EEG 10–20 locations fewer than dipoles; repeating last location to match count.",
                    )
                dipole_locations = dipole_locations[:n_dip]
                _append_job_output(job_status, job_id, "Using EEG 10–20 dipole locations for NYHeadModel.")
            sensor_locations = None
        elif model == "NYHeadModel":
            sensor_locations = None
        else:
            if sensor_locations is None:
                if model in {"FourSphereVolumeConductor", "InfiniteVolumeConductor"}:
                    sensor_locations = np.array([[0.0, 0.0, 90000.0]])
                elif model == "InfiniteHomogeneousVolCondMEG":
                    sensor_locations = np.array([[10000.0, 0.0, 0.0]])
                elif model == "SphericallySymmetricVolCondMEG":
                    sensor_locations = np.array([[0.0, 0.0, 92000.0]])

        if dipole_locations is None:
            if model == "FourSphereVolumeConductor":
                dipole_locations = np.array([0.0, 0.0, 78000.0])
            elif model == "SphericallySymmetricVolCondMEG":
                dipole_locations = np.array([0.0, 0.0, 90000.0])
            else:
                dipole_locations = np.zeros(3)

        p_list, loc_list = potential._normalize_cdm_and_locations(CDM, dipole_locations)
        model_kwargs = model_kwargs or {}
        is_meg = model in {"InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG"}

        matrices = []
        p_use_list = []
        n_sensors = 0
        if model == "NYHeadModel":
            if potential.nyhead is None:
                nyhead_model = potential._load_eegmegcalc_model("NYHeadModel")
                potential.nyhead = nyhead_model(**model_kwargs) if model_kwargs else nyhead_model()
            for p_i, loc_i in zip(p_list, loc_list):
                potential.nyhead.set_dipole_pos(loc_i)
                M = potential.nyhead.get_transformation_matrix()
                p_use = potential.nyhead.rotate_dipole_to_surface_normal(p_i) if align_to_surface else p_i
                matrices.append(M)
                p_use_list.append(p_use)
            if matrices:
                n_sensors = matrices[0].shape[0]
        else:
            if sensor_locations is None:
                raise ValueError("sensor_locations must be provided for this model.")
            sensor_locations = np.asarray(sensor_locations, dtype=float)
            if sensor_locations.ndim != 2 or sensor_locations.shape[1] != 3:
                raise ValueError("sensor_locations must have shape (n_sensors, 3).")
            if model == "FourSphereVolumeConductor":
                FourSphere = potential._load_eegmegcalc_model("FourSphereVolumeConductor")
                model_obj = FourSphere(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            elif model == "InfiniteVolumeConductor":
                InfiniteVol = potential._load_eegmegcalc_model("InfiniteVolumeConductor")
                model_obj = InfiniteVol(**model_kwargs)
                def get_M(loc):
                    r = sensor_locations - loc
                    return model_obj.get_transformation_matrix(r)
            elif model == "InfiniteHomogeneousVolCondMEG":
                IHVCMEG = potential._load_eegmegcalc_model("InfiniteHomogeneousVolCondMEG")
                model_obj = IHVCMEG(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            elif model == "SphericallySymmetricVolCondMEG":
                SSVMEG = potential._load_eegmegcalc_model("SphericallySymmetricVolCondMEG")
                model_obj = SSVMEG(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            else:
                raise ValueError(f"Unknown model '{model}'.")

            for p_i, loc_i in zip(p_list, loc_list):
                matrices.append(get_M(loc_i))
                p_use_list.append(p_i)
            if matrices:
                n_sensors = matrices[0].shape[0]

        if not matrices or n_sensors <= 0:
            raise ValueError("Unable to determine number of sensors/electrodes.")

        n_times = p_use_list[0].shape[1]
        if is_meg:
            meeg = np.zeros((n_sensors, 3, n_times))
        else:
            meeg = np.zeros((n_sensors, n_times))

        log_every = max(1, n_sensors // 50)
        progress_start = 50
        progress_end = 90
        for idx in range(n_sensors):
            if is_meg:
                acc = np.zeros((3, n_times))
                for M, p_i in zip(matrices, p_use_list):
                    acc = acc + (M[idx] @ p_i)
                meeg[idx] = acc
            else:
                acc = np.zeros((n_times,))
                for M, p_i in zip(matrices, p_use_list):
                    acc = acc + (M[idx] @ p_i)
                meeg[idx] = acc

            if (idx + 1) % log_every == 0 or (idx + 1) == n_sensors:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Computed electrodes: {idx + 1}/{n_sensors}",
                )
            progress = progress_start + int((idx + 1) / n_sensors * (progress_end - progress_start))
            job_status[job_id]["progress"] = progress

        output_root = "/tmp/field_potential_meeg"
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)
        meeg_path = os.path.join(run_dir, "meeg.pkl")
        meeg_dt_ms = _safe_float(cdm_meta.get("dt_ms"))
        meeg_decimation = int(cdm_meta.get("decimation_factor", 1))
        meeg_fs = _safe_float(cdm_meta.get("fs_hz"))
        pd.DataFrame([{
            "data": meeg,
            "dt_ms": meeg_dt_ms,
            "decimation_factor": meeg_decimation,
            "fs_hz": meeg_fs,
            "metadata": {"dt_ms": meeg_dt_ms, "decimation_factor": meeg_decimation, "fs_hz": meeg_fs},
            "source_cdm_file": os.path.basename(cdm_path),
        }]).to_pickle(meeg_path)

        _append_job_output(job_status, job_id, f"Saved M/EEG to {meeg_path}")
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": meeg_path,
                "error": False
            })
    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })
    cleanup_temp_files(params.get('file_paths', {}))
