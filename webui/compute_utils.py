import os
import glob
import pandas as pd
import pickle
import numpy as np
import sys

# Prefer local repository package over globally installed ncpi.
_webui_dir = os.path.dirname(os.path.realpath(__file__))
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
import threading
import time
import traceback
from collections.abc import Mapping as MappingABC
import tmp_paths
from tmp_paths import configure_temp_environment, tmp_subdir

configure_temp_environment()


_MODULE_TMP_NAMES = {"simulation", "field_potential", "features", "inference", "analysis"}


def _module_tmp_subdir(module_name, *parts, create=False):
    module_key = str(module_name or "").strip().lower()
    if module_key not in _MODULE_TMP_NAMES:
        raise ValueError(f"Unknown module tmp root: {module_name}")
    clean_parts = [module_key]
    for raw in parts:
        token = str(raw or "").strip().replace("\\", "/").strip("/")
        if not token:
            continue
        clean_parts.extend(seg for seg in token.split("/") if seg and seg != ".")
    rel_path = os.path.join(*clean_parts)
    return tmp_subdir(rel_path, create=create)


def _field_potential_output_dir(kind, create=False):
    kind_key = str(kind or "").strip().lower()
    if kind_key not in {"proxy", "kernel", "meeg"}:
        raise ValueError(f"Unsupported field potential output kind: {kind}")
    return _module_tmp_subdir("field_potential", kind_key, create=create)


def _field_potential_kernel_search_roots():
    roots = [_field_potential_output_dir("kernel", create=False)]
    ordered = []
    seen = set()
    for path in roots:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


sim_data_path = 'zenodo_sim_files/data/'
model_scaler_path = 'zenodo_sim_files/ML_models/4_param/MLP'
TMP_ROOT = tmp_paths.TMP_ROOT
DEFAULT_SIM_DATA_DIR = _module_tmp_subdir("simulation", "data")
FEATURES_DATA_DIR = _module_tmp_subdir("features", "data")
PREDICTIONS_DATA_DIR = _module_tmp_subdir("inference", "predictions")
INFERENCE_TRAINING_DATA_DIR = _module_tmp_subdir("inference", "training_data")
MAX_OUTPUT_LINES = 200
SIMULATION_BUNDLE_FILE = "simulation.pkl"
SIMULATION_LEGACY_BUNDLE_FILES = {"sim_data.pkl"}
SIMULATION_BUNDLE_FILES = {SIMULATION_BUNDLE_FILE, *SIMULATION_LEGACY_BUNDLE_FILES}
SIMULATION_BUNDLE_FIELD_BY_FILE = {
    "times.pkl": "times",
    "gids.pkl": "gids",
    "dt.pkl": "dt",
    "tstop.pkl": "tstop",
    "network.pkl": "network",
    "population_sizes.pkl": "population_sizes",
    "vm.pkl": "vm",
    "ampa.pkl": "ampa",
    "gaba.pkl": "gaba",
    "exc_state_events.pkl": "exc_state_events",
}


def refresh_tmp_paths():
    global TMP_ROOT, DEFAULT_SIM_DATA_DIR, FEATURES_DATA_DIR, PREDICTIONS_DATA_DIR, INFERENCE_TRAINING_DATA_DIR

    TMP_ROOT = configure_temp_environment()
    DEFAULT_SIM_DATA_DIR = _module_tmp_subdir("simulation", "data")
    FEATURES_DATA_DIR = _module_tmp_subdir("features", "data")
    PREDICTIONS_DATA_DIR = _module_tmp_subdir("inference", "predictions")
    INFERENCE_TRAINING_DATA_DIR = _module_tmp_subdir("inference", "training_data")
    return TMP_ROOT

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
    path = FEATURES_DATA_DIR
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
        if default_name in SIMULATION_BUNDLE_FIELD_BY_FILE:
            bundle_path = os.path.join(DEFAULT_SIM_DATA_DIR, SIMULATION_BUNDLE_FILE)
            if os.path.exists(bundle_path):
                return bundle_path
            for legacy_name in SIMULATION_LEGACY_BUNDLE_FILES:
                legacy_bundle_path = os.path.join(DEFAULT_SIM_DATA_DIR, legacy_name)
                if os.path.exists(legacy_bundle_path):
                    return legacy_bundle_path

    if required:
        raise FileNotFoundError(
            f"Missing required input '{key}'. Upload a file or place {default_name} in {DEFAULT_SIM_DATA_DIR}."
        )
    return None


def _coerce_simulation_bundle_payload(payload):
    required_keys = {"times", "gids", "dt", "tstop", "network"}

    if isinstance(payload, list):
        if not payload:
            return None
        first = payload[0]
        if not isinstance(first, MappingABC) or not required_keys.issubset(first.keys()):
            return None
        normalized = {
            key: [item.get(key) for item in payload]
            for key in required_keys
        }
        population_sizes = [item.get("population_sizes") for item in payload]
        if any(value is not None for value in population_sizes):
            normalized["population_sizes"] = population_sizes
        return normalized

    if not isinstance(payload, MappingABC):
        return None
    if not required_keys.issubset(payload.keys()):
        return None

    normalized = {key: payload.get(key) for key in required_keys}
    if payload.get("population_sizes") is not None:
        normalized["population_sizes"] = payload.get("population_sizes")
    if payload.get("vm") is not None:
        normalized["vm"] = payload.get("vm")
    if payload.get("ampa") is not None:
        normalized["ampa"] = payload.get("ampa")
    if payload.get("gaba") is not None:
        normalized["gaba"] = payload.get("gaba")
    if payload.get("exc_state_events") is not None:
        normalized["exc_state_events"] = payload.get("exc_state_events")
    return normalized


def _is_simulation_bundle_path(path):
    return os.path.basename(str(path or "")).lower() in SIMULATION_BUNDLE_FILES


def _load_simulation_component_from_path(path, default_file_name):
    payload = read_file(path)
    if not _is_simulation_bundle_path(path):
        return payload

    field = SIMULATION_BUNDLE_FIELD_BY_FILE.get(str(default_file_name or "").lower())
    if not field:
        return payload

    bundle = _coerce_simulation_bundle_payload(payload)
    if bundle is None:
        raise ValueError(f"Invalid simulation bundle file: {path}")
    if field not in bundle:
        raise ValueError(f"Simulation bundle {path} does not contain '{field}'.")
    return bundle[field]


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


def _announce_saved_output_folders(job_status, job_id, module_name, *paths):
    folders = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        candidate = os.path.realpath(str(raw))
        folder = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
        if not folder:
            continue
        folder = os.path.realpath(folder)
        if folder in seen:
            continue
        seen.add(folder)
        folders.append(folder)
    if folders:
        message = f"[{module_name}] output folder(s): {', '.join(folders)}"
        _append_job_output(job_status, job_id, message)
        print(message, flush=True)


def _mark_job_failed(job_status, job_id, exc):
    message = str(exc).strip()
    if message:
        error_text = f"{type(exc).__name__}: {message}"
    else:
        error_text = type(exc).__name__
    _append_job_output(job_status, job_id, f"Error: {error_text}")

    tb_text = traceback.format_exc().strip()
    if tb_text and tb_text != "NoneType: None":
        for line in tb_text.splitlines():
            _append_job_output(job_status, job_id, line)

    status = job_status.get(job_id)
    if isinstance(status, dict):
        status.update({
            "status": "failed",
            "error": error_text,
            "progress": status.get("progress", 0),
        })


_PROGRESS_PERCENT_RE = re.compile(r"(?:^|\s)(\d{1,3})%")
_FOLD_PROGRESS_RE = re.compile(r"Fold\s+(\d+)\s*/\s*(\d+)")
FILE_EXTRACTED_VIRTUAL_FIELD = "__file_extracted_label__"
FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX = "__file_extracted_chain_"
FILE_TOKEN_VIRTUAL_FIELD_PREFIX = "__file_token_"
FILE_TOKEN_VIRTUAL_FIELD_SUFFIX = "__"
FILE_ID_METADATA_LITERAL = "file_ID"


def _extract_filename_text_label(file_name):
    chains = _extract_filename_text_chains(file_name)
    return chains[-1] if chains else ""


def _extract_filename_text_chains(file_name):
    stem = os.path.splitext(os.path.basename(str(file_name or "")))[0]
    if not stem:
        return []
    # Keep runtime parsing aligned with webui/app.py inspection logic:
    # preserve numeric tokens and chain order from underscore-separated names.
    return [tok.strip() for tok in stem.split("_") if tok.strip()]


def _file_extracted_chain_index(locator):
    if not isinstance(locator, str):
        return None
    if locator == FILE_EXTRACTED_VIRTUAL_FIELD:
        return -1
    if not locator.startswith(FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX):
        return None
    suffix = locator[len(FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX):].strip()
    if not suffix.isdigit():
        return None
    return int(suffix)


def _resolve_file_extracted_value(file_name, locator):
    index = _file_extracted_chain_index(locator)
    if index is None:
        return None
    chains = _extract_filename_text_chains(file_name)
    if not chains:
        return None
    if index < 0:
        return chains[-1]
    if index >= len(chains):
        return None
    return chains[index]


_FILENAME_FORMAT_TOKEN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _file_token_locator_name(locator):
    if not isinstance(locator, str):
        return None
    raw = str(locator).strip()
    if not raw.startswith(FILE_TOKEN_VIRTUAL_FIELD_PREFIX):
        return None
    token = raw[len(FILE_TOKEN_VIRTUAL_FIELD_PREFIX):]
    if token.endswith(FILE_TOKEN_VIRTUAL_FIELD_SUFFIX):
        token = token[: -len(FILE_TOKEN_VIRTUAL_FIELD_SUFFIX)]
    token = token.strip().lower()
    if not token or not _FILENAME_FORMAT_TOKEN_NAME_RE.fullmatch(token):
        return None
    return token


def _parse_filename_format_spec(raw_format):
    value = str(raw_format or "").strip()
    if not value:
        return None
    parts = []
    tokens = []
    seen = set()
    i = 0
    while i < len(value):
        ch = value[i]
        if ch != "$":
            parts.append(("literal", ch))
            i += 1
            continue
        j = value.find("$", i + 1)
        if j < 0:
            raise ValueError("Invalid filename format: missing closing '$'.")
        token_raw = value[i + 1:j].strip()
        if not token_raw:
            raise ValueError("Invalid filename format: empty token '$$' is not allowed.")
        token = token_raw.lower()
        if not _FILENAME_FORMAT_TOKEN_NAME_RE.fullmatch(token):
            raise ValueError(
                f"Invalid filename format token '{token_raw}'. Use letters/numbers/underscore and start with a letter."
            )
        if token in seen:
            raise ValueError(f"Invalid filename format: token '{token_raw}' is duplicated.")
        seen.add(token)
        tokens.append(token)
        parts.append(("token", token))
        i = j + 1
    if not tokens:
        raise ValueError("Filename format must include at least one token, e.g. $id$.")

    regex_parts = []
    group_map = []
    token_idx = 0
    for kind, value_part in parts:
        if kind == "literal":
            regex_parts.append(re.escape(value_part))
            continue
        group_name = f"tok{token_idx}"
        token_idx += 1
        group_map.append((group_name, value_part))
        regex_parts.append(f"(?P<{group_name}>.+?)")
    return {
        "raw": value,
        "tokens": tokens,
        "group_map": group_map,
        "regex": re.compile("^" + "".join(regex_parts) + "$"),
    }


def _extract_filename_format_tokens(file_name, format_spec):
    if not format_spec:
        return None
    basename = os.path.basename(str(file_name or "").strip())
    if not basename:
        return None
    matcher = format_spec.get("regex")
    if matcher is None:
        return None
    match = matcher.fullmatch(basename)
    if not match:
        return None
    out = {}
    for group_name, token_name in (format_spec.get("group_map") or []):
        out[str(token_name)] = str(match.group(group_name) or "").strip()
    return out


def _resolve_file_token_value(file_name, locator, filename_format_spec):
    token = _file_token_locator_name(locator)
    if not token:
        return None
    extracted = _extract_filename_format_tokens(file_name, filename_format_spec)
    if not extracted:
        return None
    return extracted.get(token)


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


def _is_matlab_v73_error(exc):
    msg = str(exc or "").lower()
    return "please use hdf reader for matlab v7.3 files" in msg


class _HDF5MatLazyMapping(MappingABC):
    __lazy_hdf5__ = True

    def __init__(self, file_path, group_path="/"):
        self.file_path = str(file_path)
        self.group_path = str(group_path or "/")

    def _keys(self):
        import h5py
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            return [str(k) for k in grp.keys() if not str(k).startswith("#")]

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def __getitem__(self, key):
        key_str = str(key)
        import h5py
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            if key_str not in grp:
                raise KeyError(key_str)
            node = grp[key_str]
            return _hdf5_mat_to_python_node(node, h5f, self.file_path)

    def __contains__(self, key):
        key_str = str(key)
        import h5py
        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[self.group_path]
            return key_str in grp

    def keys(self):
        return self._keys()

    def items(self):
        for k in self._keys():
            yield k, self[k]

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default


def _hdf5_mat_to_python_value(value, h5file, file_path=None):
    import h5py

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


def _hdf5_mat_to_python_node(node, h5file, file_path=None):
    import h5py

    if isinstance(node, h5py.Group):
        if file_path:
            return _HDF5MatLazyMapping(file_path, node.name)
        out = {}
        for key in node.keys():
            if str(key).startswith("#"):
                continue
            out[str(key)] = _hdf5_mat_to_python_node(node[key], h5file, file_path)
        return out

    if isinstance(node, h5py.Dataset):
        return _hdf5_mat_to_python_value(node[()], h5file, file_path)

    return node


def _load_mat_with_fallback(source, *, in_memory=False, source_name="mat file"):
    scipy_exc = None
    try:
        import scipy.io as sio
        try:
            if in_memory:
                raw = source if isinstance(source, (bytes, bytearray, memoryview)) else bytes(source or b"")
                return sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
            return sio.loadmat(source, squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            scipy_exc = exc
            if not _is_matlab_v73_error(exc):
                raise
    except Exception as exc:
        if scipy_exc is None:
            scipy_exc = exc

    try:
        import h5py
    except Exception as exc:
        if _is_matlab_v73_error(scipy_exc):
            raise ValueError(
                f"Failed to parse MATLAB v7.3 file '{source_name}'. Install h5py or provide a pre-v7.3 .mat file. "
                f"Original error: {scipy_exc}"
            )
        raise ValueError(f"Failed to parse MATLAB file '{source_name}': {scipy_exc}") from exc

    try:
        if in_memory:
            raw = source if isinstance(source, (bytes, bytearray, memoryview)) else bytes(source or b"")
            with h5py.File(io.BytesIO(raw), "r") as h5f:
                return {
                    str(key): _hdf5_mat_to_python_node(h5f[key], h5f, None)
                    for key in h5f.keys()
                    if not str(key).startswith("#")
                }
        return _HDF5MatLazyMapping(str(source), "/")
    except Exception as exc:
        if scipy_exc is not None:
            raise ValueError(
                f"Failed to parse MATLAB file '{source_name}' with scipy and h5py. "
                f"scipy: {scipy_exc}; h5py: {exc}"
            )
        raise


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
    if ext == ".tsv":
        return pd.read_csv(io.BytesIO(raw), sep="\t")

    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))

    if ext == ".feather":
        return pd.read_feather(io.BytesIO(raw))

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))

    if ext == ".mat":
        return _load_mat_with_fallback(raw, in_memory=True, source_name=safe_name)

    if ext == ".edf":
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="ncpi_edf_", suffix=".edf", delete=False) as handle:
                handle.write(raw)
                temp_path = handle.name
            return _load_uploaded_source_path(temp_path, name=safe_name, ext=".edf")
        finally:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if ext == ".fif":
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="ncpi_fif_", suffix=".fif", delete=False) as handle:
                handle.write(raw)
                temp_path = handle.name
            return _load_uploaded_source_path(temp_path, name=safe_name, ext=".fif")
        finally:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    raise ValueError(f"Unsupported empirical file extension '{ext}' for '{safe_name}'.")


def _load_edf_with_parser(path):
    from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig

    parser = EphysDatasetParser(ParseConfig())
    source_obj, _ = parser._load_source(path)
    return source_obj


def _load_uploaded_source_path(path, name=None, ext=None):
    safe_path = str(path or "")
    if not safe_path or not os.path.exists(safe_path):
        raise ValueError(f"Uploaded file path does not exist: '{safe_path}'.")
    if not os.path.isfile(safe_path):
        raise ValueError(f"Uploaded path is not a regular file: '{safe_path}'.")
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
    if ext == ".tsv":
        return pd.read_csv(safe_path, sep="\t")

    if ext == ".parquet":
        return pd.read_parquet(safe_path)

    if ext == ".feather":
        return pd.read_feather(safe_path)

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(safe_path)

    if ext == ".mat":
        return _load_mat_with_fallback(safe_path, in_memory=False, source_name=safe_name)

    if ext == ".set":
        try:
            import mne
        except Exception as exc:
            raise ValueError(f"mne is required to parse .set files: {exc}")
        return mne.io.read_raw_eeglab(safe_path, preload=True)

    if ext == ".fif":
        try:
            import mne
        except Exception as exc:
            raise ValueError(f"mne is required to parse .fif files: {exc}")
        return mne.io.read_raw_fif(safe_path, preload=True, verbose=False)

    if ext == ".edf":
        try:
            return _load_edf_with_parser(safe_path)
        except ImportError as exc:
            raise ValueError(f"pyEDFlib is required to parse .edf files: {exc}")

    raise ValueError(f"Unsupported empirical file extension '{ext}' for '{safe_name}'.")


def _flatten_matlab_ch_names(raw):
    """Flatten a potentially nested MATLAB channel names array into a flat list of strings.

    MATLAB .mat files often produce deeply nested numpy arrays of dtype=object for cell arrays
    of strings (e.g., shape (N,1) where each element is array(['name'], dtype='<Ux')).
    This function handles all such nesting levels and returns a clean list of strings.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    arr = np.asarray(raw)
    flat = arr.flatten()
    names = []
    for elem in flat:
        # Unwrap nested arrays until we reach a scalar
        while isinstance(elem, np.ndarray):
            elem = elem.flatten()
            if elem.size == 0:
                elem = ""
                break
            elem = elem.item() if elem.size == 1 else elem[0]
        names.append(str(elem).strip())
    return [n for n in names if n]


def _try_resolve_companion_ch_names(ch_locator, upload_items):
    """Look for a companion file whose filename stem matches ch_locator.

    Search order:
    1. In upload_items by logical name (covers local-upload mode when channels.mat is in uploads).
    2. In sibling directories of the first data file (covers server-path mode where
       channels.mat lives in an adjacent folder like channels/ next to the data folder).

    Returns (list_of_names, set_of_upload_indices_to_skip) or (None, set()).
    """
    if not ch_locator or not isinstance(ch_locator, str):
        return None, set()

    def _load_dict_and_extract(path, name, ext):
        try:
            obj = _load_uploaded_source_path(path, name, ext)
        except Exception:
            return None
        if not isinstance(obj, MappingABC):
            return None
        key = ch_locator if ch_locator in obj else next(
            (k for k in obj if isinstance(k, str) and not k.startswith("_") and k.lower() == ch_locator.lower()),
            None,
        )
        if key is None:
            return None
        return _flatten_matlab_ch_names(obj[key]) or None

    # Step 1: search upload_items by logical name
    for idx, payload in enumerate(upload_items):
        name = str(payload.get("name") or "")
        stem = os.path.splitext(os.path.basename(name))[0].lower()
        if stem != ch_locator.lower():
            continue
        source_path = payload.get("path")
        ext = str(payload.get("ext") or os.path.splitext(name)[1]).lower()
        names = _load_dict_and_extract(source_path, name, ext)
        if names:
            return names, {idx}

    # Step 2: upward filesystem search from the data directory.
    # Walk up the directory tree (up to 5 levels), and at each ancestor level do a
    # shallow recursive scan (up to depth 2 within that ancestor). This handles cases
    # where channels.mat is in a sibling branch of the dataset tree.
    # Only for real (non-temp) paths — i.e., server-path mode.
    if upload_items:
        first_path = str(upload_items[0].get("path") or "")
        if first_path and os.path.isfile(first_path):
            real_first = os.path.realpath(first_path)
            real_tmp = os.path.realpath(str(TMP_ROOT))
            if not real_first.startswith(real_tmp + os.sep):
                data_dir = os.path.dirname(real_first)
                visited_roots = set()
                search_root = data_dir
                for _level in range(5):
                    if search_root in visited_roots:
                        break
                    visited_roots.add(search_root)
                    try:
                        for root, dirs, files in os.walk(search_root):
                            rel = os.path.relpath(root, search_root)
                            depth = 0 if rel == "." else rel.count(os.sep) + 1
                            if depth >= 2:
                                dirs[:] = []  # prune: don't go deeper than depth 2
                            for ext_try in [".mat", ".npy", ".pkl", ".json", ".csv"]:
                                if (ch_locator + ext_try) in files:
                                    candidate = os.path.join(root, ch_locator + ext_try)
                                    names = _load_dict_and_extract(candidate, ch_locator + ext_try, ext_try)
                                    if names:
                                        return names, set()
                    except Exception:
                        pass
                    next_root = os.path.dirname(search_root)
                    if next_root == search_root:
                        break
                    search_root = next_root

    return None, set()


def _try_resolve_additional_metadata_ch_names(ch_locator, additional_metadata_paths):
    """Resolve channel names from Additional Files tabular metadata columns.

    Returns list[str] or None.
    """
    if not ch_locator or not isinstance(ch_locator, str):
        return None
    entries = list(additional_metadata_paths or [])
    if not entries:
        return None

    values = []
    seen = set()
    for entry in entries:
        path = entry.get("path")
        name = entry.get("name")
        if not path:
            continue
        try:
            obj = _load_uploaded_source_path(path, name, None)
        except Exception:
            continue
        if not isinstance(obj, pd.DataFrame):
            continue
        if ch_locator not in obj.columns:
            continue
        series = obj[ch_locator].dropna()
        for raw in series.tolist():
            token = str(raw).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            values.append(token)

    return values if values else None


def _structure_signature_for_value(value, depth=0, max_depth=2):
    if depth > max_depth:
        return ("depth_limit", type(value).__name__)

    if value is None:
        return ("none",)

    if isinstance(value, pd.DataFrame):
        cols = []
        for col in value.columns.tolist():
            series = value[col]
            cols.append((str(col), str(series.dtype)))
        return ("dataframe", tuple(cols))

    if isinstance(value, pd.Series):
        return ("series", str(value.dtype))

    if isinstance(value, np.ndarray):
        return ("ndarray", int(value.ndim), str(value.dtype))

    if isinstance(value, dict):
        keys = sorted(str(key) for key in value.keys() if not str(key).startswith("__"))
        items = []
        for key in keys[:40]:
            nested = value.get(key)
            nested_sig = _structure_signature_for_value(nested, depth + 1, max_depth)
            items.append((key, nested_sig))
        return ("dict", tuple(items))

    if isinstance(value, (list, tuple)):
        first_non_none = None
        for item in value:
            if item is not None:
                first_non_none = item
                break
        nested_sig = _structure_signature_for_value(first_non_none, depth + 1, max_depth) if first_non_none is not None else None
        return (type(value).__name__, nested_sig)

    if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating)):
        return ("scalar", type(value).__name__)

    if hasattr(value, "__dict__"):
        attrs = sorted([name for name in vars(value).keys() if not str(name).startswith("_")])[:40]
        return ("object", type(value).__name__, tuple(attrs))

    return ("other", type(value).__name__)


def _build_source_structure_signature(source_obj):
    return _structure_signature_for_value(source_obj, depth=0, max_depth=2)


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
    def _collect_numeric_chunks(value, chunks):
        if value is None:
            return
        if isinstance(value, dict):
            for nested in value.values():
                _collect_numeric_chunks(nested, chunks)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                _collect_numeric_chunks(nested, chunks)
            return
        arr = np.asarray(value).ravel()
        if arr.size == 0:
            return
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError):
            return
        if arr.size > 0:
            chunks.append(arr)

    chunks = []
    _collect_numeric_chunks(data, chunks)
    if not chunks:
        return np.asarray([])
    return np.concatenate(chunks)


def _extract_kernel_presynaptic_populations(kernels):
    pops = []
    seen = set()
    if not isinstance(kernels, dict):
        return pops
    for key in kernels.keys():
        if not isinstance(key, str) or ":" not in key:
            continue
        _, pop = key.split(":", 1)
        pop = str(pop).strip()
        if not pop or pop in seen:
            continue
        seen.add(pop)
        pops.append(pop)
    return pops


def _to_1d_numeric_array(value):
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=float)
    try:
        return arr.astype(float, copy=False)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _prepare_spike_times_for_kernel_convolution(spike_times, required_populations):
    source = "as_is"
    data = spike_times

    # If spike times contain multiple trials, use the first dictionary-like trial.
    if isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                data = item
                source = "trial_list_first"
                break

    if not isinstance(data, dict):
        return data, source

    required = [str(pop).strip() for pop in required_populations if str(pop).strip()]
    if not required:
        return data, source

    # Already in expected format: {population: times}
    if all(pop in data for pop in required):
        normalized = {pop: _to_1d_numeric_array(data.get(pop)) for pop in required}
        return normalized, "population"

    # Four-area style format: {area: {population: times}}
    if any(isinstance(val, dict) for val in data.values()):
        aggregated = {}
        missing = []
        for pop in required:
            chunks = []
            for area_payload in data.values():
                if not isinstance(area_payload, dict) or pop not in area_payload:
                    continue
                arr = _to_1d_numeric_array(area_payload.get(pop))
                if arr.size:
                    chunks.append(arr)
            if chunks:
                aggregated[pop] = np.concatenate(chunks)
            else:
                aggregated[pop] = np.asarray([], dtype=float)
                missing.append(pop)
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Missing spike times for population(s): {missing_str}.")
        return aggregated, "area_aggregated"

    return data, source


def _prepare_population_sizes_for_kernel_convolution(population_sizes, required_populations):
    if population_sizes is None or not isinstance(population_sizes, dict):
        return population_sizes, "as_is"

    required = [str(pop).strip() for pop in required_populations if str(pop).strip()]
    if not required:
        return population_sizes, "as_is"

    # Already in expected format: {population: size}
    if all(pop in population_sizes and isinstance(population_sizes.get(pop), (int, float, np.integer, np.floating))
           for pop in required):
        return {pop: float(population_sizes[pop]) for pop in required}, "population"

    # Four-area style format: {area: {population: size}}
    if any(isinstance(val, dict) for val in population_sizes.values()):
        aggregated = {}
        found_any = False
        for pop in required:
            total = 0.0
            found = False
            for area_payload in population_sizes.values():
                if not isinstance(area_payload, dict) or pop not in area_payload:
                    continue
                try:
                    total += float(area_payload[pop])
                    found = True
                    found_any = True
                except (TypeError, ValueError):
                    continue
            if found:
                aggregated[pop] = total
        if found_any:
            return aggregated, "area_aggregated"

    return population_sizes, "as_is"


def _parse_population_area_label(label):
    token = str(label).strip()
    if token.endswith(")") and "(" in token:
        split_idx = token.rfind("(")
        pop = token[:split_idx].strip()
        area = token[split_idx + 1:-1].strip()
        if pop and area:
            return pop, area
    return token, None


def _extract_area_population_layout_from_spike_times(spike_times):
    source = "as_is"
    data = spike_times

    # If spike times contain multiple trials, use the first dictionary-like trial.
    if isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                data = item
                source = "trial_list_first"
                break

    if not isinstance(data, dict):
        return data, [], [], source

    if not data:
        return data, [], [], source

    # Four-area style format: {area: {population: times}}
    if all(isinstance(val, dict) for val in data.values()):
        areas = [str(area).strip() for area in data.keys() if str(area).strip()]
        populations = []
        seen = set()
        for area in areas:
            payload = data.get(area, {})
            if not isinstance(payload, dict):
                continue
            for pop in payload.keys():
                pop_name = str(pop).strip()
                if not pop_name or pop_name in seen:
                    continue
                seen.add(pop_name)
                populations.append(pop_name)
        if areas and populations:
            return data, areas, populations, "area_structured"

    return data, [], [], source


def _scale_kernel_payload(kernel_payload, scale):
    if not isinstance(kernel_payload, dict):
        return kernel_payload
    scaled = {}
    for probe_name, probe_kernel in kernel_payload.items():
        arr = np.asarray(probe_kernel)
        if np.issubdtype(arr.dtype, np.number):
            scaled[probe_name] = np.array(arr, dtype=float, copy=True) * float(scale)
        else:
            scaled[probe_name] = np.array(arr, copy=True)
    return scaled


def _compute_inter_area_kernel_scales(network_obj, populations, areas=None):
    if not isinstance(network_obj, dict):
        return {}

    pop_names = network_obj.get("X")
    area_names = network_obj.get("areas")
    if not isinstance(area_names, (list, tuple)) or not area_names:
        area_names = list(areas or [])
    local_c = network_obj.get("C_YX")
    local_j = network_obj.get("J_YX")
    inter_area = network_obj.get("inter_area", {})
    inter_c = inter_area.get("C_YX") if isinstance(inter_area, dict) else None
    inter_j = inter_area.get("J_YX") if isinstance(inter_area, dict) else None

    if not isinstance(pop_names, (list, tuple)):
        return {}
    if local_c is None or local_j is None or inter_c is None or inter_j is None:
        return {}

    index_by_pop = {str(pop).strip(): idx for idx, pop in enumerate(pop_names)}
    scales = {}
    pop_list = [str(pop).strip() for pop in populations if str(pop).strip()]
    pop_count = len(pop_names)
    area_count = len(area_names)

    # Population-level inter-area matrices (shared for all area pairs).
    try:
        if len(inter_c) == pop_count and len(inter_j) == pop_count:
            for pre in pop_list:
                for post in pop_list:
                    i = index_by_pop.get(pre)
                    j = index_by_pop.get(post)
                    if i is None or j is None:
                        continue
                    local_strength = abs(float(local_c[i][j])) * abs(float(local_j[i][j]))
                    inter_strength = abs(float(inter_c[i][j])) * abs(float(inter_j[i][j]))
                    if local_strength > 0:
                        scales[(pre, post)] = inter_strength / local_strength
                    elif inter_strength == 0:
                        scales[(pre, post)] = 0.0
            return scales
    except (TypeError, ValueError, IndexError, KeyError):
        pass

    # Full area-population matrices (per source/target area-pop pair).
    full_size = pop_count * area_count
    try:
        if area_count > 0 and len(inter_c) == full_size and len(inter_j) == full_size:
            for pre_area_idx, pre_area in enumerate(area_names):
                pre_area_name = str(pre_area).strip()
                for post_area_idx, post_area in enumerate(area_names):
                    post_area_name = str(post_area).strip()
                    if not pre_area_name or not post_area_name:
                        continue
                    for pre in pop_list:
                        for post in pop_list:
                            i = index_by_pop.get(pre)
                            j = index_by_pop.get(post)
                            if i is None or j is None:
                                continue
                            src_idx = pre_area_idx * pop_count + i
                            tgt_idx = post_area_idx * pop_count + j
                            local_strength = abs(float(local_c[i][j])) * abs(float(local_j[i][j]))
                            inter_strength = abs(float(inter_c[src_idx][tgt_idx])) * abs(float(inter_j[src_idx][tgt_idx]))
                            if local_strength > 0:
                                scales[(pre_area_name, pre, post_area_name, post)] = inter_strength / local_strength
                            elif inter_strength == 0:
                                scales[(pre_area_name, pre, post_area_name, post)] = 0.0
            return scales
    except (TypeError, ValueError, IndexError, KeyError):
        pass

    return scales


def _expand_kernels_for_area_combinations(base_kernels, areas, inter_area_scales=None):
    if not isinstance(base_kernels, dict):
        return base_kernels, "as_is"

    area_names = [str(area).strip() for area in areas if str(area).strip()]
    if not area_names:
        return base_kernels, "as_is"

    expanded = {}
    expanded_any = False
    inter_area_scales = inter_area_scales or {}

    for key, kernel_payload in base_kernels.items():
        if not isinstance(key, str) or ":" not in key:
            expanded[key] = kernel_payload
            continue

        post_label, pre_label = key.split(":", 1)
        post_pop, post_area = _parse_population_area_label(post_label)
        pre_pop, pre_area = _parse_population_area_label(pre_label)

        # Keep existing area-resolved kernels unchanged.
        if post_area is not None or pre_area is not None:
            expanded[key] = kernel_payload
            continue

        for post_area_name in area_names:
            for pre_area_name in area_names:
                scale = 1.0
                if post_area_name != pre_area_name:
                    scale = inter_area_scales.get(
                        (pre_area_name, pre_pop, post_area_name, post_pop),
                        inter_area_scales.get((pre_pop, post_pop), 1.0),
                    )
                area_key = f"{post_pop}({post_area_name}):{pre_pop}({pre_area_name})"
                expanded[area_key] = _scale_kernel_payload(kernel_payload, scale)
                expanded_any = True

    if expanded_any:
        return expanded, "area_expanded"
    return base_kernels, "as_is"


def _flatten_area_spike_times_for_kernels(spike_times, areas, populations):
    if not isinstance(spike_times, dict):
        return spike_times
    flattened = {}
    missing = []
    for area in areas:
        payload = spike_times.get(area)
        for pop in populations:
            token = f"{pop}({area})"
            if not isinstance(payload, dict) or pop not in payload:
                flattened[token] = np.asarray([], dtype=float)
                missing.append(token)
                continue
            flattened[token] = _to_1d_numeric_array(payload.get(pop))
    if missing:
        missing_preview = ", ".join(missing[:8])
        if len(missing) > 8:
            missing_preview += ", ..."
        raise KeyError(f"Missing spike times for area/population entry(ies): {missing_preview}.")
    return flattened


def _flatten_area_population_sizes_for_kernels(population_sizes, areas, populations):
    if population_sizes is None or not isinstance(population_sizes, dict):
        return population_sizes, "as_is"

    flattened = {}
    found_any = False
    for area in areas:
        area_payload = population_sizes.get(area)
        for pop in populations:
            token = f"{pop}({area})"
            value = None
            if isinstance(area_payload, dict) and pop in area_payload:
                value = _safe_float(area_payload.get(pop))
            elif pop in population_sizes:
                value = _safe_float(population_sizes.get(pop))
            if value is None:
                continue
            flattened[token] = float(value)
            found_any = True

    if found_any:
        return flattened, "area_flattened"
    return population_sizes, "as_is"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_trial_sequence(value):
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    first = value[0]
    if isinstance(first, (dict, pd.DataFrame, np.ndarray)):
        return True
    if isinstance(first, (list, tuple)):
        if len(first) == 0:
            return False
        nested_first = first[0]
        return isinstance(nested_first, (dict, pd.DataFrame, np.ndarray, list, tuple))
    return False


def _pick_trial_item(value, trial_idx):
    if _is_trial_sequence(value):
        if len(value) == 0:
            return None
        if 0 <= trial_idx < len(value):
            return value[trial_idx]
        return value[-1]
    return value


def _infer_trial_count_from_values(*values):
    count = 1
    for value in values:
        if _is_trial_sequence(value):
            count = max(count, len(value))
    return max(1, count)


def _coerce_dt_ms(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            dt_ms = _coerce_dt_ms(item)
            if dt_ms is not None:
                return dt_ms
        return None

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
        dt_obj = _load_simulation_component_from_path(dt_path, "dt.pkl")
    except Exception:
        return None, dt_path
    return _coerce_dt_ms(dt_obj), dt_path


def _coerce_proxy_series_array(value, label):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([[float(arr)]], dtype=float)
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D numeric array, got shape {arr.shape}.")
    return arr


def _extract_proxy_component_from_payload(payload, component_key):
    if isinstance(payload, list):
        return [_extract_proxy_component_from_payload(item, component_key) for item in payload]

    if isinstance(payload, dict):
        if component_key in payload:
            return _coerce_proxy_series_array(payload[component_key], component_key)
        if component_key == "Vm" and "V_m" in payload:
            return _coerce_proxy_series_array(payload["V_m"], "V_m")

    return _coerce_proxy_series_array(payload, component_key)


def _first_numeric_from_column(df, column_name):
    if column_name not in df.columns:
        return None
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _extract_signal_and_meta_from_source(obj):
    meta = {"dt_ms": None, "decimation_factor": 1, "fs_hz": None}
    dipole_probe_priority = ("CurrentDipoleMoment", "KernelApproxCurrentDipoleMoment")

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
        signal_column = "sum" if "sum" in obj.columns else ("data" if "data" in obj.columns else None)
        if signal_column is not None and not obj.empty:
            selected_row = obj.iloc[0]
            if "probe" in obj.columns:
                for probe_name in dipole_probe_priority:
                    matched = obj[obj["probe"] == probe_name]
                    if not matched.empty:
                        selected_row = matched.iloc[0]
                        break
            return selected_row.get(signal_column), meta
        if not obj.empty:
            return obj.iloc[0, 0], meta
        return None, meta

    if isinstance(obj, MappingABC):
        dt_ms = _safe_float(obj.get("dt_ms"))
        decimation_factor = _safe_float(obj.get("decimation_factor"))
        fs_hz = _safe_float(obj.get("fs_hz"))
        metadata = obj.get("metadata")
        if isinstance(metadata, MappingABC):
            if dt_ms is None:
                dt_ms = _safe_float(metadata.get("dt_ms"))
            if decimation_factor is None:
                decimation_factor = _safe_float(metadata.get("decimation_factor"))
            if fs_hz is None:
                fs_hz = _safe_float(metadata.get("fs_hz"))
        if decimation_factor is None or decimation_factor <= 0:
            decimation_factor = 1
        meta["dt_ms"] = dt_ms
        meta["decimation_factor"] = int(decimation_factor)
        meta["fs_hz"] = fs_hz
        if "sum" in obj:
            return obj.get("sum"), meta
        if "data" in obj:
            return obj.get("data"), meta
        probe_outputs = obj.get("probe_outputs")
        if isinstance(probe_outputs, MappingABC) and probe_outputs:
            for probe_name in dipole_probe_priority:
                candidate = probe_outputs.get(probe_name)
                if candidate is not None:
                    return _extract_signal_and_meta_from_source(candidate)
            first_candidate = next(iter(probe_outputs.values()))
            return _extract_signal_and_meta_from_source(first_candidate)

    return obj, meta


def _signal_time_length(value):
    if isinstance(value, MappingABC):
        lengths = [_signal_time_length(item) for item in value.values()]
        lengths = [length for length in lengths if length is not None]
        return min(lengths) if lengths else None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    return int(arr.shape[-1])


def _clip_signal_time_length(value, length):
    if isinstance(value, MappingABC):
        return {key: _clip_signal_time_length(item, length) for key, item in value.items()}
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.ndim == 0:
        return value
    slicer = [slice(None)] * arr.ndim
    slicer[-1] = slice(0, int(length))
    clipped = arr[tuple(slicer)]
    return np.array(clipped, copy=True)


def _decimate_signal_time(value, factor):
    step = int(factor) if factor is not None else 1
    if step <= 1:
        return value
    if isinstance(value, MappingABC):
        return {key: _decimate_signal_time(item, step) for key, item in value.items()}
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.ndim == 0:
        return value
    slicer = [slice(None)] * arr.ndim
    slicer[-1] = slice(None, None, step)
    decimated = arr[tuple(slicer)]
    return np.array(decimated, copy=True)


def _clip_trial_dataframe_payloads(payloads, signal_columns=("data",)):
    payload_list = [frame.copy() for frame in list(payloads or []) if isinstance(frame, pd.DataFrame)]
    if not payload_list:
        return payload_list, None

    min_length = None
    for frame in payload_list:
        for col in signal_columns:
            if col not in frame.columns:
                continue
            for value in frame[col].tolist():
                length = _signal_time_length(value)
                if length is None:
                    continue
                min_length = length if min_length is None else min(min_length, length)

    if min_length is None:
        return payload_list, None

    for frame in payload_list:
        for col in signal_columns:
            if col not in frame.columns:
                continue
            frame[col] = frame[col].map(lambda value: _clip_signal_time_length(value, min_length))
    return payload_list, int(min_length)


def _sum_signal_dict(signal_dict):
    total = None
    for value in signal_dict.values():
        arr = np.asarray(value)
        if total is None:
            total = np.array(arr, copy=True)
        else:
            total = total + arr
    return total


def _sum_signal_dict_by_post_area(signal_dict, areas):
    if not isinstance(signal_dict, dict):
        return {}
    normalized_areas = [str(area).strip() for area in (areas or []) if str(area).strip()]
    if not normalized_areas:
        return {}
    by_area = {}
    for key, value in signal_dict.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        post_label, _ = key.split(":", 1)
        _, post_area = _parse_population_area_label(post_label)
        if post_area is None:
            continue
        post_area = str(post_area).strip()
        if post_area not in normalized_areas:
            continue
        arr = np.asarray(value)
        if post_area not in by_area:
            by_area[post_area] = np.array(arr, copy=True)
        else:
            by_area[post_area] = by_area[post_area] + arr
    return by_area


def _looks_like_area_mapping(value, areas):
    if not isinstance(value, dict):
        return False
    normalized_areas = [str(area).strip() for area in (areas or []) if str(area).strip()]
    if not normalized_areas:
        return False
    keyset = {str(k).strip() for k in value.keys()}
    return any(area in keyset for area in normalized_areas)


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


class _KernelParamsCreateKernelAdapter:
    """Adapt list/array kernel params (e.g., Cavallari) to create_kernel's per-post loop."""

    def __init__(self, kernel_params):
        self._kernel_params = kernel_params

    def __getattr__(self, name):
        return getattr(self._kernel_params, name)

    @staticmethod
    def _current_post_index():
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        while frame is not None:
            if frame.f_code.co_name == "create_kernel" and "j" in frame.f_locals:
                try:
                    return int(frame.f_locals["j"])
                except Exception:
                    return None
            frame = frame.f_back
        return None

    @property
    def extSynapseParameters(self):
        ext_params = getattr(self._kernel_params, "extSynapseParameters", None)
        post_idx = self._current_post_index()
        if post_idx is not None and isinstance(ext_params, (list, tuple)):
            if 0 <= post_idx < len(ext_params):
                return ext_params[post_idx]
        return ext_params

    @property
    def netstim_interval(self):
        interval = getattr(self._kernel_params, "netstim_interval", None)
        post_idx = self._current_post_index()
        if isinstance(interval, (list, tuple, np.ndarray)):
            arr = np.asarray(interval, dtype=float).ravel()
            if arr.size == 0:
                return interval
            if post_idx is not None and 0 <= post_idx < arr.size:
                return float(arr[post_idx])
            if arr.size == 1:
                return float(arr[0])
            return arr
        return interval


def _adapt_kernel_params_for_create_kernel(kernel_params):
    ext_params = getattr(kernel_params, "extSynapseParameters", None)
    interval = getattr(kernel_params, "netstim_interval", None)
    uses_ext_sequence = isinstance(ext_params, (list, tuple))
    uses_interval_sequence = isinstance(interval, (list, tuple, np.ndarray))
    if not (uses_ext_sequence or uses_interval_sequence):
        return kernel_params, []
    adapted = _KernelParamsCreateKernelAdapter(kernel_params)
    reasons = []
    if uses_ext_sequence:
        reasons.append("extSynapseParameters")
    if uses_interval_sequence:
        reasons.append("netstim_interval")
    return adapted, reasons


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
    method = (raw_method or "").strip()
    return method


def _resolve_sampling_frequency(params, key, df, label):
    fs = _parse_float_param(params, key, default=None)
    if fs is None:
        fs = _resolve_fs_from_df(df)
    if fs is None:
        raise ValueError(f"Sampling frequency is required for {label}.")
    return float(fs)


def _build_feature_method_params(method, params, df):
    method_params = {}
    n_jobs = _parse_int_param(params, "features_n_jobs", default=1)
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

    else:
        raise ValueError(
            f"Unsupported features method '{method}'. Only 'catch22' and 'specparam' are available in webui."
        )

    return method_params, {
        "n_jobs": n_jobs,
        "chunksize": chunksize,
        "start_method": start_method,
    }




#############################################################
##########        COMPUTATION FUNCTIONS           ###########
#############################################################


def _apply_additional_file_metadata(
    df,
    additional_metadata_paths,
    link_field,
    job_status,
    job_id,
    metadata_locators=None,
):
    """Join canonical metadata from an additional tabular file into df.

    Matches df["subject_id"] to additional_df[link_field], then overwrites
    group, condition, species, recording_type (and any other canonical metadata
    columns present in the additional file) in df for each matched subject.
    """
    try:
        from ncpi.EphysDatasetParser import DEFAULT_COLUMNS

        if "subject_id" not in df.columns:
            _append_job_output(
                job_status,
                job_id,
                "subject_id column not present in parsed data — subject mapping skipped.",
            )
            return df

        def _is_missing(value):
            return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)

        def _key_variants(value):
            if _is_missing(value):
                return []
            out = []
            seen = set()

            def _add(token):
                marker = repr(token)
                if marker in seen:
                    return
                seen.add(marker)
                out.append(token)

            _add(value)
            text = str(value).strip()
            if not text:
                return out
            _add(text)

            if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
                try:
                    intval = int(float(text))
                    _add(intval)
                    _add(str(intval))
                except Exception:
                    pass

            if re.fullmatch(r"\d+", text):
                _add(text.lstrip("0") or "0")

            return out

        # Canonical columns to fill (skip data signals, time/freq axes, and the link itself).
        _signal_cols = {
            "data", "fs", "source_file", "epoch", "sensor",
            "t0", "t1", "f0", "f1", "data_domain", "spectral_kind",
        }

        loaded_frames = []
        for entry in additional_metadata_paths:
            add_df = _load_uploaded_source_path(entry["path"], entry["name"], None)
            if not isinstance(add_df, pd.DataFrame):
                _append_job_output(
                    job_status,
                    job_id,
                    f"Additional metadata file '{entry.get('name', 'unknown')}' is not tabular — skipped.",
                )
                continue
            if link_field not in add_df.columns:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Link field '{link_field}' not found in '{entry.get('name', 'unknown')}' — skipped.",
                )
                continue
            loaded_frames.append(add_df)

        if not loaded_frames:
            _append_job_output(
                job_status,
                job_id,
                "No valid additional metadata files available for mapping.",
            )
            return df

        metadata_locators = dict(metadata_locators or {})

        # Decide how each canonical metadata column should be sourced from
        # additional files: prefer explicit locator selected by the user
        # (e.g. group <- "correct"), then fallback to same-name canonical column.
        target_to_source = {}
        for target_col in DEFAULT_COLUMNS:
            target_name = str(target_col)
            if target_name in _signal_cols or target_name in {"subject_id"}:
                continue

            locator = metadata_locators.get(target_name)
            locator_name = str(locator).strip() if isinstance(locator, str) else ""
            source_candidates = []
            if (
                locator_name
                and locator_name not in {FILE_ID_METADATA_LITERAL}
                and not locator_name.startswith(FILE_EXTRACTED_VIRTUAL_FIELD_PREFIX)
            ):
                source_candidates.append(locator_name)
            source_candidates.append(target_name)

            for source_name in source_candidates:
                if any(source_name in add_df.columns for add_df in loaded_frames):
                    target_to_source[target_name] = source_name
                    break

        if not target_to_source:
            _append_job_output(
                job_status,
                job_id,
                "No mappable metadata columns found in additional file(s).",
            )
            return df

        # Build resilient lookup from every additional file; merge non-null values per subject.
        lookup = {}
        for add_df in loaded_frames:
            use_cols = [link_field] + [
                src for src in target_to_source.values() if src in add_df.columns
            ]
            for row in add_df[use_cols].to_dict(orient="records"):
                for key in _key_variants(row.get(link_field)):
                    bucket = lookup.setdefault(key, {})
                    for target_col, source_col in target_to_source.items():
                        if source_col not in row:
                            continue
                        value = row.get(source_col)
                        if _is_missing(value):
                            continue
                        if target_col not in bucket or _is_missing(bucket.get(target_col)):
                            bucket[target_col] = value

        def _lookup_col(sid, col_name):
            for key in _key_variants(sid):
                value = lookup.get(key, {}).get(col_name)
                if not _is_missing(value):
                    return value
            return np.nan

        # Complement existing dataframe values without erasing unmatched rows.
        for col in sorted(target_to_source.keys()):
            mapped = df["subject_id"].map(lambda sid, c=col: _lookup_col(sid, c))
            if col in df.columns:
                df[col] = mapped.where(mapped.notna(), df[col])
            else:
                df[col] = mapped

        _append_job_output(
            job_status,
            job_id,
            "Additional file metadata mapped to dataframe for: "
            + ", ".join(
                f"{target}<-{source}" if target != source else target
                for target, source in sorted(target_to_source.items())
            )
            + ".",
        )
        return df
    except Exception as exc:
        _append_job_output(job_status, job_id,
                           f"Warning: additional file metadata mapping failed: {exc}")
        return df


def features_computation(job_id, job_status, params, temp_uploaded_files):
    output_df_path = None
    try:
        _append_job_output(job_status, job_id, "Starting features computation.")
        if job_id in job_status:
            # Keep progress at 0 during data loading/preparation.
            job_status[job_id]["progress"] = 0
        _append_job_output(job_status, job_id, "Loading data for features...")
        for warning in (params.get("filename_format_filter_warnings") or []):
            message = str(warning or "").strip()
            if message:
                _append_job_output(job_status, job_id, f"Warning: {message}")

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
            filename_format_spec = _parse_filename_format_spec(params.get("parser_filename_format"))
            metadata_locators = (
                dict(parse_cfg.fields.metadata or {})
                if getattr(parse_cfg, "fields", None) is not None
                else {}
            )
            file_id_metadata_fields = []
            for meta_key, locator in metadata_locators.items():
                if isinstance(locator, str) and locator == FILE_ID_METADATA_LITERAL:
                    file_id_metadata_fields.append(str(meta_key))
            file_extracted_metadata_fields = {}
            for meta_key, locator in metadata_locators.items():
                if _file_extracted_chain_index(locator) is not None:
                    file_extracted_metadata_fields[str(meta_key)] = str(locator)
            file_token_metadata_fields = {}
            for meta_key, locator in metadata_locators.items():
                if _file_token_locator_name(locator) is not None:
                    file_token_metadata_fields[str(meta_key)] = str(locator)
            if file_token_metadata_fields and not filename_format_spec:
                fields = ", ".join(sorted(file_token_metadata_fields.keys()))
                raise ValueError(
                    "Filename format is required because metadata fields use filename token locators: "
                    f"{fields}."
                )

            empirical_uploads = list(params.get("empirical_upload_paths") or [])

            # Detect companion channel-names file: if ch_names is a string locator and
            # one of the uploaded files has that exact stem (e.g. "channels.mat"), load
            # it, extract the names, patch the parse config, and exclude it from data parsing.
            if (
                getattr(parse_cfg, "fields", None) is not None
                and isinstance(getattr(parse_cfg.fields, "ch_names", None), str)
                and parse_cfg.fields.ch_names not in {"__self__", ""}
            ):
                _ch_locator_name = str(parse_cfg.fields.ch_names)
                _comp_names, _comp_indices = _try_resolve_companion_ch_names(
                    _ch_locator_name, empirical_uploads
                )
                if _comp_names:
                    import dataclasses
                    from ncpi.EphysDatasetParser import CanonicalFields as _CF, ParseConfig as _PC
                    parse_cfg = dataclasses.replace(
                        parse_cfg,
                        fields=dataclasses.replace(parse_cfg.fields, ch_names=_comp_names),
                    )
                    parser = EphysDatasetParser(parse_cfg)
                    empirical_uploads = [
                        p for i, p in enumerate(empirical_uploads) if i not in _comp_indices
                    ]
                    _append_job_output(
                        job_status, job_id,
                        f"Loaded {len(_comp_names)} channel name(s) from companion file."
                    )
                else:
                    _add_meta_paths = list(params.get("additional_metadata_paths") or [])
                    _meta_names = _try_resolve_additional_metadata_ch_names(
                        _ch_locator_name,
                        _add_meta_paths,
                    )
                    if _meta_names:
                        import dataclasses
                        parse_cfg = dataclasses.replace(
                            parse_cfg,
                            fields=dataclasses.replace(parse_cfg.fields, ch_names=_meta_names),
                        )
                        parser = EphysDatasetParser(parse_cfg)
                        _append_job_output(
                            job_status,
                            job_id,
                            f"Loaded {len(_meta_names)} channel name(s) from Additional Files column "
                            f"'{_ch_locator_name}'."
                        )

            total_uploads = len(empirical_uploads)
            if total_uploads == 0:
                raise ValueError("No empirical uploads were provided.")

            _append_job_output(job_status, job_id, f"Parsing {total_uploads} input file(s)...")
            parsed_frames = []
            structure_reference_by_folder = {}
            log_every = max(1, total_uploads // 20)
            for idx, payload in enumerate(empirical_uploads, start=1):
                name = payload.get("name") or f"file_{idx}"
                source_path = payload.get("path")
                ext = str(payload.get("ext") or os.path.splitext(str(name))[1]).lower()
                size_mb = 0.0
                if source_path and os.path.exists(source_path):
                    try:
                        size_mb = os.path.getsize(source_path) / (1024 * 1024)
                    except OSError:
                        size_mb = 0.0
                _append_job_output(
                    job_status,
                    job_id,
                    f"Loading input file {idx}/{total_uploads}: {name} ({size_mb:.2f} MB)."
                )
                source_obj = _load_uploaded_source_path(
                    source_path,
                    payload.get("name"),
                    payload.get("ext"),
                )
                if job_id in job_status:
                    # Keep users informed during potentially long parsing stages.
                    stage_progress = min(8, max(1, int(8 * idx / max(1, total_uploads))))
                    job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), stage_progress)

                current_signature = _build_source_structure_signature(source_obj)
                folder_key = str(
                    payload.get("folder_path")
                    or payload.get("folder_name")
                    or "__ungrouped__"
                ).strip() or "__ungrouped__"
                folder_label = str(
                    payload.get("folder_name")
                    or payload.get("folder_path")
                    or folder_key
                ).strip() or folder_key
                structure_key = (folder_key, ext)
                if structure_key not in structure_reference_by_folder:
                    structure_reference_by_folder[structure_key] = (str(name), current_signature)
                else:
                    baseline_name, baseline_signature = structure_reference_by_folder[structure_key]
                    if current_signature != baseline_signature:
                        raise ValueError(
                            f"Input file structure mismatch detected in folder '{folder_label}' "
                            f"(extension '{ext}'): '{name}' does not match reference file '{baseline_name}'. "
                            "Ensure files within each folder share the same structure."
                        )

                parse_running = {"flag": True, "last_log": time.time()}
                parse_started = time.time()

                def _parse_heartbeat():
                    while parse_running["flag"]:
                        time.sleep(10.0)
                        if not parse_running["flag"]:
                            break
                        now = time.time()
                        elapsed = int(now - parse_started)
                        if now - parse_running["last_log"] >= 20.0:
                            _append_job_output(
                                job_status,
                                job_id,
                                f"Still parsing '{name}'... ({elapsed}s elapsed)"
                            )
                            parse_running["last_log"] = now

                heartbeat = threading.Thread(target=_parse_heartbeat, daemon=True)
                heartbeat.start()
                try:
                    parsed = parser.parse(source_obj)
                finally:
                    parse_running["flag"] = False
                    heartbeat.join(timeout=0.2)
                _append_job_output(
                    job_status,
                    job_id,
                    f"Finished parsing '{name}' in {time.time() - parse_started:.1f}s."
                )
                if not isinstance(parsed, pd.DataFrame):
                    raise ValueError(f"Parser output for '{name}' is not a dataframe.")
                if file_id_metadata_fields:
                    file_id_value = str(name)
                    for col_name in file_id_metadata_fields:
                        parsed[col_name] = file_id_value
                if file_extracted_metadata_fields:
                    for col_name, locator in file_extracted_metadata_fields.items():
                        parsed[col_name] = _resolve_file_extracted_value(name, locator)
                if file_token_metadata_fields:
                    for col_name, locator in file_token_metadata_fields.items():
                        token_value = _resolve_file_token_value(name, locator, filename_format_spec)
                        if token_value is None:
                            token_name = _file_token_locator_name(locator) or locator
                            raise ValueError(
                                f"File '{name}' does not match filename format for token '{token_name}'."
                            )
                        parsed[col_name] = token_value
                parsed_frames.append(parsed)

                # Do not advance global progress bar during data loading.
                if idx == 1 or idx == total_uploads or (idx % log_every == 0):
                    _append_job_output(job_status, job_id, f"Parsed input file {idx}/{total_uploads}.")

            df = pd.concat(parsed_frames, ignore_index=True)
            _append_job_output(job_status, job_id, f"Merged empirical parsed dataframe shape: {df.shape}.")

            # Apply additional file metadata lookup (subject-level cross-reference)
            _add_meta_paths = params.get("additional_metadata_paths")
            _add_link_field = params.get("additional_file_link_field", "")
            if _add_meta_paths and _add_link_field and "subject_id" in df.columns:
                _meta_locators = (
                    dict(parse_cfg.fields.metadata or {})
                    if getattr(parse_cfg, "fields", None) is not None
                    else {}
                )
                df = _apply_additional_file_metadata(
                    df,
                    _add_meta_paths,
                    _add_link_field,
                    job_status,
                    job_id,
                    metadata_locators=_meta_locators,
                )
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

        sig = inspect.signature(features.compute_features)
        supports_progress = "progress_callback" in sig.parameters
        progress_state = {
            "callback_seen": False,
            "running": True,
            "last_heartbeat_log": time.time(),
        }

        def _on_feature_progress(completed, total, pct):
            if total <= 0:
                return
            progress_state["callback_seen"] = True
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

        if job_id in job_status:
            job_status[job_id]["progress"] = max(job_status[job_id].get("progress", 0), 10)

        def _feature_progress_heartbeat():
            while progress_state["running"]:
                time.sleep(2.0)
                if not progress_state["running"]:
                    break
                # If callbacks are working, trust them and stop synthetic progress updates.
                if progress_state["callback_seen"]:
                    continue
                if job_id in job_status:
                    current = int(job_status[job_id].get("progress", 0))
                    if current < 95:
                        job_status[job_id]["progress"] = current + 1
                now = time.time()
                if now - progress_state["last_heartbeat_log"] >= 15.0:
                    _append_job_output(
                        job_status,
                        job_id,
                        "Feature extraction is still running..."
                    )
                    progress_state["last_heartbeat_log"] = now

        heartbeat_thread = threading.Thread(target=_feature_progress_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            computed = _compute_features_with_compat(
                features_obj=features,
                samples=samples,
                exec_opts=exec_opts,
                progress_callback=_on_feature_progress,
                log_callback=_on_feature_log,
            )
        finally:
            progress_state["running"] = False
            heartbeat_thread.join(timeout=0.2)
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
        _announce_saved_output_folders(job_status, job_id, "features", output_df_path, persisted_dashboard_path)

        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": output_df_path, # Return to the client the output filepath
                "dashboard_features_path": persisted_dashboard_path,
                "error": False
            })

    except Exception as e:
        _mark_job_failed(job_status, job_id, e)

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
        loaders.append(("mat.load_with_fallback", lambda p: _load_mat_with_fallback(p, in_memory=False, source_name=os.path.basename(str(p)))))

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
        generic_fallbacks.append(("mat.load_with_fallback", lambda p: _load_mat_with_fallback(p, in_memory=False, source_name=os.path.basename(str(p)))))
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

    temp_root = tempfile.mkdtemp(prefix="ncpi_train_loader_", dir=TMP_ROOT)
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
        if model_assets_source not in {"upload", "server-path"}:
            raise ValueError(f"Unsupported prediction artifacts source mode: {model_assets_source}")

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
        inference_n_jobs = _parse_int_param(params, "inference_n_jobs", default=1)
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
            predictions_dir = PREDICTIONS_DATA_DIR
            os.makedirs(predictions_dir, exist_ok=True)
            persisted_name = "predictions.pkl"
            persisted_predictions_path = os.path.join(predictions_dir, persisted_name)
            output_df.to_pickle(persisted_predictions_path)
            _append_job_output(job_status, job_id, f"Persisted dashboard predictions file: {persisted_predictions_path}")
        except Exception as persist_exc:
            _append_job_output(job_status, job_id, f"Warning: could not persist dashboard predictions file: {persist_exc}")
        _append_job_output(job_status, job_id, f"Saved predictions dataframe: {output_df_path}")
        _announce_saved_output_folders(job_status, job_id, "inference", output_df_path, persisted_predictions_path)
        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "estimated_time_remaining": 0,
            "results": output_df_path,
            "error": False,
            "dashboard_predictions_path": persisted_predictions_path,
        })

    except Exception as e:
        _mark_job_failed(job_status, job_id, e)
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
            persisted_dir = INFERENCE_TRAINING_DATA_DIR
            os.makedirs(persisted_dir, exist_ok=True)
            persisted_zip_path = os.path.join(persisted_dir, "training_artifacts.zip")
            shutil.copy2(zip_path, persisted_zip_path)
            _append_job_output(job_status, job_id, f"Persisted training artifacts to: {persisted_zip_path}")
        except Exception as persist_exc:
            _append_job_output(
                job_status,
                job_id,
                f"Warning: could not persist training artifacts to {TMP_ROOT}: {persist_exc}",
            )
        _announce_saved_output_folders(job_status, job_id, "inference_training", persisted_zip_path or zip_path)

        job_status[job_id].update({
            "status": "finished",
            "progress": 100,
            "estimated_time_remaining": 0,
            "results": persisted_zip_path or zip_path,
            "error": False,
        })

    except Exception as e:
        _mark_job_failed(job_status, job_id, e)
    finally:
        cleanup_temp_files(file_paths)
        if os.path.isdir(artifacts_dir):
            shutil.rmtree(artifacts_dir, ignore_errors=True)


def analysis_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        results_path = os.path.join(temp_uploaded_files, "LFP_predictions.png")
        # Save the image in the module upload folder.
        # LFP_predictions_webversion.run_full_pipeline([params['method-plot']], params['method'])
        
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": results_path, # Return to the client the output filepath
                "error": False
            })

    except Exception as e:
        _mark_job_failed(job_status, job_id, e)

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
        proxy_decimation_factor = _parse_int_param(params, "proxy_decimation_factor", default=1)
        if proxy_decimation_factor is None or proxy_decimation_factor < 1:
            raise ValueError("proxy_decimation_factor must be an integer >= 1.")
        proxy_decimation_factor = int(proxy_decimation_factor)
        if proxy_decimation_factor > 1:
            _append_job_output(
                job_status,
                job_id,
                f"Applying proxy decimation factor x{proxy_decimation_factor}.",
            )

        excitatory_only_value = params.get('excitatory_only', 'default').lower()
        if excitatory_only_value == 'true':
            excitatory_only = True
        elif excitatory_only_value == 'false':
            excitatory_only = False
        else:
            excitatory_only = None

        file_paths = params.get('file_paths', {})
        sim_data = {}
        fr_times = None
        fr_gids = None
        proxy_network_areas = []
        nu_ext_mode = str(params.get("nu_ext_mode") or "shared").strip().lower()
        if nu_ext_mode not in {"shared", "per-trial"}:
            nu_ext_mode = "shared"

        def _is_truthy(value):
            return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

        def _resolve_proxy_file(key, default_name, required=True):
            ignore_default = _is_truthy(params.get(f"{key}_ignore_default"))
            fallback_name = None if ignore_default else default_name
            return _resolve_sim_file(file_paths, key, fallback_name, required=required)

        def _resolve_proxy_file_candidates(key, default_names, required=True):
            ignore_default = _is_truthy(params.get(f"{key}_ignore_default"))
            uploaded_path = file_paths.get(key)
            if uploaded_path and allowed_file(uploaded_path) and os.path.exists(uploaded_path):
                return uploaded_path
            if not ignore_default:
                for default_name in default_names:
                    if not default_name:
                        continue
                    default_path = os.path.join(DEFAULT_SIM_DATA_DIR, default_name)
                    if os.path.exists(default_path):
                        return default_path
                bundle_path = os.path.join(DEFAULT_SIM_DATA_DIR, SIMULATION_BUNDLE_FILE)
                if os.path.exists(bundle_path):
                    return bundle_path
                for legacy_name in SIMULATION_LEGACY_BUNDLE_FILES:
                    legacy_bundle_path = os.path.join(DEFAULT_SIM_DATA_DIR, legacy_name)
                    if os.path.exists(legacy_bundle_path):
                        return legacy_bundle_path
            if required:
                expected = ", ".join(default_names)
                raise FileNotFoundError(
                    f"Missing required input '{key}'. Upload a file or place one of [{expected}] in {DEFAULT_SIM_DATA_DIR}."
                )
            return None

        def _load_proxy_component(component_key, file_key, default_names, label):
            source_path = _resolve_proxy_file_candidates(file_key, default_names, required=True)
            payload = read_file(source_path)
            if _is_simulation_bundle_path(source_path):
                bundle_payload = _coerce_simulation_bundle_payload(payload)
                if bundle_payload is None:
                    raise ValueError(f"Invalid simulation bundle file: {source_path}")
                bundle_field_map = {"AMPA": "ampa", "GABA": "gaba", "Vm": "vm"}
                preferred_field = bundle_field_map.get(component_key)
                payload = bundle_payload.get(preferred_field)
                if payload is None:
                    payload = bundle_payload.get("exc_state_events")
                if payload is None:
                    raise ValueError(
                        f"Simulation bundle {os.path.basename(source_path)} does not contain {component_key} data."
                    )
            component = _extract_proxy_component_from_payload(payload, component_key)
            source_name = os.path.basename(source_path)
            if source_name == "exc_state_events.pkl":
                _append_job_output(
                    job_status,
                    job_id,
                    f"Using {component_key} extracted from {source_name}.",
                )
            return component

        def _coerce_area_list(values):
            ordered = []
            seen = set()
            for value in values or []:
                area = str(value or "").strip()
                if not area or area in seen:
                    continue
                seen.add(area)
                ordered.append(area)
            return ordered

        dt_from_stage, dt_path = _load_simulation_dt_ms(file_paths)
        proxy_sim_step = None
        dt_source = None
        try:
            network_path = _resolve_proxy_file("network_file", "network.pkl", required=False)
            if network_path:
                network_payload = _load_simulation_component_from_path(network_path, "network.pkl")
                network_trial = _pick_trial_item(network_payload, 0)
                if isinstance(network_trial, dict):
                    proxy_network_areas = _coerce_area_list(network_trial.get("areas"))
        except Exception:
            proxy_network_areas = []

        if method == 'FR':
            _append_job_output(job_status, job_id, "Loading spike times and gids...")
            times_path = _resolve_proxy_file('times_file', 'times.pkl', required=True)
            gids_path = _resolve_proxy_file('gids_file', 'gids.pkl', required=True)
            fr_times = _load_simulation_component_from_path(times_path, "times.pkl")
            fr_gids = _load_simulation_component_from_path(gids_path, "gids.pkl")
            proxy_sim_step = float(bin_size)
            dt_source = "bin_size_ms"

        elif method == 'AMPA':
            _append_job_output(job_status, job_id, "Loading AMPA currents...")
            sim_data['AMPA'] = _load_proxy_component(
                "AMPA",
                "ampa_file",
                ["ampa.pkl", "exc_state_events.pkl"],
                "AMPA currents",
            )

        elif method == 'GABA':
            _append_job_output(job_status, job_id, "Loading GABA currents...")
            sim_data['GABA'] = _load_proxy_component(
                "GABA",
                "gaba_file",
                ["gaba.pkl", "exc_state_events.pkl"],
                "GABA currents",
            )

        elif method == 'Vm':
            _append_job_output(job_status, job_id, "Loading membrane potentials...")
            sim_data['Vm'] = _load_proxy_component(
                "Vm",
                "vm_file",
                ["vm.pkl", "exc_state_events.pkl"],
                "membrane potentials",
            )

        elif method in {'I', 'I_abs', 'LRWS', 'ERWS1', 'ERWS2'}:
            _append_job_output(job_status, job_id, "Loading AMPA and GABA currents...")
            sim_data['AMPA'] = _load_proxy_component(
                "AMPA",
                "ampa_file",
                ["ampa.pkl", "exc_state_events.pkl"],
                "AMPA currents",
            )
            sim_data['GABA'] = _load_proxy_component(
                "GABA",
                "gaba_file",
                ["gaba.pkl", "exc_state_events.pkl"],
                "GABA currents",
            )

            if method == 'ERWS2':
                _append_job_output(job_status, job_id, f"Preparing nu_ext values ({nu_ext_mode} mode)...")
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
                    "Simulation step could not be determined. Provide sim_step or ensure dt is present in simulation outputs "
                    f"(either dt.pkl or {SIMULATION_BUNDLE_FILE})."
                )

        _append_job_output(job_status, job_id, f"Effective proxy sampling step: {proxy_sim_step:g} ms ({dt_source}).")

        trial_count = _infer_trial_count_from_values(
            fr_times,
            fr_gids,
            *(sim_data.values()),
        )
        if job_id in job_status:
            job_status[job_id]["simulation_total"] = int(trial_count)
            job_status[job_id]["simulation_completed"] = 0
            job_status[job_id]["progress"] = 0

        def _set_sample_progress(completed_samples):
            if job_id not in job_status:
                return
            total = max(1, int(trial_count))
            completed = max(0, min(int(completed_samples), total))
            job_status[job_id]["simulation_completed"] = completed
            job_status[job_id]["progress"] = int((completed / total) * 100)

        if trial_count > 1:
            _append_job_output(job_status, job_id, f"Detected {trial_count} trial(s); computing proxy for each trial.")

        if method == "ERWS2":
            if nu_ext_mode == "per-trial":
                raw_values = params.get("nu_ext_values_json")
                try:
                    parsed_values = json.loads(raw_values) if raw_values else []
                except Exception:
                    raise ValueError("Per-trial nu_ext values must be valid JSON.")
                if not isinstance(parsed_values, list) or not parsed_values:
                    raise ValueError("Provide one nu_ext value per detected trial.")
                numeric_values = []
                for idx, value in enumerate(parsed_values):
                    numeric = _safe_float(value)
                    if numeric is None:
                        raise ValueError(f"Invalid nu_ext value for trial {idx + 1}: {value}")
                    numeric_values.append(float(numeric))
                if len(numeric_values) != trial_count:
                    raise ValueError(
                        f"Per-trial nu_ext values count ({len(numeric_values)}) does not match detected trials ({trial_count})."
                    )
                sim_data["nu_ext"] = numeric_values
                _append_job_output(job_status, job_id, f"Using per-trial nu_ext values for {trial_count} trial(s).")
            else:
                nu_ext_value = _safe_float(params.get("nu_ext_value"))
                if nu_ext_value is None:
                    raise ValueError("Provide a numeric nu_ext value for ERWS2.")
                sim_data["nu_ext"] = float(nu_ext_value)
                _append_job_output(job_status, job_id, f"Using shared nu_ext value: {float(nu_ext_value):g}.")

        _append_job_output(job_status, job_id, "Computing proxy with ncpi.FieldPotential.compute_proxy...")
        potential = ncpi.FieldPotential()
        trial_proxy_payloads = []
        for trial_idx in range(trial_count):
            area_proxy_signals = {}
            if method == "FR":
                trial_times = _pick_trial_item(fr_times, trial_idx)
                trial_gids = _pick_trial_item(fr_gids, trial_idx)
                trial_sim_data = {"FR": _compute_mean_firing_rate(trial_times, trial_gids, bin_size)}
                if isinstance(trial_times, dict) and isinstance(trial_gids, dict):
                    area_keys = [
                        str(key).strip()
                        for key, value in trial_times.items()
                        if isinstance(value, dict) and isinstance(trial_gids.get(key), dict)
                    ]
                    area_keys = _coerce_area_list(area_keys)
                    if len(area_keys) >= 2:
                        for area_name in area_keys:
                            area_sim_data = {
                                "FR": _compute_mean_firing_rate(
                                    trial_times.get(area_name, {}),
                                    trial_gids.get(area_name, {}),
                                    bin_size,
                                )
                            }
                            area_proxy_signals[area_name] = potential.compute_proxy(
                                method,
                                area_sim_data,
                                proxy_sim_step,
                                excitatory_only=excitatory_only,
                            )
                        if area_proxy_signals:
                            _append_job_output(
                                job_status,
                                job_id,
                                f"Computed area-resolved proxy sums for trial {trial_idx + 1}: {', '.join(area_proxy_signals.keys())}.",
                            )
            else:
                trial_sim_data = {
                    key: _pick_trial_item(value, trial_idx)
                    for key, value in sim_data.items()
                    if key != "nu_ext"
                }
                if method == "ERWS2":
                    nu_ext_payload = sim_data.get("nu_ext")
                    if isinstance(nu_ext_payload, (list, tuple)):
                        idx = trial_idx if trial_idx < len(nu_ext_payload) else len(nu_ext_payload) - 1
                        trial_sim_data["nu_ext"] = float(nu_ext_payload[idx])
                    else:
                        trial_sim_data["nu_ext"] = nu_ext_payload

                candidate_areas = list(proxy_network_areas)
                for value in trial_sim_data.values():
                    if isinstance(value, dict):
                        candidate_areas.extend(str(key).strip() for key in value.keys())
                area_keys = _coerce_area_list(candidate_areas)
                if area_keys:
                    for area_name in area_keys:
                        area_trial_sim_data = {}
                        has_area_specific_component = False
                        for key, value in trial_sim_data.items():
                            if _looks_like_area_mapping(value, area_keys) and area_name in value:
                                area_trial_sim_data[key] = value[area_name]
                                has_area_specific_component = True
                            else:
                                area_trial_sim_data[key] = value
                        if not has_area_specific_component:
                            continue
                        area_proxy_signals[area_name] = potential.compute_proxy(
                            method,
                            area_trial_sim_data,
                            proxy_sim_step,
                            excitatory_only=excitatory_only,
                        )
                    if area_proxy_signals:
                        _append_job_output(
                            job_status,
                            job_id,
                            f"Computed area-resolved proxy sums for trial {trial_idx + 1}: {', '.join(area_proxy_signals.keys())}.",
                        )

            proxy = potential.compute_proxy(method, trial_sim_data, proxy_sim_step, excitatory_only=excitatory_only)
            proxy_area_payload = None
            proxy_area_raw = None
            if area_proxy_signals:
                ordered_area_names = _coerce_area_list(
                    proxy_network_areas if proxy_network_areas else list(area_proxy_signals.keys())
                )
                if not ordered_area_names:
                    ordered_area_names = _coerce_area_list(list(area_proxy_signals.keys()))
                proxy_area_payload = {}
                proxy_area_raw = {}
                for area_name in ordered_area_names:
                    if area_name not in area_proxy_signals:
                        continue
                    signal = np.asarray(area_proxy_signals[area_name], dtype=float)
                    proxy_area_payload[area_name] = signal
                    proxy_area_raw[f"proxy({area_name})"] = signal
            elif isinstance(proxy, MappingABC):
                candidate_areas = _coerce_area_list(proxy_network_areas)
                if not candidate_areas:
                    for value in trial_sim_data.values():
                        if isinstance(value, dict):
                            candidate_areas.extend(str(key).strip() for key in value.keys())
                    candidate_areas = _coerce_area_list(candidate_areas)
                area_sum_payload = _sum_signal_dict_by_post_area(proxy, candidate_areas)
                if area_sum_payload and all(area in area_sum_payload for area in candidate_areas):
                    proxy_area_payload = area_sum_payload
                    proxy_area_raw = dict(proxy)
            dt_ms = _safe_float(proxy_sim_step)
            row = {
                "data": proxy_area_payload if proxy_area_payload else proxy,
                "proxy_method": method,
                "dt_ms": dt_ms,
                "decimation_factor": proxy_decimation_factor,
                "fs_hz": None,
                "metadata": {"dt_ms": dt_ms, "decimation_factor": proxy_decimation_factor, "fs_hz": None, "dt_source": dt_source},
            }
            if proxy_area_raw:
                row["raw_signals"] = proxy_area_raw
                row["sum"] = proxy_area_payload
            if proxy_decimation_factor > 1:
                row["data"] = _decimate_signal_time(row["data"], proxy_decimation_factor)
                if "raw_signals" in row:
                    row["raw_signals"] = _decimate_signal_time(row["raw_signals"], proxy_decimation_factor)
                if "sum" in row:
                    row["sum"] = _decimate_signal_time(row["sum"], proxy_decimation_factor)
            dt_val = row.get("dt_ms")
            if dt_val is not None and dt_val > 0:
                fs_hz = 1000.0 / (float(dt_val) * float(proxy_decimation_factor))
                row["fs_hz"] = fs_hz
                row["metadata"] = {
                    "dt_ms": float(dt_val),
                    "decimation_factor": proxy_decimation_factor,
                    "fs_hz": fs_hz,
                    "dt_source": dt_source,
                }
            if trial_count > 1:
                row["trial_index"] = int(trial_idx)

            trial_proxy_payloads.append(pd.DataFrame([row]))
            _set_sample_progress(trial_idx + 1)
            if trial_count > 1:
                _append_job_output(job_status, job_id, f"Computed proxy trial {trial_idx + 1}/{trial_count}.")

        trial_proxy_payloads, clipped_length = _clip_trial_dataframe_payloads(
            trial_proxy_payloads,
            signal_columns=("data", "raw_signals", "sum"),
        )
        if clipped_length is not None:
            _append_job_output(
                job_status,
                job_id,
                f"Clipped proxy time-series to common minimum length {clipped_length} samples across trials.",
            )

        output_root = _field_potential_output_dir("proxy")
        os.makedirs(output_root, exist_ok=True)
        # Keep canonical short filenames for proxy outputs.
        for name in ('proxy.pkl', SIMULATION_BUNDLE_FILE, *SIMULATION_LEGACY_BUNDLE_FILES):
            old_path = os.path.join(output_root, name)
            if os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        proxy_path = os.path.join(output_root, 'proxy.pkl')

        with open(proxy_path, "wb") as f:
            pickle.dump(trial_proxy_payloads, f)

        _append_job_output(job_status, job_id, f"Saved proxy.pkl to {output_root}")
        _announce_saved_output_folders(job_status, job_id, "field_potential_proxy", output_root)
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": proxy_path,
                "error": False
            })

    except Exception as e:
        _mark_job_failed(job_status, job_id, e)

    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_kernel_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting kernel computation.")
        mc_folder = str(params.get("mc_folder") or "").strip()
        output_sim_path = str(params.get("output_sim_path") or "").strip()
        params_path = str(params.get("kernel_params_module") or "").strip()

        if not mc_folder:
            raise ValueError("Multicompartment neuron model folder is required.")
        if not os.path.isdir(mc_folder):
            raise FileNotFoundError(f"Multicompartment neuron model folder not found: {mc_folder}")
        if not params_path:
            raise ValueError("Kernel parameters module is required.")
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"Kernel parameters module not found: {params_path}")

        _append_job_output(job_status, job_id, f"MC folder: {mc_folder}")
        _append_job_output(job_status, job_id, f"Params module: {params_path}")
        _append_job_output(
            job_status,
            job_id,
            f"Output simulation path: {output_sim_path if output_sim_path else '(not provided)'}",
        )

        module = _load_module_from_path(params_path, name="kernel_params")
        KernelParams = getattr(module, "KernelParams", None)
        if KernelParams is None:
            raise AttributeError("KernelParams not found in params module.")
        KernelParams, adapter_reasons = _adapt_kernel_params_for_create_kernel(KernelParams)
        if adapter_reasons:
            _append_job_output(
                job_status,
                job_id,
                "Detected sequence-based kernel parameters for "
                + ", ".join(adapter_reasons)
                + "; applying per-population adapter for create_kernel compatibility.",
            )
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
        elif isinstance(biophys, list) and len(biophys) == 0:
            cell_parameters = getattr(KernelParams, "cellParameters", None)
            custom_fun = cell_parameters.get("custom_fun") if isinstance(cell_parameters, dict) else None
            if callable(custom_fun):
                biophys = [custom_fun]
                _append_job_output(
                    job_status,
                    job_id,
                    "Biophysical membrane properties left empty; using KernelParams.cellParameters.custom_fun.",
                )
            elif isinstance(custom_fun, (list, tuple)) and len(custom_fun) > 0:
                biophys = list(custom_fun)
                _append_job_output(
                    job_status,
                    job_id,
                    "Biophysical membrane properties left empty; using KernelParams.cellParameters.custom_fun.",
                )
            else:
                biophys = ["set_Ih_linearized_hay2011", "make_cell_uniform"]
                _append_job_output(
                    job_status,
                    job_id,
                    "Biophysical membrane properties left empty and no KernelParams.cellParameters.custom_fun found; "
                    "using default webui biophys functions.",
                )

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

        # electrodeParameters should always be loaded from the KernelParams (analysis_params).
        # Do NOT allow passing electrode parameters via uploaded files or params from the web UI.
        electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
        # If probes are selected and GaussCylinderPotential is not among them, we don't need electrodeParameters
        if probe_selection_present and not probe_gauss_cylinder:
            electrodeParameters = None
        # If GaussCylinderPotential is selected ensure electrodeParameters exist in KernelParams
        if probe_selection_present and probe_gauss_cylinder and electrodeParameters is None:
            raise ValueError("electrodeParameters must be provided in KernelParams when GaussCylinderPotential is selected.")

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

        if output_sim_path and not os.path.exists(output_sim_path):
            raise FileNotFoundError(f"Output simulation path not found: {output_sim_path}")
        if not output_sim_path:
            if mean_nu_x is None or vrest is None:
                raise FileNotFoundError(
                    "Multicompartment network simulation outputs path was not provided. "
                    "Provide mean_nu_X and Vrest, or select a valid output_sim_path."
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

        output_root = _field_potential_output_dir("kernel")
        os.makedirs(output_root, exist_ok=True)
        for name in os.listdir(output_root):
            lower = name.lower()
            if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                continue
            old_path = os.path.join(output_root, name)
            if os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        _append_job_output(job_status, job_id, "Computing CDM/LFP from kernels...")
        spike_times_path = _resolve_sim_file(
            file_paths, "kernel_spike_times_file", "times.pkl", required=True
        )
        spike_times_raw = _load_simulation_component_from_path(spike_times_path, "times.pkl")
        manual_population_sizes = {}
        manual_pop_e = _safe_float(params.get("kernel_population_size_e"))
        manual_pop_i = _safe_float(params.get("kernel_population_size_i"))
        if manual_pop_e is not None:
            manual_population_sizes["E"] = float(manual_pop_e)
        if manual_pop_i is not None:
            manual_population_sizes["I"] = float(manual_pop_i)

        if manual_population_sizes:
            population_sizes_raw = manual_population_sizes
            _append_job_output(
                job_status,
                job_id,
                "Using manual population sizes for kernel convolution normalization.",
            )
        else:
            pop_sizes_path = _resolve_sim_file(
                file_paths, "kernel_population_sizes_file", "population_sizes.pkl", required=False
            )
            population_sizes_raw = None
            if pop_sizes_path:
                try:
                    population_sizes_raw = _load_simulation_component_from_path(
                        pop_sizes_path, "population_sizes.pkl"
                    )
                except ValueError as exc:
                    if (
                        _is_simulation_bundle_path(pop_sizes_path)
                        and "does not contain 'population_sizes'" in str(exc)
                    ):
                        population_sizes_raw = None
                        _append_job_output(
                            job_status,
                            job_id,
                            "No population_sizes found in simulation bundle; proceeding without spike-rate normalization.",
                        )
                    else:
                        raise

        network_path = _resolve_sim_file(
            file_paths, "kernel_network_file", "network.pkl", required=False
        )
        if not network_path:
            candidate_network = os.path.join(os.path.dirname(spike_times_path), "network.pkl")
            if os.path.exists(candidate_network):
                network_path = candidate_network
            else:
                candidate_bundle = os.path.join(os.path.dirname(spike_times_path), SIMULATION_BUNDLE_FILE)
                if os.path.exists(candidate_bundle):
                    network_path = candidate_bundle
        network_obj = None
        if network_path and os.path.exists(network_path):
            try:
                network_obj = _load_simulation_component_from_path(network_path, "network.pkl")
            except Exception as exc:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Warning: failed to load network file for inter-area scaling ({exc}).",
                )

        trial_count = _infer_trial_count_from_values(spike_times_raw, population_sizes_raw)
        if job_id in job_status:
            job_status[job_id]["simulation_total"] = int(trial_count)
            job_status[job_id]["simulation_completed"] = 0
            job_status[job_id]["progress"] = 0

        def _set_sample_progress(completed_samples):
            if job_id not in job_status:
                return
            total = max(1, int(trial_count))
            completed = max(0, min(int(completed_samples), total))
            job_status[job_id]["simulation_completed"] = completed
            job_status[job_id]["progress"] = int((completed / total) * 100)

        if trial_count > 1:
            _append_job_output(job_status, job_id, f"Detected {trial_count} trial(s); computing CDM/LFP for each trial.")

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
        if component_val in (None, "", "None"):
            component = None
        else:
            component_token = str(component_val).strip().lower()
            axis_to_index = {"x": 0, "y": 1, "z": 2}
            if component_token in axis_to_index:
                component = axis_to_index[component_token]
            elif component_token in {"xyz", "all", "3", "3d"}:
                component = None
            else:
                component = int(float(component_val))
        component_label = "xyz (all)" if component is None else str(component)
        _append_job_output(job_status, job_id, f"CDM/LFP component selection: {component_label}.")
        mode = params.get("cdm_mode") or "same"
        scale = _parse_float_param(params, "cdm_scale", default=(float(dt) * 1e-3))
        if scale is None or not np.isfinite(scale):
            raise ValueError("cdm_scale must be a valid numeric value.")
        cdm_decimation_factor = _parse_int_param(params, "cdm_decimation_factor", default=1)
        if cdm_decimation_factor is None or cdm_decimation_factor < 1:
            raise ValueError("cdm_decimation_factor must be an integer >= 1.")
        cdm_decimation_factor = int(cdm_decimation_factor)
        kernels_for_save = None
        probe_trial_payloads = {probe_name: [] for probe_name in probe_names}
        logged_area_detected = False
        logged_scaling_applied = False
        logged_scaling_warning = False
        logged_area_aggregated = False
        logged_population_flattened = False
        logged_population_aggregated = False
        logged_component_shape = False
        logged_target_area_sum = False

        fs_hz = (1000.0 / float(cdm_dt)) if float(cdm_dt) > 0 else None
        fs_hz_after_decimation = (fs_hz / cdm_decimation_factor) if (fs_hz is not None and cdm_decimation_factor > 1) else fs_hz
        if cdm_decimation_factor > 1:
            _append_job_output(
                job_status,
                job_id,
                f"Applying CDM/LFP decimation factor x{cdm_decimation_factor} after convolution.",
            )
        for trial_idx in range(trial_count):
            spike_times_trial_raw = _pick_trial_item(spike_times_raw, trial_idx)
            spike_times_input, area_names, area_populations, _ = _extract_area_population_layout_from_spike_times(
                spike_times_trial_raw
            )

            kernels_for_convolution = kernels
            if area_names and area_populations:
                if not logged_area_detected:
                    _append_job_output(
                        job_status,
                        job_id,
                        f"Detected area-wise spike times ({', '.join(area_names)}); expanding kernels per area/population pair.",
                    )
                    logged_area_detected = True
                inter_area_scales = {}
                if network_obj is not None:
                    try:
                        network_trial_obj = _pick_trial_item(network_obj, trial_idx)
                        inter_area_scales = _compute_inter_area_kernel_scales(
                            network_trial_obj,
                            area_populations,
                            areas=area_names,
                        )
                        if inter_area_scales and not logged_scaling_applied:
                            _append_job_output(
                                job_status,
                                job_id,
                                "Applied inter-area kernel scaling from network connectivity (network.pkl).",
                            )
                            logged_scaling_applied = True
                    except Exception as exc:
                        if not logged_scaling_warning:
                            _append_job_output(
                                job_status,
                                job_id,
                                f"Warning: failed to compute inter-area scaling ({exc}). Using unscaled area kernels.",
                            )
                            logged_scaling_warning = True

                kernels_for_convolution, kernel_mode = _expand_kernels_for_area_combinations(
                    kernels,
                    area_names,
                    inter_area_scales=inter_area_scales,
                )
                if kernel_mode == "area_expanded" and kernels_for_save is None:
                    _append_job_output(
                        job_status,
                        job_id,
                        f"Expanded kernel dictionary from {len(kernels)} to {len(kernels_for_convolution)} entries.",
                    )
                if kernel_mode == "area_expanded":
                    spike_times_input = _flatten_area_spike_times_for_kernels(
                        spike_times_input,
                        area_names,
                        area_populations,
                    )

            required_populations = _extract_kernel_presynaptic_populations(kernels_for_convolution)
            if (
                area_names
                and area_populations
                and isinstance(spike_times_input, dict)
                and any(_parse_population_area_label(pop)[1] is not None for pop in required_populations)
                and not all(pop in spike_times_input for pop in required_populations)
            ):
                spike_times_input = _flatten_area_spike_times_for_kernels(
                    spike_times_input,
                    area_names,
                    area_populations,
                )
            spike_times, spike_times_mode = _prepare_spike_times_for_kernel_convolution(
                spike_times_input,
                required_populations,
            )
            if spike_times_mode == "area_aggregated" and not logged_area_aggregated:
                _append_job_output(
                    job_status,
                    job_id,
                    "Detected area-wise spike times; aggregated spikes across areas by population for kernel convolution.",
                )
                logged_area_aggregated = True

            population_sizes = _pick_trial_item(population_sizes_raw, trial_idx)
            if population_sizes is not None:
                if area_names and area_populations:
                    population_sizes, pop_sizes_layout = _flatten_area_population_sizes_for_kernels(
                        population_sizes,
                        area_names,
                        area_populations,
                    )
                    if pop_sizes_layout == "area_flattened" and not logged_population_flattened:
                        _append_job_output(
                            job_status,
                            job_id,
                            "Detected area-wise population sizes; mapped them to area/population kernel keys.",
                        )
                        logged_population_flattened = True
                population_sizes, pop_sizes_mode = _prepare_population_sizes_for_kernel_convolution(
                    population_sizes,
                    required_populations,
                )
                if pop_sizes_mode == "area_aggregated" and not logged_population_aggregated:
                    _append_job_output(
                        job_status,
                        job_id,
                        "Detected area-wise population sizes; aggregated sizes across areas by population.",
                    )
                    logged_population_aggregated = True

            if kernels_for_save is None:
                kernels_for_save = kernels_for_convolution

            for probe_name in probe_names:
                cdm_signals = potential.compute_cdm_lfp_from_kernels(
                    kernels_for_convolution,
                    spike_times,
                    cdm_dt,
                    cdm_tstop,
                    population_sizes=population_sizes,
                    transient=transient,
                    probe=probe_name,
                    component=component,
                    mode=mode,
                    scale=scale,
                )
                if cdm_decimation_factor > 1:
                    cdm_signals = _decimate_signal_time(cdm_signals, cdm_decimation_factor)
                if not logged_component_shape and cdm_signals:
                    sample_key = next(iter(cdm_signals.keys()))
                    sample_arr = np.asarray(cdm_signals[sample_key])
                    _append_job_output(
                        job_status,
                        job_id,
                        f"Sample convolved signal shape for '{sample_key}': {tuple(sample_arr.shape)}.",
                    )
                    logged_component_shape = True
                area_sum_payload = {}
                if area_names:
                    area_sum_payload = _sum_signal_dict_by_post_area(cdm_signals, area_names)
                    if area_sum_payload and not all(
                        str(area).strip() in area_sum_payload for area in area_names
                    ):
                        area_sum_payload = {}
                    if area_sum_payload and not logged_target_area_sum:
                        _append_job_output(
                            job_status,
                            job_id,
                            "Summed CDM/LFP signals by target area for four-area outputs.",
                        )
                        logged_target_area_sum = True
                row = {
                    "sum": area_sum_payload if area_sum_payload else _sum_signal_dict(cdm_signals),
                    "raw_signals": cdm_signals,
                    "probe": probe_name,
                    "dt_ms": float(cdm_dt),
                    "decimation_factor": cdm_decimation_factor,
                    "fs_hz": fs_hz_after_decimation,
                    "metadata": {
                        "dt_ms": float(cdm_dt),
                        "decimation_factor": cdm_decimation_factor,
                        "fs_hz": fs_hz_after_decimation,
                    },
                }
                if trial_count > 1:
                    row["trial_index"] = int(trial_idx)
                payload_df = pd.DataFrame([row])
                probe_trial_payloads[probe_name].append(payload_df)
            _set_sample_progress(trial_idx + 1)
            if trial_count > 1:
                _append_job_output(job_status, job_id, f"Computed kernel convolution trial {trial_idx + 1}/{trial_count}.")

        clipped_reported = False
        for probe_name in probe_names:
            clipped_payloads, clipped_length = _clip_trial_dataframe_payloads(
                probe_trial_payloads[probe_name],
                signal_columns=("sum", "raw_signals"),
            )
            probe_trial_payloads[probe_name] = clipped_payloads
            if clipped_length is not None and not clipped_reported:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Clipped kernel field-potential time-series to common minimum length {clipped_length} samples across trials.",
                )
                clipped_reported = True

        kernels_path = os.path.join(output_root, "kernels.pkl")
        kernel_payload = kernels_for_save if kernels_for_save is not None else kernels
        with open(kernels_path, "wb") as f:
            pickle.dump([kernel_payload for _ in range(trial_count)], f)
        _append_job_output(job_status, job_id, f"Saved kernels to {kernels_path}")

        def _row_dict_from_trial_payload(payload):
            if isinstance(payload, pd.DataFrame):
                if payload.empty:
                    return {}
                return dict(payload.iloc[0].to_dict())
            if isinstance(payload, MappingABC):
                return dict(payload)
            return {"sum": payload}

        cdm_path = os.path.join(output_root, "cdm.pkl")
        cdm_trial_payloads = []
        preferred_probe_order = (
            "CurrentDipoleMoment",
            "KernelApproxCurrentDipoleMoment",
            "GaussCylinderPotential",
        )
        for trial_idx in range(trial_count):
            probe_outputs = {}
            for probe_name in probe_names:
                trial_payload = _pick_trial_item(probe_trial_payloads.get(probe_name, []), trial_idx)
                row_payload = _row_dict_from_trial_payload(trial_payload)
                if row_payload:
                    probe_outputs[probe_name] = row_payload

            if not probe_outputs:
                cdm_trial_payloads.append({})
                continue

            selected_probe = None
            for probe_name in preferred_probe_order:
                if probe_name in probe_outputs:
                    selected_probe = probe_name
                    break
            if selected_probe is None:
                selected_probe = next(iter(probe_outputs.keys()))

            selected_payload = dict(probe_outputs[selected_probe])
            selected_payload["probe"] = selected_probe
            remaining_probe_outputs = {
                probe_name: payload
                for probe_name, payload in probe_outputs.items()
                if probe_name != selected_probe
            }
            if remaining_probe_outputs:
                selected_payload["probe_outputs"] = remaining_probe_outputs
            selected_payload["probe_order"] = [selected_probe] + list(remaining_probe_outputs.keys())
            if trial_count > 1:
                selected_payload.setdefault("trial_index", int(trial_idx))
            cdm_trial_payloads.append(selected_payload)

        with open(cdm_path, "wb") as f:
            pickle.dump(cdm_trial_payloads, f)
        _append_job_output(job_status, job_id, f"Saved unified CDM/LFP outputs to {cdm_path}")
        results_path = cdm_path

        _announce_saved_output_folders(job_status, job_id, "field_potential_kernel", output_root, results_path)
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": results_path,
                "error": False
            })
    except Exception as e:
        _mark_job_failed(job_status, job_id, e)
    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_meeg_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting M/EEG computation.")
        if job_id in job_status:
            job_status[job_id]["progress"] = 0
        file_paths = params.get("file_paths", {})

        cdm_path = file_paths.get("meeg_cdm_file")
        if cdm_path and os.path.exists(cdm_path):
            cdm_name = os.path.basename(cdm_path).lower()
            if not (cdm_name == "cdm.pkl" or cdm_name.endswith("_cdm.pkl")):
                raise ValueError("M/EEG input must be cdm.pkl generated by the Field Potential kernel module.")
        if not cdm_path or not os.path.exists(cdm_path):
            preferred = []
            for kernel_root in _field_potential_kernel_search_roots():
                direct_cdm = os.path.join(kernel_root, "cdm.pkl")
                if os.path.isfile(direct_cdm):
                    preferred.append(direct_cdm)
                preferred.extend(glob.glob(os.path.join(kernel_root, "*", "cdm.pkl")))
            if preferred:
                cdm_path = max(preferred, key=os.path.getmtime)
        if not cdm_path or not os.path.exists(cdm_path):
            raise FileNotFoundError("CDM input is required (upload .pkl or compute kernels first).")
        CDM_obj = read_file(cdm_path)
        cdm_trials_raw = list(CDM_obj) if _is_trial_sequence(CDM_obj) else [CDM_obj]
        trial_count = max(1, len(cdm_trials_raw))
        if trial_count > 1:
            _append_job_output(job_status, job_id, f"Detected {trial_count} CDM trial(s); computing M/EEG for each trial.")
        if job_id in job_status:
            job_status[job_id]["simulation_total"] = int(trial_count)
            job_status[job_id]["simulation_completed"] = 0
            job_status[job_id]["meeg_trial_total"] = int(trial_count)
            job_status[job_id]["meeg_trial_index"] = 0
            job_status[job_id]["meeg_sensors_total"] = 0
            job_status[job_id]["meeg_sensors_completed"] = 0

        def _parse_xyz_locations(raw_value, label):
            parsed = _parse_literal_value(raw_value, None)
            if parsed is None or parsed == "":
                return None
            arr = np.asarray(parsed, dtype=float)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                if arr.shape[0] != 3:
                    raise ValueError(f"{label} must have shape (3,) or (n, 3).")
                arr = arr.reshape(1, 3)
            elif arr.ndim == 2:
                if arr.shape[1] != 3:
                    raise ValueError(f"{label} must have shape (n, 3).")
            else:
                raise ValueError(f"{label} must have shape (3,) or (n, 3).")
            return np.array(arr, dtype=float, copy=True)

        dipole_locations = _parse_xyz_locations(params.get("meeg_dipole_locations"), "meeg_dipole_locations")
        sensor_locations = _parse_xyz_locations(params.get("meeg_sensor_locations"), "meeg_sensor_locations")

        model = params.get("meeg_model") or "NYHeadModel"
        model_kwargs = _parse_literal_value(params.get("meeg_model_kwargs"), None)
        requested_forward_mode = str(params.get("meeg_forward_mode") or "simultaneous_all_dipoles").strip().lower()
        if requested_forward_mode not in {"simultaneous_all_dipoles", "per_sensor_independent"}:
            requested_forward_mode = "simultaneous_all_dipoles"
        align_to_surface = True
        simulation_model_hint = str(params.get("meeg_simulation_model") or "").strip().lower()
        is_four_area_simulation = simulation_model_hint == "four_area"
        required_four_area_order = ("frontal", "parietal", "temporal", "occipital")
        if is_four_area_simulation and model == "NYHeadModel":
            raise ValueError("NYHeadModel is not supported for the four-area simulation model.")
        force_per_sensor_mode = model == "NYHeadModel"
        use_per_sensor_independent = force_per_sensor_mode or (requested_forward_mode == "per_sensor_independent")

        _append_job_output(job_status, job_id, f"Model: {model}")
        if force_per_sensor_mode:
            _append_job_output(
                job_status,
                job_id,
                "NYHeadModel uses fixed per-sensor independent forward simulation mode.",
            )
        else:
            _append_job_output(
                job_status,
                job_id,
                "Forward simulation mode: "
                + ("per-sensor independent" if use_per_sensor_independent else "simultaneous all dipoles"),
            )
        potential = ncpi.FieldPotential()
        model_kwargs = model_kwargs or {}
        is_meg = model in {"InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG"}
        trial_payloads = []
        warned_cdm_1d = False
        warned_cdm_2d = False
        warned_cdm_replicated = False

        def _coerce_area_cdm_entry(value, area_name):
            arr = np.asarray(value)
            if arr.ndim == 1:
                return np.array(arr, copy=True)
            if arr.ndim == 2:
                if arr.shape[0] == 3:
                    return np.array(arr, copy=True)
                if arr.shape[1] == 3:
                    return np.array(arr.T, copy=True)
                if arr.shape[0] == 1:
                    return np.array(arr[0], copy=True)
                if arr.shape[1] == 1:
                    return np.array(arr[:, 0], copy=True)
            raise ValueError(
                f"Four-area CDM entry '{area_name}' must be shaped (n_times,) or (3, n_times)."
            )

        def _extract_four_area_cdm_map(value):
            if not isinstance(value, MappingABC):
                return None
            mapped = {}
            for area_name in required_four_area_order:
                for key, entry in value.items():
                    if str(key).strip().lower() != area_name:
                        continue
                    mapped[area_name] = _coerce_area_cdm_entry(entry, area_name)
                    break
            if len(mapped) == len(required_four_area_order):
                return mapped
            return None

        def _replicate_single_cdm_for_dipoles(cdm_value, n_dipoles):
            arr = np.asarray(cdm_value)
            if n_dipoles <= 1:
                return arr
            if arr.ndim == 1:
                return np.repeat(arr[np.newaxis, :], n_dipoles, axis=0)
            if arr.ndim == 2:
                if arr.shape[0] == 3 or arr.shape[1] == 3:
                    base = arr if arr.shape[0] == 3 else arr.T
                    return np.repeat(base[np.newaxis, :, :], n_dipoles, axis=0)
                seed = arr[:1, :] if arr.shape[0] > 0 else np.zeros((1, arr.shape[-1]))
                return np.repeat(seed, n_dipoles, axis=0)
            if arr.ndim == 3:
                seed = arr[:1, :, :] if arr.shape[0] > 0 else np.zeros((1, 3, 1))
                return np.repeat(seed, n_dipoles, axis=0)
            return arr

        def _dipole_below_sensor(sensor_xyz, model_key, brain_radius=None):
            sensor = np.asarray(sensor_xyz, dtype=float)
            norm = float(np.linalg.norm(sensor))
            if model_key == "FourSphereVolumeConductor":
                r1 = _safe_float(brain_radius)
                if r1 is None or r1 <= 0:
                    r1 = 79000.0
                if norm <= 0.0:
                    return np.array([0.0, 0.0, 0.98 * r1], dtype=float)
                # Strictly inside the brain sphere: |r_dipole| < r1
                target_norm = min(norm * 0.98, r1 * 0.98)
                return sensor * (target_norm / norm)
            if model_key == "SphericallySymmetricVolCondMEG":
                # Keep dipole strictly inside the sensor radius.
                if norm > 0.0:
                    return sensor * 0.98
                return np.array([0.0, 0.0, 1000.0], dtype=float)
            if norm > 0.0:
                return sensor + np.array([0.0, 0.0, -2000.0], dtype=float)
            return np.array([0.0, 0.0, -2000.0], dtype=float)

        for trial_idx, cdm_trial_raw in enumerate(cdm_trials_raw):
            CDM, cdm_meta = _extract_signal_and_meta_from_source(cdm_trial_raw)
            trial_four_area_map = _extract_four_area_cdm_map(CDM)
            if trial_four_area_map is not None:
                _append_job_output(
                    job_status,
                    job_id,
                    "Detected four-area CDM dictionary; mapping frontal/parietal/temporal/occipital to dipoles.",
                )
                CDM = np.stack([trial_four_area_map[area_name] for area_name in required_four_area_order], axis=0)
            elif isinstance(CDM, dict):
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

            trial_dipole_locations = np.array(dipole_locations, dtype=float, copy=True) if dipole_locations is not None else None
            trial_sensor_locations = np.array(sensor_locations, dtype=float, copy=True) if sensor_locations is not None else None

            if is_four_area_simulation:
                if trial_dipole_locations is None:
                    raise ValueError(
                        "Four-area model requires dipole locations for frontal, parietal, temporal, occipital."
                    )
                if trial_dipole_locations.shape != (4, 3):
                    raise ValueError(
                        "Four-area model requires exactly 4 dipole locations with shape (4, 3), "
                        "ordered as frontal, parietal, temporal, occipital."
                    )
                cdm_arr = np.asarray(CDM)
                if trial_four_area_map is None and not (cdm_arr.ndim == 3 and cdm_arr.shape[0] == 4):
                    raise ValueError(
                        "Four-area M/EEG requires area-resolved CDM with 4 entries (frontal, parietal, temporal, occipital)."
                    )

            CDM = np.asarray(CDM)
            if CDM.ndim == 2 and CDM.shape[0] != 3 and CDM.shape[1] == 3:
                CDM = CDM.T
            if CDM.ndim == 3 and CDM.shape[1] != 3 and CDM.shape[2] == 3:
                CDM = np.transpose(CDM, (0, 2, 1))
            if CDM.ndim == 1:
                if not warned_cdm_1d:
                    _append_job_output(job_status, job_id, "CDM is 1D; assuming z-axis dipole (x=y=0).")
                    warned_cdm_1d = True
                CDM = np.vstack([np.zeros_like(CDM), np.zeros_like(CDM), CDM])
            elif CDM.ndim == 2 and CDM.shape[0] != 3:
                if not warned_cdm_2d:
                    _append_job_output(job_status, job_id, "CDM has 1 component per dipole; assuming z-axis dipoles (x=y=0).")
                    warned_cdm_2d = True
                zeros = np.zeros_like(CDM)
                CDM = np.stack([zeros, zeros, CDM], axis=1)
            if not ((CDM.ndim == 2 and CDM.shape[0] == 3) or (CDM.ndim == 3 and CDM.shape[1] == 3)):
                raise ValueError(
                    "CDM must have 3 components. Ensure you computed a dipole-moment probe (e.g., "
                    "KernelApproxCurrentDipoleMoment or CurrentDipoleMoment), not a scalar potential."
                )

            if trial_dipole_locations is not None and trial_dipole_locations.ndim == 1:
                trial_dipole_locations = trial_dipole_locations.reshape(1, 3)
            n_dipoles = int(trial_dipole_locations.shape[0]) if trial_dipole_locations is not None else 1
            if not is_four_area_simulation and n_dipoles > 1:
                expanded_cdm = _replicate_single_cdm_for_dipoles(CDM, n_dipoles)
                if not warned_cdm_replicated:
                    _append_job_output(
                        job_status,
                        job_id,
                        "Replicating the same CDM across all configured dipole locations for M/EEG computation.",
                    )
                    warned_cdm_replicated = True
                CDM = expanded_cdm

            if model == "NYHeadModel":
                trial_sensor_locations = None
                ny_dipoles, _ = potential._get_eeg_1020_locations()
                ny_dipoles = np.asarray(ny_dipoles, dtype=float)
                if ny_dipoles.ndim != 2 or ny_dipoles.shape[1] != 3:
                    raise ValueError("NYHeadModel auto dipole locations are invalid.")
                n_dip = 1 if CDM.ndim == 2 else int(CDM.shape[0])
                if n_dip <= 1:
                    trial_dipole_locations = np.array(ny_dipoles[0], dtype=float, copy=True)
                else:
                    if ny_dipoles.shape[0] < n_dip:
                        needed = n_dip - ny_dipoles.shape[0]
                        tail = np.repeat(ny_dipoles[-1:].copy(), needed, axis=0)
                        ny_dipoles = np.vstack([ny_dipoles, tail])
                    trial_dipole_locations = np.array(ny_dipoles[:n_dip], dtype=float, copy=True)
            else:
                if trial_sensor_locations is None:
                    if model in {"FourSphereVolumeConductor", "InfiniteVolumeConductor"}:
                        trial_sensor_locations = np.array([[0.0, 0.0, 90000.0]])
                    elif model == "InfiniteHomogeneousVolCondMEG":
                        trial_sensor_locations = np.array([[10000.0, 0.0, 0.0]])
                    elif model == "SphericallySymmetricVolCondMEG":
                        trial_sensor_locations = np.array([[0.0, 0.0, 92000.0]])

            if trial_dipole_locations is None:
                if model == "FourSphereVolumeConductor":
                    default_loc = np.array([0.0, 0.0, 78000.0], dtype=float)
                elif model == "SphericallySymmetricVolCondMEG":
                    default_loc = np.array([0.0, 0.0, 90000.0], dtype=float)
                else:
                    default_loc = np.array([0.0, 0.0, 0.0], dtype=float)
                if CDM.ndim == 3 and CDM.shape[0] > 1:
                    trial_dipole_locations = np.repeat(default_loc.reshape(1, 3), int(CDM.shape[0]), axis=0)
                else:
                    trial_dipole_locations = default_loc

            p_list, loc_list = potential._normalize_cdm_and_locations(CDM, trial_dipole_locations)

            n_sensors = 0
            p_use_list = []
            matrices = []
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
                trial_sensor_locations = np.asarray(trial_sensor_locations, dtype=float)
                if trial_sensor_locations.ndim == 1:
                    if trial_sensor_locations.shape[0] != 3:
                        raise ValueError("sensor_locations must have shape (n_sensors, 3).")
                    trial_sensor_locations = trial_sensor_locations.reshape(1, 3)
                if trial_sensor_locations.ndim != 2 or trial_sensor_locations.shape[1] != 3:
                    raise ValueError("sensor_locations must have shape (n_sensors, 3).")

                if model == "FourSphereVolumeConductor":
                    FourSphere = potential._load_eegmegcalc_model("FourSphereVolumeConductor")
                    model_obj = FourSphere(trial_sensor_locations, **model_kwargs)
                    four_sphere_r1 = _safe_float(getattr(model_obj, "r1", None))
                    def get_M(loc):
                        return model_obj.get_transformation_matrix(loc)
                elif model == "InfiniteVolumeConductor":
                    InfiniteVol = potential._load_eegmegcalc_model("InfiniteVolumeConductor")
                    model_obj = InfiniteVol(**model_kwargs)
                    def get_M(loc):
                        r = trial_sensor_locations - loc
                        return model_obj.get_transformation_matrix(r)
                elif model == "InfiniteHomogeneousVolCondMEG":
                    IHVCMEG = potential._load_eegmegcalc_model("InfiniteHomogeneousVolCondMEG")
                    model_obj = IHVCMEG(trial_sensor_locations, **model_kwargs)
                    def get_M(loc):
                        return model_obj.get_transformation_matrix(loc)
                elif model == "SphericallySymmetricVolCondMEG":
                    SSVMEG = potential._load_eegmegcalc_model("SphericallySymmetricVolCondMEG")
                    model_obj = SSVMEG(trial_sensor_locations, **model_kwargs)
                    def get_M(loc):
                        return model_obj.get_transformation_matrix(loc)
                else:
                    raise ValueError(f"Unknown model '{model}'.")
                n_sensors = int(trial_sensor_locations.shape[0])
                if use_per_sensor_independent:
                    if len(p_list) == 0:
                        raise ValueError("No CDM signal available for M/EEG computation.")
                    if len(p_list) == 1:
                        p_reference = p_list[0]
                    else:
                        _append_job_output(
                            job_status,
                            job_id,
                            f"Collapsing {len(p_list)} dipole CDM entries into one equivalent CDM by summing components for per-sensor simulation.",
                        )
                        p_reference = np.sum(np.stack(p_list, axis=0), axis=0)
                    p_use_list = [p_reference]
                    _append_job_output(
                        job_status,
                        job_id,
                        "Using per-sensor independent forward simulation: for each sensor, the same CDM is evaluated at a dipole location below that sensor.",
                    )
                else:
                    for p_i, loc_i in zip(p_list, loc_list):
                        matrices.append(get_M(loc_i))
                        p_use_list.append(p_i)
                    _append_job_output(
                        job_status,
                        job_id,
                        "Using simultaneous multi-dipole forward simulation: all dipoles are simulated together across all sensors.",
                    )

            if n_sensors <= 0:
                raise ValueError("Unable to determine number of sensors/electrodes.")

            n_times = p_use_list[0].shape[1]
            _append_job_output(
                job_status,
                job_id,
                f"M/EEG workload (trial {trial_idx + 1}/{trial_count}): {n_sensors} sensors/electrodes x {n_times} time points.",
            )
            if is_meg:
                meeg = np.zeros((n_sensors, 3, n_times))
            else:
                meeg = np.zeros((n_sensors, n_times))
            if job_id in job_status:
                job_status[job_id]["meeg_trial_index"] = int(trial_idx + 1)
                job_status[job_id]["meeg_sensors_total"] = int(n_sensors)
                job_status[job_id]["meeg_sensors_completed"] = 0

            _append_job_output(
                job_status,
                job_id,
                f"Computing electrodes/sensors for trial {trial_idx + 1}/{trial_count}: 0/{n_sensors}",
            )
            log_every = 1 if n_sensors <= 300 else max(1, n_sensors // 100)
            for idx in range(n_sensors):
                if model == "NYHeadModel":
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
                else:
                    if use_per_sensor_independent:
                        loc_sensor = trial_sensor_locations[idx]
                        loc_dip = _dipole_below_sensor(
                            loc_sensor,
                            model,
                            brain_radius=four_sphere_r1 if model == "FourSphereVolumeConductor" else None,
                        )
                        M_local = get_M(loc_dip)
                        if is_meg:
                            meeg[idx] = M_local[idx] @ p_use_list[0]
                        else:
                            meeg[idx] = M_local[idx] @ p_use_list[0]
                    else:
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
                        f"Computed electrodes/sensors: {idx + 1}/{n_sensors} (trial {trial_idx + 1}/{trial_count})",
                    )
                total_trials = max(1, int(trial_count))
                sensor_fraction = float(idx + 1) / float(max(1, n_sensors))
                global_fraction = (float(trial_idx) + sensor_fraction) / float(total_trials)
                progress = int(global_fraction * 100.0)
                job_status[job_id]["progress"] = max(0, min(100, progress))
                if job_id in job_status:
                    job_status[job_id]["meeg_sensors_completed"] = int(idx + 1)

            meeg_dt_ms = _safe_float(cdm_meta.get("dt_ms"))
            meeg_decimation = int(cdm_meta.get("decimation_factor", 1))
            meeg_fs = _safe_float(cdm_meta.get("fs_hz"))
            row = {
                "data": meeg,
                "dt_ms": meeg_dt_ms,
                "decimation_factor": meeg_decimation,
                "fs_hz": meeg_fs,
                "metadata": {"dt_ms": meeg_dt_ms, "decimation_factor": meeg_decimation, "fs_hz": meeg_fs},
                "source_cdm_file": os.path.basename(cdm_path),
            }
            if trial_count > 1:
                row["trial_index"] = int(trial_idx)
            trial_payloads.append(pd.DataFrame([row]))
            if job_id in job_status:
                job_status[job_id]["simulation_completed"] = int(trial_idx + 1)
                job_status[job_id]["progress"] = int(((trial_idx + 1) / max(1, trial_count)) * 100)
            if trial_count > 1:
                _append_job_output(job_status, job_id, f"Computed M/EEG trial {trial_idx + 1}/{trial_count}.")

        output_root = _field_potential_output_dir("meeg")
        os.makedirs(output_root, exist_ok=True)
        for name in os.listdir(output_root):
            lower = name.lower()
            if not (lower.endswith(".pkl") or lower.endswith(".pickle")):
                continue
            old_path = os.path.join(output_root, name)
            if os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        meeg_path = os.path.join(output_root, "meeg.pkl")
        with open(meeg_path, "wb") as f:
            pickle.dump(trial_payloads, f)

        _append_job_output(job_status, job_id, f"Saved M/EEG to {meeg_path}")
        _announce_saved_output_folders(job_status, job_id, "field_potential_meeg", output_root, meeg_path)
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": meeg_path,
                "error": False
            })
    except Exception as e:
        _mark_job_failed(job_status, job_id, e)
    cleanup_temp_files(params.get('file_paths', {}))
