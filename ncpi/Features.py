import os
import numpy as np
import scipy.signal as scipy_signal
from ncpi import tools
from types import SimpleNamespace
from scipy import stats


_WORKER_FEATURES_OBJ = None

def _features_worker_init(method, params):
    """
    Initializer that runs once per worker process.
    Creates a worker-local Features object to avoid pickling 'self'.

    Parameters
    ----------
    method: str
        Method to compute features (e.g., 'catch22', 'specparam', 'dfa', 'fEI').
    params: dict
        Dictionary containing the parameters for the feature computation.
    """
    global _WORKER_FEATURES_OBJ
    _WORKER_FEATURES_OBJ = Features(method=method, params=params)


def _compute_one_feature(sample):
    """
    Compute features for a single sample using the worker-local Features object.

    Parameters
    ----------
    sample: array-like
        Input time-series data for a single sample.

    Returns
    -------
    features: output of the feature computation method (e.g., array of catch22 features,
    dict of specparam results, etc.)
    """
    global _WORKER_FEATURES_OBJ

    if _WORKER_FEATURES_OBJ is None:
        raise RuntimeError("Worker not initialized. Did you forget initializer?")

    x = np.asarray(sample).squeeze()
    if x.ndim != 1:
        raise ValueError("Each sample must be 1D.")

    # z-score normalization (if a different scheme is needed, compute externally and pass pre-normalized data, then
    # set normalize=False)
    normalize = bool(_WORKER_FEATURES_OBJ.params.get("normalize", False))
    if normalize:
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if not np.isfinite(sigma) or sigma == 0.0:
            # Avoid division-by-zero; return NaNs with a consistent schema
            if _WORKER_FEATURES_OBJ.method == "catch22":
                return [np.nan] * 22

            if _WORKER_FEATURES_OBJ.method == "specparam":
                # Keep consistent output shape/keys
                return {
                    "aperiodic_params": np.array([np.nan, np.nan]),
                    "peak_cf": np.nan,
                    "peak_pw": np.nan,
                    "peak_bw": np.nan,
                    "n_peaks": 0,
                    "selected_peaks": np.empty((0, 3)),
                    "all_peaks": np.empty((0, 3)),
                    "metrics": {"gof_rsquared": np.nan},
                }

            if _WORKER_FEATURES_OBJ.method == "dfa":
                return _WORKER_FEATURES_OBJ._dfa_nan_output(x.shape[0])

            if _WORKER_FEATURES_OBJ.method == "fEI":
                return _WORKER_FEATURES_OBJ._fei_nan_output(x.shape[0])

            raise ValueError(f"Unknown method: {_WORKER_FEATURES_OBJ.method}")

        x = (x - mu) / sigma

    if _WORKER_FEATURES_OBJ.method == "catch22":
        return _WORKER_FEATURES_OBJ.catch22(x)

    if _WORKER_FEATURES_OBJ.method == "specparam":
        return _WORKER_FEATURES_OBJ.specparam(sample=x)

    if _WORKER_FEATURES_OBJ.method == "dfa":
        return _WORKER_FEATURES_OBJ.dfa(sample=x)

    if _WORKER_FEATURES_OBJ.method == "fEI":
        return _WORKER_FEATURES_OBJ.fEI(sample=x)

    raise ValueError(f"Unknown method: {_WORKER_FEATURES_OBJ.method}")


#########################################
## Features Class (public API surface) ##
#########################################

class Features:
    """
    Class for computing features from electrophysiological data recordings.
    """

    def __init__(self, method='catch22', params=None):
        """
        Constructor method.

        Parameters
        ----------
        method: str
            Method to compute features. Default is 'catch22'.
        params: dict
            Dictionary containing the parameters for the feature computation.
        """

        # Assert that the method is a string
        if not isinstance(method, str):
            raise ValueError("The method must be a string.")

        # Check if the method is valid
        if method not in ['catch22', 'specparam', 'dfa', 'fEI', 'hctsa']:
            raise ValueError(
                "Invalid method. Please use 'catch22', 'specparam', 'dfa', 'fEI', or 'hctsa'."
            )

        # Check if params is a dictionary
        if params is not None and not isinstance(params, dict):
            raise ValueError("params must be a dictionary.")

        self.method = method
        # Normalize to a dict so downstream code can always assume a mapping
        self.params = {} if params is None else dict(params)

        # Default normalization (False)
        if "normalize" not in self.params:
            self.params["normalize"] = False

        # Import the required modules based on the method
        if method == 'catch22':
            if not tools.ensure_module("pycatch22"):
                raise ImportError(
                    "pycatch22 is required for computing catch22 features but is not installed."
                )
            # pycatch22 >= 0.6 changes feature values
            installed = tools._distribution_version("pycatch22")
            if installed is not None and not tools._version_satisfies(installed, "<0.6"):
                raise ImportError(
                    f"pycatch22 {installed} is not supported, install pycatch22<0.6."
                )
            self._pycatch22 = tools.dynamic_import("pycatch22")

        elif method == 'specparam':
            if not tools.ensure_module("specparam"):
                raise ImportError(
                    "specparam is required for computing spectral features but is not installed."
                )
            self._specparam = tools.dynamic_import("specparam")
        elif method == 'hctsa':
            matlab_engine = tools.dynamic_import("matlab.engine", raise_on_error=False)
            if matlab_engine is None:
                raise ImportError(
                    "MATLAB Engine for Python is required (module 'matlab.engine'). "
                    "Install it from your MATLAB distribution at "
                    "<MATLAB_ROOT>/extern/engines/python, or via a compatible "
                    "`matlabengine` pip package (e.g., `pip install \"matlabengine==24.2.2\"`) "
                    "depending on your MATLAB distribution (do not use the PyPI 'matlab' stub)."
                )

            if not tools.ensure_module("h5py"):
                raise ImportError("h5py is required for reading HCTSA.mat outputs.")

            self.matlab = tools.dynamic_import("matlab")
            self.matlabengine = matlab_engine
            self.h5py = tools.dynamic_import("h5py")

        # Use stdlib multiprocessing
        self.multiprocessing = tools.dynamic_import("multiprocessing")

        # Check if tqdm is installed
        if not tools.ensure_module("tqdm"):
            self.tqdm_inst = False
        else:
            self.tqdm_inst = True
            self.tqdm = tools.dynamic_import("tqdm", "tqdm")


    ##############################################
    ## Helper functions for feature computation ##
    ##############################################

    def _resolve_param(self, name, value, default=None):
        """
        Resolve a parameter value with priority:
        explicit argument > self.params[name] > default

        Parameters
        ----------
        name: str
            Name of the parameter to resolve.
        value: any
            Explicit value passed to the method (can be None).
        default: any
            Default value to use if neither explicit value nor self.params[name] is provided.

        Returns
        -------
        resolved_value: any
            The resolved parameter value based on the priority.
        """
        if value is not None:
            return value
        if name in self.params:
            return self.params[name]
        return default


    def _ensure_mne(self):
        """ Ensure that the MNE library is available for DFA/fEI computations. If not, raise an ImportError."""
        if not tools.ensure_module("mne"):
            raise ImportError(
                "mne is required for DFA/fEI."
            )
        return tools.dynamic_import("mne")


    def _as_2d(self, x):
        """Ensure the input is a 2D array. If it's 1D, add a new axis.
        Return the array and a flag indicating if it was originally 1D.

        Parameters
        ----------
        x: array-like
            Input data.

        Returns
        -------
        x_2d: np.ndarray
            2D array version of the input.
        was_1d: bool
            True if the original input was 1D, False if it was already 2D.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return x[np.newaxis, :], True
        if x.ndim == 2:
            return x, False
        raise ValueError("`sample` must be a 1D or 2D array.")


    # DFA/fEI computation was developed separately from the original papers.
    # Source module SHA-256 before integration:
    # 8abb7a2bef68e074209fe4c3b8d5897fda65eba595bc927a824789b3735ba259
    # DFA: Hardstone et al. (2012), doi:10.3389/fphys.2012.00450.
    # fEI: Bruining et al. (2020), doi:10.1038/s41598-020-65500-4.
    # Optional generalized ESD: Rosner (1983), doi:10.1080/00401706.1983.10487848.
    # Window rounding, default scales and filter design are implementation choices.

    @staticmethod
    def _dfa_fei_array(data, envelope=True):
        raw = np.asarray(data)
        if np.iscomplexobj(raw):
            raise ValueError("data must be real")
        try:
            x = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("data must be numeric") from exc
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or not x.shape[0] or x.shape[1] < 4:
            raise ValueError("expected (samples,) or (channels, samples), at least four samples")
        if not np.all(np.isfinite(x)) or (envelope and np.any(x < 0)):
            raise ValueError("data must be finite" + (" and nonnegative" if envelope else ""))
        if np.any(np.ptp(x, axis=1) == 0):
            raise ValueError("constant channels are not supported")
        return x

    @staticmethod
    def _dfa_fei_positive(value, name):
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return v

    @staticmethod
    def _dfa_fei_size(seconds, fs, n):
        value = Features._dfa_fei_positive(seconds, "window size") * fs
        if not np.isfinite(value) or value > n + 1:
            raise ValueError("window exceeds signal length")
        size = int(np.rint(value))
        if size < 4 or size > n:
            raise ValueError("window must contain 4 to signal length samples")
        return size

    @staticmethod
    def _dfa_fei_starts(n, size, overlap, minimum):
        try:
            overlap = float(overlap)
        except (ValueError, TypeError) as exc:
            raise ValueError("overlap must be in [0, 1)") from exc
        if not np.isfinite(overlap) or not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        step = max(1, int(np.rint(size * (1 - overlap))))
        starts = np.arange(0, n - size + 1, step)
        if len(starts) < minimum:
            raise ValueError(f"at least {minimum} complete windows are required")
        return starts

    @staticmethod
    def _dfa_fei_profile(x):
        return np.cumsum(x - x.mean(axis=-1, keepdims=True), axis=-1)

    @staticmethod
    def _dfa_fei_residual_std(windows):
        t = np.arange(windows.shape[-1], dtype=float)
        t -= t.mean()
        centered = windows - windows.mean(axis=-1, keepdims=True)
        slope = np.sum(centered * t, axis=-1, keepdims=True) / np.dot(t, t)
        residual = centered - slope * t
        return np.sqrt(np.mean(residual ** 2, axis=-1))

    @staticmethod
    def _dfa_fei_dfa(envelope, fs, window_seconds, fit_interval, overlap=0.5):
        x = Features._dfa_fei_array(envelope)
        fs = Features._dfa_fei_positive(fs, "fs")
        try:
            requested = np.asarray(window_seconds, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("window_seconds must be a sequence") from exc
        if requested.ndim != 1 or len(requested) < 2:
            raise ValueError("at least two window sizes required")
        sizes = np.array([Features._dfa_fei_size(v, fs, x.shape[1]) for v in requested])
        if np.any(np.diff(sizes) <= 0):
            raise ValueError("window sizes must be strictly increasing after rounding")
        try:
            bounds = np.asarray(fit_interval, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("fit_interval must contain two bounds") from exc
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0 or bounds[0] >= bounds[1]:
            raise ValueError("fit_interval must contain increasing positive bounds")
        seconds = sizes / fs
        mask = (seconds >= bounds[0]) & (seconds <= bounds[1])
        if mask.sum() < 2:
            raise ValueError("fit interval must include at least two window sizes")
        profile = Features._dfa_fei_profile(x)
        fluct = np.empty((len(x), len(sizes)))
        counts = []
        for j, size in enumerate(sizes):
            starts = Features._dfa_fei_starts(x.shape[1], size, overlap, 2)
            counts.append(len(starts))
            windows = profile[:, starts[:, None] + np.arange(size)]
            fluct[:, j] = Features._dfa_fei_residual_std(windows).mean(axis=1)
        exponent = np.full(len(x), np.nan)
        intercept = exponent.copy()
        fitted = np.full_like(fluct, np.nan)
        for c in range(len(x)):
            if np.all(fluct[c, mask] > 0):
                exponent[c], intercept[c] = np.polyfit(np.log10(seconds[mask]), np.log10(fluct[c, mask]), 1)
                fitted[c] = 10 ** (intercept[c] + exponent[c] * np.log10(seconds))
        return SimpleNamespace(window_samples=sizes, window_seconds=seconds, window_counts=np.array(counts), fluctuation=fluct, fit_mask=mask, exponent=exponent, intercept=intercept, fitted=fitted)

    @staticmethod
    def _dfa_fei_esd(values, alpha, maximum):
        indices = np.arange(len(values))
        candidates, passed = [], []
        for _ in range(maximum):
            current = values[indices]
            sd = np.std(current, ddof=1)
            if sd == 0:
                break
            deviations = np.abs(current - np.mean(current))
            k = int(np.argmax(deviations))
            count = len(current)
            statistic = deviations[k] / sd
            quantile = stats.t.ppf(1 - alpha / (2 * count), count - 2)
            critical = (count - 1) * quantile / np.sqrt(count * (count - 2 + quantile ** 2))
            candidates.append(indices[k])
            passed.append(statistic > critical)
            indices = np.delete(indices, k)
        result = np.zeros(len(values), dtype=bool)
        if any(passed):
            last = np.flatnonzero(passed)[-1] + 1
            result[np.asarray(candidates[:last])] = True
        return result

    @staticmethod
    def _dfa_fei_fei(envelope, fs, dfa_exponent=None, window_seconds=5.0, overlap=0.8,
            threshold=0.6, outliers="none", outlier_alpha=0.05, max_outliers=None):
        x = Features._dfa_fei_array(envelope)
        fs = Features._dfa_fei_positive(fs, "fs")
        size = Features._dfa_fei_size(window_seconds, fs, x.shape[1])
        starts = Features._dfa_fei_starts(x.shape[1], size, overlap, 3)
        if outliers not in ("none", "esd"):
            raise ValueError("outliers must be 'none' or 'esd'")
        if outliers == "esd":
            if not isinstance(max_outliers, (int, np.integer)) or isinstance(max_outliers, (bool, np.bool_)) or not 1 <= max_outliers <= len(starts) - 3:
                raise ValueError("max_outliers must leave at least three windows")
            if not np.isfinite(outlier_alpha) or not 0 < outlier_alpha < 1:
                raise ValueError("outlier_alpha must be in (0, 1)")
        if threshold is None:
            eligible = np.ones(len(x), dtype=bool)
        else:
            if not np.isscalar(threshold) or not np.isfinite(threshold):
                raise ValueError("threshold must be finite or None")
            if dfa_exponent is None:
                raise ValueError("dfa_exponent is required when threshold is enabled")
            exponents = np.asarray(dfa_exponent, dtype=float)
            if exponents.ndim == 0 and len(x) == 1:
                exponents = exponents.reshape(1)
            if exponents.shape != (len(x),) or np.any(np.isinf(exponents)):
                raise ValueError("dfa_exponent must have one value per channel; NaN is allowed")
            eligible = exponents > threshold
        indices = starts[:, None] + np.arange(size)
        amp = x[:, indices].mean(axis=-1)
        profile_windows = Features._dfa_fei_profile(x)[:, indices]
        invalid = amp == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = Features._dfa_fei_residual_std(profile_windows / amp[..., None])
        invalid = invalid | ~np.isfinite(norm)
        ao = np.zeros_like(invalid)
        fo = np.zeros_like(invalid)
        corr = np.full(len(x), np.nan)
        for c in range(len(x)):
            valid_indices = np.flatnonzero(~invalid[c] & np.isfinite(norm[c]))
            if outliers == "esd" and len(valid_indices) >= max_outliers + 3:
                ao[c, valid_indices] = Features._dfa_fei_esd(amp[c, valid_indices], outlier_alpha, max_outliers)
                fo[c, valid_indices] = Features._dfa_fei_esd(norm[c, valid_indices], outlier_alpha, max_outliers)
            elif outliers == "esd" and len(valid_indices) < max_outliers + 3:
                raise ValueError("too few valid windows for max_outliers")
            keep = ~(invalid[c] | ao[c] | fo[c]) & np.isfinite(norm[c])
            if eligible[c] and keep.sum() >= 3 and np.ptp(amp[c, keep]) > 0 and np.ptp(norm[c, keep]) > 0:
                corr[c] = np.clip(np.corrcoef(amp[c, keep], norm[c, keep])[0, 1], -1, 1)
        return SimpleNamespace(window_samples=size, starts=starts, amplitude=amp, normalized_fluctuation=norm, invalid=invalid, amplitude_outlier=ao, fluctuation_outlier=fo, excluded=invalid | ao | fo, correlation=corr, fei=1 - corr, criterion_passed=eligible)

    @staticmethod
    def _dfa_fei_amplitude_envelope(data, fs, band, order=4):
        x = Features._dfa_fei_array(data, envelope=False)
        fs = Features._dfa_fei_positive(fs, "fs")
        try:
            band = np.asarray(band, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("band must contain two frequencies") from exc
        if band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] < fs / 2:
            raise ValueError("band must satisfy 0 < low < high < Nyquist")
        if not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 1:
            raise ValueError("order must be a positive integer")
        sos = scipy_signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
        try:
            filtered = scipy_signal.sosfiltfilt(sos, x, axis=-1)
        except ValueError as exc:
            raise ValueError("signal is too short for filter padding") from exc
        return np.abs(scipy_signal.hilbert(filtered, axis=-1))

    def _compute_band_envelope(self, signal_matrix, sampling_frequency, frequency_range,
                               *, filter_kwargs=None, trim_seconds=1.0, hilbert_n_fft=None):
        if filter_kwargs or hilbert_n_fft is not None:
            raise ValueError("MNE filter_kwargs and hilbert_n_fft are unsupported by the independent Butterworth filter; prepare the envelope separately if needed.")
        envelope = self._dfa_fei_amplitude_envelope(signal_matrix, sampling_frequency, frequency_range)
        trim_seconds = float(trim_seconds)
        if not np.isfinite(trim_seconds) or trim_seconds < 0:
            raise ValueError("trim_seconds must be finite and non-negative")
        trim = int(trim_seconds * sampling_frequency)
        if trim:
            if envelope.shape[1] <= 2 * trim:
                raise ValueError("Signal is too short after trimming for filtering artifacts.")
            envelope = envelope[:, trim:-trim]
        return envelope


    def _resolve_sampling_frequency(self, sampling_frequency=None):
        """Resolve the sampling frequency from the explicit argument or self.params.

        Parameters
        ----------
        sampling_frequency: float or None
            Explicit sampling frequency passed to the method. If None, will attempt to resolve from self.params.

        Returns
        -------
        sampling_frequency: float or None
            The resolved sampling frequency, or None if it cannot be resolved.
        """
        sampling_frequency = self._resolve_param("sampling_frequency", sampling_frequency)
        if sampling_frequency is None:
            sampling_frequency = self._resolve_param("fs", None)
        return sampling_frequency


    def _dfa_window_sizes(self, sampling_frequency, compute_interval):
        """Generate up to 20 log-spaced integer scales within the explicit interval."""
        fs = float(sampling_frequency)
        bounds = np.asarray(compute_interval, dtype=float)
        if not np.isfinite(fs) or fs <= 0 or bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("compute_interval must have two positive increasing bounds and sampling_frequency must be positive")
        sizes = np.unique(np.rint(np.geomspace(bounds[0], bounds[1], 20) * fs).astype(int))
        return sizes[(sizes >= 4) & (sizes / fs >= bounds[0]) & (sizes / fs <= bounds[1])]


    def _dfa_nan_output(self, length_signal):
        """Generate a consistent NaN output structure for DFA features when computation cannot be performed.

        Parameters
        ----------
        length_signal: int
            Length of the input signal (in samples), used to determine the shape of window_sizes and fluctuations arrays.

        Returns
        ----------
        output: dict
            A dictionary containing NaN values for DFA features and appropriately shaped empty arrays for
            window_sizes and fluctuations.
        """
        sampling_frequency = self._resolve_sampling_frequency(None)
        compute_interval = self._resolve_param("compute_interval", None, None)
        window_sizes = np.array([], dtype=int)
        fluctuations = np.array([], dtype=float)

        if sampling_frequency is not None and compute_interval is not None:
            try:
                window_sizes = self._dfa_window_sizes(float(sampling_frequency), compute_interval)
                fluctuations = np.full(window_sizes.shape, np.nan, dtype=float)
            except Exception:
                window_sizes = np.array([], dtype=int)
                fluctuations = np.array([], dtype=float)

        return {
            "dfa": np.nan,
            "window_sizes": window_sizes,
            "fluctuations": fluctuations,
            "dfa_intercept": np.nan,
        }


    def _fei_nan_output(self, length_signal):
        """Generate a consistent NaN output structure for fEI features when computation cannot be performed.

        Parameters
        ----------
        length_signal: int
            Length of the input signal (in samples), used to determine the shape of w_amp and w_dnf arrays.

        Returns
        ----------
        output: dict
            A dictionary containing NaN values for fEI features and appropriately shaped empty arrays for w_amp
            and w_dnf.
        """
        sampling_frequency = self._resolve_sampling_frequency(None)
        window_size_sec = self._resolve_param("window_size_sec", None, 5.0)
        window_overlap = self._resolve_param("window_overlap", None, 0.8)

        num_windows = 0
        if sampling_frequency is not None and window_size_sec is not None:
            try:
                window_size = int(np.rint(float(window_size_sec) * float(sampling_frequency)))
                if window_size > 0:
                    window_offset = max(1, int(np.rint(window_size * (1 - float(window_overlap)))))
                    if window_offset > 0:
                        num_windows = len(np.arange(0, length_signal - window_size + 1, window_offset))
            except Exception:
                num_windows = 0

        w_amp = np.full((num_windows,), np.nan, dtype=float)
        w_dnf = np.full((num_windows,), np.nan, dtype=float)

        return {
            "fEI_outliers_removed": np.nan,
            "fEI_val": np.nan,
            "num_outliers": np.nan,
            "wAmp": w_amp,
            "wDNF": w_dnf,
        }


    def _dfa_single(self, sample, sampling_frequency, fit_interval, compute_interval, overlap):
        if sampling_frequency is None or fit_interval is None or compute_interval is None:
            raise ValueError("sampling_frequency, fit_interval and compute_interval are required for DFA")
        bounds = np.asarray(fit_interval, dtype=float)
        compute = np.asarray(compute_interval, dtype=float)
        if bounds.shape != (2,) or compute.shape != (2,) or bounds[0] < compute[0] or bounds[1] > compute[1]:
            raise ValueError("fit_interval must be contained in compute_interval")
        sizes = self._dfa_window_sizes(sampling_frequency, compute_interval)
        result = self._dfa_fei_dfa(sample, sampling_frequency, sizes / float(sampling_frequency), fit_interval, overlap=0.5 if overlap else 0.0)
        return {"dfa": result.exponent[0], "window_sizes": result.window_samples,
                "window_seconds": result.window_seconds, "window_counts": result.window_counts,
                "fit_mask": result.fit_mask, "fluctuations": result.fluctuation[0],
                "dfa_intercept": result.intercept[0], "fitted": result.fitted[0]}


    def _dfa_multichannel(self, signal_matrix, sampling_frequency, fit_interval,
                          compute_interval, overlap, bad_idxes=None):
        """Apply the independent DFA implementation to the selected channels."""
        data, _ = self._as_2d(signal_matrix)
        ignored = set([] if bad_idxes is None else bad_idxes)
        selected = [i for i in range(len(data)) if i not in ignored]
        slopes = np.full(len(data), np.nan)
        intercepts = np.full(len(data), np.nan)
        sizes = self._dfa_window_sizes(sampling_frequency, compute_interval)
        fluctuations = np.full((len(data), len(sizes)), np.nan)
        for channel in selected:
            result = self._dfa_single(data[channel], sampling_frequency, fit_interval,
                                      compute_interval, overlap)
            slopes[channel] = result["dfa"]
            intercepts[channel] = result["dfa_intercept"]
            fluctuations[channel] = result["fluctuations"]
        return {"DFA": slopes, "window_sizes": sizes, "fluctuations": fluctuations,
                "dfa_intercept": intercepts}


    def _fei_single(self, sample, sampling_frequency, window_size_sec, window_overlap,
                    dfa_value, dfa_threshold, dfa_fit_interval=None,
                    dfa_compute_interval=None, dfa_overlap=None):
        if dfa_value is None:
            dfa_value = self._dfa_single(sample, sampling_frequency, dfa_fit_interval,
                                         dfa_compute_interval, True if dfa_overlap is None else dfa_overlap)["dfa"]
        result = self._dfa_fei_fei(sample, sampling_frequency, dfa_exponent=dfa_value,
                                         window_seconds=window_size_sec, overlap=window_overlap,
                                         threshold=dfa_threshold, outliers="none")
        return {"fEI_val": result.fei[0], "fEI_outliers_removed": np.nan,
                "num_outliers": 0, "wAmp": result.amplitude[0],
                "wDNF": result.normalized_fluctuation[0], "dfa": dfa_value,
                "window_starts": result.starts, "invalid": result.invalid[0],
                "excluded": result.excluded[0]}


    def _fei_multichannel(self, signal_matrix, sampling_frequency, window_size_sec,
                          window_overlap, dfa_array, dfa_threshold, bad_idxes=None,
                          dfa_fit_interval=None, dfa_compute_interval=None,
                          dfa_overlap=None):
        """Apply the independent fEI implementation to the selected channels."""
        data, _ = self._as_2d(signal_matrix)
        count, length = data.shape
        size = int(np.rint(float(window_size_sec) * float(sampling_frequency)))
        step = max(1, int(np.rint(size * (1 - float(window_overlap)))))
        windows = len(range(0, max(0, length - size + 1), step))
        values = np.full((count, 1), np.nan)
        removed = np.full((count, 1), np.nan)
        num_outliers = np.full((count, 1), np.nan)
        amplitude = np.full((count, windows), np.nan)
        fluctuation = np.full((count, windows), np.nan)
        ignored = set([] if bad_idxes is None else bad_idxes)
        for channel in (i for i in range(count) if i not in ignored):
            exponent = None if dfa_array is None else float(dfa_array[channel])
            result = self._fei_single(data[channel], sampling_frequency, window_size_sec,
                                      window_overlap, exponent, dfa_threshold,
                                      dfa_fit_interval, dfa_compute_interval, dfa_overlap)
            values[channel, 0] = result["fEI_val"]
            num_outliers[channel, 0] = result["num_outliers"]
            amplitude[channel] = result["wAmp"]
            fluctuation[channel] = result["wDNF"]
        return {"fEI_val": values, "fEI_outliers_removed": removed,
                "num_outliers": num_outliers, "wAmp": amplitude, "wDNF": fluctuation}


    ######################################
    ## Main feature computation methods ##
    ######################################

    def catch22(self, sample):
        """
        Compute catch22 features from a single time-series sample.

        Parameters
        ----------
        sample: np.array
            Sample data.

        Returns
        -------
        features: np.array
            Array with the catch22 feature values.
        """

        result = self._pycatch22.catch22_all(sample)

        if not isinstance(result, dict) or 'values' not in result:
            raise RuntimeError("Unexpected output format from pycatch22.catch22_all")

        return result['values']


    def specparam(
            self,
            sample=None,
            *,
            freqs=None,
            power_spectrum=None,
            fs=None,
            freq_range=None,
            welch_kwargs=None,
            model_kwargs=None,
            select_peak=None,
            debug=None,
            metric_thresholds=None,
            metric_policy=None,
    ):
        """
        Parameterize a power spectrum using specparam.SpectralModel.

        You can pass either:
          - sample (time series) + fs (Welch power spectrum is computed internally), OR
          - freqs + power_spectrum (already-computed PSD)

        Parameter precedence:
          explicit argument > self.params[...] > internal default

        Metric thresholding:
          - metric_thresholds: dict like {"gof_rsquared": 0.9}
          - metric_policy: {"reject", "flag"} (default: "reject")
            * reject -> return NaNs and valid=False
            * flag   -> keep results but set valid=False

        Parameters
        ----------
        sample: array-like
            1D array of time-series data. Mutually exclusive with freqs/power_spectrum.
        freqs: array-like
            1D array of frequency values corresponding to the power_spectrum. Mutually exclusive with sample.
        power_spectrum: array-like
            1D array of power values corresponding to the freqs. Mutually exclusive with sample.
        fs: float
            Sampling frequency of the time-series data (required if sample is passed).
        freq_range: tuple or list
            (fmin, fmax) frequency range (in Hz) to fit the model. Default is (1.0, 45.0).
        welch_kwargs: dict
            Additional keyword arguments to pass to scipy.signal.welch when computing the power spectrum from the
            time series. Default is None (uses internal defaults).
        model_kwargs: dict
            Additional keyword arguments to pass to specparam.SpectralModel. Default is None (uses internal defaults).
        select_peak: str
            Strategy for selecting a single peak to report (in addition to all peaks):
              - "all": do not select, leave selected_peaks empty and report all_peaks
              - "max_pw": select the peak with maximum power
              - "max_cf_in_range": select the peak with maximum center frequency within the freq_range
            Default is "max_pw".
        debug: bool
            If True, print debug information and plots. Default is False.
        metric_thresholds: dict
            Optional dict of metric thresholds for quality control, e.g., {"gof_rsquared": 0.9}.
            If provided, metrics will be compared against these thresholds.
        metric_policy: str
            Policy for handling metric threshold failures. One of:
                - "reject": if any metric fails the threshold, return NaNs and valid=False
                - "flag": if any metric fails the threshold, keep results but set valid=False
                Default is "reject".

        Returns
        -------
        out: dict
            Dictionary containing the following keys:
              - "aperiodic_params": array of aperiodic parameters (e.g., [offset, exponent])
              - "peak_cf": center frequency of the selected peak (or NaN if no peak selected)
              - "peak_pw": power of the selected peak (or NaN if no peak selected)
              - "peak_bw": bandwidth of the selected peak (or NaN if no peak selected)
              - "n_peaks": total number of peaks identified by specparam
              - "selected_peaks": array of shape (num_selected_peaks, 3) with [CF, PW, BW] for each selected peak
              - "all_peaks": array of shape (num_all_peaks, 3) with [CF, PW, BW] for all identified peaks
              - "metrics": dict of fit metrics from specparam results
              - "valid": bool indicating whether the fit passed metric thresholds (if provided)
              - "thresholds": dict of metric thresholds used (only included if metric_thresholds was provided)
              - "threshold_values": dict of actual metric values for the checked thresholds (only included if
              metric_thresholds was provided)
        """

        # -------------------------
        # Resolve params (explicit > self.params > defaults)
        # -------------------------
        fs = self._resolve_param("fs", fs)
        freq_range = self._resolve_param("freq_range", freq_range, (1.0, 45.0))
        welch_kwargs = self._resolve_param("welch_kwargs", welch_kwargs, None)
        model_kwargs = self._resolve_param("model_kwargs", model_kwargs, None)
        select_peak = self._resolve_param("select_peak", select_peak, "max_pw")
        debug = self._resolve_param("debug", debug, False)
        freqs = self._resolve_param("freqs", freqs, None)
        power_spectrum = self._resolve_param("power_spectrum", power_spectrum, None)

        # Correct resolution of metric_thresholds / metric_policy
        metric_thresholds = self._resolve_param("metric_thresholds", metric_thresholds, None)
        metric_policy = self._resolve_param("metric_policy", metric_policy, "reject")

        # Validate metric_policy
        if metric_policy not in {"reject", "flag"}:
            raise ValueError("metric_policy must be one of {'reject', 'flag'}")

        # Validate metric_thresholds
        if metric_thresholds is not None:
            if not isinstance(metric_thresholds, dict):
                raise ValueError("metric_thresholds must be a dict like {'gof_rsquared': 0.9}")
            for k, v in metric_thresholds.items():
                if not isinstance(k, str):
                    raise ValueError("metric_thresholds keys must be metric names (str).")
                try:
                    metric_thresholds[k] = float(v)
                except Exception as e:
                    raise ValueError(f"metric_thresholds['{k}'] must be numeric.") from e

        # -------------------------
        # Validate mutually exclusive inputs
        # -------------------------
        has_ts = sample is not None
        has_freqs = freqs is not None
        has_psd = power_spectrum is not None

        if has_ts:
            if has_freqs or has_psd:
                raise ValueError("Pass either `sample` (+ `fs`) OR (`freqs` and `power_spectrum`), not both.")
        else:
            if not (has_freqs and has_psd):
                missing = []
                if not has_freqs:
                    missing.append("freqs")
                if not has_psd:
                    missing.append("power_spectrum")
                raise ValueError(
                    "When `sample` is not passed, you must pass both `freqs` and `power_spectrum`. "
                    f"Missing: {', '.join(missing)}."
                )

        # -------------------------
        # Get PSD (either compute via Welch or validate provided PSD)
        # -------------------------
        if has_ts:
            if fs is None:
                raise ValueError("`fs` must be provided (argument or self.params['fs']) when `sample` is passed.")

            x = np.asarray(sample).squeeze()
            if x.ndim != 1:
                raise ValueError("`sample` must be a 1D array.")

            wkw = {} if welch_kwargs is None else dict(welch_kwargs)
            nperseg = int(0.5 * float(fs))
            nperseg = max(1, min(nperseg, x.shape[0]))
            wkw.setdefault("nperseg", nperseg)

            freqs, power_spectrum = scipy_signal.welch(x, fs=float(fs), **wkw)
        else:
            freqs = np.asarray(freqs).squeeze()
            power_spectrum = np.asarray(power_spectrum).squeeze()
            if freqs.ndim != 1 or power_spectrum.ndim != 1:
                raise ValueError("`freqs` and `power_spectrum` must be 1D arrays.")
            if freqs.shape[0] != power_spectrum.shape[0]:
                raise ValueError("`freqs` and `power_spectrum` must have the same length.")

        # Drop non-finite and non-positive values (specparam fits log power)
        m = np.isfinite(freqs) & np.isfinite(power_spectrum) & (freqs > 0) & (power_spectrum > 0)
        freqs = freqs[m]
        power_spectrum = power_spectrum[m]
        if freqs.size < 5:
            raise ValueError("Not enough valid frequency points to fit.")

        # Clip freq_range to support
        fmin, fmax = map(float, freq_range)
        if fmin >= fmax:
            raise ValueError("`freq_range` must be (fmin, fmax) with fmin < fmax.")
        fmin = max(fmin, float(np.min(freqs)))
        fmax = min(fmax, float(np.max(freqs)))
        if fmin >= fmax:
            raise ValueError("`freq_range` is outside the available frequency support.")

        # -------------------------
        # Build model settings
        # -------------------------
        base_model_kwargs = dict(self.params.get("specparam_model", {}))
        if model_kwargs is not None:
            base_model_kwargs.update(dict(model_kwargs))

        fm = self._specparam.SpectralModel(**base_model_kwargs)

        # -------------------------
        # Fit
        # -------------------------
        fm.fit(freqs, power_spectrum, [fmin, fmax])

        # -------------------------
        # Extract aperiodic params
        # -------------------------
        try:
            aperiodic_params = np.asarray(fm.results.params.aperiodic.params, dtype=float).squeeze()
        except Exception:
            if debug:
                print("Warning: could not extract aperiodic parameters from specparam results container.")
            aperiodic_params = np.array([np.nan, np.nan], dtype=float)
        aperiodic_params = np.atleast_1d(aperiodic_params).astype(float)

        # -------------------------
        # Extract peaks (rows: [CF, PW, BW])
        # -------------------------
        try:
            peaks = np.asarray(fm.results.params.periodic.params, dtype=float)
        except Exception:
            if debug:
                print("Warning: could not extract peak parameters from specparam results container.")
            peaks = np.empty((0, 3), dtype=float)

        peaks = np.asarray(peaks, dtype=float)
        if peaks.ndim == 1 and peaks.size == 3:
            peaks = peaks.reshape(1, 3)
        elif peaks.size == 0:
            peaks = np.empty((0, 3), dtype=float)

        try:
            n_peaks = int(fm.results.n_peaks)
        except Exception:
            n_peaks = int(peaks.shape[0])

        # -------------------------
        # Select peaks according to strategy
        # -------------------------
        peak_cf = peak_pw = peak_bw = np.nan
        all_peaks = np.empty((0, 3), dtype=float)
        selected_peaks = np.empty((0, 3), dtype=float)

        if n_peaks > 0 and peaks.shape[0] > 0:
            cfs = peaks[:, 0]
            pws = peaks[:, 1]

            if select_peak == "all":
                all_peaks = peaks
            elif select_peak == "max_pw":
                if not np.all(np.isnan(pws)):
                    idx = int(np.nanargmax(pws))
                    selected_peaks = peaks[idx:idx + 1]
            elif select_peak == "max_cf_in_range":
                mask = (cfs >= fmin) & (cfs <= fmax)
                if np.any(mask):
                    idx = int(np.nanargmax(cfs[mask]))
                    selected_peaks = peaks[mask][idx:idx + 1]
            else:
                raise ValueError("select_peak must be one of {'all', 'max_pw', 'max_cf_in_range'}")

        if selected_peaks.shape[0] == 1:
            peak_cf, peak_pw, peak_bw = map(float, selected_peaks[0])

        # -------------------------
        # Fit metrics
        # -------------------------
        metric_values = fm.results.metrics.results

        # Choose which metrics to output:
        # - if user set SpectralModel(metrics=[...]) inside specparam_model, try to mirror those
        requested_metrics = base_model_kwargs.get("metrics", None)
        if requested_metrics:
            metrics_out = {m: float(metric_values.get(m, np.nan)) for m in requested_metrics}
        else:
            metrics_out = {"gof_rsquared": float(metric_values.get("gof_rsquared", np.nan))}

        # -------------------------
        # Metric thresholding (optional)
        # -------------------------
        valid_flag = True
        threshold_checked = {}

        if metric_thresholds:
            for name, th in metric_thresholds.items():
                val = float(metric_values.get(name, np.nan))
                threshold_checked[name] = val
                if (not np.isfinite(val)) or (val < th):
                    valid_flag = False

            if (not valid_flag) and (metric_policy == "reject"):
                out = {
                    "aperiodic_params": np.array([np.nan, np.nan]),
                    "peak_cf": np.nan,
                    "peak_pw": np.nan,
                    "peak_bw": np.nan,
                    "n_peaks": 0,
                    "selected_peaks": np.empty((0, 3)),
                    "all_peaks": np.empty((0, 3)),
                    "metrics": metrics_out,
                    "valid": False,
                    "thresholds": dict(metric_thresholds),
                    "threshold_values": threshold_checked,
                }
                if debug:
                    import pprint
                    pprint.pprint(out, sort_dicts=False)
                return out

        # -------------------------
        # Output
        # -------------------------
        out = {
            "aperiodic_params": aperiodic_params,
            "peak_cf": peak_cf,
            "peak_pw": peak_pw,
            "peak_bw": peak_bw,
            "n_peaks": n_peaks,
            "selected_peaks": selected_peaks,
            "all_peaks": all_peaks,
            "metrics": metrics_out,
            "valid": bool(valid_flag),
        }

        # Include threshold info when used (helps debugging downstream)
        if metric_thresholds:
            out["thresholds"] = dict(metric_thresholds)
            out["threshold_values"] = threshold_checked

        if debug:
            import pprint
            import matplotlib.pyplot as plt

            pprint.pprint(out, sort_dicts=False)

            fig, (ax_fit, ax_txt) = plt.subplots(nrows=1, ncols=2, figsize=(24, 8), constrained_layout=True)

            plt.sca(ax_fit)
            try:
                fm.plot(plot_peaks="shade", peak_kwargs={"color": "green"}, ax=ax_fit)
            except TypeError:
                fm.plot(plot_peaks="shade", peak_kwargs={"color": "green"})
            ax_fit.set_title("Specparam fit")

            ax_txt.axis("off")
            text = pprint.pformat(out, sort_dicts=False, width=60)
            ax_txt.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=16,
                        transform=ax_txt.transAxes)
            ax_txt.set_title("Returned results")

            plt.show()

        return out

    def dfa(
        self,
        sample,
        *,
        sampling_frequency=None,
        fit_interval=None,
        compute_interval=None,
        overlap=None,
        frequency_range=None,
        spectrum_range=None,
        bad_idxes=None,
        input_is_envelope=None,
        filter_kwargs=None,
        trim_seconds=None,
        hilbert_n_fft=None,
    ):
        """
        Compute DFA from a time series sample using the Python implementation.
        Note: ncpi does not provide a C runtime for DFA.
        Based on Hardstone et al. (2012), doi:10.3389/fphys.2012.00450,
        and Bruining et al. (2020), doi:10.1038/s41598-020-65500-4.

        If `frequency_range` is provided, the method will:
          1) band-pass filter the signal
          2) compute amplitude envelope (Hilbert)
          3) compute DFA on the envelope

        Otherwise, the input is treated as an amplitude envelope.

        Parameters
        ----------
        sample: array-like
            Input time series. Can be 1D (single channel) or 2D (n_channels, n_timepoints).
        sampling_frequency: float, optional
            Sampling frequency of the signal (in Hz). Required for band/spectrum workflows.
        fit_interval: list or tuple, optional
            Fitting interval (in seconds) for DFA. Required when computing DFA on an envelope.
        compute_interval: list or tuple, optional
            Compute interval (in seconds) for DFA. Required when computing DFA on an envelope.
        overlap: bool, optional
            Whether to use 50% overlapping windows. Default True.
        frequency_range: list or tuple, optional
            Frequency range [low, high] in Hz for band-pass DFA. Mutually exclusive with spectrum_range.
        spectrum_range: list or tuple, optional
            Automatic spectrum ranges are no longer supported.
        bad_idxes: list, optional
            Channel indices to ignore in multichannel inputs.
        input_is_envelope: bool, optional
            Whether the input is already an amplitude envelope. Only used for metadata.
        filter_kwargs: dict, optional
            Unsupported by the independent filter; prepare an envelope externally if needed.
        trim_seconds: float, optional
            Seconds trimmed from both ends after filtering to reduce edge artifacts.
        hilbert_n_fft: int, optional
            Unsupported by the independent filter.

        Returns
        -------
        out: dict
            For 1D envelope input: keys `dfa`, `window_sizes`, `fluctuations`, `dfa_intercept`.
            For multichannel envelope input: keys `DFA`, `window_sizes`, `fluctuations`, `dfa_intercept`.
            For band input: multichannel envelope output plus `frequency_range`, `fit_interval`, `compute_interval`.
        """
        sampling_frequency = self._resolve_sampling_frequency(sampling_frequency)
        fit_interval = self._resolve_param("fit_interval", fit_interval, None)
        compute_interval = self._resolve_param("compute_interval", compute_interval, None)
        overlap = self._resolve_param("overlap", overlap, True)
        frequency_range = self._resolve_param("frequency_range", frequency_range, None)
        spectrum_range = self._resolve_param("spectrum_range", spectrum_range, None)
        bad_idxes = self._resolve_param("bad_idxes", bad_idxes, None)
        input_is_envelope = self._resolve_param("input_is_envelope", input_is_envelope, True)
        filter_kwargs = self._resolve_param("filter_kwargs", filter_kwargs, None)
        trim_seconds = self._resolve_param("trim_seconds", trim_seconds, 1.0)
        hilbert_n_fft = self._resolve_param("hilbert_n_fft", hilbert_n_fft, None)

        if frequency_range is not None and spectrum_range is not None:
            raise ValueError("Pass only one of `frequency_range` or `spectrum_range`.")

        if frequency_range is None and spectrum_range is None:
            x = np.asarray(sample).squeeze()
            if x.ndim == 1:
                return self._dfa_single(
                    x,
                    sampling_frequency=sampling_frequency,
                    fit_interval=fit_interval,
                    compute_interval=compute_interval,
                    overlap=overlap,
                )

            out = self._dfa_multichannel(
                x,
                sampling_frequency=sampling_frequency,
                fit_interval=fit_interval,
                compute_interval=compute_interval,
                overlap=overlap,
                bad_idxes=bad_idxes,
            )
            out["input_is_envelope"] = bool(input_is_envelope)
            return out

        if sampling_frequency is None:
            raise ValueError("sampling_frequency (or fs) must be provided for DFA.")

        signal_matrix, _ = self._as_2d(sample)

        if spectrum_range is not None:
            raise ValueError("Automatic spectrum ranges are unavailable; choose an explicit frequency band.")

        band_fit_interval = fit_interval
        if band_fit_interval is None:
            raise ValueError("An explicit DFA fit_interval is required for each frequency band.")
        band_compute_interval = compute_interval or band_fit_interval
        envelope = self._compute_band_envelope(
            signal_matrix,
            sampling_frequency,
            frequency_range,
            filter_kwargs=filter_kwargs,
            trim_seconds=trim_seconds,
            hilbert_n_fft=hilbert_n_fft,
        )

        out = self._dfa_multichannel(
            envelope,
            sampling_frequency=sampling_frequency,
            fit_interval=band_fit_interval,
            compute_interval=band_compute_interval,
            overlap=overlap,
            bad_idxes=bad_idxes,
        )
        out["frequency_range"] = frequency_range
        out["fit_interval"] = band_fit_interval
        out["compute_interval"] = band_compute_interval
        return out


    def fEI(
        self,
        sample,
        *,
        sampling_frequency=None,
        window_size_sec=None,
        window_overlap=None,
        dfa_value=None,
        dfa_threshold=None,
        dfa_fit_interval=None,
        dfa_compute_interval=None,
        dfa_overlap=None,
        frequency_range=None,
        spectrum_range=None,
        bad_idxes=None,
        input_is_envelope=None,
        filter_kwargs=None,
        trim_seconds=None,
        hilbert_n_fft=None,
    ):
        """
        Compute fEI from a time series sample using the Python implementation.
        Note: ncpi does not provide a C runtime for fEI.
        Based on Hardstone et al. (2012), doi:10.3389/fphys.2012.00450,
        and Bruining et al. (2020), doi:10.1038/s41598-020-65500-4.

        If `frequency_range` is provided, the method will:
          1) band-pass filter the signal
          2) compute amplitude envelope (Hilbert)
          3) compute DFA (for thresholding)
          4) compute fEI on the envelope

        Otherwise, the input is treated as an amplitude envelope.

        Parameters
        ----------
        sample: array-like
            Input time series. Can be 1D (single channel) or 2D (n_channels, n_timepoints).
        sampling_frequency: float, optional
            Sampling frequency of the signal (in Hz). Required for band/spectrum workflows.
        window_size_sec: float, optional
            Window size in seconds for fEI. Default 5 seconds.
        window_overlap: float, optional
            Fractional overlap between windows in [0, 1). Default 0.8.
        dfa_value: float, optional
            DFA value for thresholding. Only valid for single-channel envelope input.
        dfa_threshold: float, optional
            DFA threshold below which fEI is set to NaN. Default 0.6.
        dfa_fit_interval: list or tuple, optional
            Fit interval (in seconds) to compute DFA internally when dfa_value is None.
        dfa_compute_interval: list or tuple, optional
            Compute interval (in seconds) to compute DFA internally when dfa_value is None.
        dfa_overlap: bool, optional
            Whether to use 50% overlapping windows for DFA. Default True.
        frequency_range: list or tuple, optional
            Frequency range [low, high] in Hz for band-pass fEI. Mutually exclusive with spectrum_range.
        spectrum_range: list or tuple, optional
            Automatic spectrum ranges are no longer supported.
        bad_idxes: list, optional
            Channel indices to ignore in multichannel inputs.
        input_is_envelope: bool, optional
            Whether the input is already an amplitude envelope. Only used for metadata.
        filter_kwargs: dict, optional
            Unsupported by the independent filter; prepare an envelope externally if needed.
        trim_seconds: float, optional
            Seconds trimmed from both ends after filtering to reduce edge artifacts.
        hilbert_n_fft: int, optional
            Unsupported by the independent filter.

        Returns
        -------
        out: dict
            For 1D envelope input: keys `fEI_outliers_removed`, `fEI_val`, `num_outliers`, `wAmp`, `wDNF`, `dfa`.
            For multichannel envelope input: keys `fEI_outliers_removed`, `fEI_val`, `num_outliers`, `wAmp`, `wDNF`,
            plus `DFA`.
            For band input: multichannel envelope output plus `frequency_range`, `fit_interval`, `compute_interval`,
            and `fEI` (without outlier rejection).
        """
        sampling_frequency = self._resolve_sampling_frequency(sampling_frequency)
        window_size_sec = self._resolve_param("window_size_sec", window_size_sec, 5.0)
        window_overlap = self._resolve_param("window_overlap", window_overlap, 0.8)
        dfa_value = self._resolve_param("dfa_value", dfa_value, None)
        dfa_threshold = self._resolve_param("dfa_threshold", dfa_threshold, 0.6)
        dfa_fit_interval = self._resolve_param("dfa_fit_interval", dfa_fit_interval, None)
        dfa_compute_interval = self._resolve_param("dfa_compute_interval", dfa_compute_interval, None)
        dfa_overlap = self._resolve_param("dfa_overlap", dfa_overlap, True)
        frequency_range = self._resolve_param("frequency_range", frequency_range, None)
        spectrum_range = self._resolve_param("spectrum_range", spectrum_range, None)
        bad_idxes = self._resolve_param("bad_idxes", bad_idxes, None)
        input_is_envelope = self._resolve_param("input_is_envelope", input_is_envelope, True)
        filter_kwargs = self._resolve_param("filter_kwargs", filter_kwargs, None)
        trim_seconds = self._resolve_param("trim_seconds", trim_seconds, 1.0)
        hilbert_n_fft = self._resolve_param("hilbert_n_fft", hilbert_n_fft, None)

        if frequency_range is not None and spectrum_range is not None:
            raise ValueError("Pass only one of `frequency_range` or `spectrum_range`.")

        if frequency_range is None and spectrum_range is None:
            x = np.asarray(sample).squeeze()
            if x.ndim == 1:
                if dfa_value is None:
                    if dfa_fit_interval is None:
                        dfa_fit_interval = self._resolve_param("fit_interval", None, None)
                    if dfa_compute_interval is None:
                        dfa_compute_interval = self._resolve_param("compute_interval", None, None)
                out = self._fei_single(
                    x,
                    sampling_frequency=sampling_frequency,
                    window_size_sec=window_size_sec,
                    window_overlap=window_overlap,
                    dfa_value=dfa_value,
                    dfa_threshold=dfa_threshold,
                    dfa_fit_interval=dfa_fit_interval,
                    dfa_compute_interval=dfa_compute_interval,
                    dfa_overlap=dfa_overlap,
                )
                return out

            dfa_array = None
            if dfa_value is not None:
                raise ValueError("dfa_value must be None when providing a multi-channel sample.")
            if dfa_fit_interval is None:
                dfa_fit_interval = self._resolve_param("fit_interval", None, None)
            if dfa_compute_interval is None:
                dfa_compute_interval = self._resolve_param("compute_interval", None, None)
            dfa_out = self._dfa_multichannel(
                x,
                sampling_frequency=sampling_frequency,
                fit_interval=dfa_fit_interval,
                compute_interval=dfa_compute_interval,
                overlap=dfa_overlap,
                bad_idxes=bad_idxes,
            )
            dfa_array = dfa_out["DFA"]
            fei_out = self._fei_multichannel(
                x,
                sampling_frequency=sampling_frequency,
                window_size_sec=window_size_sec,
                window_overlap=window_overlap,
                dfa_array=dfa_array,
                dfa_threshold=dfa_threshold,
                bad_idxes=bad_idxes,
                dfa_fit_interval=dfa_fit_interval,
                dfa_compute_interval=dfa_compute_interval,
                dfa_overlap=dfa_overlap,
            )
            fei_out["DFA"] = dfa_array
            fei_out["input_is_envelope"] = bool(input_is_envelope)
            return fei_out

        if sampling_frequency is None:
            raise ValueError("sampling_frequency (or fs) must be provided for fEI.")

        signal_matrix, _ = self._as_2d(sample)

        if spectrum_range is not None:
            raise ValueError("Automatic spectrum ranges are unavailable; choose an explicit frequency band.")

        band_fit_interval = dfa_fit_interval
        if band_fit_interval is None:
            raise ValueError("An explicit DFA fit_interval is required for each frequency band.")
        band_compute_interval = dfa_compute_interval or band_fit_interval
        envelope = self._compute_band_envelope(
            signal_matrix,
            sampling_frequency,
            frequency_range,
            filter_kwargs=filter_kwargs,
            trim_seconds=trim_seconds,
            hilbert_n_fft=hilbert_n_fft,
        )

        dfa_out = self._dfa_multichannel(
            envelope,
            sampling_frequency=sampling_frequency,
            fit_interval=band_fit_interval,
            compute_interval=band_compute_interval,
            overlap=dfa_overlap,
            bad_idxes=bad_idxes,
        )
        dfa_array = dfa_out["DFA"]

        fei_out = self._fei_multichannel(
            envelope,
            sampling_frequency=sampling_frequency,
            window_size_sec=window_size_sec,
            window_overlap=window_overlap,
            dfa_array=dfa_array,
            dfa_threshold=dfa_threshold,
            bad_idxes=bad_idxes,
            dfa_fit_interval=band_fit_interval,
            dfa_compute_interval=band_compute_interval,
            dfa_overlap=dfa_overlap,
        )
        fei_out["DFA"] = dfa_array
        fei_out["fEI"] = np.squeeze(fei_out["fEI_val"])
        fei_out["frequency_range"] = frequency_range
        fei_out["fit_interval"] = band_fit_interval
        fei_out["compute_interval"] = band_compute_interval
        return fei_out


    def hctsa(self, samples, hctsa_folder, workers=None, return_meta=False):
        """
        Compute hctsa features using the MATLAB implementation.

        Parameters
        ----------
        samples: ndarray | list
            Input samples. Either a 2D array of shape (n_samples, n_timepoints) or
            a list of 1D arrays.
        hctsa_folder: str
            Path to the hctsa folder (must contain the hctsa startup script).
        workers: int | None
            Number of MATLAB workers for the parallel pool. If None, no parpool is created.
        return_meta: bool
            If True, return a dict with additional metadata. If False, return a list
            of feature vectors per sample.

        Returns
        -------
        features: list | dict
            If return_meta is False, returns a list of feature vectors (one per sample).
            If return_meta is True, returns a dict with keys:
            - "features": list of feature vectors per sample
            - "feature_matrix": np.ndarray of shape (n_samples, n_valid_features)
            - "valid_feature_mask": boolean mask over all computed features
            - "num_valid_features": int
            - "num_total_features": int
        """
        if hctsa_folder is None:
            raise ValueError(
                "hctsa_folder must be provided. Make sure hctsa is installed and its setup "
                "instructions have been followed."
            )
        if not os.path.isdir(hctsa_folder):
            raise ValueError(f"hctsa_folder does not exist: {hctsa_folder}")

        if not hasattr(self, "matlab") or not hasattr(self, "matlabengine") or not hasattr(self, "h5py"):
            matlab_engine = tools.dynamic_import("matlab.engine", raise_on_error=False)
            if matlab_engine is None:
                raise ImportError(
                    "MATLAB Engine for Python is required (module 'matlab.engine'). "
                    "Install it from your MATLAB distribution at "
                    "<MATLAB_ROOT>/extern/engines/python, or via a compatible "
                    "`matlabengine` pip package (e.g., `pip install \"matlabengine==24.2.2\"`) "
                    "depending on your MATLAB distribution (do not use the PyPI 'matlab' stub)."
                )
            if not tools.ensure_module("h5py"):
                raise ImportError("h5py is required for reading HCTSA.mat outputs.")
            self.matlab = tools.dynamic_import("matlab")
            self.matlabengine = matlab_engine
            self.h5py = tools.dynamic_import("h5py")

        if isinstance(samples, np.ndarray):
            if samples.ndim == 1:
                samples_list = [samples]
            elif samples.ndim == 2:
                samples_list = [samples[i, :] for i in range(samples.shape[0])]
            else:
                raise ValueError("samples must be 1D or 2D.")
        else:
            samples_list = list(samples)

        n_samples = len(samples_list)
        if n_samples == 0:
            empty = [] if not return_meta else {
                "features": [],
                "feature_matrix": np.empty((0, 0), dtype=float),
                "valid_feature_mask": np.array([], dtype=bool),
                "num_valid_features": 0,
                "num_total_features": 0,
            }
            return empty

        eng = self.matlabengine.start_matlab()

        try:
            hctsa_mat = os.path.join(hctsa_folder, "HCTSA.mat")
            if os.path.isfile(hctsa_mat):
                os.remove(hctsa_mat)

            eng.cd(hctsa_folder)
            eng.startup(nargout=0)

            eng.eval(f"timeSeriesData = cell(1,{n_samples});", nargout=0)
            eng.eval(f"labels = cell(1,{n_samples});", nargout=0)
            eng.eval(f"keywords = cell(1,{n_samples});", nargout=0)

            for s_idx, s in enumerate(samples_list, start=1):
                s_arr = np.asarray(s).squeeze()
                if s_arr.ndim != 1:
                    raise ValueError("Each sample must be 1D.")
                eng.workspace["aux"] = self.matlab.double(list(map(float, s_arr.tolist())))
                eng.eval(f"timeSeriesData{{1,{s_idx}}} = aux;", nargout=0)
                eng.eval(f"labels{{1,{s_idx}}} = '{s_idx}';", nargout=0)
                eng.eval(f"keywords{{1,{s_idx}}} = '{s_idx}';", nargout=0)

            inp_mat = "INP_ncpi_ts.mat"
            inp_mat_path = os.path.join(hctsa_folder, inp_mat)
            if os.path.isfile(inp_mat_path):
                os.remove(inp_mat_path)
            eng.eval(f"save('{inp_mat}','timeSeriesData','labels','keywords');", nargout=0)
            eng.eval(f"load('{inp_mat}');", nargout=0)

            eng.TS_Init(inp_mat, "hctsa", self.matlab.logical([False, False, False]), nargout=0)

            if workers is not None and int(workers) > 1:
                n_workers = int(workers)
                eng.eval(
                    f"p = gcp('nocreate'); if isempty(p), parpool({n_workers}); end",
                    nargout=0,
                )

            eng.eval("TS_Compute(true);", nargout=0)

            with self.h5py.File(hctsa_mat, "r") as f:
                ts_data = np.array(f.get("TS_DataMat"))
                ts_quality = f.get("TS_Quality")
                ts_quality = np.array(ts_quality) if ts_quality is not None else None

            if ts_data.ndim != 2:
                raise RuntimeError("Unexpected TS_DataMat shape in HCTSA.mat.")

            if ts_data.shape[1] == n_samples:
                data = ts_data
            elif ts_data.shape[0] == n_samples:
                data = ts_data.T
            else:
                data = ts_data

            valid_mask = np.isfinite(data).all(axis=1)

            if ts_quality is not None:
                tq = ts_quality
                if tq.shape != data.shape and tq.T.shape == data.shape:
                    tq = tq.T
                if tq.shape == data.shape:
                    valid_mask &= (tq == 0).all(axis=1)

            data_valid = data[valid_mask, :]
            features = [data_valid[:, s].tolist() for s in range(data_valid.shape[1])]

            if return_meta:
                return {
                    "features": features,
                    "feature_matrix": data_valid.T,
                    "valid_feature_mask": valid_mask,
                    "num_valid_features": int(np.sum(valid_mask)),
                    "num_total_features": int(data.shape[0]),
                }

            return features

        finally:
            try:
                eng.quit()
            except Exception:
                pass


    def compute_features(
        self,
        samples,
        n_jobs=None,
        chunksize=None,
        start_method="spawn",
        progress_callback=None,
        log_callback=None,
    ):
        """
        Compute features from a collection of 1D samples in parallel.

        Parameters
        ----------
        samples : list | np.ndarray
            Iterable of 1D samples (time series).
        n_jobs : int | None
            Number of worker processes. Defaults to all CPUs.
        chunksize : int | None
            Chunk size for multiprocessing scheduling.
        start_method : {"spawn","fork","forkserver"}
            Process start method. "spawn" is most robust across platforms.
        progress_callback : callable | None
            Optional callback receiving progress updates as
            ``progress_callback(completed, total, percent)``.
            If the callback only accepts one argument, ``percent`` is passed.
        log_callback : callable | None
            Optional callback receiving textual progress messages.

        Returns
        -------
        list
            One feature output per sample.
            - catch22: list of 22-length arrays
            - specparam: list of dicts
            - dfa: list of dicts
            - fEI: list of dicts
            - hctsa: list of feature vectors (uses MATLAB backend)
        """
        # Materialize as list once (so we know length and can iterate multiple times if needed)
        samples_list = list(samples) if not isinstance(samples, np.ndarray) else list(samples)
        n = len(samples_list)
        if n == 0:
            return []

        if log_callback:
            try:
                log_callback(f"Starting feature computation ({self.method}) on {n} sample(s).")
            except Exception:
                pass

        if progress_callback:
            try:
                progress_callback(0, n, 0)
            except TypeError:
                progress_callback(0)
            except Exception:
                pass

        if self.method == "hctsa":
            hctsa_folder = self.params.get("hctsa_folder", None)
            if hctsa_folder is None:
                raise ValueError(
                    "hctsa_folder must be provided in params for hctsa. "
                    "Make sure hctsa is installed and its setup instructions have been followed."
                )
            workers = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
            if log_callback:
                try:
                    log_callback(f"Running hctsa with {workers} worker(s).")
                except Exception:
                    pass
            result = self.hctsa(samples_list, hctsa_folder=hctsa_folder, workers=workers)
            if progress_callback:
                try:
                    progress_callback(n, n, 100)
                except TypeError:
                    progress_callback(100)
                except Exception:
                    pass
            if log_callback:
                try:
                    log_callback("Feature computation finished.")
                except Exception:
                    pass
            return result

        # Determine number of processes
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1
        n_jobs = max(1, int(n_jobs))

        # Adaptive chunksize: enough to amortize IPC, small enough to load-balance
        if chunksize is None:
            # aim for ~8 chunks/worker (tune if needed)
            chunksize = max(1, n // (n_jobs * 8))

        ctx = self.multiprocessing.get_context(start_method)

        with ctx.Pool(
            processes=n_jobs,
            initializer=_features_worker_init,
            initargs=(self.method, self.params),
        ) as pool:

            it = pool.imap(_compute_one_feature, samples_list, chunksize=chunksize)

            if self.tqdm_inst:
                it = self.tqdm(it, total=n, desc=f"Computing {self.method} features")
            results = []
            last_percent = -1
            next_log_percent = 5
            for completed, feature_out in enumerate(it, start=1):
                results.append(feature_out)
                percent = int((completed * 100) / n)
                if percent != last_percent and progress_callback:
                    try:
                        progress_callback(completed, n, percent)
                    except TypeError:
                        progress_callback(percent)
                    except Exception:
                        pass
                    last_percent = percent

                if log_callback and (percent >= next_log_percent or completed == n):
                    try:
                        log_callback(f"Feature progress: {completed}/{n} ({percent}%).")
                    except Exception:
                        pass
                    while next_log_percent <= percent:
                        next_log_percent += 5

            if log_callback:
                try:
                    log_callback("Feature computation finished.")
                except Exception:
                    pass
            return results
