import os
import numpy as np
import numpy.ma as np_ma
import scipy.signal as scipy_signal
from scipy.stats import t as scipy_t
from ncpi import tools


### Worker helper functions ###
_WORKER_FEATURES_OBJ = None


def _create_window_indices(length_signal, length_window, window_offset):
    window_starts = np.arange(0, length_signal - length_window, window_offset)
    num_windows = len(window_starts)

    one_window_index = np.arange(0, length_window)
    all_window_index = np.tile(one_window_index, (num_windows, 1)).astype(int)

    all_window_index = all_window_index + np.tile(
        np.transpose(window_starts[np.newaxis, :]), (1, length_window)
    ).astype(int)

    return all_window_index


def _generalized_esd(x, max_ols, alpha=0.05, full_output=False, ubvar=False):
    """
    Generalized ESD test for outliers.
    Adapted from crosci/outliers.py (MIT).
    """
    if max_ols < 1:
        raise ValueError(
            "Maximum number of outliers, `max_ols`, must be > 1. "
            "Specify, e.g., max_ols = 2."
        )
    xm = np_ma.array(x)
    n = len(xm)

    r_vals = []
    l_vals = []
    max_indices = []
    for i in range(max_ols + 1):
        xmean = xm.mean()
        xstd = xm.std(ddof=int(ubvar))
        rr = np.abs((xm - xmean) / xstd)
        max_indices.append(np.argmax(rr))
        r_vals.append(rr[max_indices[-1]])
        if i >= 1:
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            per_point = scipy_t.ppf(p, n - i - 1)
            l_vals.append(
                (n - i) * per_point / np.sqrt((n - i - 1 + per_point**2) * (n - i + 1))
            )
        xm[max_indices[-1]] = np_ma.masked

    r_vals.pop(-1)
    ofound = False
    for i in range(max_ols - 1, -1, -1):
        if r_vals[i] > l_vals[i]:
            ofound = True
            break

    if ofound:
        if not full_output:
            return i + 1, max_indices[0: i + 1]
        return i + 1, max_indices[0: i + 1], r_vals, l_vals, max_indices

    if not full_output:
        return 0, []
    return 0, [], r_vals, l_vals, max_indices


def _features_worker_init(method, params):
    """
    Initializer that runs once per worker process.
    Creates a worker-local Features object to avoid pickling 'self'.
    """
    global _WORKER_FEATURES_OBJ
    _WORKER_FEATURES_OBJ = Features(method=method, params=params)


def _compute_one_feature(sample):
    """
    Compute features for a single sample using the worker-local Features object.
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
        # specparam reads parameters from self.params via _resolve_param
        return _WORKER_FEATURES_OBJ.specparam(sample=x)

    if _WORKER_FEATURES_OBJ.method == "dfa":
        return _WORKER_FEATURES_OBJ.dfa(sample=x)

    if _WORKER_FEATURES_OBJ.method == "fEI":
        return _WORKER_FEATURES_OBJ.fEI(sample=x)

    raise ValueError(f"Unknown method: {_WORKER_FEATURES_OBJ.method}")


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
        if method not in ['catch22', 'specparam', 'dfa', 'fEI']:
            raise ValueError("Invalid method. Please use 'catch22', 'specparam', 'dfa', or 'fEI'.")

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
            self._pycatch22 = tools.dynamic_import("pycatch22")

        elif method == 'specparam':
            if not tools.ensure_module("specparam"):
                raise ImportError(
                    "specparam is required for computing spectral features but is not installed."
                )
            self._specparam = tools.dynamic_import("specparam")
        elif method in {"dfa", "fEI"}:
            # Python implementation is used for DFA/fEI.
            self._crosci_run_dfa = None
            self._crosci_run_fei = None

        # elif method == 'hctsa':
        #     # Check if MATLAB engine for Python is installed but do not try to install it
        #     if not tools.ensure_module("matlab", raise_on_error=False):
        #         raise ImportError("MATLAB Engine for Python is required (provides 'matlab' and 'matlab.engine').")
        #
        #     matlab_engine = tools.dynamic_import("matlab.engine", raise_on_error=False)
        #
        #     if matlab_engine is None:
        #         raise ImportError("MATLAB Engine is not importable as 'matlab.engine'.")
        #     if not tools.ensure_module("h5py"):
        #         raise ImportError("h5py is required ...")
        #
        #     self.matlab = tools.dynamic_import("matlab")
        #     self.matlabengine = matlab_engine
        #     self.h5py = tools.dynamic_import("h5py")

        # Use stdlib multiprocessing
        self.multiprocessing = tools.dynamic_import("multiprocessing")

        # Check if tqdm is installed
        if not tools.ensure_module("tqdm"):
            self.tqdm_inst = False
        else:
            self.tqdm_inst = True
            self.tqdm = tools.dynamic_import("tqdm", "tqdm")


    def _resolve_param(self, name, value, default=None):
        """
        Resolve a parameter value with priority:
        explicit argument > self.params[name] > default
        """
        if value is not None:
            return value
        if name in self.params:
            return self.params[name]
        return default


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


    def _resolve_sampling_frequency(self, sampling_frequency=None):
        sampling_frequency = self._resolve_param("sampling_frequency", sampling_frequency)
        if sampling_frequency is None:
            sampling_frequency = self._resolve_param("fs", None)
        return sampling_frequency


    def _dfa_window_sizes(self, sampling_frequency, compute_interval):
        window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(int)
        window_sizes = np.sort(np.unique(window_sizes))
        window_sizes = window_sizes[
            (window_sizes >= compute_interval[0] * sampling_frequency)
            & (window_sizes <= compute_interval[1] * sampling_frequency)
        ]
        return window_sizes


    def _dfa_nan_output(self, length_signal):
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
        sampling_frequency = self._resolve_sampling_frequency(None)
        window_size_sec = self._resolve_param("window_size_sec", None, None)
        window_overlap = self._resolve_param("window_overlap", None, 0.0)

        num_windows = 0
        if sampling_frequency is not None and window_size_sec is not None:
            try:
                window_size = int(float(window_size_sec) * float(sampling_frequency))
                if window_size > 0:
                    window_offset = int(np.floor(window_size * (1 - float(window_overlap))))
                    if window_offset > 0:
                        num_windows = len(np.arange(0, length_signal - window_size, window_offset))
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


    def dfa(
        self,
        sample,
        *,
        sampling_frequency=None,
        fit_interval=None,
        compute_interval=None,
        overlap=None,
        runtime=None,
    ):
        """
        Compute DFA for a 1D signal (amplitude envelope).

        Parameters follow crosci.biomarkers.DFA with a 1D input.
        Only runtime="python" is supported.
        """
        sampling_frequency = self._resolve_sampling_frequency(sampling_frequency)
        fit_interval = self._resolve_param("fit_interval", fit_interval, None)
        compute_interval = self._resolve_param("compute_interval", compute_interval, None)
        overlap = self._resolve_param("overlap", overlap, True)
        runtime = self._resolve_param("runtime", runtime, "python")

        if sampling_frequency is None:
            raise ValueError("sampling_frequency (or fs) must be provided for DFA.")
        if fit_interval is None or compute_interval is None:
            raise ValueError("fit_interval and compute_interval must be provided for DFA.")

        x = np.asarray(sample).squeeze()
        if x.ndim != 1:
            raise ValueError("`sample` must be a 1D array.")

        fit_interval = [float(fit_interval[0]), float(fit_interval[1])]
        compute_interval = [float(compute_interval[0]), float(compute_interval[1])]

        if not (fit_interval[0] >= compute_interval[0] and fit_interval[1] <= compute_interval[1]):
            raise ValueError("fit_interval should be included in compute_interval.")
        if compute_interval[0] < 0.1 or compute_interval[1] > 1000:
            raise ValueError("compute_interval should be between 0.1 and 1000 seconds.")
        if compute_interval[1] / float(sampling_frequency) > x.shape[0]:
            raise ValueError("compute_interval should not extend beyond the length of the signal.")

        window_sizes = self._dfa_window_sizes(float(sampling_frequency), compute_interval)
        fluctuations = np.full(window_sizes.shape, np.nan, dtype=float)
        dfa_val = np.nan
        dfa_intercept = np.nan

        if window_sizes.size == 0 or np.max(window_sizes) > x.shape[0]:
            return {
                "dfa": dfa_val,
                "window_sizes": window_sizes,
                "fluctuations": fluctuations,
                "dfa_intercept": dfa_intercept,
            }

        if runtime != "python":
            raise ValueError("runtime must be 'python' for DFA.")

        window_overlap = 0.5 if overlap else 0.0
        window_offset = np.floor(window_sizes * (1 - window_overlap)).astype(int)
        signal_profile = np.cumsum(x - np.mean(x))

        for i_window_size, window_size in enumerate(window_sizes):
            offset = int(window_offset[i_window_size])
            if offset <= 0:
                continue
            all_window_index = _create_window_indices(x.shape[0], int(window_size), offset)
            if all_window_index.size == 0:
                continue
            x_signal = signal_profile[all_window_index]
            _, fluc, _, _, _ = np.polyfit(
                np.arange(window_size), np.transpose(x_signal), deg=1, full=True
            )
            fluctuations[i_window_size] = np.mean(np.sqrt(fluc / window_size))

        try:
            fit_interval_first_window = np.argwhere(
                window_sizes >= fit_interval[0] * sampling_frequency
            )[0][0]
            fit_interval_last_window = np.argwhere(
                window_sizes <= fit_interval[1] * sampling_frequency
            )[-1][0]
        except Exception as e:
            raise ValueError("fit_interval does not match any computed window sizes.") from e

        if fit_interval_first_window > 0:
            if (
                np.abs(
                    window_sizes[fit_interval_first_window - 1] / sampling_frequency
                    - fit_interval[0]
                )
                <= fit_interval[0] / 100
            ):
                if np.abs(
                    window_sizes[fit_interval_first_window - 1] / sampling_frequency
                    - fit_interval[0]
                ) < np.abs(
                    window_sizes[fit_interval_first_window] / sampling_frequency
                    - fit_interval[0]
                ):
                    fit_interval_first_window = fit_interval_first_window - 1

        x_fit = np.log10(
            window_sizes[fit_interval_first_window: fit_interval_last_window + 1]
        )
        y_fit = np.log10(
            fluctuations[fit_interval_first_window: fit_interval_last_window + 1]
        )
        model = np.polyfit(x_fit, y_fit, 1)
        dfa_intercept = model[1]
        dfa_val = model[0]

        return {
            "dfa": dfa_val,
            "window_sizes": window_sizes,
            "fluctuations": fluctuations,
            "dfa_intercept": dfa_intercept,
        }


    def fEI(
        self,
        sample,
        *,
        sampling_frequency=None,
        window_size_sec=None,
        window_overlap=None,
        dfa_value=None,
        dfa_threshold=None,
        runtime=None,
        dfa_fit_interval=None,
        dfa_compute_interval=None,
        dfa_overlap=None,
        dfa_runtime=None,
    ):
        """
        Compute fEI for a 1D signal (amplitude envelope).

        Parameters follow crosci.biomarkers.fEI with a 1D input.
        Only runtime="python" is supported.
        """
        sampling_frequency = self._resolve_sampling_frequency(sampling_frequency)
        window_size_sec = self._resolve_param("window_size_sec", window_size_sec, None)
        window_overlap = self._resolve_param("window_overlap", window_overlap, 0.0)
        runtime = self._resolve_param("runtime", runtime, "python")
        dfa_value = self._resolve_param("dfa_value", dfa_value, None)
        dfa_threshold = self._resolve_param("dfa_threshold", dfa_threshold, 0.6)

        if sampling_frequency is None:
            raise ValueError("sampling_frequency (or fs) must be provided for fEI.")
        if window_size_sec is None:
            raise ValueError("window_size_sec must be provided for fEI.")

        x = np.asarray(sample).squeeze()
        if x.ndim != 1:
            raise ValueError("`sample` must be a 1D array.")

        window_size = int(float(window_size_sec) * float(sampling_frequency))
        if window_size <= 0:
            raise ValueError("window_size_sec must yield a positive window size.")
        if not (0.0 <= float(window_overlap) < 1.0):
            raise ValueError("window_overlap must be in [0, 1).")

        window_offset = int(np.floor(window_size * (1 - float(window_overlap))))
        if window_offset <= 0:
            raise ValueError("window_overlap yields a non-positive window offset.")

        all_window_index = _create_window_indices(x.shape[0], window_size, window_offset)
        if all_window_index.size == 0:
            return self._fei_nan_output(x.shape[0])

        if np.min(x) == np.max(x):
            return self._fei_nan_output(x.shape[0])

        if runtime != "python":
            raise ValueError("runtime must be 'python' for fEI.")

        signal_profile = np.cumsum(x - np.mean(x))
        w_original_amp = np.mean(x[all_window_index], axis=1)

        x_amp = np.tile(np.transpose(w_original_amp[np.newaxis, :]), (1, window_size))
        x_signal = signal_profile[all_window_index]
        x_signal = np.divide(x_signal, x_amp)

        _, fluc, _, _, _ = np.polyfit(
            np.arange(window_size), np.transpose(x_signal), deg=1, full=True
        )
        w_dnf = np.sqrt(fluc / window_size)

        fei_val = np.nan
        fei_outliers_removed = np.nan
        num_outliers = np.nan

        if w_original_amp.size >= 2 and w_dnf.size >= 2:
            fei_val = 1 - np.corrcoef(w_original_amp, w_dnf)[0, 1]

            gesd_alpha = 0.05
            max_outliers_percentage = 0.025
            max_num_outliers = max(int(np.round(max_outliers_percentage * len(w_original_amp))), 2)

            outlier_indexes_wamp = _generalized_esd(w_original_amp, max_num_outliers, gesd_alpha)[1]
            outlier_indexes_wdnf = _generalized_esd(w_dnf, max_num_outliers, gesd_alpha)[1]
            outlier_union = outlier_indexes_wamp + outlier_indexes_wdnf
            num_outliers = len(outlier_union)
            not_outlier_both = np.setdiff1d(np.arange(len(w_original_amp)), np.array(outlier_union))
            if not_outlier_both.size >= 2:
                fei_outliers_removed = 1 - np.corrcoef(
                    w_original_amp[not_outlier_both], w_dnf[not_outlier_both]
                )[0, 1]

        if dfa_value is None:
            dfa_fit_interval = self._resolve_param("dfa_fit_interval", dfa_fit_interval, None)
            if dfa_fit_interval is None:
                dfa_fit_interval = self._resolve_param("fit_interval", None, None)
            dfa_compute_interval = self._resolve_param("dfa_compute_interval", dfa_compute_interval, None)
            if dfa_compute_interval is None:
                dfa_compute_interval = self._resolve_param("compute_interval", None, None)
            dfa_overlap = self._resolve_param("dfa_overlap", dfa_overlap, True)
            dfa_runtime = self._resolve_param("dfa_runtime", dfa_runtime, runtime)

            if dfa_fit_interval is None or dfa_compute_interval is None:
                raise ValueError(
                    "dfa_value not provided; dfa_fit_interval and dfa_compute_interval are required."
                )
            dfa_out = self.dfa(
                x,
                sampling_frequency=sampling_frequency,
                fit_interval=dfa_fit_interval,
                compute_interval=dfa_compute_interval,
                overlap=dfa_overlap,
                runtime=dfa_runtime,
            )
            dfa_value = float(dfa_out["dfa"])

        if dfa_value <= float(dfa_threshold):
            fei_val = np.nan
            fei_outliers_removed = np.nan

        return {
            "fEI_outliers_removed": fei_outliers_removed,
            "fEI_val": fei_val,
            "num_outliers": num_outliers,
            "wAmp": np.asarray(w_original_amp, dtype=float),
            "wDNF": np.asarray(w_dnf, dtype=float),
        }

    # def hctsa(self, samples, hctsa_folder, workers=32):
    #     """
    #     Compute hctsa features.
    #
    #     Parameters
    #     ----------
    #     samples: ndarray/list of shape (n_samples, times-series length)
    #         A set of samples of time-series data.
    #     hctsa_folder: str
    #         Folder where hctsa is installed.
    #     workers: int
    #         Number of MATLAB workers of the parallel pool.
    #
    #     Returns
    #     -------
    #     feats: list of shape (n_samples, n_features)
    #         hctsa features.
    #
    #     Debugging
    #     ---------
    #     This function has been debugged by approximating results shown
    #     in https://github.com/benfulcher/hctsaTutorial_BonnEEG.
    #     """
    #
    #     feats = []
    #
    #     # start Matlab engine
    #     print("\n--> Starting Matlab engine ...")
    #     eng = self.matlabengine.start_matlab()
    #
    #     try:
    #         # Remove hctsa file
    #         if os.path.isfile(os.path.join(hctsa_folder, 'HCTSA.mat')):
    #             os.remove(os.path.join(hctsa_folder, 'HCTSA.mat'))
    #
    #         # Change to hctsa folder
    #         eng.cd(hctsa_folder)
    #
    #         # Startup hctsa script
    #         print("\n--> hctsa startup ...")
    #         st = eng.startup(nargout=0)
    #         print(st)
    #
    #         # Check if samples is a list and convert it to a numpy array
    #         if isinstance(samples, list):
    #             samples = np.array(samples)
    #
    #         # Create the input variables in Matlab
    #         eng.eval(f'timeSeriesData = cell(1,{samples.shape[0]});', nargout=0)
    #         eng.eval(f'labels = cell(1,{samples.shape[0]});', nargout=0)
    #         eng.eval(f'keywords = cell(1,{samples.shape[0]});', nargout=0)
    #
    #         # Transfer time-series data to Matlab workspace
    #         for s in range(samples.shape[0]):
    #             eng.workspace['aux'] = self.matlab.double(list(samples[s]))
    #             eng.eval('timeSeriesData{1,%s} = aux;' % (s + 1), nargout=0)
    #
    #         # Fill in the other 2 Matlab structures with the index of the sample
    #         for s in range(samples.shape[0]):
    #             eng.eval('labels{1,%s} = \'%s\';' % (str(s + 1), str(s + 1)), nargout=0)
    #             eng.eval('keywords{1,%s} = \'%s\';' % (str(s + 1), str(s + 1)), nargout=0)
    #
    #         # Save variables into a mat file
    #         eng.eval('save INP_ccpi_ts.mat timeSeriesData labels keywords;', nargout=0)
    #
    #         # Load mat file
    #         eng.eval('load INP_ccpi_ts.mat;', nargout=0)
    #
    #         # Initialize an hctsa calculation
    #         print("\n--> hctsa TS_Init ...")
    #         eng.TS_Init('INP_ccpi_ts.mat',
    #                     'hctsa',
    #                     self.matlab.logical([False, False, False]),
    #                     nargout=0)
    #
    #         # Open a parallel pool of a specific size
    #         if workers > 1:
    #             eng.parpool(workers)
    #
    #         # Compute features
    #         print("\n--> hctsa TS_Compute ...")
    #         # eng.TS_Compute(matlab.logical([True]),nargout = 0)
    #         eng.eval('TS_Compute(true);', nargout=0)
    #
    #         # Load hctsa file
    #         f = self.h5py.File(os.path.join(hctsa_folder, 'HCTSA.mat'), 'r')
    #         TS_DataMat = np.array(f.get('TS_DataMat'))
    #         # TS_Quality = np.array(f.get('TS_Quality'))
    #
    #         # Create the array of features to return
    #         print(f'\n--> Formatting {TS_DataMat.shape[0]} features...')
    #         for s in range(samples.shape[0]):
    #             feats.append(list(TS_DataMat[:, s]))
    #
    #         # Stop Matlab engine
    #         print("\n--> Stopping Matlab engine ...")
    #         eng.quit()
    #
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         # Stop Matlab engine
    #         print("\n--> Stopping Matlab engine ...")
    #         eng.quit()
    #         raise e
    #
    #     return feats


    def compute_features(self, samples, n_jobs=None, chunksize=None, start_method="spawn"):
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

        Returns
        -------
        list
            One feature output per sample.
            - catch22: list of 22-length arrays
            - specparam: list of dicts
            - dfa: list of dicts
            - fEI: list of dicts
        """
        # Materialize as list once (so we know length and can iterate multiple times if needed)
        samples_list = list(samples) if not isinstance(samples, np.ndarray) else list(samples)
        n = len(samples_list)
        if n == 0:
            return []

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

            return list(it)
