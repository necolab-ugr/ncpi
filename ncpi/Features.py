import os
import numpy as np
import numpy.ma as np_ma
import scipy.signal as scipy_signal
from scipy.stats import t as scipy_t
from ncpi import tools

############################################################
## DFA/fEI Core Helpers (windowing, outlier detection)    ##
############################################################

def _create_window_indices(length_signal, length_window, window_offset):
    """Create a 2D array of indices for sliding windows.
    Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

    Parameters
    ----------
    length_signal: int
        Total length of the signal.
    length_window: int
        Length of each window.
    window_offset: int
        Step size between the starts of consecutive windows.

    Returns
    -------
    all_window_index: np.ndarray
        2D array of shape (num_windows, length_window) where each row contains the
        indices for that window.
    """
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
    Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

    Parameters
    ----------
    x: array-like
        Input data.
    max_ols: int
        Maximum number of outliers to test for.
    alpha: float
        Significance level for the test (default: 0.05).
    full_output: bool
        If True, return additional diagnostic information (default: False).
    ubvar: bool
        If True, use unbiased variance estimator (ddof=1) instead of population estimator (
        ddof=0) when computing standard deviation (default: False).

    Returns
    -------
    num_outliers: int
        Number of outliers detected.
    outlier_indices: list
        List of indices in `x` that are identified as outliers.
    r_vals: list (optional)
        List of test statistic values for each iteration (returned if full_output=True).
    l_vals: list (optional)
        List of critical values for each iteration (returned if full_output=True).
    max_indices: list (optional)
        List of indices of the maximum test statistic at each iteration (returned if full_output=True).
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


################################################
## Multiprocessing Helpers (worker lifecycle) ##
################################################

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


    def _get_frequency_bins(self, frequency_range):
        """ Get frequency bins for spectral analysis based on the specified frequency range.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        frequency_range: list or tuple
            A list or tuple containing the lower and upper bounds of the frequency range (in Hz).

        Returns
        -------
        frequency_bins: list of lists
            A list of frequency bins, where each bin is represented as a list [lower_bound, upper_bound].
        """
        assert frequency_range[0] >= 1.0 and frequency_range[1] <= 150.0, (
            "The frequency range should cannot be less than 1 Hz or more than 150 Hz"
        )

        frequency_bin_delta = [1.0, 4.0]
        frequency_range_full = [frequency_bin_delta[1], 150]
        n_bins_full = 16

        frequencies_full = np.logspace(
            np.log10(frequency_range_full[0]),
            np.log10(frequency_range_full[-1]),
            n_bins_full,
        )
        frequencies = np.append(frequency_bin_delta[0], frequencies_full)
        myfrequencies = frequencies[
            np.where(
                (np.round(frequencies, 4) >= frequency_range[0])
                & (np.round(frequencies, 4) <= frequency_range[1])
            )[0]
        ]

        frequency_bins = [
            [myfrequencies[i], myfrequencies[i + 1]]
            for i in range(len(myfrequencies) - 1)
        ]

        return frequency_bins


    def _get_DFA_fitting_interval(self, frequency_interval):
        """ Get the fitting interval for DFA based on the specified frequency interval.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        frequency_interval: list or tuple
            A list or tuple containing the lower and upper bounds of the frequency interval (in Hz).

        Returns
        fit_interval: list
            A list containing the lower and upper bounds of the fitting interval (in seconds).
        """
        upper_fit = 30
        default_lower_fits = [
            5.0,
            5.0,
            5.0,
            3.981,
            3.162,
            2.238,
            1.412,
            1.122,
            0.794,
            0.562,
            0.398,
            0.281,
            0.141,
            0.1,
            0.1,
            0.1,
        ]

        frequency_bins = self._get_frequency_bins([1, 150])
        idx_freq = np.where(
            (np.array(frequency_bins)[:, 0] <= frequency_interval[0])
        )[0][-1]

        fit_interval = [default_lower_fits[idx_freq], upper_fit]

        return fit_interval


    def _compute_band_envelope(
        self,
        signal_matrix,
        sampling_frequency,
        frequency_range,
        *,
        filter_kwargs=None,
        trim_seconds=1.0,
        hilbert_n_fft=None,
    ):
        """Compute the amplitude envelope of a signal in a specified frequency band using bandpass filtering and
        the Hilbert transform.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        signal_matrix: np.ndarray
            2D array of shape (n_channels, n_timepoints) containing the input signal.
        sampling_frequency: float
            Sampling frequency of the input signal (in Hz).
        frequency_range: list or tuple
            A list or tuple containing the lower and upper bounds of the frequency range (in Hz) for bandpass filtering.
        filter_kwargs: dict, optional
            Additional keyword arguments to pass to the MNE filter_data function (default: None).
        trim_seconds: float, optional
            Number of seconds to trim from the start and end of the filtered signal to mitigate edge artifacts
            (default: 1.0).
        hilbert_n_fft: int, optional
            Length of the FFT to use for the Hilbert transform. If None, the next fast length after the number of
            timepoints is used (default: None).

        Returns
        -------
        amplitude_envelope: np.ndarray
            2D array of shape (n_channels, n_timepoints_trimmed) containing the amplitude envelope of the signal in the
            specified frequency band, where n_timepoints_trimmed is the number of timepoints after trimming.
        """

        mne = self._ensure_mne()

        fk = {
            "filter_length": "auto",
            "l_trans_bandwidth": "auto",
            "h_trans_bandwidth": "auto",
            "fir_window": "hamming",
            "phase": "zero",
            "fir_design": "firwin",
            "pad": "reflect_limited",
            "verbose": 0,
        }
        if filter_kwargs:
            fk.update(dict(filter_kwargs))

        filtered_signal = mne.filter.filter_data(
            data=signal_matrix,
            sfreq=sampling_frequency,
            l_freq=frequency_range[0],
            h_freq=frequency_range[1],
            **fk,
        )

        trim = int(float(trim_seconds) * float(sampling_frequency))
        if trim > 0:
            if filtered_signal.shape[1] <= 2 * trim:
                raise ValueError(
                    "Signal is too short after trimming for filtering artifacts."
                )
            filtered_signal = filtered_signal[:, trim:-trim]

        num_timepoints = filtered_signal.shape[1]
        if hilbert_n_fft is None:
            hilbert_n_fft = mne.filter.next_fast_len(num_timepoints)

        analytic = scipy_signal.hilbert(filtered_signal, N=int(hilbert_n_fft), axis=-1)
        analytic = analytic[..., :num_timepoints]
        amplitude_envelope = np.abs(np.array(analytic))

        return amplitude_envelope


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
        """Compute window sizes for DFA based on the sampling frequency and fitting interval.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        sampling_frequency: float
            Sampling frequency of the signal (in Hz).
        compute_interval: list or tuple
            A list or tuple containing the lower and upper bounds of the fitting interval (in seconds) for DFA.

        Returns
        ----------
        window_sizes: np.ndarray
            1D array of integer window sizes (in samples) to use for DFA, filtered to be within the compute_interval.
        """
        window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(int)
        window_sizes = np.sort(np.unique(window_sizes))
        window_sizes = window_sizes[
            (window_sizes >= compute_interval[0] * sampling_frequency)
            & (window_sizes <= compute_interval[1] * sampling_frequency)
        ]
        return window_sizes


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


    def _dfa_single(
        self,
        sample,
        sampling_frequency,
        fit_interval,
        compute_interval,
        overlap,
    ):
        """Compute DFA for a single 1D amplitude envelope using the Python implementation.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        sample: array-like
            1D amplitude envelope signal.
        sampling_frequency: float
            Sampling frequency of the signal (in Hz).
        fit_interval: list or tuple
            Fitting interval (in seconds) for DFA.
        compute_interval: list or tuple
            Compute interval (in seconds) for DFA.
        overlap: bool
            Whether to use 50% window overlap.

        Returns
        -------
        output: dict
            Dictionary with keys: dfa, window_sizes, fluctuations, dfa_intercept.
        """
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


    def _dfa_multichannel(
        self,
        signal_matrix,
        sampling_frequency,
        fit_interval,
        compute_interval,
        overlap,
        bad_idxes=None,
    ):
        """Compute DFA for a multichannel amplitude envelope.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        signal_matrix: np.ndarray
            2D array of shape (n_channels, n_timepoints) containing the amplitude envelope.
        sampling_frequency: float
            Sampling frequency of the signal (in Hz).
        fit_interval: list or tuple
            Fitting interval (in seconds) for DFA.
        compute_interval: list or tuple
            Compute interval (in seconds) for DFA.
        overlap: bool
            Whether to use 50% window overlap.
        bad_idxes: list or None
            Channel indices to ignore.

        Returns
        -------
        output: dict
            Dictionary with keys: DFA, window_sizes, fluctuations, dfa_intercept.
        """
        signal_matrix, _ = self._as_2d(signal_matrix)
        num_chans, _ = np.shape(signal_matrix)
        bad_idxes = [] if bad_idxes is None else list(bad_idxes)
        channels_to_ignore = set(bad_idxes)

        dfa_array = np.zeros(num_chans)
        dfa_array[:] = np.nan
        dfa_intercept = np.zeros(num_chans)
        dfa_intercept[:] = np.nan

        window_sizes = None
        fluctuations = None

        for ch_idx in range(num_chans):
            if ch_idx in channels_to_ignore:
                continue
            out = self._dfa_single(
                signal_matrix[ch_idx, :],
                sampling_frequency=sampling_frequency,
                fit_interval=fit_interval,
                compute_interval=compute_interval,
                overlap=overlap,
            )
            dfa_array[ch_idx] = out["dfa"]
            dfa_intercept[ch_idx] = out["dfa_intercept"]

            if window_sizes is None:
                window_sizes = out["window_sizes"]
                fluctuations = np.full((num_chans, window_sizes.shape[0]), np.nan, dtype=float)

            fluc = out["fluctuations"]
            if fluctuations is not None:
                n = min(fluc.shape[0], fluctuations.shape[1])
                fluctuations[ch_idx, :n] = fluc[:n]

        if window_sizes is None:
            window_sizes = np.array([], dtype=int)
            fluctuations = np.empty((num_chans, 0), dtype=float)

        return {
            "DFA": dfa_array,
            "window_sizes": window_sizes,
            "fluctuations": fluctuations,
            "dfa_intercept": dfa_intercept,
        }


    def _fei_single(
        self,
        sample,
        sampling_frequency,
        window_size_sec,
        window_overlap,
        dfa_value,
        dfa_threshold,
        dfa_fit_interval=None,
        dfa_compute_interval=None,
        dfa_overlap=None,
    ):
        """Compute fEI for a single 1D amplitude envelope using the Python implementation.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        sample: array-like
            1D amplitude envelope signal.
        sampling_frequency: float
            Sampling frequency of the signal (in Hz).
        window_size_sec: float
            Window size in seconds.
        window_overlap: float
            Fractional overlap between windows in [0, 1).
        dfa_value: float or None
            DFA value for thresholding. If None, DFA is computed internally.
        dfa_threshold: float
            DFA threshold below which fEI is set to NaN.
        dfa_fit_interval: list or tuple, optional
            Fit interval for DFA if dfa_value is None.
        dfa_compute_interval: list or tuple, optional
            Compute interval for DFA if dfa_value is None.
        dfa_overlap: bool, optional
            Whether to use 50% overlap when computing DFA.

        Returns
        -------
        output: dict
            Dictionary with keys: fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF, dfa.
        """
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
            if dfa_fit_interval is None or dfa_compute_interval is None:
                raise ValueError(
                    "dfa_value not provided; dfa_fit_interval and dfa_compute_interval are required."
                )
            if dfa_overlap is None:
                dfa_overlap = True
            dfa_out = self._dfa_single(
                x,
                sampling_frequency=sampling_frequency,
                fit_interval=dfa_fit_interval,
                compute_interval=dfa_compute_interval,
                overlap=dfa_overlap,
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
            "dfa": dfa_value,
        }


    def _fei_multichannel(
        self,
        signal_matrix,
        sampling_frequency,
        window_size_sec,
        window_overlap,
        dfa_array,
        dfa_threshold,
        bad_idxes=None,
        dfa_fit_interval=None,
        dfa_compute_interval=None,
        dfa_overlap=None,
    ):
        """Compute fEI for a multichannel amplitude envelope.
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        Parameters
        ----------
        signal_matrix: np.ndarray
            2D array of shape (n_channels, n_timepoints) containing the amplitude envelope.
        sampling_frequency: float
            Sampling frequency of the signal (in Hz).
        window_size_sec: float
            Window size in seconds.
        window_overlap: float
            Fractional overlap between windows in [0, 1).
        dfa_array: np.ndarray or None
            DFA values per channel for thresholding. If None, DFA is computed internally per channel.
        dfa_threshold: float
            DFA threshold below which fEI is set to NaN.
        bad_idxes: list or None
            Channel indices to ignore.
        dfa_fit_interval: list or tuple, optional
            Fit interval for DFA if dfa_array is None.
        dfa_compute_interval: list or tuple, optional
            Compute interval for DFA if dfa_array is None.
        dfa_overlap: bool, optional
            Whether to use 50% overlap when computing DFA.

        Returns
        -------
        output: dict
            Dictionary with keys: fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF.
        """
        signal_matrix, _ = self._as_2d(signal_matrix)
        num_chans, length_signal = np.shape(signal_matrix)
        bad_idxes = [] if bad_idxes is None else list(bad_idxes)
        channels_to_ignore = set(bad_idxes)

        window_size = int(float(window_size_sec) * float(sampling_frequency))
        window_offset = int(np.floor(window_size * (1 - float(window_overlap))))
        num_windows = 0
        if window_size > 0 and window_offset > 0:
            num_windows = len(np.arange(0, length_signal - window_size, window_offset))

        fEI_val = np.full((num_chans, 1), np.nan, dtype=float)
        fEI_outliers_removed = np.full((num_chans, 1), np.nan, dtype=float)
        num_outliers = np.full((num_chans, 1), np.nan, dtype=float)
        wAmp = np.full((num_chans, num_windows), np.nan, dtype=float)
        wDNF = np.full((num_chans, num_windows), np.nan, dtype=float)

        for ch_idx in range(num_chans):
            if ch_idx in channels_to_ignore:
                continue
            dfa_value = None if dfa_array is None else float(dfa_array[ch_idx])
            out = self._fei_single(
                signal_matrix[ch_idx, :],
                sampling_frequency=sampling_frequency,
                window_size_sec=window_size_sec,
                window_overlap=window_overlap,
                dfa_value=dfa_value,
                dfa_threshold=dfa_threshold,
                dfa_fit_interval=dfa_fit_interval,
                dfa_compute_interval=dfa_compute_interval,
                dfa_overlap=dfa_overlap,
            )

            fEI_outliers_removed[ch_idx, 0] = out["fEI_outliers_removed"]
            fEI_val[ch_idx, 0] = out["fEI_val"]
            num_outliers[ch_idx, 0] = out["num_outliers"]

            w_amp = out["wAmp"]
            w_dnf = out["wDNF"]
            n = min(w_amp.shape[0], wAmp.shape[1])
            wAmp[ch_idx, :n] = w_amp[:n]
            n = min(w_dnf.shape[0], wDNF.shape[1])
            wDNF[ch_idx, :n] = w_dnf[:n]

        return {
            "fEI_outliers_removed": fEI_outliers_removed,
            "fEI_val": fEI_val,
            "num_outliers": num_outliers,
            "wAmp": wAmp,
            "wDNF": wDNF,
        }


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
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        If `frequency_range` or `spectrum_range` is provided, the method will:
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
            Frequency range [low, high] in Hz for spectrum DFA (bins computed internally).
        bad_idxes: list, optional
            Channel indices to ignore in multichannel inputs.
        input_is_envelope: bool, optional
            Whether the input is already an amplitude envelope. Only used for metadata.
        filter_kwargs: dict, optional
            Extra keyword arguments for MNE filter_data.
        trim_seconds: float, optional
            Seconds trimmed from both ends after filtering to reduce edge artifacts.
        hilbert_n_fft: int, optional
            FFT length for Hilbert transform. If None, uses next_fast_len.

        Returns
        -------
        out: dict
            For 1D envelope input: keys `dfa`, `window_sizes`, `fluctuations`, `dfa_intercept`.
            For multichannel envelope input: keys `DFA`, `window_sizes`, `fluctuations`, `dfa_intercept`.
            For band input: multichannel envelope output plus `frequency_range`, `fit_interval`, `compute_interval`.
            For spectrum input: keys `DFA`, `frequency_bins`.
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
            frequency_bins = self._get_frequency_bins(spectrum_range)
            num_chans = signal_matrix.shape[0]
            dfa_matrix = np.full((num_chans, len(frequency_bins)), np.nan, dtype=float)

            for idx_frequency, freq_bin in enumerate(frequency_bins):
                band_fit_interval = fit_interval or self._get_DFA_fitting_interval(freq_bin)
                band_compute_interval = compute_interval or band_fit_interval
                envelope = self._compute_band_envelope(
                    signal_matrix,
                    sampling_frequency,
                    freq_bin,
                    filter_kwargs=filter_kwargs,
                    trim_seconds=trim_seconds,
                    hilbert_n_fft=hilbert_n_fft,
                )
                out_band = self._dfa_multichannel(
                    envelope,
                    sampling_frequency=sampling_frequency,
                    fit_interval=band_fit_interval,
                    compute_interval=band_compute_interval,
                    overlap=overlap,
                    bad_idxes=bad_idxes,
                )
                dfa_matrix[:, idx_frequency] = out_band["DFA"]

            return {
                "DFA": dfa_matrix,
                "frequency_bins": frequency_bins,
            }

        band_fit_interval = fit_interval or self._get_DFA_fitting_interval(frequency_range)
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
        Adapted from crosci module (https://github.com/Critical-Brain-Dynamics/crosci/tree/main).

        If `frequency_range` or `spectrum_range` is provided, the method will:
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
            Window size in seconds for fEI.
        window_overlap: float, optional
            Fractional overlap between windows in [0, 1). Default 0.0.
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
            Frequency range [low, high] in Hz for spectrum fEI (bins computed internally).
        bad_idxes: list, optional
            Channel indices to ignore in multichannel inputs.
        input_is_envelope: bool, optional
            Whether the input is already an amplitude envelope. Only used for metadata.
        filter_kwargs: dict, optional
            Extra keyword arguments for MNE filter_data.
        trim_seconds: float, optional
            Seconds trimmed from both ends after filtering to reduce edge artifacts.
        hilbert_n_fft: int, optional
            FFT length for Hilbert transform. If None, uses next_fast_len.

        Returns
        -------
        out: dict
            For 1D envelope input: keys `fEI_outliers_removed`, `fEI_val`, `num_outliers`, `wAmp`, `wDNF`, `dfa`.
            For multichannel envelope input: keys `fEI_outliers_removed`, `fEI_val`, `num_outliers`, `wAmp`, `wDNF`,
            plus `DFA`.
            For band input: multichannel envelope output plus `frequency_range`, `fit_interval`, `compute_interval`,
            and `fEI` (outliers-removed).
            For spectrum input: keys `DFA`, `fEI`, `frequency_bins`.
        """
        sampling_frequency = self._resolve_sampling_frequency(sampling_frequency)
        window_size_sec = self._resolve_param("window_size_sec", window_size_sec, None)
        window_overlap = self._resolve_param("window_overlap", window_overlap, 0.0)
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
            frequency_bins = self._get_frequency_bins(spectrum_range)
            num_chans = signal_matrix.shape[0]
            dfa_matrix = np.full((num_chans, len(frequency_bins)), np.nan, dtype=float)
            fei_matrix = np.full((num_chans, len(frequency_bins)), np.nan, dtype=float)

            for idx_frequency, freq_bin in enumerate(frequency_bins):
                band_fit_interval = dfa_fit_interval or self._get_DFA_fitting_interval(freq_bin)
                band_compute_interval = dfa_compute_interval or band_fit_interval
                envelope = self._compute_band_envelope(
                    signal_matrix,
                    sampling_frequency,
                    freq_bin,
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
                dfa_matrix[:, idx_frequency] = dfa_array
                fei_matrix[:, idx_frequency] = np.squeeze(fei_out["fEI_outliers_removed"])

            return {
                "DFA": dfa_matrix,
                "fEI": fei_matrix,
                "frequency_bins": frequency_bins,
            }

        band_fit_interval = dfa_fit_interval or self._get_DFA_fitting_interval(frequency_range)
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
        fei_out["fEI"] = np.squeeze(fei_out["fEI_outliers_removed"])
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
