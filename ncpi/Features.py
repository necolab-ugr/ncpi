import os
import numpy as np
import scipy.signal as scipy_signal
from ncpi import tools


### Worker helper functions ###
_WORKER_FEATURES_OBJ = None
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

            raise ValueError(f"Unknown method: {_WORKER_FEATURES_OBJ.method}")

        x = (x - mu) / sigma

    if _WORKER_FEATURES_OBJ.method == "catch22":
        return _WORKER_FEATURES_OBJ.catch22(x)

    if _WORKER_FEATURES_OBJ.method == "specparam":
        # specparam reads parameters from self.params via _resolve_param
        return _WORKER_FEATURES_OBJ.specparam(sample=x)

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
        if method not in ['catch22', 'specparam']:
            raise ValueError("Invalid method. Please use 'catch22' or 'specparam'.")

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
    ):
        """
        Parameterize a power spectrum using specparam.SpectralModel.

        You can pass either:
          - sample (time series) + fs (Welch power spectrum is computed internally), OR
          - freqs + power_spectrum (already-computed PSD)

        Parameter precedence:
          explicit argument > self.params[...] > internal default

        Returns
        -------
        out : dict
            Dictionary with aperiodic parameters, selected peak parameters, plus fit metrics.
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

        # -------------------------
        # Validate mutually exclusive inputs
        # -------------------------
        has_ts = sample is not None
        has_freqs = freqs is not None
        has_psd = power_spectrum is not None

        if has_ts:
            # If using time-series mode, PSD inputs must NOT be provided
            if has_freqs or has_psd:
                raise ValueError(
                    "Pass either `sample` (+ `fs`) OR (`freqs` and `power_spectrum`), not both."
                )
        else:
            # If not using time-series mode, require BOTH PSD inputs
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
            # Default to 0.5 s segments if not provided
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

        # Ensure freq_range is sane and clipped to available freqs (avoids fit edge issues)
        fmin, fmax = map(float, freq_range)
        if fmin >= fmax:
            raise ValueError("`freq_range` must be (fmin, fmax) with fmin < fmax.")
        # Clip to the PSD support
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
        metrics = base_model_kwargs.get("metrics", None)
        metric_values = fm.results.metrics.results

        if metrics:
            metrics_out = {m: float(metric_values.get(m, np.nan)) for m in metrics}
        else:
            metrics_out = {"gof_rsquared": float(metric_values.get("gof_rsquared", np.nan))}

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
        }

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

