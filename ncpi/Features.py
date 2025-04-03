import os
import numpy as np
from numpy.matlib import repmat # for compatibility with the original fE/I code
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt
from ncpi import tools


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
        if method not in ['catch22', 'power_spectrum_parameterization', 'fEI', 'DFA', 'hctsa']:
            raise ValueError("Invalid method. Please use 'catch22', 'power_spectrum_parameterization', 'fEI', 'DFA', "
                             "or 'hctsa'.")

        # Import the required modules based on the method
        if method == 'catch22':
            if not tools.ensure_module("pycatch22"):
                raise ImportError("pycatch22 is required for computing catch22 features but is not installed.")
            self.pycatch22 = tools.dynamic_import("pycatch22")

        elif method == 'power_spectrum_parameterization':
            if not tools.ensure_module("fooof"):
                raise ImportError("fooof is required for computing power_spectrum_parameterization features but is "
                                  "not installed.")
            self.FOOOF = tools.dynamic_import("fooof", "FOOOF")

        elif method == 'fEI' or method == 'DFA':
            if not tools.ensure_module("PyAstronomy"):
                raise ImportError("PyAstronomy is required for computing fEI or DFA features but is not installed.")
            self.generalizedESD = tools.dynamic_import("PyAstronomy.pyasl", "generalizedESD")

        elif method == 'hctsa':
            if not tools.ensure_module("matlab"):
                raise ImportError("matlab is required for computing hctsa features but is not installed.")
            if not tools.ensure_module("matlabengine"):
                raise ImportError("matlabengine is required for computing hctsa features but is not installed.")
            if not tools.ensure_module("h5py"):
                raise ImportError("h5py is required for computing hctsa features but is not installed.")
            self.matlab = tools.dynamic_import("matlab")
            self.matlabengine = tools.dynamic_import("matlab","engine")
            self.h5py = tools.dynamic_import("h5py")

        # Check if params is a dictionary
        if not isinstance(params, dict) and params is not None:
            raise ValueError("params must be a dictionary.")

        self.method = method
        self.params = params

        # Check if pathos is installed. If not, use the default Python multiprocessing library
        if not tools.ensure_module("pathos"):
            self.pathos_inst = False
            self.multiprocessing = tools.dynamic_import("multiprocessing")
        else:
            self.pathos_inst = True
            self.pathos = tools.dynamic_import("pathos", "pools")

        # Check if tqdm is installed
        if not tools.ensure_module("tqdm"):
            self.tqdm_inst = False
        else:
            self.tqdm_inst = True
            self.tqdm = tools.dynamic_import("tqdm", "tqdm")

    def catch22(self,sample):
        """
        Compute the catch22 features from a time-series sample.

        Parameters
        ----------
        sample: np.array
            Sample data.

        Returns
        -------
        features: np.array
            Array with the catch22 feature values.
        """

        features = self.pycatch22.catch22_all(sample)

        return features['values']

    def power_spectrum_parameterization(self, sample, fs, fmin, fmax, fooof_setup, r_squared_th=0.9,
                                        freq_range=[30., 200.], nperseg=-1, compute_knee=False):
        """
        Power spectrum parameterization of a time-series sample using the FOOOF algorithm.

        Parameters
        ----------
        sample: np.array
            Times-series sample.
        fs: float
            Sampling frequency.
        fmin: float
            Minimum frequency for the power spectrum fit.
        fmax: float
            Maximum frequency for the power spectrum fit.
        fooof_setup: dict
            Dictionary containing the parameters for the FOOOF algorithm.
                - peak_threshold: float
                - min_peak_height: float
                - max_n_peaks: int
                - peak_width_limits: tuple
        r_squared_th: float
            Threshold for the r_squared value. Default is 0.9.
        freq_range: list
            Frequency range for the search of the peak parameters. Default is [30., 200.].
        nperseg: int
            Length of each segment for the power spectrum. Default is -1 (half a second).
        compute_knee: bool
            If True, the knee parameter is computed. Default is False.

        Returns
        -------
        features: np.array
            Array with the aperiodic components and peak parameters, plus the mean power:
            features[0:2] = aperiodic_params_fixed
            features[2:5] = peak_params_fixed
            features[5:8] = aperiodic_params_knee
            features[8:11] = peak_params_knee
            features[11] = mean power
        """

        debug = False
        features = np.full(12, np.nan)

        # Check that the length of the sample is at least 2 seconds
        if len(sample) >= 2 * fs:
            # Estimate power spectral density using Welch’s method
            if nperseg == -1:
                fxx, Pxx = welch(sample, fs, nperseg=int(0.5 * fs))
            else:
                fxx, Pxx = welch(sample, fs, nperseg=nperseg)

            if fmin >= fxx[0] and fmax <= fxx[-1]:
                f1 = np.where(fxx >= fmin)[0][0]
                f2 = np.where(fxx >= fmax)[0][0]
            else:
                print(
                    'Warning: fmin and fmax are out of the frequency range of the power spectrum. Adjusting fmin and fmax '
                    'to the minimum and maximum frequencies of the power spectrum.')
                f1 = fxx[0]
                f2 = fxx[-1]

            # Ensure the input data has no 0s
            if not np.any(Pxx == 0):
                # Fit the power spectrum using FOOOF for both aperiodic modes (fixed and knee)
                for ii, aperiodic_mode in enumerate(['fixed', 'knee'] if compute_knee else ['fixed']):
                    fm = self.FOOOF(peak_threshold=fooof_setup['peak_threshold'],
                               min_peak_height=fooof_setup['min_peak_height'],
                               max_n_peaks=fooof_setup['max_n_peaks'],
                               aperiodic_mode=aperiodic_mode,
                               peak_width_limits=fooof_setup['peak_width_limits'])
                    try:
                        fm.fit(fxx[f1:f2], Pxx[f1:f2])
                    except:
                        print('Error fitting the power spectrum.')
                        return np.full(12, np.nan)

                    # Discard fits with negative exponents
                    if fm.aperiodic_params_[-1] <= 0.:
                        fm.r_squared_ = 0.
                    # Discard nan r_squared
                    if np.isnan(fm.r_squared_):
                        fm.r_squared_ = 0.

                    # Print parameters and plot the fit
                    if debug:
                        print('fm.aperiodic_params_ = ', fm.aperiodic_params_)
                        print('fm.peak_params_ = ', fm.peak_params_)
                        print('fm.r_squared_ = ', fm.r_squared_)

                        fm.plot(plot_peaks='shade', peak_kwargs={'color': 'green'})

                        plt.title(f'aperiodic_params = {fm.aperiodic_params_}\n'
                                  f'peak_params = {fm.peak_params_}\n'
                                  f'r_squared = {fm.r_squared_}', fontsize=12)
                        plt.show()

                    # Collect the aperiodic and peak parameters
                    if fm.r_squared_ >= r_squared_th:
                        if ii == 0:
                            features[0:2] = fm.aperiodic_params_
                        else:
                            features[5:8] = fm.aperiodic_params_

                        if fm.peak_params_ is not None:
                            # Find peaks within the frequency range
                            pos_freq = np.where((fm.peak_params_[:, 0] >= freq_range[0]) &
                                                (fm.peak_params_[:, 0] <= freq_range[1]))[0]
                            # spectral shapes that have an oscillatory peak
                            if len(pos_freq) > 0:
                                peak_params = fm.peak_params_[pos_freq, :]

                                # Find the peak with the maximum power
                                pos_max = np.argmax(peak_params[:, 1])
                                peak_freq = peak_params[pos_max, 0]
                                peak_power = peak_params[pos_max, 1]
                                peak_BW = peak_params[pos_max, 2]

                                if ii == 0:
                                    features[2:5] = [peak_freq, peak_power, peak_BW]
                                else:
                                    features[8:11] = [peak_freq, peak_power, peak_BW]

                # Collect mean power
                features[11] = np.mean(Pxx[f1:f2])

        return features

    def _create_window_indices(self, length_signal, length_window, window_offset):

        """
        Function to create window indices for a signal.

        This code was downloaded from https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under
        creative commons license CC-BY-NC https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

        Parameters
        ----------
        length_signal: int
            Length of the signal.
        length_window: int
            Length of the window.
        window_offset: int
            Offset for the window.

        Returns
        -------
        all_window_index: np.array
            Array with the window indices.
        """

        window_starts = np.arange(0, length_signal - length_window, window_offset)
        num_windows = len(window_starts)

        one_window_index = np.arange(0, length_window)
        all_window_index = repmat(one_window_index, num_windows, 1).astype(int)

        all_window_index = all_window_index + repmat(np.transpose(window_starts[np.newaxis, :]), 1,
                                                     length_window).astype(int)

        return all_window_index

    def fEI(self, signal, sampling_frequency, window_size_sec, window_overlap, DFA_array, bad_idxes=[]):
        """
        Calculates fEI (on a set window size) for a signal

            Steps refer to description of fEI algorithm in Figure 2D of paper:
              Measurement of excitation inhibition ratio in autism spectrum disorder using critical brain dynamics
              Scientific Reports (2020)
              Hilgo Bruining*, Richard Hardstone*, Erika L. Juarez-Martinez*, Jan Sprengers*, Arthur-Ervin Avramiea,
              Sonja Simpraga, Simon J. Houtman, Simon-Shlomo Poil5, Eva Dallares, Satu Palva, Bob Oranje, J. Matias Palva,
              Huibert D. Mansvelder & Klaus Linkenkaer-Hansen
              (*Joint First Author)

        This code was downloaded from https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under
        creative commons license CC-BY-NC https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

        Originally created by Richard Hardstone (2020), rhardstone@gmail.com. Please note that commercial use of this
        algorithm is protected by Patent claim (PCT/NL2019/050167) “Method of determining brain activity”,
        with priority date 16 March 2018.


        Parameters
        ----------
        signal: array, shape(n_channels,n_times)
            amplitude envelope for all channels
        sampling_frequency: integer
            sampling frequency of the signal
        window_size_sec: float
            window size in seconds
        window_overlap: float
            fraction of overlap between windows (0-1)
        DFA_array: array, shape(n_channels)
            array of DFA values, with corresponding value for each channel, used for thresholding fEI
        bad_idxes: array, shape(n_channels)
            channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
            case in which all channels are computed

        Returns
        -------
        fEI_outliers_removed: array, shape(n_channels)
            fEI values, with outliers removed
        fEI_val: array, shape(n_channels)
            fEI values, with outliers included
        num_outliers: integer
            number of detected outliers
        wAmp: array, shape(n_channels, num_windows)
            windowed amplitude, computed across all channels/windows
        wDNF: array, shape(n_channels, num_windows)
            windowed detrended normalized fluctuation, computed across all channels/windows
        """

        window_size = int(window_size_sec * sampling_frequency)

        num_chans = np.shape(signal)[0]
        length_signal = np.shape(signal)[1]

        channels_to_ignore = [False] * num_chans

        for bad_idx in bad_idxes:
            channels_to_ignore[bad_idx] = True

        window_offset = int(np.floor(window_size * (1 - window_overlap)))
        all_window_index = _create_window_indices(length_signal, window_size, window_offset)
        num_windows = np.shape(all_window_index)[0]

        fEI_val = np.zeros((num_chans, 1))
        fEI_val[:] = np.NAN
        fEI_outliers_removed = np.zeros((num_chans, 1))
        fEI_outliers_removed[:] = np.NAN
        num_outliers = np.zeros((num_chans, 1))
        num_outliers[:] = np.NAN
        wAmp = np.zeros((num_chans, num_windows))
        wAmp[:] = np.NAN
        wDNF = np.zeros((num_chans, num_windows))
        wDNF[:] = np.NAN

        for ch_idx in range(num_chans):
            if channels_to_ignore[ch_idx]:
                continue

            original_amp = signal[ch_idx, :]

            if np.min(original_amp) == np.max(original_amp):
                print('Problem computing fEI for channel idx ' + str(ch_idx))
                continue

            signal_profile = np.cumsum(original_amp - np.mean(original_amp))
            w_original_amp = np.mean(original_amp[all_window_index], axis=1)

            x_amp = repmat(np.transpose(w_original_amp[np.newaxis, :]), 1, window_size)
            x_signal = signal_profile[all_window_index]
            x_signal = np.divide(x_signal, x_amp)

            # Calculate local trend, as the line of best fit within the time window
            _, fluc, _, _, _ = np.polyfit(np.arange(window_size), np.transpose(x_signal), deg=1, full=True)
            # Convert to root-mean squared error, from squared error
            w_detrendedNormalizedFluctuations = np.sqrt(fluc / window_size)

            fEI_val[ch_idx] = 1 - np.corrcoef(w_original_amp, w_detrendedNormalizedFluctuations)[0, 1]

            gesd_alpha = 0.05
            max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
            # smallest value for max number of outliers is 2 for generalizedESD
            max_num_outliers = max(int(np.round(max_outliers_percentage * len(w_original_amp))), 2)
            outlier_indexes_wAmp = self.generalizedESD(w_original_amp, max_num_outliers, gesd_alpha)[1]
            outlier_indexes_wDNF = self.generalizedESD(w_detrendedNormalizedFluctuations, max_num_outliers, gesd_alpha)[1]
            outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
            num_outliers[ch_idx, :] = len(outlier_union)
            not_outlier_both = np.setdiff1d(np.arange(len(w_original_amp)), np.array(outlier_union))
            fEI_outliers_removed[ch_idx] = 1 - np.corrcoef(w_original_amp[not_outlier_both], \
                                                           w_detrendedNormalizedFluctuations[not_outlier_both])[0, 1]

            wAmp[ch_idx, :] = w_original_amp
            wDNF[ch_idx, :] = w_detrendedNormalizedFluctuations

        fEI_val[DFA_array <= 0.6] = np.nan
        fEI_outliers_removed[DFA_array <= 0.6] = np.nan

        return (fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF)

    def DFA(self, signal, sampling_frequency, fit_interval, compute_interval, overlap=True, bad_idxes=[]):
        """ Calculates DFA of a signal.

        This code was downloaded from https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under
        creative commons license CC-BY-NC https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

        Parameters
        ----------
        signal: array, shape(n_channels,n_times)
            amplitude envelope for all channels
        sampling_frequency: integer
            sampling frequency of the signal
        fit_interval: list, length 2
            interval (in seconds) over which the DFA exponent is fit. should be included in compute_interval
        compute_interval: list, length 2
            interval (in seconds) over which DFA is computed
        overlap: boolean
            if set to True, then windows are generated with an overlap of 50%
        bad_idxes: array, shape(n_channels)
            channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
            case in which all channels are computed

        Returns
        -------
        dfa_array, window_sizes, fluctuations, dfa_intercept
        dfa_array: array, shape(n_channels)
            DFA value for each channel
        window_sizes: array, shape(num_windows)
            window sizes over which the fluctuation function is computed
        fluctuations: array, shape(num_windows)
            fluctuation function value at each computed window size
        dfa_intercept: array, shape(n_channels)
            DFA intercept for each channel
        """

        num_chans, num_timepoints = np.shape(signal)

        channels_to_ignore = [False] * num_chans
        for bad_idx in bad_idxes:
            channels_to_ignore[bad_idx] = True

        length_signal = np.shape(signal)[1]

        assert fit_interval[0] >= compute_interval[0] and fit_interval[1] <= compute_interval[
            1], 'CalcInterval should be included in ComputeInterval'
        assert compute_interval[0] >= 0.1 and compute_interval[
            1] <= 1000, 'ComputeInterval should be between 0.1 and 1000 seconds'
        assert compute_interval[1] / sampling_frequency <= num_timepoints, \
            'ComputeInterval should not extend beyond the length of the signal'

        # compute DFA window sizes for the given CalcInterval
        window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(
            int)  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds

        # make sure there are no duplicates after rounding
        window_sizes = np.sort(np.unique(window_sizes))

        window_sizes = window_sizes[(window_sizes >= compute_interval[0] * sampling_frequency) & \
                                    (window_sizes <= compute_interval[1] * sampling_frequency)]

        dfa_array = np.zeros(num_chans)
        dfa_array[:] = np.NAN
        dfa_intercept = np.zeros(num_chans)
        dfa_intercept[:] = np.NAN
        fluctuations = np.zeros((num_chans, len(window_sizes)))
        fluctuations[:] = np.NAN

        if max(window_sizes) <= num_timepoints:
            for ch_idx in range(num_chans):
                if channels_to_ignore[ch_idx]:
                    continue

                signal_for_channel = signal[ch_idx, :]

                for i_window_size in range(len(window_sizes)):
                    if overlap == True:
                        window_overlap = 0.5
                    else:
                        window_overlap = 0

                    window_size = window_sizes[i_window_size]
                    window_offset = np.floor(window_size * (1 - window_overlap))
                    all_window_index = _create_window_indices(length_signal, window_sizes[i_window_size], window_offset)
                    # First we convert the time series into a series of fluctuations y(i) around the mean.
                    demeaned_signal = signal_for_channel - np.mean(signal_for_channel)
                    # Then we integrate the above fluctuation time series ('y').
                    signal_profile = np.cumsum(demeaned_signal)

                    x_signal = signal_profile[all_window_index]

                    # Calculate local trend, as the line of best fit within the time window -> fluc is the sum of
                    # squared residuals
                    _, fluc, _, _, _ = np.polyfit(np.arange(window_size), np.transpose(x_signal), deg=1, full=True)

                    # Peng's formula - Convert to root-mean squared error, from squared error
                    # det_fluc = np.sqrt(np.mean(fluc / window_size))
                    # Richard's formula
                    det_fluc = np.mean(np.sqrt(fluc / window_size))
                    fluctuations[ch_idx, i_window_size] = det_fluc

                # get the positions of the first and last window sizes used for fitting
                fit_interval_first_window = np.argwhere(window_sizes >= fit_interval[0] * sampling_frequency)[0][0]
                fit_interval_last_window = np.argwhere(window_sizes <= fit_interval[1] * sampling_frequency)[-1][0]

                # take the previous to the first window size if the difference between the lower end of fitting and
                # the previous window is no more than 1% of the lower end of fitting and if the difference between the lower
                # end of fitting and the previous window is less than the difference between the lower end of fitting and
                # the current first window
                if (np.abs(window_sizes[fit_interval_first_window - 1] / sampling_frequency - fit_interval[0]) <=
                        fit_interval[0] / 100):
                    if np.abs(window_sizes[fit_interval_first_window - 1] / sampling_frequency - fit_interval[0]) < \
                            np.abs(window_sizes[fit_interval_first_window] / sampling_frequency - fit_interval[0]):
                        fit_interval_first_window = fit_interval_first_window - 1

                x = np.log10(window_sizes[fit_interval_first_window:fit_interval_last_window + 1])
                y = np.log10(fluctuations[ch_idx, fit_interval_first_window:fit_interval_last_window + 1])
                model = np.polyfit(x, y, 1)
                dfa_intercept[ch_idx] = model[1]
                dfa_array[ch_idx] = model[0]

        return (dfa_array, window_sizes, fluctuations, dfa_intercept)

    def hctsa(self, samples, hctsa_folder, workers=32):
        """
        Compute hctsa features.

        Parameters
        ----------
        samples: ndarray/list of shape (n_samples, times-series length)
            A set of samples of time-series data.
        hctsa_folder: str
            Folder where hctsa is installed.
        workers: int
            Number of MATLAB workers of the parallel pool.

        Returns
        -------
        feats: list of shape (n_samples, n_features)
            hctsa features.

        Debugging
        ---------
        This function has been debugged by approximating results shown
        in https://github.com/benfulcher/hctsaTutorial_BonnEEG.
        """

        feats = []

        # start Matlab engine
        print("\n--> Starting Matlab engine ...")
        eng = self.matlabengine.start_matlab()

        try:
            # Remove hctsa file
            if os.path.isfile(os.path.join(hctsa_folder, 'HCTSA.mat')):
                os.remove(os.path.join(hctsa_folder, 'HCTSA.mat'))

            # Change to hctsa folder
            eng.cd(hctsa_folder)

            # Startup hctsa script
            print("\n--> hctsa startup ...")
            st = eng.startup(nargout=0)
            print(st)

            # Check if samples is a list and convert it to a numpy array
            if isinstance(samples, list):
                samples = np.array(samples)

            # Create the input variables in Matlab
            eng.eval(f'timeSeriesData = cell(1,{samples.shape[0]});', nargout=0)
            eng.eval(f'labels = cell(1,{samples.shape[0]});', nargout=0)
            eng.eval(f'keywords = cell(1,{samples.shape[0]});', nargout=0)

            # Transfer time-series data to Matlab workspace
            for s in range(samples.shape[0]):
                eng.workspace['aux'] = self.matlab.double(list(samples[s]))
                eng.eval('timeSeriesData{1,%s} = aux;' % (s + 1), nargout=0)

            # Fill in the other 2 Matlab structures with the index of the sample
            for s in range(samples.shape[0]):
                eng.eval('labels{1,%s} = \'%s\';' % (str(s + 1), str(s + 1)), nargout=0)
                eng.eval('keywords{1,%s} = \'%s\';' % (str(s + 1), str(s + 1)), nargout=0)

            # Save variables into a mat file
            eng.eval('save INP_ccpi_ts.mat timeSeriesData labels keywords;', nargout=0)

            # Load mat file
            eng.eval('load INP_ccpi_ts.mat;', nargout=0)

            # Initialize an hctsa calculation
            print("\n--> hctsa TS_Init ...")
            eng.TS_Init('INP_ccpi_ts.mat',
                        'hctsa',
                        self.matlab.logical([False, False, False]),
                        nargout=0)

            # Open a parallel pool of a specific size
            if workers > 1:
                eng.parpool(workers)

            # Compute features
            print("\n--> hctsa TS_Compute ...")
            # eng.TS_Compute(matlab.logical([True]),nargout = 0)
            eng.eval('TS_Compute(true);', nargout=0)

            # Load hctsa file
            f = self.h5py.File(os.path.join(hctsa_folder, 'HCTSA.mat'), 'r')
            TS_DataMat = np.array(f.get('TS_DataMat'))
            # TS_Quality = np.array(f.get('TS_Quality'))

            # Create the array of features to return
            print(f'\n--> Formatting {TS_DataMat.shape[0]} features...')
            for s in range(samples.shape[0]):
                feats.append(list(TS_DataMat[:, s]))

            # Stop Matlab engine
            print("\n--> Stopping Matlab engine ...")
            eng.quit()

        except Exception as e:
            print(f"An error occurred: {e}")
            # Stop Matlab engine
            print("\n--> Stopping Matlab engine ...")
            eng.quit()
            raise e

        return feats


    def compute_features(self, data, hctsa_folder = None):
        """
        Function to compute features from the data.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing the data. The time-series samples must be in the 'Data' column.
        hctsa_folder: str
            Folder where hctsa is installed (if method is 'hctsa').

        Returns
        -------
        data: pd.DataFrame
            DataFrame containing the data with the features appended.
        """

        def process_batch(batch_tuple):
            """
            Function to process a batch of samples.

            Parameters
            ----------
            batch: list
                List of samples.

            Returns
            -------
            features: list
                List of features.
            """

            batch_index, batch = batch_tuple
            # Normalize the batch
            if self.method != 'fEI':
                batch = [(sample - np.mean(sample)) / np.std(sample) for sample in batch]
            features = []

            # Compute features of each sample in the batch
            if self.method != 'hctsa':
                for sample in batch:
                    if self.method == 'catch22':
                        features.append(self.catch22(sample))
                    elif self.method == 'power_spectrum_parameterization':
                        features.append(self.power_spectrum_parameterization(sample,
                                                                        self.params['fs'],
                                                                        self.params['fmin'],
                                                                        self.params['fmax'],
                                                                        self.params['fooof_setup'],
                                                                        self.params['r_squared_th']))
                    elif self.method == 'DFA':
                        features.append(self.DFA(sample,
                                            self.params['fs'],
                                            self.params['fit_interval'],
                                            self.params['compute_interval'],
                                            self.params['overlap'],
                                            self.params['bad_idxes']))

                    elif self.method == 'fEI':
                        features.append(self.fEI(sample,
                                            self.params['fs'],
                                            self.params['window_size_sec'],
                                            self.params['window_overlap'],
                                            self.params['DFA_array'],
                                            self.params['bad_idxes']))

            # Compute hctsa for the whole batch to avoid starting the Matlab engine multiple times
            else:
                if self.method == 'hctsa':
                    features = self.hctsa(batch,hctsa_folder)

            return batch_index,features

        # Split the data into batches using the number of available CPUs
        num_cpus = os.cpu_count()
        if self.method == 'hctsa':
            factor = 0.5 # decrease this factor to avoid memory issues with MATLAB engine
        else:
            factor = 10 # more chunks than available CPUs (10 is a factor to update the progress bar more frequently)

        batch_size = len(data['Data']) // int(num_cpus*factor)
        if batch_size == 0:
            batch_size = 1
        batches = [(i, data['Data'][i:i + batch_size]) for i in range(0, len(data['Data']), batch_size)]

        # Compute the features in parallel using all available CPUs with pathos
        if self.pathos_inst:
            with self.pathos.ProcessPool(num_cpus) as pool:
                if self.tqdm_inst:
                    results = list(self.tqdm(pool.imap(process_batch, batches), total=len(batches),
                                        desc="Computing features"))
                else:
                    results = list(pool.imap(process_batch, batches))
        # If pathos is not installed, use the default Python multiprocessing library
        else:
            with self.multiprocessing.Pool(num_cpus) as pool:
                if self.tqdm_inst:
                    results = list(self.tqdm(pool.imap(process_batch, batches), total=len(batches),
                                        desc="Computing features"))
                else:
                    results = list(pool.imap(process_batch, batches))

        # Sort the features based on the original index
        results.sort(key=lambda x: x[0])
        features = [feature for _, batch_features in results for feature in batch_features]

        # Append the features to the DataFrame
        pd_feat = pd.DataFrame({'Features': features})
        data = pd.concat([data, pd_feat], axis=1)
        return data