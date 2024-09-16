import importlib
import signal as sgnl
import subprocess
import pandas as pd
import os
import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt


def install(module_name):
    """
    Function to install a Python module.

    Parameters
    ----------
    module_name: str
        Module name.
    """
    subprocess.check_call(['pip', 'install', module_name])
    print(f"The module {module_name} was installed!")


def module(module_name):
    """
    Function to dynamically import a Python module.

    Parameters
    ----------
    module_name: str
        Name of the module to import.

    Returns
    -------
    module
        The imported module.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"{module_name} is not installed!")
        install(module_name)
        module = importlib.import_module(module_name)
    return module


def catch22(sample):
    """
    Function to compute the catch22 features from a time-series sample.

    Parameters
    ----------
    sample: np.array
        Sample data.

    Returns
    -------
    features: np.array
        Array with the catch22 features.
    """

    # Dynamically import the pycatch22 module
    pycatch22 = module('pycatch22')

    # Compute the catch22 features
    features = pycatch22.catch22_all(sample)

    return features['values']

def power_spectrum_parameterization(sample,fs,fmin,fmax,fooof_setup,r_squared_th = 0.9):
    """
    Function to compute the power spectrum parameterization of a time-series sample using the FOOOF algorithm.

    Parameters
    ----------
    sample: np.array
        Times-series sample.
    fs: float
        Sampling frequency.
    fmin: float
        Minimum frequency for the power spectrum.
    fmax: float
        Maximum frequency for the power spectrum.
    fooof_setup: dict
        Dictionary containing the parameters for the FOOOF algorithm.
            - peak_threshold: float
            - min_peak_height: float
            - max_n_peaks: int
            - peak_width_limits: tuple
    r_squared_th: float
        Threshold for the r_squared value. Default is 0.9.

    Returns
    -------
    features: np.array
        Array with the aperiodic and peak parameters.
    """

    debug=False

    # Dynamically import the fooof module
    # from fooof import FOOOF
    FOOOF = getattr(module('fooof'), 'FOOOF')

    # Check that the length of the sample is at least 2 seconds
    if len(sample) >= 2 * fs:
        # Estimate power spectral density using Welch’s method
        fxx, Pxx = welch(sample, fs, nperseg=int(0.2*fs))

        if fmin >= fxx[0] and fmax <= fxx[-1]:
            f1 = np.where(fxx >= fmin)[0][0]
            f2 = np.where(fxx >= fmax)[0][0]
        else:
            print('Warning: fmin and fmax are out of the frequency range of the power spectrum.')
            f1 = fxx[0]
            f2 = fxx[-1]

        # Ensure the input data has no 0s
        if not np.any(Pxx == 0):
            # Fit the power spectrum using FOOOF
            fm = FOOOF(peak_threshold=fooof_setup['peak_threshold'],
                       min_peak_height=fooof_setup['min_peak_height'],
                       max_n_peaks=fooof_setup['max_n_peaks'],
                       aperiodic_mode='fixed',
                       peak_width_limits=fooof_setup['peak_width_limits'])
            try:
                fm.fit(fxx[f1:f2], Pxx[f1:f2])
            except:
                print('Error fitting the power spectrum.')
                return np.full(2, np.nan)

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

                fm.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})

                plt.title(f'aperiodic_params = {fm.aperiodic_params_}\n'
                         f'peak_params = {fm.peak_params_}\n'
                         f'r_squared = {fm.r_squared_}', fontsize=12)
                plt.show()

            # Concatenate the aperiodic and peak parameters
            if fm.peak_params_ is None:
                features = fm.aperiodic_params_
            else:
                features = np.concatenate((fm.aperiodic_params_, fm.peak_params_.flatten()))

            if fm.r_squared_ < r_squared_th:
                return np.full_like(features, np.nan)
            else:
                return features
        else:
            return np.full(2, np.nan)
    else:
        return np.full(2, np.nan)

def bandpass(sample, fmin, fmax, fs):
    """
    Function to bandpass filter a time-series sample.

    Parameters
    ----------
    sample: np.array
        Times-series sample.
    fmin: float
        Minimum frequency for the bandpass filter.
    fmax: float
        Maximum frequency for the bandpass filter.
    fs: int
        Sampling frequency.

    Returns
    -------
    sample_bandpassed: np.array
        Bandpassed sample.
    """

    # Compute the bandpass filter
    b, a = butter(4, [fmin/(fs/2), fmax/(fs/2)], 'band')
    sample_bandpassed = filtfilt(b, a, sample)

    return sample_bandpassed


def fEI(samples,fs,fmin,fmax,fEI_folder):
    """
    Function to compute the fEI from a list of time-series samples.

    Parameters
    ----------
    samples: list
        List of time-series samples.
    fs: float
        Sampling frequency.
    fmin: float
        Minimum frequency for the bandpass filter.
    fmax: float
        Maximum frequency for the bandpass filter.
    fEI_folder: str
        Path to the folder containing the fEI function.

    Returns
    -------
    features: list
        List of fEI features.
    """

    debug = False

    # Compute the amplitude envelope of the signals
    envelopes = []
    for sample in samples:
        # Bandpass the signal
        sample_alpha = bandpass(sample, fmin, fmax, fs)
        # Compute the amplitude envelope using the Hilbert transform
        amplitude_envelope = np.abs(hilbert(sample_alpha))
        envelopes.append(list(amplitude_envelope))

    # Start the matlab engine
    matlab_engine = module('matlab.engine')
    try:
        eng = matlab_engine.start_matlab()

        # # Start a parallel pool with the number of available physical cores
        # num_cpus = os.cpu_count()/2
        # eng.parpool(num_cpus)

        # Import matlab
        matlab = module('matlab')

        # Add the folder containing the fEI function to the Matlab path
        eng.addpath(fEI_folder)

        features = []
        for i, envelope in enumerate(envelopes):
            # Compute fEI
            try:
                eng.workspace['aux'] = matlab.double(envelope)
                eng.eval(f'signal = zeros({len(envelope)},1);', nargout=0)
                eng.eval(f'signal(:,1) = aux;', nargout=0)
                eng.eval(f'[EI, wAmp, wDNF] = calculateFEI(signal,{int(len(envelope) / 10)},0.8);',
                         nargout=0)
                fEI = eng.workspace['EI']
            except:
                fEI = np.nan
            features.append(fEI)

            # Plot the amplitude envelope over the original signal
            if debug:
                plt.figure()
                plt.plot(bandpass(samples[i], fmin, fmax, fs))
                plt.plot(envelope)
                plt.title(f'fEI = {fEI}')
                plt.show()

    finally:
        # Stop the MATLAB engine
        eng.quit()
        # Delete the engine object
        del eng

    return features


    #     # Compute fEI for each sample
    #     eng.workspace['envelopes'] = matlab.double(envelopes)
    #     eng.eval(f'all_EI = zeros({len(envelopes)},1);', nargout=0)
    #     eng.eval(f'for i = 1:{len(envelopes)}; '
    #              f'aux = envelopes(i,:); '
    #              f'signal = aux\'; '
    #              f'[EI, wAmp, wDNF] = calculateFEI(signal,{int(len(envelopes[0]) / 10)},0.8); '
    #              f'all_EI(i) = EI; end', nargout=0)
    #     all_EI = np.array(eng.workspace['all_EI'])
    #     features = [all_EI[i][0] for i in range(len(all_EI))]
    #
    #     # Plot the amplitude envelope over the original signal
    #     if debug:
    #         for i, envelope in enumerate(envelopes):
    #             plt.figure()
    #             plt.plot(bandpass(samples[i], fmin, fmax, fs))
    #             plt.plot(envelope)
    #             plt.title(f'fEI = {features[i]}')
    #             plt.show()
    #
    #     # Stop the MATLAB engine
    #     eng.quit()
    #     # Delete the engine object
    #     del eng
    #
    # except:
    #     print('Error computing fEI.')
    #     features = [np.nan for _ in range(len(samples))]
    #
    # return features


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
        if method not in ['catch22', 'power_spectrum_parameterization', 'fEI']:
            raise ValueError("Invalid method. Please use 'catch22', 'power_spectrum_parameterization' or 'fEI'.")

        # Check if params is a dictionary
        if not isinstance(params, dict) and params is not None:
            raise ValueError("params must be a dictionary.")

        self.method = method
        self.params = params


    def compute_features(self, data):
        """
        Function to compute features from the data.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing the data. The time-series samples must be in the 'Data' column.

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
            batch = [(sample - np.mean(sample)) / np.std(sample) for sample in batch]
            features = []

            # Compute features of each sample in the batch
            if self.method != 'fEI':
                for sample in batch:
                    if self.method == 'catch22':
                        features.append(catch22(sample))
                    elif self.method == 'power_spectrum_parameterization':
                        features.append(power_spectrum_parameterization(sample,
                                                                        self.params['fs'],
                                                                        self.params['fmin'],
                                                                        self.params['fmax'],
                                                                        self.params['fooof_setup'],
                                                                        self.params['r_squared_th']))

            # Compute the fEI for the whole batch to avoid starting the Matlab engine multiple times
            else:
                features = fEI(batch,
                               self.params['fs'],
                               self.params['fmin'],
                               self.params['fmax'],
                               self.params['fEI_folder'])

            return batch_index,features

        # Split the data into batches using the number of available CPUs
        num_cpus = os.cpu_count()
        if self.method == 'fEI':
            factor = 0.5
        else:
            factor = 10 # 10 is a factor to update the progress bar more frequently

        batch_size = len(data['Data']) // int(num_cpus*factor)
        if batch_size == 0:
            batch_size = 1
        batches = [(i, data['Data'][i:i + batch_size]) for i in range(0, len(data['Data']), batch_size)]

        # Import the multiprocessing module and create a ProcessingPool
        Pool = getattr(module('pathos'), 'multiprocessing').ProcessingPool

        # Compute the features in parallel using all available CPUs
        with Pool(num_cpus) as pool:
            results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Computing features"))

        # Sort the features based on the original index
        results.sort(key=lambda x: x[0])
        features = [feature for _, batch_features in results for feature in batch_features]

        # Append the features to the DataFrame
        pd_feat = pd.DataFrame({'Features': features})
        data = pd.concat([data, pd_feat], axis=1)
        return data

    def create_df(self, data, epoch_l_samp, df_pat, group, ID):

        '''
        Create epochs of the data and save them in a dataframe containing all data.

        Parameters
        ----------
        data: np.array
            Time-series data.
        epoch_l_samp: int
            Length of the epoch in samples.

        Returns
        -------
        dataframe

        '''
        n_channels = data.shape[1]
        n_epochs = len(data) // epoch_l_samp

        for i in range(n_channels):
            data_epochs = []
            for l in range(n_epochs):
                data_epochs.append(data[l * epoch_l_samp: (l + 1) * epoch_l_samp, i])

            df_new = pd.DataFrame({
                'ID': [ID] * n_epochs,
                'Group': [group] * n_epochs,
                'Epoch': np.arange(n_epochs),
                'Sensor': [i] * n_epochs,
                'Data': data_epochs
            })

            df_pat = pd.concat([df_pat, df_new], ignore_index=True)

        return df_pat

    def load_data(self, data_path, recording_type, data_format, epoch_l):

        '''
        Create a dataframe with the following columns:
            - ID (subject/animal ID).
            - Epoch (epoch number).
            - Group (e.g. HC/AD in neuroimaging, P2/P4... in development, Control/Opto in optogenetics).
            - Sensor (electrode number for EEG, ROI for MEG and, perhaps, 0 for LFP that only has 1 electrode normally).
            - Data (time-series data).

            and the following attributes:
            - fs (sampling frequency).
            - Recording (type of recording: LFP, EEG, MEG...).

        Parameters
        ----------
        data_path: str
            Path to the folder containing the data. One archive per patient with shape (n_samples, n_channels).
        recording_type: str
            Type of data to be loaded. Options: 'EEG', 'LFP', 'MEG'.
        data_format: str
            Format of the data. Options: 'mat', 'csv', 'txt', 'set'.
                - .mat structure: {'data': np.array, 'fs': int, 'group': string}
        epoch_l: int
            Length of the epoch in seconds.

        Returns
        -------
        dataframe

        '''

        ''' 

        TO DO: 

        - Check that a .set can be read with mne.io.read_raw_eeglab
        - Add to the .mat that we read the variable fs to automate the creation of epochs based on this.
        For now we assume that it is 500 Hz.
        - Implement the reading of OpenNeuro and LFP databases.

        '''

        # Initialize an empty DataFrame
        df = pd.DataFrame(columns=['ID', 'Group', 'Epoch', 'Sensor', 'Data'])
        fs = 500

        if data_format == 'mat':

            loadmat = module('scipy.io').loadmat

            file_list = [f for f in os.listdir(data_path) if f.endswith('.mat')]
            # Load the data
            data_tot = []
            ID = 0

            for file_name in file_list:
                ts = {'data': [], 'group': []}
                print(f"Loading file: {file_name}")
                file_full_path = os.path.join(data_path, file_name)
                mat_data = loadmat(file_full_path)
                # Dictionary with the data
                ts['data'] = mat_data['data'][0][0]['signal']
                print(f"Data loaded for patient {file_name}")
                ts['group'] = mat_data['group']
                epoch_l_samp = epoch_l * fs
                # print(type(df))  # Should print <class 'pandas.core.frame.DataFrame'>
                df = self.create_df(ts['data'], epoch_l_samp, df, ts['group'], ID)
                # print(df)
                ID += 1

            df.Recording = recording_type
            df.fs = fs

        return df
