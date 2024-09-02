import importlib
import subprocess
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


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
    pass

def fEI(samples,fs,fEI_folder):
    pass

class Features:
    """
    Class for computing features from electrophysiological data recordings.
    """

    def __init__(self, method='catch22'):
        """
        Constructor method.

        Parameters
        ----------
        method: str
            Method to compute features. Default is 'catch22'.
        """
        self.method = method

        # Assert that the method is a string
        if not isinstance(self.method, str):
            raise ValueError("The method must be a string.")

        # Check if the method is valid
        if self.method not in ['catch22']:
            raise ValueError("Invalid method. Please use 'catch22'.")


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
            features = []
            for sample in batch:
                # Normalize the sample
                sample = (sample - np.mean(sample)) / np.std(sample)
                if self.method == 'catch22':
                    features.append(catch22(sample))
            return batch_index,features

        # Split the data into batches using the number of available CPUs
        num_cpus = os.cpu_count()
        batch_size = len(data['Data']) // (num_cpus*10) # 10 is a factor to update the progress bar more frequently
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
