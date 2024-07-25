import importlib
import subprocess
import pandas as pd
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def install(module):
    """
    Function to install a Python module.

    Parameters
    ----------
    module: str
        Module name.
    """
    subprocess.check_call(['pip', 'install', module])
    print(f"The module {module} was installed!")


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
            DataFrame containing the data.

        Returns
        -------
        data: pd.DataFrame
            DataFrame containing the data with the features appended.
        """

        def compute_sample_features(index, sample):
            """ Compute the features for a single sample."""
            # Normalize the sample
            sample = (sample - np.mean(sample)) / np.std(sample)
            if self.method == 'catch22':
                return index, self.catch22(sample)
            return index, []

        # Compute the features in parallel using all available CPUs
        features = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(compute_sample_features, idx, sample) for idx, sample in enumerate(data['Data'])]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing features"):
                features.append(future.result())

        # Sort the features based on the original index
        features.sort(key=lambda x: x[0])
        features = [feature[1] for feature in features]

        # Append the features to the DataFrame
        pd_feat = pd.DataFrame({'Features': features})
        data = pd.concat([data, pd_feat], axis=1)
        return data

    def catch22(self, sample):
        """
        Function to compute the catch22 features.

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
        try:
            pycatch22 = importlib.import_module('pycatch22')
        except ImportError:
            print("pycatch22 is not installed!")
            install('pycatch22')

        features = pycatch22.catch22_all(sample)
        return features['values']

    def create_df_patient(self, data, epoch_l_samp, df_pat, group, ID):

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

            remaining_d = len(data) % epoch_l_samp
            if remaining_d == epoch_l_samp:
                last_epoch = data[n_epochs * epoch_l_samp:, i]
                data_epochs.extend(last_epoch)
                n_epochs += 1
                print(f"Last epoch has {remaining_d} samples. Adding it to the dataframe.")
            else:
                print(f"Last epoch has {remaining_d} samples. Discarding it.")

            # epoch_data = pd.DataFrame({'Data': data_epochs, 'Group': [group] * n_epochs})
            # df = pd.concat([df, epoch_data], ignore_index=True)

        return df_pat

    def create_dataframe(self, data_path, recording_type, data_format, epoch_l):

        '''
        Create a dataframe with the following columns:
            - ID (subject/animal ID).
            - Epoch (epoch number).
            - Group (e.g. HC/AD in neuroimaging, P2/P4... in development, Control/Opto in optogenetics).
            - Location (electrode number for EEG, ROI for MEG and, perhaps, 0 for LFP that only has 1 electrode normally).
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

        import matlab.engine

        # Initialize an empty DataFrame
        df = pd.DataFrame(columns=['ID', 'Group', 'Epoch', 'Sensor', 'Data'])
        fs = 500

        if data_format == 'mat':

            file_list = [f for f in os.listdir(data_path) if f.endswith('.mat')]
            # Initialize MATLAB engine
            eng = matlab.engine.start_matlab()
            # Load the data
            data_tot = []
            ID = 0

            for file_name in file_list:
                ts = {'data': [], 'group': []}
                print(f"Loading file: {file_name}")
                file_full_path = os.path.join(data_path, file_name)
                eng.load(file_full_path, nargout=0)
                # Dictionary with the data
                ts['data'] = np.array(eng.eval('data.signal', nargout=1))
                print(f"Data loaded for patient {file_name}")
                ts['group'] = eng.eval('group', nargout=1)
                epoch_l_samp = epoch_l * fs
                print(type(df))  # Should print <class 'pandas.core.frame.DataFrame'>
                df = self.create_df_patient(ts['data'], epoch_l_samp, df, ts['group'], ID)
                print(df)  # Should also print <class 'pandas.core.frame.DataFrame'>
                ID += 1

            df.Recording = recording_type
            df.fs = fs

        return df
