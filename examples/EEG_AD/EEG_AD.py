import os
import pickle
import pandas as pd
import scipy
import numpy as np
import shutil
import ncpi
from ncpi import tools
from ncpi.tools import timer

# Select either raw EEG data or source-reconstructed EEG data. This study used the raw EEG data for all analyses.
raw = True
if raw:
    data_path = os.path.join(os.sep, 'DATOS','pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SENSORS') # Specify you data path here
else:
    data_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'empirical_datasets', 'POCTEP_data', 'CLEAN', 'SOURCES', 'dSPM', 'DK') # Specify you data path here

# Choose to either download data from Zenodo (True) or load it from a local path (False).
# Important: the zenodo downloads will take a while, so if you have already downloaded the data, set this to False and
# configure the zenodo_dir variables to point to the local paths where the data is stored.
zenodo_dw_sim = True # simulation data

# Zenodo URL that contains the simulation data and ML models (used if zenodo_dw_sim is True)
zenodo_URL_sim = "https://zenodo.org/api/records/15351118"

# Paths to zenodo simulation files
zenodo_dir_sim = os.path.join("/home/pablomc","zenodo_sim_files")

# Methods used to compute the features
all_methods = ['catch22','power_spectrum_parameterization_1']

# ML model used to compute the predictions
ML_model = 'MLPRegressor'


def create_POCTEP_dataframe(data_path):
    '''
    Load the POCTEP dataset and create a DataFrame with the data.

    Parameters
    ----------
    data_path: str
        Path to the directory containing the POCTEP data files.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the POCTEP data.
    '''


    # List files in the directory
    ldir = os.listdir(data_path)

    ID = []
    group = []
    epoch = []
    sensor = []
    EEG = []

    for pt,file in enumerate(ldir):
        print(f'\rProcessing {file} - {pt + 1 }/{len(ldir)}', end="", flush=True)

        # load data
        data = scipy.io.loadmat(data_path + '/' + file)['data']
        signal = data['signal'][0, 0]

        # get sampling frequency
        fs = data['cfg'][0, 0]['fs'][0, 0][0, 0]

        # Electrodes (raw data)/ regions (if source data)
        regions = np.arange(signal.shape[1])

        # get channels
        ch_names = data['cfg'][0, 0]['channels'][0, 0][0]
        ch_names = [ch_names[ll][0] for ll in range(len(ch_names))]

        # 5-second epochs
        epochs = np.arange(0, signal.shape[0], int(fs * 5))

        for i in range(len(epochs) - 1):
            ep = signal[epochs[i]:epochs[i + 1], :]
            # z-score normalization
            ep = (ep - np.mean(ep, axis=0)) / np.std(ep, axis=0)

            # Append data
            for rg in regions:
                ID.append(pt)
                group.append(file.split('_')[0])
                epoch.append(i)
                sensor.append(ch_names[rg])
                EEG.append(ep[:, rg])

    # Create the Pandas DataFrame
    df = pd.DataFrame({'ID': ID,
                       'Group': group,
                       'Epoch': epoch,
                       'Sensor': sensor,
                       'Data': EEG})
    df['Recording'] = 'EEG'
    df['fs'] = fs

    return df

@timer("Loading simulation data.")
def load_model_features(method, zenodo_dir_sim):
    """ Load model parameters (theta) and features (X) from simulation data."""
    try:
        with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_theta'), 'rb') as file:
            theta = pickle.load(file)
        with open(os.path.join(zenodo_dir_sim, 'data', method, 'sim_X'), 'rb') as file:
            X = pickle.load(file)
    except Exception as e:
        print(f"Error loading simulation data: {e}")

    # Print info
    print('theta:')
    for key, value in theta.items():
        if isinstance(value, np.ndarray):
            print(f'--Shape of {key}: {value.shape}')
        else:
            print(f'--{key}: {value}')
    print(f'Shape of X: {X.shape}')

    return X, theta


@timer("Feature extraction")
def feature_extraction(method, df):
    """ Extract features from empirical data using specified method."""
    if "Data" not in df.columns:
        raise ValueError("Expected input DataFrame to contain a 'Data' column with 1D samples.")

    samples = df["Data"].to_list()

    if method == "catch22":
        features = ncpi.Features(method="catch22", params={"normalize": True})
        feats = features.compute_features(samples)
        emp_data = df.copy()
        emp_data["Features"] = feats
        return emp_data

    if method == "power_spectrum_parameterization_1":
        fooof_setup_emp = {
            "peak_threshold": 1.0,
            "min_peak_height": 0.0,
            "max_n_peaks": 5,
            "peak_width_limits": (10.0, 50.0),
        }

        params = {
            "fs": float(df["fs"].iloc[0]),
            "freq_range": (5.0, 45.0),
            "specparam_model": dict(fooof_setup_emp),
            "r_squared_th": 0.9,
        }

        features = ncpi.Features(method="specparam", params=params)
        feats = features.compute_features(samples)

        emp_data = df.copy()
        emp_data["Features"] = [float(np.asarray(d["aperiodic_params"])[1]) for d in feats]
        return emp_data

    raise ValueError(f"Unknown method: {method}")


@timer('Computing predictions...')
def compute_predictions_neural_circuit(emp_data, method, ML_model, X, theta, zenodo_dir_sim, sensor_list=None):
    emp_data['Predictions'] = np.nan

    if sensor_list is None:
        sensor_list = [
            'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
            'O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
        ]

    # Create folder to save results
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', method)):
        os.makedirs(os.path.join('data', method))

    # Create inference object
    inference = ncpi.Inference(model=ML_model)
    # Not sure if this is really needed
    inference.add_simulation_data(X, theta['data'])

    for s, sensor in enumerate(sensor_list):
        print(f'--- Sensor: {sensor}')

        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'model'),
            os.path.join('data', 'model.pkl')
        )

        shutil.copy(
            os.path.join(zenodo_dir_sim, 'ML_models/EEG', sensor, method, 'scaler'),
            os.path.join('data', 'scaler.pkl')
        )

        sensor_df = emp_data[emp_data['Sensor'].isin([sensor, s])]

        predictions = inference.predict(np.array(sensor_df['Features'].to_list()))

        sensor_df['Predictions'] = [list(pred) for pred in predictions]
        emp_data.update(sensor_df['Predictions'])

    return emp_data

def save_data(emp_data, method):
    # Save the data including predictions of all parameters
    emp_data.to_pickle(os.path.join('data', method, 'emp_data_all.pkl'))

    # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
    E_I_net = emp_data['Predictions'].apply(lambda x: (x[0]/x[2]) / (x[1]/x[3]))
    others = emp_data['Predictions'].apply(lambda x: x[4:])
    emp_data['Predictions'] = (np.concatenate((E_I_net.values.reshape(-1,1),
                                                np.array(others.tolist())), axis=1)).tolist()

    # Save the data including predictions of (E/I)_net
    emp_data.to_pickle(os.path.join('data', method, 'emp_data_reduced.pkl'))

if __name__ == "__main__":
    # Download simulation data and ML models
    if zenodo_dw_sim:
        tools.timer("Downloading simulation data and ML models from Zenodo")(
            tools.download_zenodo_record
        )(zenodo_URL_sim, download_dir=zenodo_dir_sim)

    # Go through all methods
    for method in all_methods:
        print(f'\n\n--- Method: {method}')

        # Load parameters of the model (theta) and features (X) from simulation data
        X, theta = load_model_features(method, zenodo_dir_sim)

        # Load empirical data and create DataFrame
        df = create_POCTEP_dataframe(data_path=data_path)

        ##########################
        #   FEATURE EXTRACTION   #
        ##########################

        emp_data = feature_extraction(method, df)

        #######################################################
        #   PREDICTIONS OF PARAMETERS OF THE NEURAL CIRCUIT   #
        #######################################################

        emp_data = compute_predictions_neural_circuit(emp_data, method, ML_model, X, theta, zenodo_dir_sim)

        # Save the data including predictions of all parameters  
        save_data(emp_data, method)      
