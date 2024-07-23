import sys 
import os

ccpi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ccpi'))
sys.path.insert(0, ccpi_path)

# from ccpi.Inference import inference
import ccpi
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time

def read_data(path):
    '''
    Load data from file
    :param path: path of data
                string
    :return: data
    '''

    with open(path, 'rb') as f:
        return pickle.load(f)

def save(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)
    print(f"File saved in: {path}")

def feature_index(feature, ftype):
    if ftype == 'CATCH22':
        # catch22
        features_ID = ['mode_5',
                       'mode_10',
                       'acf_timescale',
                       'acf_first_min',
                       'ami2',
                       'trev',
                       'high_fluctuation',
                       'stretch_high',
                       'transition_variance',
                       'periodicity',
                       'embedding_dist',
                       'ami_timescale',
                       'whiten_timescale',
                       'outlier_timing_pos',
                       'outlier_timing_neg',
                       'low_freq_power',
                       'stretch_decreasing',
                       'entropy_pairs',
                       'rs_range',
                       'dfa',
                       'centroid_freq',
                       'forecast_error']
    elif ftype == 'FOOOF':
        # fooof
        features_ID = ['offset (knee)',
                       'knee (knee)',
                       'exponent (knee)',
                       'CF (knee)',
                       'PW (knee)',
                       'BW (knee)',
                       'offset (fixed)',
                       'knee (fixed)',
                       'exponent (fixed)',
                       'CF (fixed)',
                       'PW (fixed)',
                       'BW (fixed)']
    if feature == 'all':
        return features_ID
    if feature in features_ID:
        return features_ID.index(feature)
    else:
        return None
def get_data_for_feature(data_feature, desired_features, databse):

    features = np.zeros((data_feature.shape[0], len(desired_features)))
    print(f'Initial features shape {features.shape}')
    for f, feature_name in enumerate(desired_features):
        desired_feature = feature_index(feature_name, database)
        data = data_feature[:, desired_feature]
        print(data.shape)
        features[:, f] = data
    print(f'Features shape {features.shape}')
    return features

def separate_data(data, database):
    if database == 'POCTEP':
        HC = data[:51]
        ADMIL = data[51:101]
        ADMOD = data[101:151]
        ADSEV = data[151:201]
        return [HC, ADMIL, ADMOD, ADSEV]
    elif database == 'OpenNeuro':
        HC = data[:29]
        FTD = data[29:52]
        AD = data[52:88]
        return [HC, FTD, AD]


database = 'POCTEP'

features_set = 'FOOOF'
desired_features = [
    'exponent (fixed)',
]

files = [

]

# Training data
features_label = feature_index('all', features_set)

features_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/ccpi_features/fooof/sim_X"
parameters_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/ccpi_features/fooof/sim_theta"

# features_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/sim_X"
# parameters_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/sim_theta"

features = read_data(features_file)
parameters = read_data(parameters_file)

df_features = pd.DataFrame(features, columns=features_label)
df_parameters = pd.DataFrame(parameters['data'], columns=parameters['parameters'])

# Data for inference
inference_path = f"/home/alejandro/CCPI/DATA/features/databases/{database}/trials"
# inference_path = f"/home/alejandro/CCPI/ml_inference/{database}/features_raw/PABLO/{database}_feat_{features_set}.pkl"


features_data = df_features.loc[:, desired_features].copy()
parameters_data = df_parameters.iloc[:, :4].copy()

x_train = features_data.to_numpy()
y_train = parameters_data.to_numpy()

x_train_filtered = np.array([])
y_train_filtered = np.array([])

no_cero = np.any(x_train != 0.0, axis=1)

x_train_filtered = x_train[no_cero]
y_train_filtered = y_train[no_cero]

# HERE START THE USE OF CCPI LIBRARY
##MLPRegressor
# params = {
#     'hidden_layer_sizes': (25, 25),
#     'activation': 'relu',
#     'solver': 'adam',
#     'tol': 1e-1,
#     'n_iter_no_change': 2,
#     'random_state': 0,
#     'verbose': True
# }

## Lasso
# params = {
#     'alpha': 1.0,
#     'tol': 1e-1,
#     'random_state': 0,
# }

## Ridge
# params = {
#     'alpha': 1.0,
#     'tol': 1e-1,
#     'random_state': 0,
# }

inference = ccpi.inference(framework='scikit-learn', model='Ridge', hyperparams=None)
inference.train_data(x_train_filtered, y_train_filtered)
inference.train(gridsearch=True, param_grid=None)

entries = os.listdir(inference_path)
files = [entry for entry in entries if 'fooof' in entry]
print(files)
for file in files:
    print(file)
    data_path = os.path.join(inference_path, file)

    inference_data = read_data(data_path)


#     # # DATA = separate_data(inference_data, database)
#     # # PREDICTIONS = []
#     # # for pt in DATA:
#     # #     pt_pred = []
#     # #     for patient in pt:
#     # #         results = np.zeros((len(patient), 19, 4)) # CADA PACIENTE TIENE UN NUMERO DE ELECTRODOS DIFERENTE!!!
#     # #         for epoch in range(len(patient)):
#     # #             for electrode in range(len(patient[epoch])):
#     # #                 for feat in desired_features:
#     # #                     f = feature_index(feat, features_set)
#     # #                     results[epoch, electrode, :] = inference.predict(patient[epoch][electrode][f].reshape(-1, 1)
#     # #                     )
#     # #         pt_pred.append(results)
#     # #     PREDICTIONS.extend(pt_pred)

    DATA = separate_data(inference_data, database)
    PREDICTIONS = []
    for pt in DATA:
        pt_pred = []
        for patient in pt:
            results = np.zeros((len(patient), len(patient[0]), 4))
            for electrode in range(patient.shape[1]):
                for feat in desired_features:
                        # print(f'Infering {feat} of patient {pt} {pt.index(patient)}')
                        f = feature_index(feat, features_set)
                        for epoch in range(patient.shape[0]):     
                            if patient[epoch, electrode, f] == 0:
                                results[epoch, electrode, :] = [0, 0, 0, 0]
                                # patient[epoch, electrode, f] = np.array([np.nan]).reshape(-1, 1)
                                # print(patient[epoch, electrode, f])
                                # results[epoch, electrode, :] = inference.predict(patient[epoch, electrode, f].reshape(-1, 1))
                            else:       
                                results[epoch, electrode, :] = inference.predict(patient[epoch, electrode, f].reshape(-1, 1))
            pt_pred.append(results)
        PREDICTIONS.extend(pt_pred)

    # save(PREDICTIONS, f"/home/alejandro/CCPI/DATA/parameters/{database}/ccpi_{file}_v2")