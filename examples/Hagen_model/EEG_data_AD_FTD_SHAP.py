import sys 
import os

ccpi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ccpi'))
sys.path.insert(0, ccpi_path)

# from ccpi.Inference import inference
import ccpi
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import shap 
import pandas as pd
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
                       'slope',
                       'CF (fixed)',
                       'PW (fixed)',
                       'BW (fixed)']
    if feature == 'all':
        return features_ID
    if feature in features_ID:
        return features_ID.index(feature)
    else:
        return None
def get_data_for_feature(data_feature, desired_features):

    features = np.zeros((data_feature.shape[0], len(desired_features)))
    print(f'Initial features shape {features.shape}')
    for f, feature_name in enumerate(desired_features):
        desired_feature = feature_index(feature_name, 'CATCH22')
        data = data_feature[:, desired_feature]
        print(data.shape)
        features[:, f] = data
    print(f'Features shape {features.shape}')
    return features

features_set = 'CATCH22'
desired_features = [
    'ami2',
    'dfa',
    'rs_range',
    'transition_variance'
]

features_label = feature_index('all', features_set)

features_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/sim_X"
parameters_file = f"/home/alejandro/CCPI/ml_inference/{features_set}/sim_theta"

# For testing purposes, we use a smaller dataset of the original data to speed up the process
features = read_data(features_file)
parameters = read_data(parameters_file)

df_features = pd.DataFrame(features, columns=features_label)
df_parameters = pd.DataFrame(parameters['data'], columns=parameters['parameters'])

# data_indexes = np.random.choice(features.shape[0], int(1e5), replace=False)

features_data = df_features.loc[:, desired_features].copy()
parameters_data = df_parameters.iloc[:, :4].copy()

x_train, x_test, y_train, y_test = train_test_split(
    features_data.to_numpy(),
    parameters_data.to_numpy(),
    test_size=0.20,
    shuffle=True)


params = {
    'hidden_layer_sizes': (64, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'power_t': 0.5,
    'max_iter': 200,
    'shuffle': True,
    'random_state': None,
    'tol': 1e-1,
    'verbose': True,
    'warm_start': False,
    'momentum': 0.9,
    'nesterovs_momentum': True,
    'early_stopping': True,
    'validation_fraction': 0.1
}


inference = ccpi.inference(framework='scikit-learn', model='MLPRegressor', params=params)
inference.train_data(x_train, y_train)
inference.train()
inference.predict(x_test)
save(inference, "/home/alejandro/CCPI/ccpi/examples/Hagen_model/inf_model_ami2_dfa_rs_range_transition_variance")

# sample_indices = np.random.choice(x_train.shape[0], 100, replace=False)
# x_train_sample = x_train[sample_indices]

# explainer = shap.KernelExplainer(inference.predict, x_train_sample)
# shap_values = explainer.shap_values(x_test, nsamples=100)

# feature_names = features_data.columns.tolist()
# parameter_names = parameters_data.columns.tolist()[:4]

# # Visualizar los resultados de SHAP con nombres de columnas
# shap.summary_plot(shap_values, x_test, feature_names=feature_names, class_names=parameter_names, plot_type="bar")
# shap.summary_plot(shap_values, x_test, feature_names=feature_names, class_names=parameter_names)


# # Genera el gráfico de dependencia para la primera salida (índice 0) y la característica 'dfa'
# for p, param in enumerate(parameter_names):
    
#     for f, feat in enumerate(desired_features):
#         feature_index = desired_features.index(feat)

#         shap.dependence_plot(feature_index, shap_values[p], x_test, feature_names=desired_features)