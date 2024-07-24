""" Create artificial test data and train regression models using the ccpi toolbox. """

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ccpi


def create_artificial_data(type=0, n_samples=1000):
    """ Create artificial data for testing.

    Create a Pandas DataFrame with the following columns: 'ID', 'Group', 'Epoch', 'Recording', 'Sensor',
    'Sampling_freq' and 'Data'. The 'ID' column contains the unique identifier of the subject or animal. The
    'Group' column defines the group in which the subject/animal has been classified: for example, 'Control' or
    'Patient'. The 'Epoch' column contains the epoch number. The 'Recording' column contains the type of recording
    (e.g., 'LFP'). The 'Sensor' column contains the sensor number. The 'Sampling_freq' column contains the sampling
    frequency of the data. The 'Data' column contains the time-series data values.

    Parameters
    ----------
    type : int
        Type of artificial data to generate. The following types are available:
        - 0: The first 500 samples are a sine wave with noise, while the last 500 samples are a polynomial with noise.
        - 1: Convolution of a Poisson process with an exponential decay with different exponential decay rates.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    df : Pandas DataFrame
        Pandas DataFrame containing the artificial data
    """

    # Create a list of unique identifiers.
    ID = np.repeat(np.arange(1, 11), int(n_samples/10))
    # Create a list of groups.
    Group = np.repeat(['Control', 'Optogenetics'], int(n_samples/2))
    # Create a list of epochs.
    Epoch = np.tile(np.arange(1, int(n_samples/10)+1), 10)
    # Create a list of recordings.
    Recording = np.repeat(['LFP'], n_samples)
    # Create a list of sensors.
    Sensor = np.ones(n_samples)
    # Create a list of sampling frequencies (Hz).
    Sampling_freq = 100.0 * np.ones(n_samples)
    # Create a list of random data values in which the first 500 samples are different from the last 500 samples.
    if type == 0:
        # The first 500 samples are a sine wave with noise, while the last 500 samples are a polynomial with noise.
        Data = [np.random.randn(1000) + 2. * np.sin(np.arange(1000) / 100.) for _ in range(int(n_samples/2))] + \
               [np.random.randn(1000) + (np.linspace(0, 2, 1000)) ** 2 for _ in range(int(n_samples/2))]
    elif type == 1:
        # Convolution of a Poisson process with an exponential decay with different exponential decay rates.
        Data = [np.convolve(np.random.poisson(lam=0.1, size=1200),
                            np.power(np.e, -np.linspace(0, 1, 100)*(int(i/2)+1)),
                            mode='same')[100:1100] for i in range(n_samples)]

    # Create a Pandas DataFrame.
    df = pd.DataFrame({'ID': ID, 'Group': Group, 'Epoch': Epoch, 'Recording': Recording, 'Sensor': Sensor,
                       'Sampling_freq': Sampling_freq, 'Data': Data})

    return df


def plot_df(df):
    """ Plot data from the Pandas DataFrame."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5), dpi=300)
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

    i = 0
    for column in df.columns:
        if column != 'Data':
            plt.subplot(2, 3, i + 1)
            # Plot the 'ID', 'Group', 'Sensor' and 'Sampling_freq' columns as countplots.
            if column != 'Epoch' and column != 'Features' and column != 'Recording':
                sns.countplot(x=column, data=df, palette='Set2', hue='Group', edgecolor='black', linewidth=1)
                plt.title(column)
                i += 1
            elif column == 'Features':
                # Boxplot of some random feature as a function of the group
                features = np.array(df['Features'].tolist())
                idx = np.random.randint(0, features.shape[1], 1)[0]
                sns.boxplot(x='Group', y=features[:, idx], data=df, linewidth=1, legend=False)
                plt.title(f'Feature {idx}')
                i += 1
            # Plot the 'Epoch' column as a histogram.
            elif column == 'Epoch':
                sns.histplot(df['Epoch'], color='blue', edgecolor='black', linewidth=1, bins=20)
                plt.title(column)
                i += 1

    plt.tight_layout()

    # Plot some random time series from the 'Data' column.
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5), dpi=300)

    for i in np.random.randint(0, len(df['Data']), 5):
        plt.plot(np.arange(0, len(df['Data'][i]) / df['Sampling_freq'][i], 1 / df['Sampling_freq'][i]),
                 df['Data'][i], label='ID: ' + str(df['ID'][i]) + ', Group: ' + df['Group'][i] + ', Epoch: ' +
                                      str(df['Epoch'][i]))
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel(df['Recording'][0])
    plt.title('Examples of time series data')
    plt.show()


if __name__ == '__main__':
    # Create artificial data
    data = create_artificial_data(type=0)

    # Compute features
    features = ccpi.Features(method='catch22')
    data = features.compute_features(data)

    # Plot data
    plot_df(data)

    # Split data into training and testing sets
    train_data = data.sample(frac=0.9, random_state=0)
    test_data = data.drop(train_data.index)

    # Assume that the 'Group' column can be interpreted as the parameters to be estimated
    le = LabelEncoder()
    parameters = le.fit_transform(train_data['Group'])

    # sklearn models to test
    models = ['Ridge', 'Lasso', 'ElasticNet', 'MLPRegressor']
    predictions = {}

    for model in models:
        print(f'Training {model}...')
        # Create an instance of the Inference class
        inference = ccpi.Inference(model=model)

        # Add simulation data
        inference.add_simulation_data(train_data['Features'].values, parameters)

        # Train
        inference.train(param_grid=None)

        # Predict
        predictions[model] = inference.predict(test_data['Features'].values)

    # Boxplots of the predictions as a function of the group
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5), dpi=300)
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='Group', y=predictions[model], data=test_data, linewidth=1, palette='Set2', legend=False,
                    hue='Group')
        plt.title(model)
    plt.tight_layout()
    plt.show()



