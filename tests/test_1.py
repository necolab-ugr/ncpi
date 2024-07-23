""" Create artificial test data and train a regression model using the ccpi toolbox. """

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


def create_artificial_data():
    """ Create artificial data for testing."""
    # Create a Pandas DataFrame with the following columns: 'ID', 'Group', 'Epoch', 'Recording', 'Sensor',
    # 'Sampling_freq' and 'Data'. The 'ID' column contains the unique identifier of the subject or animal. The
    # 'Group' column defines the group in which the subject/animal has been classified: for example, 'Control' or
    # 'Patient'. The 'Epoch' column contains the epoch number. The 'Recording' column contains the type of recording
    # (e.g., 'LFP'). The 'Sensor' column contains the sensor number. The 'Sampling_freq' column contains the sampling
    # frequency of the data. The 'Data' column contains the time-series data values.

    # Create a list of unique identifiers.
    ID = np.repeat(np.arange(1, 11), 100)
    # Create a list of groups.
    Group = np.repeat(['Control', 'Optogenetics'], 500)
    # Create a list of epochs.
    Epoch = np.tile(np.arange(1, 101), 10)
    # Create a list of recordings.
    Recording = np.repeat(['LFP'], 1000)
    # Create a list of sensors.
    Sensor = np.ones(1000)
    # Create a list of sampling frequencies (Hz).
    Sampling_freq = 100.0 * np.ones(1000)
    # Create a list of random data values in which the first 500 samples are different from the last 500 samples.
    Data = [np.random.randn(1000) + 2. * np.sin(np.arange(1000) / 100.) for _ in range(500)] + \
           [np.random.randn(1000) + (np.linspace(0, 2, 1000)) ** 2 for _ in range(500)]

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


def compute_features(data):
    """ Compute features from the time-series data."""
    ccpi_feat = ccpi.Features(method='catch22')
    return ccpi_feat.compute_features(data)


def split_data(data):
    """ Split the data into training and testing sets."""
    train_data = data.sample(frac=0.9, random_state=0)
    test_data = data.drop(train_data.index)
    return train_data, test_data


def train(data):
    """ Train a regression model using the training data."""
    ccpi_inf = ccpi.Inference(model='Ridge')
    # Assume that the 'Group' column can be interpreted as the parameters to be estimated
    le = LabelEncoder()
    parameters = le.fit_transform(data['Group'])

    # Append the simulation data
    ccpi_inf.append_simulation_data(data['Features'].values, parameters)

    # Train the model
    ccpi_inf.train()

    return ccpi_inf


if __name__ == '__main__':
    data = create_artificial_data()
    data = compute_features(data)
    plot_df(data)
    train_data, test_data = split_data(data)
    ccpi_inf = train(train_data)
    predictions = ccpi_inf.predict(test_data['Features'].values)

    # boxplots of predictions as a function of the group
    plt.figure(figsize=(5, 3), dpi=300)
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    sns.boxplot(x='Group', y=predictions, data=test_data, linewidth=1)
    plt.title('Predictions')
    plt.show()
