""" Create artificial test data and assess the methods of the Features and Inference classes."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_artificial_data(plot=True):
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
    # Create a list of random data values.
    Data = [np.random.randn(1000) for _ in range(1000)]

    # Create a Pandas DataFrame.
    df = pd.DataFrame({'ID': ID, 'Group': Group, 'Epoch': Epoch, 'Recording': Recording, 'Sensor': Sensor,
                       'Sampling_freq': Sampling_freq, 'Data': Data})

    if plot:
        # Countplots of the 'ID', 'Group', 'Epoch', 'Recording' and 'Sensor' columns
        sns.set(style='whitegrid')
        plt.figure(figsize=(8, 5), dpi=300)
        plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

        for i, column in enumerate(df.columns):
            if column != 'Data':
                plt.subplot(2, 3, i+1)
                if column != 'Epoch':
                    sns.countplot(x=column, data=df, palette='Set2', hue='Group', edgecolor='black', linewidth=1)
                else:   # Plot the 'Epoch' column as a histogram.
                    sns.histplot(df['Epoch'], color='blue', edgecolor='black', linewidth=1, bins=20)
                plt.title(column)
        plt.tight_layout()

        # Plot some random time series from the 'Data' column.
        sns.set(style='whitegrid')
        plt.figure(figsize=(8, 5), dpi=300)

        for i in np.random.randint(0, len(df['Data']), 5):
            plt.plot(np.arange(0, len(df['Data'][i])/df['Sampling_freq'][i], 1/df['Sampling_freq'][i]),
                     df['Data'][i], label='ID: ' + str(df['ID'][i]) + ', Group: ' + df['Group'][i] + ', Epoch: ' +
                        str(df['Epoch'][i]))
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.ylabel(df['Recording'][0])
        plt.title('Examples of time series data')
        plt.show()

    return df


if __name__ == '__main__':
    data = create_artificial_data()
