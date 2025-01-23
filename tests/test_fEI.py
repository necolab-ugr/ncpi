"""
This code is a demo to compute the fEI and DFA biomarkers for a random white noise signal. The code has been adapted
from https://github.com/arthur-ervin/crosci/tree/main. This code is licensed under creative commons license CC-BY-NC
https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.
"""

import os, sys
import multiprocessing
import numpy as np
import mne
import pandas as pd
from mne.filter import next_fast_len
from scipy.signal import hilbert
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ncpi

np.random.seed(3)

def get_frequency_bins(frequency_range):
    """ Get frequency bins for the frequency range of interest.

    Parameters
    ----------
    frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.

    Returns
    -------
    frequency_bins : list, shape (n_bins,2)
        The lower and upper range in Hz per frequency bin.
    """

    assert frequency_range[0] >= 1.0 and frequency_range[1] <= 150.0, \
        'The frequency range should cannot be less than 1 Hz or more than 150 Hz'

    frequency_bin_delta = [1.0, 4.0]
    frequency_range_full = [frequency_bin_delta[1], 150]
    n_bins_full = 16

    # Create logarithmically-spaced bins over the full frequency range
    frequencies_full = np.logspace(np.log10(frequency_range_full[0]), np.log10(frequency_range_full[-1]), n_bins_full)
    frequencies = np.append(frequency_bin_delta[0],frequencies_full)
    # Get frequencies that fall within the frequency range of interest
    myfrequencies = frequencies[np.where((np.round(frequencies, 4) >= frequency_range[0]) & (
                np.round(frequencies, 4) <= frequency_range[1]))[0]]

    # Get all frequency bin ranges
    frequency_bins = [[myfrequencies[i], myfrequencies[i + 1]] for i in range(len(myfrequencies) - 1)]

    n_bins = len(frequency_bins)

    return frequency_bins


def get_DFA_fitting_interval(frequency_interval):
    """ Get a fitting interval for DFA computation.

    Parameters
    ----------
    frequency_interval : array, shape (1,2)
        The lower and upper bound of the frequency bin in Hz for which the fitting interval will be inferred.
        The fitting interval is where the regression line is fit for log-log coordinates of the fluctuation function vs.
        time windows.

    Returns
    -------
    fit_interval : array, shape (1,2)
        The lower and upper bound of the fitting range in seconds for a frequency bin.
    """

    # Upper fitting margin in seconds
    upper_fit = 30
    # Default lower fitting margins in seconds per frequency bin
    default_lower_fits = [5., 5., 5., 3.981, 3.162, 2.238, 1.412, 1.122, 0.794,
                          0.562, 0.398, 0.281, 0.141, 0.1, 0.1, 0.1]

    frequency_bins = get_frequency_bins([1, 150])
    # Find the fitting interval. In case when frequency range is not exactly one from the defined frequency bins,
    # it finds the fitting interval of the bin for which the lowest of the provided frequencies falls into.
    idx_freq = np.where((np.array(frequency_bins)[:, 0] <= frequency_interval[0]))[0][-1]

    fit_interval = [default_lower_fits[idx_freq],upper_fit]

    return fit_interval

def compute_spectrum_biomarkers(signal_matrix, sampling_frequency, frequency_range, overlap=True, bad_idxes=[],
                                biomarkers_to_compute=['fEI','DFA']):
    """Compute spectral DFA and fEI for the frequencies provided.
    Parameters
    ----------
    signal_matrix : array, shape (n_channels,n_times)
        The signal to compute DFA and fEI for.
    biomarkers_to_compute: list[str]
        Contains the list of biomarkers to be computed. Options are DFA, fEI, bistability_index.
        If fEI is mentioned, DFA is computed by default
    sampling_frequency : float
        The sample frequency in Hz.
    frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.
    overlap : bool
        Whether 50% overlapping windows will be used.
        Default True
    bad_idxes : array, shape (1,)
        The indices of bad channels which will be ignored when DFA is computed. Will be NaNs.
        Default is empty list.

    Returns
    -------
    dfa_exponents_matrix : array, shape (n_channels,n_frequency_bins)
        Computed DFA exponents per frequency bin
    fei_values_matrix : array, shape (n_channels,n_frequency_bins)
        Computed fEI values per frequency bin
    """

    num_cores = multiprocessing.cpu_count()
    num_channels,num_timepoints = np.shape(signal_matrix)
    # Get frequency bins
    frequency_bins = get_frequency_bins(frequency_range)

    output = {}
    if 'DFA' or 'fEI' in biomarkers_to_compute:
        output['DFA']=np.zeros((num_channels,len(frequency_bins)))
    if 'fEI' in biomarkers_to_compute:
        output['fEI']=np.zeros((num_channels,len(frequency_bins)))

    # Parameters
    fEI_window_seconds = 5
    fEI_overlap = 0.8

    for idx_frequency,frequency_bin in enumerate(frequency_bins):

        # Get fit interval
        fit_interval = get_DFA_fitting_interval(frequency_bin)
        DFA_compute_interval = fit_interval

        # Filter signal in the given frequency bin
        filtered_signal = mne.filter.filter_data(data=signal_matrix,sfreq=sampling_frequency,
                                                 l_freq=frequency_bin[0],h_freq=frequency_bin[1],
                                                 filter_length='auto', l_trans_bandwidth='auto',
                                                 h_trans_bandwidth='auto', fir_window='hamming',phase='zero',
                                                 fir_design="firwin", pad='reflect_limited', verbose=0)

        filtered_signal = filtered_signal[:,1*sampling_frequency:filtered_signal.shape[1]-1*sampling_frequency]
        # Compute amplitude envelope
        n_fft = next_fast_len(num_timepoints)
        amplitude_envelope = Parallel(n_jobs=num_cores,backend='threading',verbose=0)(delayed(hilbert)
                                                                            (filtered_signal[idx_channel,:],n_fft)
                                                                            for idx_channel in range(num_channels))
        amplitude_envelope = np.abs(np.array(amplitude_envelope))


        if 'DFA' in biomarkers_to_compute or 'fEI' in biomarkers_to_compute:
            print("Computing DFA for frequency range: %.2f - %.2f Hz" % (frequency_bin[0], frequency_bin[1]))
            params = {'fs': sampling_frequency,
                      'fit_interval': fit_interval,
                      'compute_interval': DFA_compute_interval,
                      'overlap': overlap,
                      'bad_idxes': bad_idxes}
            df = pd.DataFrame({'Data': [amplitude_envelope]})
            features = ncpi.Features(method='DFA', params=params)
            df = features.compute_features(df)
            (dfa_array,window_sizes,fluctuations,dfa_intercept) = df['Features'][0]

            output['DFA'][:,idx_frequency] = dfa_array

        if 'fEI' in biomarkers_to_compute and 'DFA' in biomarkers_to_compute:
            print("Computing fEI for frequency range: %.2f - %.2f Hz" % (frequency_bin[0], frequency_bin[1]))
            params = {'fs': sampling_frequency,
                      'window_size_sec': fEI_window_seconds,
                      'window_overlap': fEI_overlap,
                      'DFA_array': dfa_array,
                      'bad_idxes': bad_idxes}
            df = pd.DataFrame({'Data': [amplitude_envelope]})
            features = ncpi.Features(method='fEI', params=params)
            df = features.compute_features(df)
            (fEI_outliers_removed,fEI_val,num_outliers,wAmp,wDNF) = df['Features'][0]

            output['fEI'][:,idx_frequency] = np.squeeze(fEI_outliers_removed)

    return output


def run_demo_white_noise():
    sampling_frequency = 250
    num_seconds = 100
    frequency_range = [1,45]

    # define random signal with specified number of channels to test the code
    # note for that because this is a random signal, most DFA values will be <0.6, and so in these cases
    # fEI will not be computed (it will have a value of NaN)
    # this isgnal can be replaced by any EEG/MEG files. you can load these files using mne
    # (https://mne.tools/stable/index.html)
    # make sure to feed only the signal matrix to the DFA&fEI algorithms
    num_channels = 10
    signal = np.random.rand(num_channels,num_seconds*sampling_frequency)

    # compute DFA and fEI
    biomarkers = compute_spectrum_biomarkers(signal,sampling_frequency,frequency_range)
    DFA_matrix = biomarkers['DFA']
    fEI_matrix = biomarkers['fEI']

    ############# PLOTTING OF RESULTS
    # idx of channel to plot
    chan_to_plot = 0

    # get all the frequency bins for which DFA/fEI were computed
    frequency_bins = get_frequency_bins(frequency_range)

    # define the
    x_labels = []
    for i in range(len(frequency_bins)):
        x_labels.append(str(round(frequency_bins[i][0],1)) + '-' + str(round(frequency_bins[i][1],1)))
    x_ticks = np.arange(len(frequency_bins))

    fig, ax = plt.subplots(1, 2, figsize=(9.6,3.2))
    ax[0].plot(x_ticks,DFA_matrix[chan_to_plot,:])
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('DFA')
    ax[0].set_xticks(x_ticks,x_labels,rotation=45)

    #note that fE/I will be NaN in this case, so nothing will be swhon
    ax[1].plot(x_ticks, fEI_matrix[chan_to_plot, :] , marker = 'x',markersize = 20)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('fEI')
    ax[1].set_xticklabels(x_labels)
    ax[1].set_xticks(x_ticks,x_labels,rotation=45)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    run_demo_white_noise()