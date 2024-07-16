#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common methods accessed by different example codes

Copyright (C) 2021 https://github.com/espenhgn

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
import os
import numpy as np
import scipy.signal as ss
from matplotlib import mlab
import neuron
import h5py


def set_active(cell, Vrest):
    """Insert HH and Ih channels across cell sections

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential (ignored)
    """
    for sec in cell.template.all:
        sec.insert('hh')
        sec.insert('Ih')
        if sec.name().rfind('soma') >= 0:
            sec.gnabar_hh = 0.12
            sec.gkbar_hh = 0.036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
            sec.ena = 50
            sec.ek = -77

            sec.gIhbar_Ih = 0.002

        if sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            # set HH channel conductancesto 10% of default in apical dendrite
            sec.gnabar_hh = 0.012
            sec.gkbar_hh = 0.0036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
            sec.ena = 50
            sec.ek = -77

            # set higher Ih-conductance in apical dendrite
            sec.gIhbar_Ih = 0.01


def set_passive(cell, Vrest):
    """Insert passive leak channel across sections

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.g_pas = 0.0003
        sec.e_pas = Vrest


def set_Ih(cell, Vrest):
    """Insert passive leak and voltage-gated Ih across sections

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('Ih')
        sec.e_pas = Vrest
        sec.g_pas = 0.0003
        if sec.name().rfind('soma') >= 0:
            sec.gIhbar_Ih = 0.002
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gIhbar_Ih = 0.01


def set_Ih_linearized(cell, Vrest):
    """Insert passive leak and linearized Ih.

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('Ih_linearized_v2')
        sec.e_pas = Vrest
        sec.g_pas = 0.0003
        sec.V_R_Ih_linearized_v2 = Vrest
        if sec.name().rfind('soma') >= 0:
            sec.gIhbar_Ih_linearized_v2 = 0.002
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gIhbar_Ih_linearized_v2 = 0.01


def set_pas_hay2011(cell, Vrest):
    """Insert passive leak as in Hay2011 model

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        if sec.name().rfind('soma') >= 0:
            sec.g_pas = 0.0000338
            sec.e_pas = -90

        if sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.g_pas = 0.0000589
            sec.e_pas = -90


def set_active_hay2011(cell, Vrest):
    """Insert passive leak, Ih, NaTa_t and SKv3_1 channels as in Hay 2011 model

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential (not used)
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('Ih')
        sec.insert('NaTa_t')
        sec.insert('SKv3_1')
        sec.ena = 50
        sec.ek = -85
        if sec.name().rfind('soma') >= 0:
            sec.gNaTa_tbar_NaTa_t = 2.04
            sec.gSKv3_1bar_SKv3_1 = 0.693
            sec.g_pas = 0.0000338
            sec.e_pas = -90
            sec.gIhbar_Ih = 0.0002

        if sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gNaTa_tbar_NaTa_t = 0.0213
            sec.gSKv3_1bar_SKv3_1 = 0.000261
            sec.g_pas = 0.0000589
            sec.e_pas = -90
            sec.gIhbar_Ih = 0.0002 * 10


def set_frozen_hay2011(cell, Vrest):
    """Set passive leak and linear passive-frozen versions of Ih, NaTa_t and
    SKv3_1 channels from Hay 2011 model

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float or list of float
        Steady state potential. If list, set Vrest per segment
    """
    i = 0
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('Ih_linearized_v2_frozen')
        if isinstance(Vrest, float):
            sec.e_pas = Vrest
            sec.V_R_NaTa_t_frozen = Vrest
            sec.V_R_SKv3_1_frozen = Vrest
            sec.V_R_Ih_linearized_v2_frozen = Vrest
            sec.ena = Vrest  # 50
            sec.ek = Vrest  # -85
        else:
            for seg in sec:
                seg.e_pas = Vrest[i]
                seg.V_R_NaTa_t_frozen = Vrest[i]
                seg.V_R_SKv3_1_frozen = Vrest[i]
                seg.V_R_Ih_linearized_v2_frozen = Vrest[i]
                seg.ena = Vrest[i]  # 50
                seg.ek = Vrest[i]  # -85
            i += 1
        if sec.name().rfind('soma') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 2.04
            sec.gSKv3_1bar_SKv3_1_frozen = 0.693
            sec.g_pas = 0.0000338
            sec.gIhbar_Ih_linearized_v2_frozen = 0.0002
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 0.0213
            sec.gSKv3_1bar_SKv3_1_frozen = 0.000261
            sec.g_pas = 0.0000589
            sec.gIhbar_Ih_linearized_v2_frozen = 0.0002 * 10


def set_frozen_hay2011_no_Ih(cell, Vrest):
    """
    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.e_pas = Vrest
        sec.V_R_NaTa_t_frozen = Vrest
        sec.V_R_SKv3_1_frozen = Vrest
        sec.ena = Vrest  # 50
        sec.ek = Vrest  # -85
        if sec.name().rfind('soma') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 2.04
            sec.gSKv3_1bar_SKv3_1_frozen = 0.693
            sec.g_pas = 0.0000338
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 0.0213
            sec.gSKv3_1bar_SKv3_1_frozen = 0.000261
            sec.g_pas = 0.0000589


def set_Ih_hay2011(cell, Vrest):
    """
    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('Ih')
        sec.e_pas = Vrest
        sec.ena = Vrest  # 50
        sec.V_R_NaTa_t_frozen = Vrest
        sec.V_R_SKv3_1_frozen = Vrest
        sec.ek = Vrest  # -85
        if sec.name().rfind('soma') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 2.04
            sec.gSKv3_1bar_SKv3_1_frozen = 0.693
            sec.g_pas = 0.0000338
            sec.gIhbar_Ih = 0.0002
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 0.0213
            sec.gSKv3_1bar_SKv3_1_frozen = 0.000261
            sec.g_pas = 0.0000589
            sec.gIhbar_Ih = 0.0002 * 10


def set_Ih_linearized_hay2011(cell, Vrest):
    """
    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float or list of float
        Steady state potential. If list, set per segment
    """
    i = 0
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('Ih_linearized_v2')
        if isinstance(Vrest, float):
            sec.e_pas = Vrest
            sec.V_R_Ih_linearized_v2 = Vrest
            sec.V_R_NaTa_t_frozen = Vrest
            sec.V_R_SKv3_1_frozen = Vrest
            sec.ena = Vrest  # 50
            sec.ek = Vrest  # -85
        else:
            for seg in sec:
                seg.e_pas = Vrest[i]
                seg.V_R_Ih_linearized_v2 = Vrest[i]
                seg.V_R_NaTa_t_frozen = Vrest[i]
                seg.V_R_SKv3_1_frozen = Vrest[i]
                seg.ena = Vrest[i]  # 50
                seg.ek = Vrest[i]  # -85
                i += 1

        if sec.name().rfind('soma') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 2.04
            sec.gSKv3_1bar_SKv3_1_frozen = 0.693
            sec.g_pas = 0.0000338
            sec.gIhbar_Ih_linearized_v2 = 0.0002
        elif sec.name().rfind('apic') >= 0 or sec.name().rfind('dend') >= 0:
            sec.gNaTa_tbar_NaTa_t_frozen = 0.0213
            sec.gSKv3_1bar_SKv3_1_frozen = 0.000261
            sec.g_pas = 0.0000589
            sec.gIhbar_Ih_linearized_v2 = 0.0002 * 10


def set_V_R_Ih_linearized_v2(cell, Vrest):
    """
    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        if neuron.h.ismembrane("Ih_linearized_v2", sec=sec):
            sec.V_R_Ih_linearized_v2 = Vrest
        elif neuron.h.ismembrane("Ih_linearized_v2_frozen", sec=sec):
            sec.V_R_Ih_linearized_v2_frozen = Vrest


def set_V_R(cell, Vrest):
    """
    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    ion_channels = [
        "Ih_linearized_v2",
        "Ih_linearized_v2_frozen",
        "Ca_LVAst_frozen",
        "Ca_HVA_frozen",
        "SKv3_1_frozen",
        "K_Tst_frozen",
        "K_Pst_frozen",
        "Nap_Et2_frozen",
        "Nap_Et2_linearized",
        "NaTa_t_frozen",
        "Im_frozen"
    ]
    for ion in ion_channels:
        i = 0
        for sec in cell.template.all:
            if neuron.h.ismembrane(ion, sec=sec):
                if isinstance(Vrest, float):
                    setattr(sec, f'V_R_{ion}', Vrest)
                else:
                    for seg in sec:
                        setattr(seg, f'V_R_{ion}', Vrest[i])
                        i += 1


def make_cell_uniform(cell, Vrest=-65):
    """
    Adjusts e_pas to enforce a uniform resting membrane potential at Vrest

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential. If list of float; only the first element
        will be used for cell initialization.
    """
    neuron.h.t = 0
    if isinstance(Vrest, float):
        neuron.h.finitialize(Vrest)
    else:
        neuron.h.finitialize()
    neuron.h.fcurrent()
    for sec in cell.allseclist:
        for seg in sec:
            seg.e_pas = seg.v
            if neuron.h.ismembrane("na_ion", sec=sec):
                seg.e_pas += seg.ina / seg.g_pas
            if neuron.h.ismembrane("k_ion", sec=sec):
                seg.e_pas += seg.ik / seg.g_pas
            if neuron.h.ismembrane("ca_ion", sec=sec):
                seg.e_pas += seg.ica / seg.g_pas
            if neuron.h.ismembrane("Ih", sec=sec):
                seg.e_pas += seg.ihcn_Ih / seg.g_pas
            if neuron.h.ismembrane("Ih_z", sec=sec):
                seg.e_pas += seg.ih_Ih_z / seg.g_pas
            if neuron.h.ismembrane("Ih_linearized_v2_frozen", sec=sec):
                seg.e_pas += seg.ihcn_Ih_linearized_v2_frozen / seg.g_pas
            if neuron.h.ismembrane("Ih_linearized_v2", sec=sec):
                seg.e_pas += seg.ihcn_Ih_linearized_v2 / seg.g_pas
            if neuron.h.ismembrane("Ih_frozen", sec=sec):
                seg.e_pas += seg.ihcn_Ih_frozen / seg.g_pas


def compute_mean_nu_X(params, OUTPUTPATH, tstop, TRANSIENT=200.):
    """
    Return the population-averaged firing rate for each population X

    Parameters
    ----------
    params: module
        `<module 'example_network_parameters'>`
    OUTPUTPATH: str
        path to directory with `spikes.h5` file
    TRANSIENT: float
        startup transient duration

    Returns
    -------
    mean_nu_X: dict of floats
        keys and values denote population names and firing rates, respectively.
    """
    mean_nu_X = dict()
    for i, (X, N_X) in enumerate(zip(params.population_names,
                                     params.population_sizes)):
        with h5py.File(os.path.join(OUTPUTPATH, 'spikes.h5'), 'r') as f:
            times = np.concatenate(f[X]['times'][()])
            times = times[times >= TRANSIENT]
            mean_nu_X[X] = (times.size / N_X
                            / (tstop - TRANSIENT)
                            * 1000)
    return mean_nu_X


def integrate_beta(tau_1, tau_2):
    """
    Return the integral of the beta function from 0 to infty

    Parameters
    ----------
    tau_1: float
        rise time constant
    tau_2: float
        decay time constant. tau_2 > tau_1

    Returns
    -------
    float
    """
    tp = (tau_1 * tau_2) / (tau_2 - tau_1) * np.log(tau_2 / tau_1)
    u = np.exp(-tp / tau_2) - np.exp(-tp / tau_1)
    return (tau_2 - tau_1) / u


def quant10(x):
    return np.quantile(x, 0.1)


def quant90(x):
    return np.quantile(x, 0.9)


def csd(x, y=None, Fs=16000, NFFT=256, noverlap=192, library='mpl', **kwargs):
    '''Compute and return the cross-spectral density (``S_xy(f)``) between
    signals ``x`` and ``y``.

    Parameters
    ----------
    x: ndarray
        signal
    y: ndarray
        signal
    Fs: float, default: 16000
        The sampling frequency (samples per time unit).  It is used to
        calculate
        the Fourier frequencies, *freqs*, in cycles per time unit.
    NFFT: int, default: 256
        The number of data points used in each block for the FFT.  A power 2 is
        most efficient.  This should *NOT* be used to get zero padding, or the
        scaling of the result will be incorrect; use *pad_to* for this instead.
    noverlap: int, default: 192 (no overlap)
        The number of points of overlap between segments.
    library: str
        if ``library=='mpl'``, use matplotlib; if ``library=='scipy'``
        use scipy
    **kwargs:
        additional arguments to the wrapped functions

    Returns
    -------
    S_xy: ndarray
        cross-spectral density
    freqs: ndarray
        frequencies
    '''
    if y is None:
        y = x.copy()
        auto = True
    else:
        auto = False
    if library == 'mpl':
        S_xy, freqs = mlab.csd(x, y, Fs=Fs, NFFT=NFFT, noverlap=noverlap,
                               **kwargs)
    elif library == 'scipy':
        freqs, S_xy = ss.csd(x, y,
                             fs=Fs, nfft=NFFT, nperseg=NFFT, noverlap=noverlap,
                             **kwargs)
    if auto:
        return abs(S_xy), freqs
    else:
        return S_xy, freqs


def coherence(x, y, Fs=16000, NFFT=256, noverlap=192, library='mpl', **kwargs):
    '''Compute the coherence ``|P_xy|**2/(P_xx, P_yy)``` wrapping either
    ``plt.mlab.csd``/``plt.mlab.psd`` or ``scipy.signal.csd`` functions

    Parameters
    ----------
    x: ndarray
        signal
    y: ndarray
        signal
    Fs: float, default: 16000
        The sampling frequency (samples per time unit).  It is used to
        calculate
        the Fourier frequencies, *freqs*, in cycles per time unit.
    NFFT: int, default: 256
        The number of data points used in each block for the FFT.  A power 2 is
        most efficient.  This should *NOT* be used to get zero padding, or the
        scaling of the result will be incorrect; use *pad_to* for this instead.
    noverlap: int, default: 192 (no overlap)
        The number of points of overlap between segments.
    library: str
        if ``library=='mpl'``, use matplotlib; if ``library=='scipy'``
        use scipy
    **kwargs:
        additional arguments to the wrapped functions

    Returns
    -------
    C_xy: ndarray
        coherence
    freqs: ndarray
        frequencies
    '''
    if library == 'mpl':
        P_xy, freqs = mlab.csd(x, y, Fs=Fs, NFFT=NFFT, noverlap=noverlap,
                               **kwargs)
        P_xx, _ = mlab.psd(x, Fs=Fs, NFFT=NFFT, noverlap=noverlap, **kwargs)
        P_yy, _ = mlab.psd(y, Fs=Fs, NFFT=NFFT, noverlap=noverlap, **kwargs)
    elif library == 'scipy':
        freqs, P_xy = ss.csd(x, y,
                             fs=Fs, nfft=NFFT, nperseg=NFFT, noverlap=noverlap,
                             **kwargs)
        _, P_xx = ss.csd(x, x, fs=Fs, nfft=NFFT, nperseg=NFFT,
                         noverlap=noverlap, **kwargs)
        _, P_yy = ss.csd(y, y, fs=Fs, nfft=NFFT, nperseg=NFFT,
                         noverlap=noverlap, **kwargs)

    return abs(P_xy)**2 / (P_xx * P_yy), freqs


def zscore(x):
    '''Returns z-scored data ``z(x)=(x - mean(x)) / std(x))``

    Parameters
    ----------
    x: ndarray

    Returns
    -------
    z(x): ndarray
    '''

    return (x - x.mean()) / x.std()


def compute_nu_X(h5file, X, T=(0, 1000), Delta_t=1.):
    '''Compute the number of spikes per time bin using `np.histogram`.
    This function defines bin edges that span
    `[t_i - Delta_t / 2, t_i + Delta_t / 2]` for each time
    `t_i = T[0] + i * Delta_t` for `i = 0, 1, ...`

    Parameters
    ----------
    h5file: str
        path to `spikes.h5` file
    X: list of str
        population names
    T: tuple
        lower and upper time limits (t_min, t_max)
    Delta_t: float
        bin size (ms)

    Returns
    -------
    nu_X: dict
        keys: X
        values: ndarrays with
    bin_edges: ndarray
        bin edges
    '''
    nu_X = {}
    bins = np.linspace(T[0] - Delta_t / 2, T[1] +
                       Delta_t / 2, int(np.diff(T) / Delta_t + 2))

    with h5py.File(h5file, 'r') as f:
        for i, X_i in enumerate(X):
            times = []

            for g, t in zip(f[X_i]['gids'], f[X_i]['times']):
                times = np.r_[times, t]

            ii = (times >= T[0]) & (times <= T[1])

            hist, bin_edges = np.histogram(times[ii], bins=bins)
            nu_X[X_i] = hist

    return nu_X, bin_edges
