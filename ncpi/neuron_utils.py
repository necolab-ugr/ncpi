"""
Utility functions for working with NEURON models, including functions to set up different channel configurations and
to compute population firing rates from simulation output.

Code downloaded from https://github.com/LFPy/LFPykernels/blob/main/examples/example_network_methods.py
Copyright (C) 2021 https://github.com/espenhgn
"""

import os
import numpy as np
import neuron
import h5py


def set_active(cell, Vrest):
    """Insert HH and Ih channels across cell sections

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
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
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('Ih_linearized_v2_frozen')
        sec.e_pas = Vrest
        sec.V_R_NaTa_t_frozen = Vrest
        sec.V_R_SKv3_1_frozen = Vrest
        sec.V_R_Ih_linearized_v2_frozen = Vrest
        sec.ena = Vrest  # 50
        sec.ek = Vrest  # -85
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
    Vrest: float
        Steady state potential
    """
    for sec in cell.template.all:
        sec.insert('pas')
        sec.insert('NaTa_t_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('Ih_linearized_v2')
        sec.e_pas = Vrest
        sec.V_R_Ih_linearized_v2 = Vrest
        sec.V_R_NaTa_t_frozen = Vrest
        sec.V_R_SKv3_1_frozen = Vrest
        sec.ena = Vrest  # 50
        sec.ek = Vrest  # -85

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
    for sec in cell.template.all:
        for ion in ion_channels:
            if neuron.h.ismembrane(ion, sec=sec):
                setattr(sec, f'V_R_{ion}', Vrest)


def make_cell_uniform(cell, Vrest=-65):
    """
    Adjusts e_pas to enforce a uniform resting membrane potential at Vrest

    Parameters
    ----------
    cell: object
        LFPy.NetworkCell like object
    Vrest: float
        Steady state potential
    """
    neuron.h.t = 0
    neuron.h.finitialize(Vrest)
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


def compute_nu_X(params, OUTPUTPATH, TRANSIENT=200.):
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
    nu_X: dict of floats
        keys and values denote population names and firing rates, respectively.
    """
    nu_X = dict()
    for i, (X, N_X) in enumerate(zip(params.population_names,
                                     params.population_sizes)):
        with h5py.File(os.path.join(OUTPUTPATH, 'spikes.h5'), 'r') as f:
            times = np.concatenate(f[X]['times'][()])
            times = times[times >= TRANSIENT]
            nu_X[X] = (times.size / N_X
                       / (params.networkParameters['tstop'] - TRANSIENT)
                       * 1000)
    return nu_X