"""
Common methods accessed by different example codes

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

import numpy as np
import scipy.stats as st
import neuron
from LFPy import NetworkCell
from network_methods import set_active_hay2011 as set_active

# testing flag. If True, run with downsized network w. dense connectivity
TESTING = False

# class NetworkCell parameters:
cellParameters = dict(
    # morphology='BallAndStick.hoc',  # set by main simulation
    templatefile='BallAndSticksTemplate.hoc',
    templatename='BallAndSticksTemplate',
    custom_fun=[set_active],
    custom_fun_args=[dict(Vrest=-65.)],  # [dict(Vrest=Vrest)] set at runtime
    templateargs=None,
    delete_sections=False,
)


# class NetworkPopulation parameters:
populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=150.,  # population radius
        loc=0.,  # population center along z-axis
        scale=75.),  # standard deviation along z-axis
    rotation_args=dict(x=0., y=0.))

# # class Network parameters:
# networkParameters = dict(
#     dt=2**-4,  # simulation time resolution (ms)
#     tstop=12000.,  # simulation duration (ms)
#     v_init=-65.,  # initial membrane voltage for all cells (mV)
#     celsius=34,  # simulation temperature (deg. C)
#     # OUTPUTPATH=OUTPUTPATH  # set in main simulation script
# )

# class RecExtElectrode parameters:
electrodeParameters = dict(
    x=np.zeros(13),  # x-coordinates of contacts
    y=np.zeros(13),  # y-coordinates of contacts
    z=np.linspace(1000., -200., 13),  # z-coordinates of contacts
    N=np.array([[0., 1., 0.] for _ in range(13)]),  # contact surface normals
    r=5.,  # contact radius
    n=100,  # n-point averaging for potential on each contact
    sigma=0.3,  # extracellular conductivity (S/m)
    method="linesource"  # use line sources
)

# class LaminarCurrentSourceDensity parameters:
csdParameters = dict(
    z=np.c_[electrodeParameters['z'] - 50.,
            electrodeParameters['z'] + 50.],  # lower and upper boundaries
    r=np.array([populationParameters['pop_args']['radius']] * 13)  # radius
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_pop_contributions=True,  # store contributions by each population
    to_memory=True,  # simulate to memory
    to_file=False  # simulate to file
)

# population names, morphologies, sizes and connection probability:
population_names = ['E', 'I']
morphologies = ['BallAndSticks_E.hoc', 'BallAndSticks_I.hoc']
if TESTING:
    population_sizes = [32, 8]
    connectionProbability = [[1., 1.], [1., 1.]]
else:
    population_sizes = [8192, 1024]
    connectionProbability = [[0.05, 0.05], [0.05, 0.05]]

# synapse model. All corresponding parameters for weights,
# connection delays, multapses and layerwise positions are
# set up as shape (2, 2) nested lists for each possible
# connection on the form:
# [["E:E", "E:I"],
#  ["I:E", "I:I"]].
# using convention "<pre>:<post>"
synapseModel = neuron.h.Exp2Syn
# synapse parameters in terms of rise- and decay time constants
# (tau1, tau2 [ms]) and reversal potential (e [mV])
synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.),
                      dict(tau1=0.2, tau2=1.8, e=0.)],
                     [dict(tau1=0.1, tau2=9.0, e=-80.),
                      dict(tau1=0.1, tau2=9.0, e=-80.)]]
# synapse max. conductance (function, mean, st.dev., min.):
weightFunction = np.random.normal
# weight_<pre><post> values set by parameters file via main simulation scripts
# weightArguments = [[dict(loc=weight_EE, scale=weight_EE / 10),
#                     dict(loc=weight_IE, scale=weight_IE / 10)],
#                    [dict(loc=weight_EI, scale=weight_EI / 10),
#                     dict(loc=weight_II, scale=weight_II / 10)]]
minweight = 0.  # weight values below this value will be redrawn

# conduction delay (function, mean, st.dev., min.) using truncated normal
# continuous random variable:
delayFunction = st.truncnorm
delayArguments = [[dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3),
                   dict(a=(0.3 - 1.4) / 0.4, b=np.inf, loc=1.4, scale=0.4)],
                  [dict(a=(0.3 - 1.4) / 0.5, b=np.inf, loc=1.3, scale=0.5),
                   dict(a=(0.3 - 1.2) / 0.6, b=np.inf, loc=1.2, scale=0.6)]]
# will be deprecated; has no effect with delayFunction = st.truncnorm:
mindelay = None

# Distributions of multapses. They are here defined via a truncated normal
# continous random variable distribution which will be used to compute a
# discrete probability distribution for integer numbers of
# synapses 1, 2, ..., 100, via a scipy.stats.rv_discrete instance
multapseFunction = st.truncnorm
multapseArguments = [[dict(a=(1 - 2.) / .4,
                           b=(10 - 2.) / .4,
                           loc=2.,
                           scale=.4),
                      dict(a=(1 - 2.) / .6,
                           b=(10 - 2.) / .6,
                           loc=2.,
                           scale=.6)],
                     [dict(a=(1 - 5.) / 0.9,
                           b=(10 - 5.) / 0.9,
                           loc=5.,
                           scale=0.9),
                      dict(a=(1 - 5.) / 1.1,
                           b=(10 - 5.) / 1.1,
                           loc=5.,
                           scale=1.1)]]

# method NetworkCell.get_rand_idx_area_and_distribution_norm
# parameters for layerwise synapse positions:
synapsePositionArguments = [[dict(section=['apic', 'dend'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=500., scale=100.)],
                                  funweights=[0.5, 1.]
                                  ),
                             dict(section=['apic', 'dend'],
                                  fun=[st.norm],
                                  funargs=[dict(loc=50., scale=100.)],
                                  funweights=[1.]
                                  )],
                            [dict(section=['soma', 'apic', 'dend'],
                                  fun=[st.norm],
                                  funargs=[dict(loc=-50., scale=100.)],
                                  funweights=[1.]
                                  ),
                             dict(section=['soma', 'apic', 'dend'],
                                  fun=[st.norm],
                                  funargs=[dict(loc=-100., scale=100.)],
                                  funweights=[1.]
                                  )]
                            ]

# Parameters for extrinsic (e.g., cortico-cortical connections) synapses
# and mean interval (ms)
extSynapseParameters = dict(
    syntype='Exp2Syn',
    weight=0.0002,
    tau1=0.2,
    tau2=1.8,
    e=0.
)
netstim_interval = 25.
