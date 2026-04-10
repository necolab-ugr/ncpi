#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the Cavallari-adapted multicompartment network simulation.

This script intentionally stays close to the LFPykernels reference script
``Hagen_et_al_2022_PLOS_Comput_Biol/example_network.py``. The main simulation
flow, population creation, connectivity, recording probes, file outputs, and
plotting blocks are kept in the same inline style as the reference.

Intentional differences from the reference script:

* Model files are loaded from ``zenodo_dir`` instead of paths relative to the
  script directory. This includes NEURON mechanisms, template files, and
  morphologies.
* Output is written under ``zenodo_dir/output_Cavallari`` instead of
  ``output/<md5>``.
* Network parameters and helper functions come from the local
  ``analysis_params.py`` file via ``KernelParams`` instead of
  ``example_network_parameters.py``, ``parameters/<md5>.txt``,
  ``ParameterSet``, or ``ParameterSpace``.
* The command-line ``md5`` parameter sweep logic from the reference script is
  not used. This file runs one Cavallari parameterization directly.
* ``weight_scaling`` and ``save_connections`` logic have been removed for this
  Cavallari version.
* MPI/NEURON initialization follows an MPI-safe pattern:
  NEURON initializes MPI before ``mpi4py`` exposes ``MPI.COMM_WORLD``, and the
  script checks that NEURON and mpi4py agree on rank and size. Since mpi4py
  auto-finalization is disabled for this initialization order, the script also
  explicitly finalizes MPI before exiting.
* The script prints an error and exits if the number of MPI ranks exceeds the
  smallest population size. In that regime some ranks own no cells for at least
  one population, which is not safe for the extracellular probe simulation
  because NEURON's fast membrane-current bookkeeping expects participating
  ranks to have local cell state.

Execution (w. MPI):

    mpirun -np 2 python example_model_simulation.py

"""
# import modules:
import os
import shutil
import sys
from copy import deepcopy
from time import time

# Force headless operation on cluster nodes before importing GUI-sensitive modules.
os.environ.pop("DISPLAY", None)
os.environ.setdefault("MPLBACKEND", "Agg")

MPI_LAUNCH_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "PMI_SIZE",
    "PMI_RANK",
    "PMIX_RANK",
    "MPI_LOCALRANKID",
)

if "mpi4py.run" in sys.modules or "mpi4py.__main__" in sys.modules:
    raise RuntimeError(
        "Do not launch this script with 'python -m mpi4py'. "
        "Use 'mpirun -np N python example_model_simulation.py' so NEURON can initialize MPI."
    )

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

try:
    import neuron
except ImportError as exc:
    if any(var in os.environ for var in MPI_LAUNCH_ENV_VARS):
        raise RuntimeError(
            "MPI launch detected, but NEURON could not be imported."
        ) from exc
    raise

if any(var in os.environ for var in MPI_LAUNCH_ENV_VARS):
    try:
        neuron.h.nrnmpi_init()
    except AttributeError as exc:
        raise RuntimeError(
            "NEURON does not expose nrnmpi_init(); this NEURON build likely lacks MPI support."
        ) from exc

try:
    from mpi4py import MPI
except ImportError as exc:
    if any(var in os.environ for var in MPI_LAUNCH_ENV_VARS):
        raise RuntimeError(
            "MPI launch detected, but mpi4py could not be imported after NEURON initialized MPI."
        ) from exc
    raise

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from LFPy import CurrentDipoleMoment, LaminarCurrentSourceDensity, Network, RecExtElectrode, Synapse
from matplotlib.gridspec import GridSpec

from analysis_params import KernelParams as params
from ncpi import tools

matplotlib.use("Agg")

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the data is stored (must be an absolute path to correctly load morphologies in NEURON)
zenodo_dir = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "multicompartment_neuron_network")
))

# Choose to either download files and precomputed outputs used in simulations of the reference multicompartment neuron
# network model (True) or load them from a local path (False)
zenodo_dw_mult = False

NEURON_PC = neuron.h.ParallelContext()
NRN_RANK = int(NEURON_PC.id())
NRN_SIZE = int(NEURON_PC.nhost())
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if any(var in os.environ for var in MPI_LAUNCH_ENV_VARS) and SIZE <= 1:
    raise RuntimeError(
        "MPI launch environment detected, but mpi4py reports COMM_WORLD size 1."
    )

if NRN_RANK != RANK or NRN_SIZE != SIZE:
    raise RuntimeError(
        "MPI mismatch between mpi4py and NEURON ParallelContext: "
        f"mpi4py rank/size={RANK}/{SIZE}, "
        f"NEURON rank/size={NRN_RANK}/{NRN_SIZE}."
    )

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

##########################################################################
# Function declarations
##########################################################################


def remove_axis_junk(ax, lines=['right', 'top']):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter_data=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1]) * dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    # subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch.%i' % (i + 1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                [0.5, -0.5], lw=2, color='k', clip_on=False)
        # bbox = ax.get_window_extent().transformed(ax.get_figure().inverted())
        fig = ax.get_figure()
        figwidth = fig.figbbox.transformed(
            fig.dpi_scale_trans.inverted()).width
        axwidth = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()).width
        # bbox.width
        # ax.text(x[-1] + (x[-1] - x[0]) / width * 0.1, 0.5, 'test')
        ax.text(tvec[tinds][-1], 0,
        '\n\n$2^{' + '{}'.format(int(round(np.log2(vlimround)))
                                ) + '}$ ' + '{0}'.format(unit),
        color='k', rotation='vertical',
        va='center', ha='center')


    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r't (ms)', labelpad=0.1)

    return vlimround


def local_first_cell(network, population_name):
    cells = network.populations[population_name].cells
    if len(cells) == 0:
        return None
    return cells[0]


def finalize_mpi():
    if MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.Finalize()


if __name__ == '__main__':
    ##########################################################################
    # Main simulation
    ##########################################################################

    if zenodo_dw_mult and RANK == 0:
        tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
    COMM.Barrier()

    model_root = os.path.abspath(os.path.expanduser(os.path.expandvars(zenodo_dir)))
    neuron.load_mechanisms(os.path.join(model_root, 'mod'))

    #######################################
    # tic tac
    #######################################
    tic = time()

    #######################################
    # Capture command line values
    #######################################
    # fetch parameter values
    weight_EE = params.MC_params['weight_EE']
    weight_IE = params.MC_params['weight_IE']
    weight_EI = params.MC_params['weight_EI']
    weight_II = params.MC_params['weight_II']
    n_ext = params.MC_params['n_ext']

    if SIZE > min(params.population_sizes):
        if RANK == 0:
            print(
                "The number of MPI ranks must not exceed the smallest population size "
                "when recording extracellular probes. "
                f"Got MPI size {SIZE} and population_sizes={params.population_sizes}.",
                file=sys.stderr,
                flush=True,
            )
        COMM.Barrier()
        finalize_mpi()
        sys.exit(1)

    ##########################################################################
    # Set up shared and population-specific parameters
    ##########################################################################
    # path for simulation output:
    OUTPUTPATH = os.path.join(model_root, 'output_Cavallari')

    if RANK == 0:
        # create directory for output:
        if not os.path.isdir(OUTPUTPATH):
            os.mkdir(OUTPUTPATH)
        # remove old simulation output if directory exist
        else:
            for fname in os.listdir(OUTPUTPATH):
                fpath = os.path.join(OUTPUTPATH, fname)
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                else:
                    os.unlink(fpath)
    COMM.Barrier()

    # tic tac
    tac = time()
    if RANK == 0:
        with open(os.path.join(OUTPUTPATH, 'tic_tac.txt'), 'w') as f:
            f.write(f'step time_s\nsetup {tac - tic}\n')

    # synapse max. conductance (function, mean, st.dev., min.):
    weightArguments = [[dict(loc=weight_EE, scale=weight_EE / 10),
                        dict(loc=weight_IE, scale=weight_IE / 10)],
                       [dict(loc=weight_EI, scale=weight_EI / 10),
                        dict(loc=weight_II, scale=weight_II / 10)]]

    # instantiate Network:
    network = Network(OUTPUTPATH=OUTPUTPATH, **params.networkParameters)

    # create E and I populations:
    for j, (name, size, morphology) in enumerate(zip(params.population_names,
                                                     params.population_sizes,
                                                     params.morphologies)):
        popParams = deepcopy(params.populationParameters)
        popParams['cell_args'].update(dict(
            templatefile=os.path.join(model_root, params.cellParameters['templatefile']),
            morphology=os.path.join(model_root, morphology),
        ))
        network.create_population(name=name, POP_SIZE=size,
                                  **popParams)

        # create excitatory background synaptic activity for each cell
        # with Poisson statistics
        for cell in network.populations[name].cells:
            idx = cell.get_rand_idx_area_norm(section='allsec', nidx=n_ext[j])
            for i in idx:
                syn = Synapse(cell=cell, idx=i,
                              **params.extSynapseParameters[j])
                syn.set_spike_times_w_netstim(
                    interval=params.netstim_interval[j],
                    seed=np.random.rand() * 2**32 - 1)

    # tic tac
    tic = time()
    if RANK == 0:
        with open(os.path.join(OUTPUTPATH, 'tic_tac.txt'), 'a') as f:
            f.write(f'create {tic - tac}\n')

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(params.population_names):
        for j, post in enumerate(params.population_names):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=params.connectionProbability[i][j]
            )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=params.synapseModel,
                synparams=params.synapseParameters[i][j],
                weightfun=params.weightFunction,
                weightargs=weightArguments[i][j],
                minweight=params.minweight,
                delayfun=params.delayFunction,
                delayargs=params.delayArguments[i][j],
                mindelay=params.mindelay,
                multapsefun=params.multapseFunction,
                multapseargs=params.multapseArguments[i][j],
                syn_pos_args=params.synapsePositionArguments[i][j],
            )

    # tic tac
    tac = time()
    if RANK == 0:
        with open(os.path.join(OUTPUTPATH, 'tic_tac.txt'), 'a') as f:
            f.write(f'connect {tac - tic}\n')

    # record membrane voltages for 1st cell per population per MPI RANK
    for name in params.population_names:
        cell = local_first_cell(network, name)
        if cell is not None:
            cell._set_voltage_recorders(1.)

    # set up extracellular recording device.
    # Here `cell` is set to None as handles to cell geometry is handled
    # internally
    electrode = RecExtElectrode(cell=None, **params.electrodeParameters)

    # set up recording of current dipole moments. Ditto with regards to
    # `cell` being set to None
    current_dipole_moment = CurrentDipoleMoment(cell=None)

    # set up recording of current source density. Ditto with regards to
    # `cell` being set to None
    csd = LaminarCurrentSourceDensity(cell=None, **params.csdParameters)

    # run simulation:
    SPIKES = network.simulate(
        probes=[electrode, current_dipole_moment, csd],
        **params.networkSimulationArguments
    )

    # gather recorded membrane potentials
    for name in params.population_names:
        cell = local_first_cell(network, name)
        if cell is not None:
            cell._collect_vmem()

    # tic tac
    tic = time()
    if RANK == 0:
        with open(os.path.join(OUTPUTPATH, 'tic_tac.txt'), 'a') as f:
            f.write(f'simulate {tic - tac}\n')

    # save spikes
    if RANK == 0:
        with h5py.File(os.path.join(OUTPUTPATH, 'spikes.h5'), 'w') as f:
            dtype = h5py.special_dtype(vlen=np.dtype('float'))
            for i, name in enumerate(params.population_names):
                subgrp = f.create_group(name)
                if len(SPIKES['gids'][i]) > 0:
                    subgrp['gids'] = np.array(SPIKES['gids'][i]).flatten()
                    dset = subgrp.create_dataset('times',
                                                 (len(SPIKES['gids'][i]),),
                                                 dtype=dtype)
                    for j, spt in enumerate(SPIKES['times'][i]):
                        dset[j] = spt
                else:
                    subgrp['gids'] = []
                    subgrp['times'] = []

    # collect somatic potentials across all RANKs to RANK 0:
    if RANK == 0:
        somavs = []
    for i, name in enumerate(params.population_names):
        somavs_pop = None
        for j, cell in enumerate(network.populations[name].cells):
            if j == 0:
                somavs_pop = ss.decimate(cell.somav,
                                         q=int(round(1 // network.dt)),
                                         axis=-1,
                                         zero_phase=True)
            # else:
            #     somavs_pop = np.vstack((somavs_pop, cell.somav))
        if RANK == 0:
            somavs_rows = []
            if somavs_pop is not None:
                somavs_rows.append(np.atleast_2d(somavs_pop))
            for j in range(1, SIZE):
                recvbuf = COMM.recv(source=j, tag=15)
                if recvbuf is not None:
                    somavs_rows.append(np.atleast_2d(recvbuf))
            if somavs_rows:
                somavs.append(np.vstack(somavs_rows))
            else:
                somavs.append(np.empty((0, 0), dtype=float))
        else:
            COMM.send(somavs_pop, dest=0, tag=15)

    if RANK == 0:
        with h5py.File(os.path.join(OUTPUTPATH, 'somav.h5'), 'w') as f:
            for i, name in enumerate(params.population_names):
                f[name] = somavs[i]

    # store lfpykit probe data to file
    if RANK == 0:
        for probe in [electrode, current_dipole_moment, csd]:
            with h5py.File(
                os.path.join(OUTPUTPATH,
                             '{}.h5'.format(probe.__class__.__name__)), 'w'
            ) as f:
                f['data'] = probe.data

    # compute mean compartmental membrane voltages across cells and store
    if RANK == 0:
        with h5py.File(os.path.join(OUTPUTPATH, 'vmem.h5'), 'w') as f:
            pass
    for name in params.population_names:
        cell = local_first_cell(network, name)
        local_vmem = None if cell is None else cell.vmem
        shapes = COMM.gather(None if local_vmem is None else local_vmem.shape, root=0)
        if RANK == 0:
            target_shape = next((shape for shape in shapes if shape is not None), None)
        else:
            target_shape = None
        target_shape = COMM.bcast(target_shape, root=0)
        if target_shape is None:
            continue

        sendbuf = np.zeros(target_shape) if local_vmem is None else local_vmem
        if RANK == 0:
            recvbuf = np.zeros(sendbuf.shape)
        else:
            recvbuf = None
        COMM.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
        counts = COMM.gather(0 if local_vmem is None else 1, root=0)
        if RANK == 0:
            with h5py.File(os.path.join(OUTPUTPATH, 'vmem.h5'), 'a') as f:
                f[name] = recvbuf / max(sum(counts), 1)

    # tic tac
    tac = time()
    if RANK == 0:
        with open(os.path.join(OUTPUTPATH, 'tic_tac.txt'), 'a') as f:
            f.write(f'save {tac - tic}\n')

    ##########################################################################
    # Plot some output on RANK 0
    ##########################################################################

    if RANK == 0:
        # spike raster
        fig, ax = plt.subplots(1, 1)
        for name, spts, gids in zip(
                params.population_names, SPIKES['times'], SPIKES['gids']):
            t = []
            g = []
            for spt, gid in zip(spts, gids):
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size) + gid]
            inds = (t >= 500) & (t <= 1000)  # show [500, 1000] ms interval
            ax.plot(t[inds], g[inds], '.', ms=3, label=name)
        ax.legend(loc=1)
        remove_axis_junk(ax, lines=['right', 'top'])
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('gid')
        ax.set_title('spike raster')
        fig.savefig(os.path.join(OUTPUTPATH, 'spike_raster.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # somatic potentials
        fig = plt.figure()
        gs = GridSpec(4, 1)
        ax = fig.add_subplot(gs[:2])
        if len(somavs[0] > 8):
            data = somavs[0][:8]
        else:
            data = somavs[0],
        draw_lineplot(ax,
                      data,
                      dt=network.dt * int(round(1 // network.dt)),
                      T=(500, 1000),
                      scaling_factor=1.,
                      vlimround=16,
                      label='E',
                      scalebar=True,
                      unit='mV',
                      ylabels=False,
                      color='C0',
                      ztransform=True
                      )
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_ylabel('E')
        ax.set_title('somatic potentials')
        ax.set_xlabel('')

        ax = fig.add_subplot(gs[2:])
        if len(somavs[1] > 8):
            data = somavs[1][:8]
        else:
            data = somavs[1],
        draw_lineplot(ax,
                      data,
                      dt=network.dt * int(round(1 // network.dt)),
                      T=(500, 1000),
                      scaling_factor=1.,
                      vlimround=16,
                      label='I',
                      scalebar=True,
                      unit='mV',
                      ylabels=False,
                      color='C1',
                      ztransform=True
                      )
        ax.set_yticks([])
        ax.set_ylabel('I')

        fig.savefig(os.path.join(OUTPUTPATH, 'soma_potentials.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # extracellular potentials, E and I contributions, sum
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
        fig.suptitle('extracellular potentials')
        for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                                  ['E', 'I', 'sum'])):
            draw_lineplot(ax,
                          ss.decimate(electrode.data[name], q=16,
                                      zero_phase=True),
                          dt=network.dt * 16,
                          T=(500, 1000),
                          scaling_factor=1.,
                          vlimround=None,
                          label=label,
                          scalebar=True,
                          unit='mV',
                          ylabels=True if i == 0 else False,
                          color='C{}'.format(i),
                          ztransform=True
                          )
            ax.set_title(label)
        fig.savefig(os.path.join(OUTPUTPATH, 'extracellular_potential.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # current-dipole moments, E and I contributions, sum
        fig, axes = plt.subplots(3, 3, figsize=(6.4, 4.8))
        fig.subplots_adjust(wspace=0.45)
        fig.suptitle('current-dipole moments')
        for i, u in enumerate(['x', 'y', 'z']):
            for j, (name, label) in enumerate(zip(['E', 'I', 'imem'],
                                                  ['E', 'I', 'sum'])):
                t = np.arange(current_dipole_moment.data.shape[1]) * network.dt
                inds = (t >= 500) & (t <= 1000)
                axes[i, j].plot(
                    t[inds][::16],
                    ss.decimate(current_dipole_moment.data[name][i, inds],
                                q=16, zero_phase=True),
                    'C{}'.format(j))

                if j == 0:
                    axes[i, j].set_ylabel(r'$\mathbf{p}\cdot\mathbf{e}_{' +
                                          '{}'.format(u) + '}$ (nA$\\mu$m)')
                if i == 0:
                    axes[i, j].set_title(label)
                if i != 2:
                    axes[i, j].set_xticklabels([])
                else:
                    axes[i, j].set_xlabel('t (ms)')
        fig.savefig(os.path.join(OUTPUTPATH, 'current_dipole_moment.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # current source density, E and I contributions, sum
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
        fig.suptitle('CSD')
        for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                                  ['E', 'I', 'sum'])):
            draw_lineplot(ax,
                          ss.decimate(csd.data[name], q=16,
                                      zero_phase=True),
                          dt=network.dt * 16,
                          T=(500, 1000),
                          scaling_factor=1.,
                          vlimround=None,
                          label=label,
                          scalebar=True,
                          unit='nA/um$^3$',
                          ylabels=True if i == 0 else False,
                          color='C{}'.format(i),
                          ztransform=True
                          )
            ax.set_title(label)
        fig.savefig(os.path.join(OUTPUTPATH, 'csd.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

    # population illustration (per RANK)
    if RANK == 0:
        try:
            fig = plt.figure(figsize=(6.4, 4.8 * 2))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=5)
            ax.plot(electrode.x, electrode.y, electrode.z, 'ko', zorder=0)
            for i, (name, pop) in enumerate(network.populations.items()):
                for cell in pop.cells:
                    c = 'C0' if name == 'E' else 'C1'
                    for x, y, z, d in zip(cell.x, cell.y, cell.z, cell.d):
                        ax.plot(x, y, z, c, lw=d / 2)
            ax.set_xlabel(r'$x$ ($\mu$m)')
            ax.set_ylabel(r'$y$ ($\mu$m)')
            ax.set_zlabel(r'$z$ ($\mu$m)')
            ax.set_title('network populations')
            fig.savefig(os.path.join(OUTPUTPATH,
                                     'population_RANK_{}.pdf'.format(RANK)),
                        bbox_inches='tight')
        except Exception:
            pass
        plt.close(fig)

    ##########################################################################
    # customary cleanup of object references - the psection() function may not
    # write correct information if NEURON still has object references in memory
    # even if Python references has been deleted. It will also allow the script
    # to be run in successive fashion.
    ##########################################################################
    network.pc.gid_clear()  # allows assigning new gids to threads
    electrode = None
    current_dipole_moment = None
    csd = None
    syn = None
    synapseModel = None
    for population in network.populations.values():
        for cell in population.cells:
            cell.__del__()
            cell = None
        population.cells = None
        population = None
    pop = None
    network = None
    neuron.h('forall delete_section()')
    COMM.Barrier()
    finalize_mpi()
