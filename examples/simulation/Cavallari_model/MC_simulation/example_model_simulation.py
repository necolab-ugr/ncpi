import os
import pickle
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

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
from matplotlib.gridspec import GridSpec
from LFPy import CurrentDipoleMoment, LaminarCurrentSourceDensity, Network, RecExtElectrode, Synapse

from analysis_params import KernelParams
from ncpi import neuron_utils, tools

matplotlib.use("Agg")

# Zenodo URL that contains the data (used if zenodo_dw_mult is True)
zenodo_URL_mult = "https://zenodo.org/api/records/15429373"

# Zenodo directory where the data is stored (must be an absolute path to correctly load morphologies in NEURON)
zenodo_dir = os.path.expandvars(os.path.expanduser(
    os.path.join("$HOME", "multicompartment_neuron_network")
))

# Choose to either download files and precomputed outputs used in simulations of the reference multicompartment neuron
# network model (True) or load them from a local path (False)
zenodo_dw_mult = True

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


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTPATH = ROOT_DIR / "output_Cavallari"
GLOBAL_SEED = 1234
np.random.seed(GLOBAL_SEED + RANK)


def _log(message):
    print(f"[rank {RANK}/{SIZE}] {message}", flush=True)


def _prepare_output_directory():
    _log(f"Preparing output directory at {OUTPUTPATH}")
    if RANK == 0:
        OUTPUTPATH.mkdir(parents=True, exist_ok=True)
        for entry in OUTPUTPATH.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        _log("Output directory prepared")
    COMM.Barrier()
    _log("Output directory barrier released")


def _ensure_model_data():
    if zenodo_dw_mult and RANK == 0:
        _log("Downloading multicompartment reference data from Zenodo")
        start_time = time.time()
        tools.download_zenodo_record(zenodo_URL_mult, download_dir=zenodo_dir)
        end_time = time.time()
        _log(f"Zenodo download completed in {(end_time - start_time) / 60:.2f} minutes")
    COMM.Barrier()

    model_root = Path(zenodo_dir).expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"Multicompartment model directory not found: {model_root}")
    _log(f"Using multicompartment model root {model_root}")
    return model_root


def _load_mechanisms(model_root):
    mod_dir = model_root / "mod"
    if not mod_dir.is_dir():
        raise FileNotFoundError(f"Mechanism folder not found: {mod_dir}")

    _log(f"Loading NEURON mechanisms from {mod_dir}")
    if neuron.load_mechanisms(str(mod_dir)):
        _log("NEURON mechanisms already available")
        return

    if RANK == 0:
        _log("Compiling NEURON mechanisms with nrnivmodl")
        subprocess.run(["nrnivmodl"], check=True, cwd=str(mod_dir))
    COMM.Barrier()
    if not neuron.load_mechanisms(str(mod_dir)):
        raise RuntimeError(f"Unable to load NEURON mechanisms from {mod_dir}")
    _log("NEURON mechanisms compiled and loaded")


def _resolve_weight_arguments():
    _log("Resolving synaptic weight distributions")
    mc_params = KernelParams.MC_params
    weight_ee = mc_params["weight_EE"]
    weight_ie = mc_params["weight_IE"]
    weight_ei = mc_params["weight_EI"]
    weight_ii = mc_params["weight_II"]
    weight_scaling = mc_params.get("weight_scaling", 1.0)

    weight_ee *= weight_scaling ** (weight_ei / weight_ee)
    weight_ie *= weight_scaling ** (weight_ii / weight_ie)
    weight_ei *= weight_scaling ** (weight_ee / weight_ei)
    weight_ii *= weight_scaling ** (weight_ie / weight_ii)

    return [
        [
            dict(loc=weight_ee, scale=abs(weight_ee) / 10.0),
            dict(loc=weight_ie, scale=abs(weight_ie) / 10.0),
        ],
        [
            dict(loc=weight_ei, scale=abs(weight_ei) / 10.0),
            dict(loc=weight_ii, scale=abs(weight_ii) / 10.0),
        ],
    ]


def _local_first_cell(network, population_name):
    cells = network.populations[population_name].cells
    if not cells:
        return None
    return cells[0]


def _expected_local_gids(first_gid, pop_size):
    return [
        gid for gid in range(first_gid, first_gid + pop_size)
        if gid % SIZE == RANK
    ]


def _save_spikes(spikes):
    _log("Saving spike output to spikes.h5")
    with h5py.File(OUTPUTPATH / "spikes.h5", "w") as handle:
        dtype = h5py.special_dtype(vlen=np.dtype("float"))
        for index, name in enumerate(KernelParams.population_names):
            group = handle.create_group(name)
            gids = spikes["gids"][index]
            times = spikes["times"][index]
            if len(gids) > 0:
                group["gids"] = np.asarray(gids).flatten()
                dataset = group.create_dataset("times", (len(gids),), dtype=dtype)
                for row, spike_times in enumerate(times):
                    dataset[row] = spike_times
            else:
                group["gids"] = []
                group["times"] = []


def remove_axis_junk(ax, lines=('right', 'top')):
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_lineplot(
    ax,
    data,
    dt=0.1,
    T=(0, 200),
    scaling_factor=1.0,
    vlimround=None,
    label='local',
    scalebar=True,
    unit='mV',
    ylabels=True,
    color='r',
    ztransform=True,
    filter_data=False,
    filterargs=None,
):
    if filterargs is None:
        filterargs = dict(N=2, Wn=0.02, btype='lowpass')

    tvec = np.arange(data.shape[1]) * dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    if ztransform:
        data_t = data.T - data.mean(axis=1)
        data = data_t.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.0 ** np.round(np.log2(vlim)) / scaling_factor

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(
                tvec[tinds],
                data[i][tinds] / vlimround + z,
                rasterized=False,
                label=label,
                clip_on=False,
                color=color,
            )
        else:
            ax.plot(
                tvec[tinds],
                data[i][tinds] / vlimround + z,
                rasterized=False,
                clip_on=False,
                color=color,
            )
        yticklabels.append(f'ch.{i + 1}')
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[tinds][-1], tvec[tinds][-1]], [0.5, -0.5], lw=2, color='k', clip_on=False)
        ax.text(
            tvec[tinds][-1],
            0,
            '\n\n$2^{' + f'{int(round(np.log2(vlimround)))}' + '}$ ' + f'{unit}',
            color='k',
            rotation='vertical',
            va='center',
            ha='center',
        )

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=('right', 'top'))
    ax.set_xlabel(r't (ms)', labelpad=0.1)
    return vlimround


def _save_probe_data(probes):
    _log("Saving probe outputs")
    for probe in probes:
        _log(f"Writing probe data for {probe.__class__.__name__}")
        with h5py.File(OUTPUTPATH / f"{probe.__class__.__name__}.h5", "w") as handle:
            handle["data"] = probe.data


def _save_recorded_potentials(network):
    _log("Collecting recorded somatic potentials")
    somavs = []
    decimation = max(1, int(round(1.0 / float(network.dt))))
    for population_name in KernelParams.population_names:
        local_cell = _local_first_cell(network, population_name)
        if local_cell is None:
            somavs_pop = None
        else:
            somavs_pop = ss.decimate(
                local_cell.somav,
                q=decimation,
                axis=-1,
                zero_phase=True,
            )
            if somavs_pop.ndim == 1:
                somavs_pop = somavs_pop[np.newaxis, :]
        if RANK == 0:
            if somavs_pop is None:
                somavs_pop = np.empty((0, 0), dtype=float)
            for source in range(1, SIZE):
                recvbuf = COMM.recv(source=source, tag=15)
                if recvbuf is not None and np.size(recvbuf) > 0:
                    if somavs_pop.size == 0:
                        somavs_pop = recvbuf
                    else:
                        somavs_pop = np.vstack((somavs_pop, recvbuf))
            somavs.append(somavs_pop)
        else:
            COMM.send(somavs_pop, dest=0, tag=15)

    if RANK == 0:
        _log("Writing somav.h5")
        with h5py.File(OUTPUTPATH / "somav.h5", "w") as handle:
            for population_name, somav in zip(KernelParams.population_names, somavs):
                handle[population_name] = somav
        return somavs
    return None


def _save_mean_vmem(network):
    _log("Saving mean compartment voltages to vmem.h5")
    if RANK == 0:
        with h5py.File(OUTPUTPATH / "vmem.h5", "w"):
            pass

    for population_name in KernelParams.population_names:
        local_cell = _local_first_cell(network, population_name)
        if local_cell is None:
            local_vmem = None
        else:
            local_vmem = np.asarray(local_cell.vmem, dtype=float)

        shapes = COMM.gather(None if local_vmem is None else local_vmem.shape, root=0)
        if RANK == 0:
            target_shape = next((shape for shape in shapes if shape is not None), None)
        else:
            target_shape = None
        target_shape = COMM.bcast(target_shape, root=0)

        if target_shape is None:
            continue

        sendbuf = np.zeros(target_shape, dtype=float) if local_vmem is None else local_vmem
        recvbuf = np.zeros(target_shape, dtype=float) if RANK == 0 else None
        COMM.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

        counts = COMM.gather(0 if local_vmem is None else 1, root=0)
        if RANK == 0:
            divisor = max(sum(counts), 1)
            with h5py.File(OUTPUTPATH / "vmem.h5", "a") as handle:
                handle[population_name] = recvbuf / divisor


def _plot_outputs(network, spikes, somavs, electrode, current_dipole_moment, csd):
    if RANK != 0:
        return

    _log("Creating summary plots")

    fig, ax = plt.subplots(1, 1)
    for name, spts, gids in zip(KernelParams.population_names, spikes['times'], spikes['gids']):
        t = []
        g = []
        for spt, gid in zip(spts, gids):
            t = np.r_[t, spt]
            g = np.r_[g, np.zeros(spt.size) + gid]
        inds = (t >= 500) & (t <= 1000)
        ax.plot(t[inds], g[inds], '.', ms=3, label=name)
    ax.legend(loc=1)
    remove_axis_junk(ax, lines=('right', 'top'))
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('gid')
    ax.set_title('spike raster')
    fig.savefig(OUTPUTPATH / 'spike_raster.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    gs = GridSpec(4, 1)
    ax = fig.add_subplot(gs[:2])
    data = somavs[0][:8] if len(somavs[0]) > 8 else somavs[0]
    draw_lineplot(
        ax,
        data,
        dt=network.dt * max(1, int(round(1.0 / float(network.dt)))),
        T=(500, 1000),
        scaling_factor=1.0,
        vlimround=16,
        label='E',
        scalebar=True,
        unit='mV',
        ylabels=False,
        color='C0',
        ztransform=True,
    )
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_ylabel('E')
    ax.set_title('somatic potentials')
    ax.set_xlabel('')

    ax = fig.add_subplot(gs[2:])
    data = somavs[1][:8] if len(somavs[1]) > 8 else somavs[1]
    draw_lineplot(
        ax,
        data,
        dt=network.dt * max(1, int(round(1.0 / float(network.dt)))),
        T=(500, 1000),
        scaling_factor=1.0,
        vlimround=16,
        label='I',
        scalebar=True,
        unit='mV',
        ylabels=False,
        color='C1',
        ztransform=True,
    )
    ax.set_yticks([])
    ax.set_ylabel('I')
    fig.savefig(OUTPUTPATH / 'soma_potentials.png', bbox_inches='tight')
    plt.close(fig)

    try:
        fig = plt.figure(figsize=(6.4, 4.8 * 2))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=5)
        ax.plot(electrode.x, electrode.y, electrode.z, 'ko', zorder=0)
        for name, pop in network.populations.items():
            for cell in pop.cells:
                c = 'C0' if name == 'E' else 'C1'
                for x, y, z, d in zip(cell.x, cell.y, cell.z, cell.d):
                    ax.plot(x, y, z, c, lw=d / 2)
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_zlabel(r'$z$ ($\mu$m)')
        ax.set_title('network populations')
        fig.savefig(OUTPUTPATH / f'population_RANK_{RANK}.png', bbox_inches='tight')
        plt.close(fig)
    except Exception:
        plt.close('all')

    _log("Summary plots saved")


def _save_runtime_metadata(model_root):
    if RANK != 0:
        return

    _log("Computing runtime metadata")
    mean_nu_x = neuron_utils.compute_nu_X(KernelParams, str(OUTPUTPATH), KernelParams.transient)
    with h5py.File(OUTPUTPATH / "somav.h5", "r") as handle:
        v_rest = {
            population_name: float(np.median(handle[population_name][()][:, 200:]))
            for population_name in KernelParams.population_names
            if handle[population_name].shape[0] > 0 and handle[population_name].shape[1] > 200
        }

    metadata = {
        "zenodo_url": zenodo_URL_mult,
        "model_root": str(model_root),
        "population_names": list(KernelParams.population_names),
        "population_sizes": list(KernelParams.population_sizes),
        "MC_params": dict(KernelParams.MC_params),
        "mean_nu_X": mean_nu_x,
        "Vrest": v_rest,
    }
    with open(OUTPUTPATH / "mc_metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)
    _log("Saved mc_metadata.pkl")


if __name__ == "__main__":
    _log(
        "Starting multicompartment Cavallari simulation "
        f"on pid {os.getpid()} "
        f"(mpi4py={RANK}/{SIZE}, neuron={NRN_RANK}/{NRN_SIZE})"
    )

    model_root = _ensure_model_data()
    _load_mechanisms(model_root)

    tic = time.time()
    _prepare_output_directory()
    tac = time.time()
    if RANK == 0:
        with open(OUTPUTPATH / "tic_tac.txt", "w", encoding="utf-8") as handle:
            handle.write("step time_s\n")
            handle.write(f"setup {tac - tic}\n")

    weight_arguments = _resolve_weight_arguments()

    _log("Creating LFPy Network object")
    network_parameters = deepcopy(KernelParams.networkParameters)
    network_parameters["OUTPUTPATH"] = str(OUTPUTPATH)
    network = Network(**network_parameters)

    n_ext = KernelParams.MC_params["n_ext"]
    save_connections = False
    first_gid = 0
    for population_index, (name, size, morphology) in enumerate(
        zip(KernelParams.population_names, KernelParams.population_sizes, KernelParams.morphologies)
    ):
        local_gids = _expected_local_gids(first_gid=first_gid, pop_size=size)
        _log(
            f"Starting creation of population {name} "
            f"(global size={size}, expected local cells={len(local_gids)})"
        )
        if local_gids:
            _log(
                f"Population {name}: local gid range on this rank starts at "
                f"{local_gids[0]} and ends at {local_gids[-1]}"
            )
        else:
            _log(f"Population {name}: no local gids assigned to this rank")

        pop_params = deepcopy(KernelParams.populationParameters)
        pop_params["cell_args"].update(
            dict(
                templatefile=str(model_root / KernelParams.cellParameters["templatefile"]),
                morphology=str(model_root / morphology),
            )
        )
        network.create_population(name=name, POP_SIZE=size, **pop_params)
        _log(
            f"Population {name}: rank {RANK} instantiated "
            f"{len(network.populations[name].cells)} local neurons"
        )

        _log(f"Attaching {n_ext[population_index]} external synapses per local {name} cell")
        ext_synapse_parameters = KernelParams.extSynapseParameters[population_index]
        netstim_interval = KernelParams.netstim_interval[population_index]
        for cell in network.populations[name].cells:
            idx = cell.get_rand_idx_area_norm(section='allsec', nidx=n_ext[population_index])
            for syn_idx in idx:
                syn = Synapse(cell=cell, idx=syn_idx, **ext_synapse_parameters)
                syn.set_spike_times_w_netstim(
                    interval=netstim_interval,
                    seed=np.random.rand() * 2 ** 32 - 1,
                )
        _log(f"Finished external synapse setup for population {name}")
        first_gid += size

    tic = time.time()
    if RANK == 0:
        with open(OUTPUTPATH / "tic_tac.txt", "a", encoding="utf-8") as handle:
            handle.write(f"create {tic - tac}\n")

    for pre_index, pre in enumerate(KernelParams.population_names):
        for post_index, post in enumerate(KernelParams.population_names):
            _log(f"Building connectivity {pre} -> {post}")
            connectivity = network.get_connectivity_rand(
                pre=pre,
                post=post,
                connprob=KernelParams.connectionProbability[pre_index][post_index],
            )
            network.connect(
                pre=pre,
                post=post,
                connectivity=connectivity,
                syntype=KernelParams.synapseModel,
                synparams=KernelParams.synapseParameters[pre_index][post_index],
                weightfun=KernelParams.weightFunction,
                weightargs=weight_arguments[pre_index][post_index],
                minweight=KernelParams.minweight,
                delayfun=KernelParams.delayFunction,
                delayargs=KernelParams.delayArguments[pre_index][post_index],
                mindelay=KernelParams.mindelay,
                multapsefun=KernelParams.multapseFunction,
                multapseargs=KernelParams.multapseArguments[pre_index][post_index],
                syn_pos_args=KernelParams.synapsePositionArguments[pre_index][post_index],
                save_connections=save_connections,
            )
            _log(f"Finished connectivity {pre} -> {post}")

    tac = time.time()
    if RANK == 0:
        with open(OUTPUTPATH / "tic_tac.txt", "a", encoding="utf-8") as handle:
            handle.write(f"connect {tac - tic}\n")

    _log("Setting voltage recorders")
    for population_name in KernelParams.population_names:
        local_cell = _local_first_cell(network, population_name)
        if local_cell is not None:
            local_cell._set_voltage_recorders(1.0)

    _log("Creating probes")
    electrode = RecExtElectrode(cell=None, **KernelParams.electrodeParameters)
    current_dipole_moment = CurrentDipoleMoment(cell=None)
    csd = LaminarCurrentSourceDensity(cell=None, **KernelParams.csdParameters)

    _log("Starting network simulation")
    spikes = network.simulate(
        probes=[electrode, current_dipole_moment, csd],
        **KernelParams.networkSimulationArguments,
    )

    for population_name in KernelParams.population_names:
        local_cell = _local_first_cell(network, population_name)
        if local_cell is not None:
            local_cell._collect_vmem()

    tic = time.time()
    if RANK == 0:
        with open(OUTPUTPATH / "tic_tac.txt", "a", encoding="utf-8") as handle:
            handle.write(f"simulate {tic - tac}\n")

    if RANK == 0:
        _save_spikes(spikes)
    somavs = _save_recorded_potentials(network)
    if RANK == 0:
        _save_probe_data([electrode, current_dipole_moment, csd])
    _save_mean_vmem(network)

    tac = time.time()
    if RANK == 0:
        with open(OUTPUTPATH / "tic_tac.txt", "a", encoding="utf-8") as handle:
            handle.write(f"save {tac - tic}\n")
        _save_runtime_metadata(model_root)
        _plot_outputs(network, spikes, somavs, electrode, current_dipole_moment, csd)

    _log("Cleaning up NEURON/LFPy objects")
    network.pc.gid_clear()
    for population in network.populations.values():
        for cell in population.cells:
            cell.__del__()
    neuron.h("forall delete_section()")
    _log("Simulation script finished")
