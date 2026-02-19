import importlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest
from ncpi import FieldPotential

def _resolve_examples_dir():
    env_path = os.environ.get("LFPYKERNELS_EXAMPLES")
    if env_path:
        return Path(env_path).expanduser()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        candidate = Path(workspace) / "tests" / "data" / "LFPykernels" / "examples"
        if candidate.exists():
            return candidate
    return Path(os.path.expanduser("~/LFPykernels/LFPykernels-main/examples"))


EXAMPLES_DIR = _resolve_examples_dir()
EXAMPLES_URL = os.environ.get(
    "LFPYKERNELS_EXAMPLES_URL",
    "https://github.com/LFPy/LFPykernels/archive/refs/heads/main.tar.gz",
)


def _import_example_modules():
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    params = importlib.import_module("example_network_parameters")
    methods = importlib.import_module("example_network_methods")
    return params, methods


def _ensure_examples_downloaded():
    if EXAMPLES_DIR.exists():
        return True
    target_parent = EXAMPLES_DIR.parent.parent
    try:
        target_parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tar_path = target_parent / "lfpykernels_examples.tar.gz"
    try:
        with urllib.request.urlopen(EXAMPLES_URL, timeout=60) as response:
            with open(tar_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_parent)
    except Exception:
        return False
    finally:
        try:
            tar_path.unlink()
        except OSError:
            pass
    return EXAMPLES_DIR.exists()


def _skip_if_missing_example_files():
    if not _ensure_examples_downloaded():
        pytest.skip(
            "Missing LFPykernels examples and unable to download them; "
            "set LFPYKERNELS_EXAMPLES or LFPYKERNELS_EXAMPLES_URL."
        )
    if not EXAMPLES_DIR.exists():
        pytest.skip(f"Missing LFPykernels examples at {EXAMPLES_DIR}")
    required = [
        EXAMPLES_DIR / "BallAndSticksTemplate.hoc",
        EXAMPLES_DIR / "BallAndSticks_E.hoc",
        EXAMPLES_DIR / "BallAndSticks_I.hoc",
        EXAMPLES_DIR / "mod",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Missing LFPykernels example files: {missing}")


def _ensure_neuron_mechanisms(mod_dir):
    import neuron

    def _load_mechanisms(path):
        try:
            return neuron.load_mechanisms(str(path))
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "already exists" in msg:
                if "exp2syni" in msg:
                    return "duplicate_exp2syni"
                return True
            raise

    def _compile_without_exp2syni(source_dir):
        source_dir = Path(source_dir)
        mod_files = [mod for mod in source_dir.glob("*.mod") if mod.name != "exp2synI.mod"]
        if not mod_files:
            raise RuntimeError("No mod files available after excluding exp2synI.mod.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_mod_dir = Path(tmp_dir) / "mod"
            tmp_mod_dir.mkdir()
            for mod in mod_files:
                shutil.copy(mod, tmp_mod_dir / mod.name)
            subprocess.run(["nrnivmodl"], check=True, cwd=str(tmp_mod_dir))
            mech_loaded = _load_mechanisms(tmp_mod_dir)
            if mech_loaded is True:
                return
            raise RuntimeError(f"Failed to load NEURON mechanisms from {tmp_mod_dir}.")

    mech_loaded = _load_mechanisms(mod_dir)
    if mech_loaded is True:
        return
    if mech_loaded == "duplicate_exp2syni":
        _compile_without_exp2syni(mod_dir)
        return
    subprocess.run(["nrnivmodl"], check=True, cwd=str(mod_dir))
    mech_loaded = _load_mechanisms(mod_dir)
    if mech_loaded is True:
        return
    if mech_loaded == "duplicate_exp2syni":
        _compile_without_exp2syni(mod_dir)
        return
    raise RuntimeError(f"Failed to load NEURON mechanisms from {mod_dir}.")


def test_compute_proxy_excitatory_key_selection():
    fp = FieldPotential()
    sim_data = {
        "FR": np.array([[7.0, 8.0, 9.0]]),
        "FR_exc": np.array([[1.0, 2.0, 3.0]]),
        "FR_all": np.array([[10.0, 20.0, 30.0]]),
        "AMPA": np.array([[3.0, 3.0, 3.0]]),
        "AMPA_exc": np.array([[1.0, 1.0, 1.0]]),
        "AMPA_all": np.array([[2.0, 2.0, 2.0]]),
        "GABA": np.array([[1.0, 1.0, 1.0]]),
        "GABA_exc": np.array([[0.5, 0.5, 0.5]]),
        "GABA_all": np.array([[1.5, 1.5, 1.5]]),
        "Vm": np.array([[2.0, 3.0, 4.0]]),
        "Vm_exc": np.array([[1.0, 2.0, 3.0]]),
        "Vm_all": np.array([[4.0, 5.0, 6.0]]),
    }

    fr_base = fp.compute_proxy("FR", sim_data, sim_step=1.0)
    fr_exc = fp.compute_proxy("FR", sim_data, sim_step=1.0, excitatory_only=True)
    fr_all = fp.compute_proxy("FR", sim_data, sim_step=1.0, excitatory_only=False)
    np.testing.assert_allclose(fr_base, np.array([7.0, 8.0, 9.0]))
    np.testing.assert_allclose(fr_exc, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(fr_all, np.array([10.0, 20.0, 30.0]))

    ampa_base = fp.compute_proxy("AMPA", sim_data, sim_step=1.0)
    ampa_exc = fp.compute_proxy("AMPA", sim_data, sim_step=1.0, excitatory_only=True)
    ampa_all = fp.compute_proxy("AMPA", sim_data, sim_step=1.0, excitatory_only=False)
    np.testing.assert_allclose(ampa_base, np.array([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(ampa_exc, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(ampa_all, np.array([2.0, 2.0, 2.0]))

    vm_base = fp.compute_proxy("Vm", sim_data, sim_step=1.0)
    vm_exc = fp.compute_proxy("Vm", sim_data, sim_step=1.0, excitatory_only=True)
    vm_all = fp.compute_proxy("Vm", sim_data, sim_step=1.0, excitatory_only=False)
    np.testing.assert_allclose(vm_base, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(vm_exc, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(vm_all, np.array([4.0, 5.0, 6.0]))


def test_compute_proxy_lrws_erws_matches_formula():
    fp = FieldPotential()
    sim_step = 0.1
    ampa = np.arange(20, dtype=float)[None, :]
    gaba = np.ones_like(ampa)
    sim_data = {
        "AMPA": ampa,
        "GABA": gaba,
        "nu_ext": 5.0,
    }

    # LRWS: tau_AMPA=6ms, alpha=1.65
    delay = int(6.0 / sim_step)
    ampa_delayed = np.array([fp.roll_with_zeros(ampa[0], delay)])
    expected_lrws = np.sum(ampa_delayed - 1.65 * gaba, axis=0)
    lrws = fp.compute_proxy("LRWS", sim_data, sim_step=sim_step, excitatory_only=False)
    np.testing.assert_allclose(lrws, expected_lrws)

    # ERWS1 (non-causal): tau_AMPA=-0.9ms, tau_GABA=2.3ms, alpha=0.3
    ampa_delay = -int(0.9 / sim_step)
    gaba_delay = int(2.3 / sim_step)
    ampa_delayed = np.array([fp.roll_with_zeros(ampa[0], ampa_delay)])
    gaba_delayed = np.array([fp.roll_with_zeros(gaba[0], gaba_delay)])
    expected_erws1 = np.sum(ampa_delayed - 0.3 * gaba_delayed, axis=0)
    erws1 = fp.compute_proxy("ERWS1", sim_data, sim_step=sim_step, excitatory_only=False)
    np.testing.assert_allclose(erws1, expected_erws1)

    # ERWS2 (non-causal): params from Table 1 in PLOS Comput Biol 17(4):e1008893
    coeff = [-0.6, 0.1, -0.4, -1.9, 0.6, 3.0, 1.4, 1.7, 0.2]
    nu_ext = sim_data["nu_ext"]
    ampa_delay = int(coeff[0] * np.power(nu_ext, -coeff[1]) + coeff[2])
    gaba_delay = int(coeff[3] * np.power(nu_ext, -coeff[4]) + coeff[5])
    alpha = coeff[6] * np.power(nu_ext, -coeff[7]) + coeff[8]
    ampa_delayed = np.array([fp.roll_with_zeros(ampa[0], ampa_delay)])
    gaba_delayed = np.array([fp.roll_with_zeros(gaba[0], gaba_delay)])
    expected_erws2 = np.sum(ampa_delayed - alpha * gaba_delayed, axis=0)
    erws2 = fp.compute_proxy("ERWS2", sim_data, sim_step=sim_step, excitatory_only=False)
    np.testing.assert_allclose(erws2, expected_erws2)


def test_compute_meeg_four_sphere_matches_lfpykit():
    pytest.importorskip("lfpykit")
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    fp = FieldPotential()
    radii = [79000.0, 80000.0, 85000.0, 90000.0]
    sigmas = [0.3, 1.5, 0.015, 0.3]
    r_electrodes = np.array([[0.0, 0.0, 90000.0], [0.0, 85000.0, 0.0]])
    dipole_location = np.array([0.0, 0.0, 78000.0])
    p = np.array([[10.0] * 10, [10.0] * 10, [10.0] * 10])

    model = FourSphereVolumeConductor(r_electrodes, radii=radii, sigmas=sigmas)
    expected = model.get_dipole_potential(p, dipole_location)
    result = fp.compute_MEEG(
        p,
        dipole_locations=dipole_location,
        sensor_locations=r_electrodes,
        model="FourSphereVolumeConductor",
        model_kwargs={"radii": radii, "sigmas": sigmas},
    )
    np.testing.assert_allclose(result, expected)


def test_compute_meeg_nyhead_matches_lfpykit():
    pytest.importorskip("lfpykit")
    from lfpykit.eegmegcalc import NYHeadModel

    fp = FieldPotential()
    locs, _ = fp._get_eeg_1020_locations()
    dipole_location = locs[0]
    p = np.array([[1.0], [2.0], [3.0]])

    try:
        nyhead = NYHeadModel()
    except Exception as exc:
        pytest.skip(f"NYHeadModel unavailable: {exc}")

    for align in (True, False):
        nyhead.set_dipole_pos(dipole_location)
        M = nyhead.get_transformation_matrix()
        p_use = nyhead.rotate_dipole_to_surface_normal(p) if align else p
        expected = M @ p_use
        result = fp.compute_MEEG(
            p,
            dipole_locations=dipole_location,
            model="NYHeadModel",
            align_to_surface=align,
        )
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected)


def test_compute_meeg_infinite_volume_matches_lfpykit():
    pytest.importorskip("lfpykit")
    from lfpykit.eegmegcalc import InfiniteVolumeConductor

    fp = FieldPotential()
    p = np.array([[10.0], [10.0], [10.0]])
    r = np.array([[1000.0, 0.0, 5000.0]])
    dipole_location = np.zeros(3)

    model = InfiniteVolumeConductor(sigma=0.3)
    expected = model.get_dipole_potential(p, r)
    result = fp.compute_MEEG(
        p,
        dipole_locations=dipole_location,
        sensor_locations=r,
        model="InfiniteVolumeConductor",
        model_kwargs={"sigma": 0.3},
    )
    np.testing.assert_allclose(result, expected)


def test_compute_meeg_meg_models_match_lfpykit():
    pytest.importorskip("lfpykit")
    from lfpykit.eegmegcalc import InfiniteHomogeneousVolCondMEG, SphericallySymmetricVolCondMEG

    fp = FieldPotential()
    p = np.array([[0.0], [1.0], [0.0]])
    dipole_location = np.array([0.0, 0.0, 90000.0])
    sensor_locations = np.array([[0.0, 0.0, 92000.0]])

    meg = InfiniteHomogeneousVolCondMEG(sensor_locations)
    expected_inf = meg.calculate_H(p, dipole_location)
    result_inf = fp.compute_MEEG(
        p,
        dipole_locations=dipole_location,
        sensor_locations=sensor_locations,
        model="InfiniteHomogeneousVolCondMEG",
    )
    np.testing.assert_allclose(result_inf, expected_inf)

    meg = SphericallySymmetricVolCondMEG(sensor_locations)
    expected_sph = meg.calculate_H(p, dipole_location)
    result_sph = fp.compute_MEEG(
        p,
        dipole_locations=dipole_location,
        sensor_locations=sensor_locations,
        model="SphericallySymmetricVolCondMEG",
    )
    np.testing.assert_allclose(result_sph, expected_sph)


@pytest.mark.slow
def test_create_kernel_pairing_lfpykernels_example():
    pytest.importorskip("lfpykernels")
    pytest.importorskip("LFPy")
    pytest.importorskip("neuron")
    if shutil.which("nrnivmodl") is None:
        pytest.skip("nrnivmodl not found; NEURON mechanisms cannot be compiled.")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Flaky on GitHub Actions due to NEURON mechanism conflicts.")

    _skip_if_missing_example_files()
    _ensure_neuron_mechanisms(EXAMPLES_DIR / "mod")
    params, methods = _import_example_modules()

    fp = FieldPotential()
    mean_nu_X = {name: 1.0 for name in params.population_names}
    weights = [[0.0001, 0.0001], [0.0001, 0.0001]]
    t_X = 4.0
    tau = 4.0
    dt = 0.5

    from copy import deepcopy
    from lfpykernels import GaussCylinderPotential, KernelApprox, KernelApproxCurrentDipoleMoment

    cell_params = deepcopy(params.cellParameters)
    cell_params["templatefile"] = str(EXAMPLES_DIR / cell_params["templatefile"])
    morph = str(EXAMPLES_DIR / params.morphologies[0])
    cell_params.update(
        dict(
            morphology=morph,
            custom_fun=[methods.set_frozen_hay2011, methods.make_cell_uniform],
            custom_fun_args=[dict(Vrest=-65.0), dict(Vrest=-65.0)],
        )
    )
    synapse_parameters = [
        dict(weight=weights[ii][0], syntype="Exp2Syn", **params.synapseParameters[ii][0])
        for ii in range(len(params.population_names))
    ]
    synapse_pos_args = [params.synapsePositionArguments[ii][0] for ii in range(len(params.population_names))]

    kernel = KernelApprox(
        X=params.population_names,
        Y=params.population_names[0],
        N_X=np.array(params.population_sizes),
        N_Y=params.population_sizes[0],
        C_YX=np.array(params.connectionProbability[0]),
        cellParameters=cell_params,
        populationParameters=params.populationParameters["pop_args"],
        multapseFunction=params.multapseFunction,
        multapseParameters=[params.multapseArguments[ii][0] for ii in range(len(params.population_names))],
        delayFunction=params.delayFunction,
        delayParameters=[params.delayArguments[ii][0] for ii in range(len(params.population_names))],
        synapseParameters=synapse_parameters,
        synapsePositionArguments=synapse_pos_args,
        extSynapseParameters=params.extSynapseParameters,
        nu_ext=1000.0 / params.netstim_interval,
        n_ext=0,
        nu_X=mean_nu_X,
    )

    gauss_cyl = GaussCylinderPotential(
        cell=None,
        z=params.electrodeParameters["z"],
        sigma=params.electrodeParameters["sigma"],
        R=params.populationParameters["pop_args"]["radius"],
        sigma_z=params.populationParameters["pop_args"]["scale"],
    )
    cdm_probe = KernelApproxCurrentDipoleMoment(cell=None)
    seed = 1234
    np.random.seed(seed)
    direct = kernel.get_kernel(
        probes=[gauss_cyl, cdm_probe],
        Vrest=-65.0,
        dt=dt,
        X=params.population_names[0],
        t_X=t_X,
        tau=tau,
        g_eff=False,
    )

    np.random.seed(seed)
    kernels = fp.create_kernel(
        MC_folder=str(EXAMPLES_DIR),
        params=params,
        biophys=[methods.set_frozen_hay2011, methods.make_cell_uniform],
        dt=dt,
        tstop=12.0,
        electrodeParameters=params.electrodeParameters,
        CDM=True,
        mean_nu_X=mean_nu_X,
        Vrest=-65.0,
        t_X=t_X,
        tau=tau,
        g_eff=False,
        n_ext=[0, 0],
        weights=weights,
    )

    key = f"{params.population_names[0]}:{params.population_names[0]}"
    assert key in kernels
    assert set(kernels[key].keys()) == set(direct.keys())
    for probe_name, arr in direct.items():
        assert kernels[key][probe_name].shape == arr.shape
        np.testing.assert_allclose(kernels[key][probe_name], arr, rtol=1e-7, atol=1e-9)


def test_compute_cdm_lfp_from_kernels_matches_manual_convolution():
    fp = FieldPotential()
    kernels = {"E:E": {"KernelApproxCurrentDipoleMoment": np.array([0.0, 1.0, 0.5])}}
    spike_times = {"E": np.array([1.0, 2.0, 2.0])}
    dt = 1.0
    tstop = 5.0
    bins = np.arange(0.0, tstop + dt, dt)
    counts, _ = np.histogram(spike_times["E"], bins=bins)
    rate = counts / (dt / 1000.0)
    expected = np.convolve(rate, kernels["E:E"]["KernelApproxCurrentDipoleMoment"], mode="same")
    result = fp.compute_cdm_lfp_from_kernels(
        kernels,
        spike_times,
        dt=dt,
        tstop=tstop,
        probe="KernelApproxCurrentDipoleMoment",
        component=None,
        mode="same",
    )
    np.testing.assert_allclose(result["E:E"], expected)
