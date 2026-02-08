import inspect
import os
import subprocess
import numpy as np
from copy import deepcopy
from ncpi import tools
from ncpi import neuron_utils


class FieldPotential:
    """
    Compute current dipole moments (CDM) and/or electrophysiological signals (LFP/EEG/MEG) from spiking network
    simulations. Two possible workflows are supported:
        1. Kernel-based CDM + LFP/EEG/MEG forward modeling
        2. Direct LFP/EEG/MEG prediction from proxies
    """

    def __init__(self):
        """
        Initialize the FieldPotential class with empty dictionaries for kernels and proxies,
        and placeholders for dependencies.
        """
        self.kernels = dict()  # Dictionary to store kernels
        self.proxies = dict()  # Dictionary to store proxies

        # Dependencies for kernel computation
        self.lfpykernels = None
        self.neuron = None  # NEURON module for loading morphologies and mechanisms
        self.h5py = None  # h5py module for reading simulation data
        self.LFPy = None  # LFPy module required by lfpykernels

        # Dependencies for EEG/MEG forward modeling
        self.eegmegcalc = None

        # NYHeadModel instance (to avoid reloading the heavy model multiple times)
        self.nyhead = None

        # Dictionary to cache EEG/MEG forward model classes
        self._eegmegcalc_models = {}

    #################################################################
    ## Helper methods for loading dependencies and format handling ##
    #################################################################

    def _load_kernel_deps(self):
        """
        Load dependencies used for kernel computation.

        This method lazily imports and caches modules on the instance:
        `lfpykernels`, `LFPy`, `h5py`, and `neuron`.
        """

        if not tools.ensure_module("LFPy", package="LFPy"):
            raise ImportError("LFPy is required for computing kernels but is not installed.")
        self.LFPy = tools.dynamic_import("LFPy")

        if not tools.ensure_module("lfpykernels", package="lfpykernels"):
            raise ImportError("lfpykernels is required for computing kernels but is not installed.")
        self.lfpykernels = tools.dynamic_import("lfpykernels")

        if not tools.ensure_module("h5py", package="h5py"):
            raise ImportError("h5py is required for computing kernels but is not installed.")
        self.h5py = tools.dynamic_import("h5py")

        if not tools.ensure_module("neuron", package="neuron"):
            raise ImportError("NEURON is required for computing kernels but is not installed.")
        self.neuron = tools.dynamic_import("neuron")


    def _load_eegmegcalc_model(self, model_name):
        """Load an EEG/MEG forward model class from lfpykit.eegmegcalc, with caching.

        Parameters
        ----------
        model_name: str
            Name of the forward model class to load (e.g., "NYHeadModel", "FourSphereVolumeConductor",
            "InfiniteVolumeConductor", "InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG").

        Returns
        -------
        model: class
            The forward model class corresponding to model_name.
        """

        # Check cache first to avoid repeated dynamic imports of lfpykit for multiple models
        if model_name in self._eegmegcalc_models:
            return self._eegmegcalc_models[model_name]

        # Check that lfpykit is available before attempting to import the eegmegcalc module
        if not tools.ensure_module("lfpykit", package="lfpykit"):
            raise ImportError("lfpykit is required for EEG/MEG forward models but is not installed.")

        # Dynamically import the eegmegcalc module from lfpykit and get the requested model class
        if self.eegmegcalc is None:
            self.eegmegcalc = tools.dynamic_import("lfpykit.eegmegcalc")
        model = getattr(self.eegmegcalc, model_name, None)
        if model is None:
            raise ImportError(f"Unable to import EEG/MEG forward model '{model_name}'.")
        self._eegmegcalc_models[model_name] = model

        return model


    def _format_cdm(self, CDM):
        """
        Normalize a CDM input to a 3xN array with components in rows.

        Parameters
        ----------
        CDM: array_like
            Current dipole moment. Must be a 2D array shaped (3, N) or (N, 3).

        Returns
        -------
        p: np.ndarray
            Array shaped (3, N) with x/y/z components in rows.
        """
        cdm = np.asarray(CDM)
        if cdm.ndim == 2:
            if cdm.shape[0] == 3:
                return cdm
            if cdm.shape[1] == 3:
                return cdm.T
        raise ValueError("CDM must be a 2D array with 3 components.")


    def _resolve_biophys(self, biophys):
        """
        Resolve biophysical setup entries into callables.

        Parameters
        ----------
        biophys: list
            A list of either callables, or strings naming functions in `ncpi.neuron_utils`.

        Returns
        -------
        resolved: list
            List of callables corresponding to the requested biophysical setup.
        """
        if not isinstance(biophys, list):
            raise TypeError("biophys must be a list.")
        resolved = []
        for item in biophys:
            if callable(item):
                resolved.append(item)
                continue
            if isinstance(item, str):
                func = getattr(neuron_utils, item, None)
                if func is None:
                    raise KeyError(f"Unknown biophys function '{item}'.")
                resolved.append(func)
                continue
            raise TypeError(
                "biophys entries must be callables or strings naming known biophys functions."
            )
        return resolved


    @staticmethod
    def roll_with_zeros(arr, shift):
        """
        Shift a 1D array, padding exposed entries with zeros.

        Parameters
        ----------
        arr: np.array
            Array to be shifted.
        shift: int
            Number of positions to shift the array.

        Returns
        -------
        result: np.array
            Shifted array.
        """
        result = np.zeros_like(arr)
        if shift > 0:
            result[shift:] = arr[:-shift]
        elif shift < 0:
            result[:shift] = arr[-shift:]
        else:
            result[:] = arr
        return result


    ############################################################
    ## Main methods for kernel creation and M/EEG computation ##
    ############################################################

    def _build_kernel_probes(self, params, electrodeParameters=None, CDM=True, probes=None, cdm_probe=None):
        """Create probe objects for kernel predictions.

        Parameters
        ----------
        params: module
            Module or object holding multicompartment network parameters.
        electrodeParameters: dict
            Electrode parameters. If None, no LFP probe is created.
        CDM: bool
            Whether to include a current dipole moment probe.
        probes: list or None
            Optional list of probe objects to use directly. If provided, this overrides
            electrodeParameters and CDM settings.
        cdm_probe: str, class, or object
            CDM probe to use when CDM=True. Supported strings:
            "KernelApproxCurrentDipoleMoment" and "CurrentDipoleMoment". If a class or
            object is provided, it is used directly.

        Returns
        -------
        probes: list
            List of probe objects to be used for kernel predictions.
        """
        if probes is not None:
            return probes

        if self.lfpykernels is None:
            self._load_kernel_deps()

        probes = []
        if electrodeParameters is not None:
            gauss_cyl_potential = self.lfpykernels.GaussCylinderPotential(
                cell=None,
                z=electrodeParameters['z'],
                sigma=electrodeParameters['sigma'],
                R=params.populationParameters['pop_args']['radius'],
                sigma_z=params.populationParameters['pop_args']['scale'],
            )
            probes.append(gauss_cyl_potential)

        if CDM:
            if cdm_probe is None:
                cdm_probe = "KernelApproxCurrentDipoleMoment"

            if inspect.isclass(cdm_probe):
                # Accept known probe classes from lfpykernels / lfpykit, but don't hard-depend on lfpykit here.
                try:
                    current_dipole_moment = cdm_probe(cell=None)
                except TypeError:
                    # Some probes may not take 'cell' kwarg
                    current_dipole_moment = cdm_probe()
            elif isinstance(cdm_probe, str):
                if cdm_probe == "KernelApproxCurrentDipoleMoment":
                    current_dipole_moment = self.lfpykernels.KernelApproxCurrentDipoleMoment(cell=None)
                elif cdm_probe == "CurrentDipoleMoment":
                    if not tools.ensure_module("lfpykit", package="lfpykit"):
                        raise ImportError("lfpykit is required for CurrentDipoleMoment but is not installed.")
                    CDP = tools.dynamic_import("lfpykit", "CurrentDipoleMoment")
                    current_dipole_moment = CDP(cell=None)
                else:
                    raise ValueError(f"Unknown CDM probe '{cdm_probe}'.")
            else:
                current_dipole_moment = cdm_probe
            probes.append(current_dipole_moment)

        return probes


    def create_kernel(
            self,
            MC_folder,
            params,
            biophys,
            dt,
            tstop,
            *,
            output_sim_path=None,
            electrodeParameters=None,
            CDM=True,
            probes=None,
            cdm_probe=None,
            mean_nu_X=None,
            Vrest=None,
            t_X=None,
            tau=None,
            g_eff=None,
            n_ext=None,
            weights=None,
    ):
        """
        Create kernels from multicompartment neuron network descriptions.

        Note: if using `output_sim_path` to load mean firing rates and resting potentials, ensure that simulations
        of the same network configuration (cell parameters, morphologies, synapse parameters, etc.) were used to
        generate the output simulation data. Mismatches in network configuration may lead to inaccurate kernels and
        predictions.

        Parameters
        ----------
        MC_folder: str
            Path to the folder containing the multicompartment neuron model descriptions (cell parameters and
            morphologies).
        params: module
            Module or object holding multicompartment network parameters
            (e.g., an example_network_parameters-like script/module).
        biophys: list
            List of biophysical membrane properties.
        dt: float
            Time step.
        tstop: float
            Simulation time.
        output_sim_path: str or None
            Path to the output folder containing multicompartment network simulation outputs.
            If None, both mean_nu_X and Vrest must be provided.
        electrodeParameters: dict
            Electrode parameters. If None, no LFP is computed.
        CDM: bool
            Compute the current dipole moment.
        probes: list or None
            Optional list of probe objects to use directly. If provided, this overrides
            electrodeParameters and CDM settings.
        cdm_probe: str or class or object
            CDM probe to use when CDM=True. Supported strings:
            "KernelApproxCurrentDipoleMoment" and "CurrentDipoleMoment".
        mean_nu_X: dict or None
            Optional mean firing rates per presynaptic population (Hz). Must be provided
            together with Vrest if not using output_sim_path.
        Vrest: float, dict, sequence, or None
            Optional resting membrane potential. If dict, keys are population names.
            Must be provided together with mean_nu_X if not using output_sim_path.
        t_X: float or None
            Optional presynaptic activation time. Defaults to params.transient.
        tau: float or None
            Optional kernel time lag. Defaults to params.tau.
        g_eff: bool or None
            If True, account for conductance effects in kernel computation. Defaults to params.MC_params['g_eff'].
        n_ext: list or None
            Optional list of external synapse counts per population. Defaults to params.MC_params['n_ext'].
        weights: list or None
            Optional 2x2 weight matrix to override params.MC_params weights. Defaults to [[weight_EE, weight_IE],
            [weight_EI, weight_II]] from params.MC_params.
        Returns
        -------
        kernels: dict
            Dictionary containing the kernels keyed by "Y:X", where Y is the postsynaptic population and X is the
            presynaptic population.
        """

        # Some basic sanity checks on inputs
        if not os.path.exists(MC_folder):
            raise FileNotFoundError(f"{MC_folder} not found.")

        if output_sim_path is None:
            if mean_nu_X is None or Vrest is None:
                raise ValueError(
                    "When output_sim_path is None, both mean_nu_X and Vrest must be provided."
                )
        else:
            if (mean_nu_X is None) != (Vrest is None):
                raise ValueError(
                    "Provide both mean_nu_X and Vrest together, or omit both to load from output_sim_path."
                )
            if not os.path.exists(output_sim_path):
                raise FileNotFoundError(f"{output_sim_path} not found.")

        # Basic numeric sanity checks
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a number.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if not isinstance(tstop, (int, float)):
            raise TypeError("tstop must be a number.")
        if tstop <= 0:
            raise ValueError("tstop must be > 0.")
        if dt >= tstop:
            raise ValueError("dt must be smaller than tstop.")

        # Check that electrodeParameters is a dictionary with required keys
        if electrodeParameters is not None:
            if not isinstance(electrodeParameters, dict):
                raise TypeError("electrodeParameters must be a dictionary.")
            missing = [key for key in ("z", "sigma") if key not in electrodeParameters]
            if missing:
                raise KeyError(f"electrodeParameters missing required keys: {missing}.")

        # Check that CDM is a boolean
        if not isinstance(CDM, bool):
            raise TypeError("CDM must be a boolean.")

        # Load dependencies for kernel computation if not already loaded
        if self.lfpykernels is None:
            self._load_kernel_deps()

        # Update paths of cellParameters and morphologies
        cellParameters = deepcopy(params.cellParameters)
        cellParameters['templatefile'] = os.path.join(MC_folder, cellParameters['templatefile'])
        morphologies = [os.path.join(MC_folder, m) for m in params.morphologies]

        # Recompile mod files if needed
        mod_dir = os.path.join(MC_folder, "mod")
        mech_loaded = self.neuron.load_mechanisms(mod_dir)
        if not mech_loaded:
            try:
                subprocess.run(["nrnivmodl"], check=True, cwd=mod_dir)
            except FileNotFoundError as exc:
                raise RuntimeError("nrnivmodl not found; cannot build NEURON mechanisms.") from exc
            self.neuron.load_mechanisms(mod_dir)

        # Presynaptic activation time
        if t_X is None:
            t_X = getattr(params, "transient", None)
        if t_X is None:
            raise ValueError("t_X must be provided when params.transient is not available.")

        # Synapse max. conductance (function, mean, st.dev., min.):
        if weights is None:
            weights = [[params.MC_params['weight_EE'], params.MC_params['weight_IE']],
                       [params.MC_params['weight_EI'], params.MC_params['weight_II']]]

        # Define biophysical membrane properties
        set_biophys = self._resolve_biophys(biophys)

        # Class RecExtElectrode/PointSourcePotential parameters:
        if electrodeParameters is not None:
            electrodeParameters = deepcopy(electrodeParameters)
            for key in ['r', 'n', 'N', 'method']:
                electrodeParameters.pop(key, None)

        probes = self._build_kernel_probes(
            params,
            electrodeParameters,
            CDM=CDM,
            probes=probes,
            cdm_probe=cdm_probe,
        )

        # Compute average firing rate of presynaptic populations X
        if mean_nu_X is None:
            mean_nu_X = neuron_utils.compute_nu_X(params, output_sim_path, params.transient)

        if tau is None:
            tau = getattr(params, "tau", None)
        if tau is None:
            raise ValueError("tau must be provided when params.tau is not available.")

        if g_eff is None:
            g_eff = params.MC_params.get("g_eff")
        if n_ext is None:
            n_ext = params.MC_params.get("n_ext")
        if g_eff is None:
            raise ValueError("g_eff must be provided when params.MC_params['g_eff'] is not available.")
        if n_ext is None:
            raise ValueError("n_ext must be provided when params.MC_params['n_ext'] is not available.")

        def resolve_vrest(population_name):
            if Vrest is None:
                if output_sim_path is None:
                    raise ValueError("output_sim_path is required to infer Vrest from somav.h5.")
                with self.h5py.File(os.path.join(output_sim_path, "somav.h5"), "r") as f:
                    return np.median(f[population_name][()][:, 200:])
            if isinstance(Vrest, dict):
                return Vrest[population_name]
            if isinstance(Vrest, (list, tuple, np.ndarray)):
                idx = params.population_names.index(population_name)
                return Vrest[idx]
            return Vrest

        # Compute kernels
        for i, (X, N_X) in enumerate(zip(params.population_names,
                                         params.population_sizes)):
            for j, (Y, N_Y, morphology) in enumerate(zip(params.population_names,
                                                         params.population_sizes,
                                                         morphologies)):
                # Extract median soma voltages from actual network simulation and
                # assume this value corresponds to Vrest.
                Vrest_Y = resolve_vrest(Y)

                cellParameters = deepcopy(cellParameters)
                cellParameters.update(dict(
                    morphology=morphology,
                    custom_fun=set_biophys,
                    custom_fun_args=[dict(Vrest=Vrest_Y), dict(Vrest=Vrest_Y)],
                ))

                # Some inputs must be lists
                synapseParameters = [
                    dict(weight=weights[ii][j],
                         syntype='Exp2Syn',
                         **params.synapseParameters[ii][j])
                    for ii in range(len(params.population_names))]
                synapsePositionArguments = [
                    params.synapsePositionArguments[ii][j]
                    for ii in range(len(params.population_names))]

                # Create kernel approximator object
                kernel = self.lfpykernels.KernelApprox(
                    X=params.population_names,
                    Y=Y,
                    N_X=np.array(params.population_sizes),
                    N_Y=N_Y,
                    C_YX=np.array(params.connectionProbability[i]),
                    cellParameters=cellParameters,
                    populationParameters=params.populationParameters['pop_args'],
                    multapseFunction=params.multapseFunction,
                    multapseParameters=[params.multapseArguments[ii][j] for ii in range(len(params.population_names))],
                    delayFunction=params.delayFunction,
                    delayParameters=[params.delayArguments[ii][j] for ii in range(len(params.population_names))],
                    synapseParameters=synapseParameters,
                    synapsePositionArguments=synapsePositionArguments,
                    extSynapseParameters=params.extSynapseParameters,
                    nu_ext=1000. / params.netstim_interval,
                    n_ext=n_ext[j],
                    nu_X=mean_nu_X,
                )

                # get kernel and store in dictionary keyed by "Y:X"
                self.kernels['{}:{}'.format(Y, X)] = kernel.get_kernel(
                    probes=probes,
                    Vrest=Vrest_Y, dt=dt, X=X, t_X=t_X, tau=tau,
                    g_eff=g_eff,
                )

        return self.kernels


    def compute_cdm_from_kernels(
            self,
            kernels,
            spike_times,
            dt,
            tstop,
            population_sizes=None,
            transient=0.0,
            probe='KernelApproxCurrentDipoleMoment',
            component=2,
            mode='same',
            scale=1.0,
            aggregate=None,
    ):
        """
        Compute the CDM from spike times and kernels via convolution.

        Parameters
        ----------
        kernels: dict
            Kernel dictionary keyed by "Y:X".
        spike_times: dict
            Spike times per presynaptic population.
        dt: float
            Time step in ms.
        tstop: float
            Simulation stop time in ms.
        population_sizes: dict or None
            Optional population sizes to normalize spike rates (Hz).
        transient: float
            Transient period to discard from spike times.
        probe: str
            Probe name to read from kernels.
        component: int or None
            Component to select (e.g., z-component). If None, use all components.
        mode: str
            Convolution mode ('same', 'full', 'valid').
        scale: float
            Optional scaling factor for the signal.
        aggregate: str or None
            If "sum", return an additional key "sum" with the total signal.

        Returns
        -------
        signals: dict
            Dictionary mapping each kernel key (e.g., "Y:X") to the convolved signal. If
            `aggregate="sum"`, an additional "sum" entry contains the sum across keys.
        """

        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if tstop <= 0:
            raise ValueError("tstop must be > 0.")

        if not isinstance(probe, str):
            probe = probe.__name__ if inspect.isclass(probe) else probe.__class__.__name__

        rate_cache = {}
        signals = {}
        for key, kernel_dict in kernels.items():
            if ":" not in key:
                continue
            _, X = key.split(":")
            if X not in spike_times:
                raise KeyError(f"Missing spike times for population {X}.")

            if X not in rate_cache:
                times = np.asarray(spike_times[X])
                times = times[times >= transient]
                bins = np.arange(transient, tstop + dt, dt)
                counts, _ = np.histogram(times, bins=bins)
                rate = counts / (dt / 1000.0)
                if population_sizes is not None and X in population_sizes:
                    rate = rate / population_sizes[X]
                rate_cache[X] = rate

            if probe not in kernel_dict:
                raise KeyError(f"Probe {probe} not found in kernel for {key}.")
            kernel = kernel_dict[probe]
            if kernel.ndim == 1:
                kernel_sel = kernel
                sig = np.convolve(rate_cache[X], kernel_sel, mode=mode) * scale
            elif component is not None:
                kernel_sel = kernel[component]
                sig = np.convolve(rate_cache[X], kernel_sel, mode=mode) * scale
            else:
                sig = np.vstack([
                    np.convolve(rate_cache[X], kernel[ch], mode=mode) * scale
                    for ch in range(kernel.shape[0])
                ])

            signals[key] = sig

        if aggregate == "sum":
            total = None
            for sig in signals.values():
                if total is None:
                    total = np.array(sig, copy=True)
                else:
                    total = total + sig
            signals["sum"] = total

        return signals


    def _get_eeg_1020_locations(self):
        """
        Return default dipole locations for the NYHeadModel based on the EEG 10–20 system.

        Each dipole location is positioned inside the NYHeadModel such that it is as
        close as possible to a corresponding EEG sensor from the 10–20 system. The
        returned sensor labels indicate the EEG electrode associated with each dipole.

        Returns
        -------
        locations : np.ndarray, shape (n_dipoles, 3)
            Dipole positions in Cartesian coordinates (x, y, z) expressed in mm.
        labels : list[str]
            EEG 10–20 electrode labels. Each label corresponds to the EEG sensor for
            which the associated dipole location is optimally placed to be closest
            within the NYHeadModel.
        """

        locations_mm = [
            np.array([-25, 65, 0]),  # Fp1
            np.array([25, 65, 0]),  # Fp2
            np.array([-50, 36, -10]),  # F7
            np.array([-39, 36, 36]),  # F3
            np.array([-3, 36, 56]),  # Fz
            np.array([39, 36, 36]),  # F4
            np.array([50, 36, -10]),  # F8
            np.array([-68, -20, 0]),  # T3
            np.array([-46, -15, 48]),  # C3
            np.array([-3, -19, 76]),  # Cz
            np.array([46, -15, 48]),  # C4
            np.array([68, -20, 0]),  # T4
            np.array([-57, -61, -17]),  # T5
            np.array([-40, -55, 50]),  # P3
            np.array([-7, -52, 72]),  # Pz
            np.array([40, -55, 50]),  # P4
            np.array([57, -61, -17]),  # T6
            np.array([-32, -90, 18]),  # O1
            np.array([-3, -92, 20]),  # Oz
            np.array([32, -90, 18]),  # O2
        ]
        labels = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2",
        ]
        return np.asarray(locations_mm), labels


    def compute_MEEG(
            self,
            CDM,
            dipole_locations=None,
            sensor_locations=None,
            model=None,
            model_kwargs=None,
            align_to_surface=True
    ):
        """
        Compute EEG/MEG from current dipole moment time series using lfpykit forward models.

        Parameters
        ----------
        CDM: np.ndarray
            Current dipole moment time series. Must be an array shaped:
            - (3, n_times) for a single dipole, or
            - (n_dipoles, 3, n_times) for multiple dipoles.
        dipole_locations: np.ndarray or None
            Dipole locations matching CDM. Must be shaped:
            - (3,) for a single dipole, or
            - (n_dipoles, 3) for multiple dipoles.
            Units: mm for NYHeadModel, µm for other models.
        sensor_locations: np.ndarray or None
            Sensor/electrode locations shaped (n_sensors, 3). Required for non-NYHead models and must be None for
            NYHeadModel (which are automatically computed from dipole locations and the head model geometry).
            Units: mm for NYHeadModel, µm for other models.
        model: str or None
            Forward model name. Supported values (see lfpykit forward models):
            - NYHeadModel (EEG)
            - FourSphereVolumeConductor (EEG)
            - InfiniteVolumeConductor (EEG)
            - InfiniteHomogeneousVolCondMEG (MEG)
            - SphericallySymmetricVolCondMEG (MEG)
        model_kwargs: dict or None
            Optional keyword arguments forwarded to the forward model constructor.
        align_to_surface: bool
            NYHeadModel only. If True, rotate dipole moment to be aligned with the local surface normal of the
            head model at the dipole location.

        Returns
        -------
        meeg: np.ndarray
            Summed contribution from all dipoles at each sensor/electrode.
            - EEG models and NYHeadModel: (n_sensors, n_times)
            - MEG models: (n_sensors, 3, n_times)
        """

        if model is None:
            model = "NYHeadModel"
        if not isinstance(model, str):
            raise TypeError("model must be a string name for the forward model.")
        model_key = model
        if model_key not in {
            "NYHeadModel",
            "FourSphereVolumeConductor",
            "InfiniteVolumeConductor",
            "InfiniteHomogeneousVolCondMEG",
            "SphericallySymmetricVolCondMEG",
        }:
            raise ValueError(f"Unknown model '{model_key}'.")

        # Set default sensor locations according to examples provided in the documentation of lfpykit forward models.
        # For InfiniteVolumeConductor, we reuse the default sensor locations from FourSphereVolumeConductor. Although
        # InfiniteVolumeConductor is defined in terms of sensor positions r relative to the dipole (rather than
        # absolute coordinates), the relative positions (sensor location minus dipole position) are computed
        # later in _compute_eegmeg_from_cdm.
        if sensor_locations is None and model_key != "NYHeadModel":
            if model_key in {"FourSphereVolumeConductor", "InfiniteVolumeConductor"}:
                sensor_locations = np.array([[0.0, 0.0, 90000.0]])
            elif model_key in {"InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG"}:
                if model_key == "InfiniteHomogeneousVolCondMEG":
                    sensor_locations = np.array([[10000.0, 0.0, 0.0]])
                else:
                    sensor_locations = np.array([[0.0, 0.0, 92000.0]])

        # Set default dipole locations according to documentation examples in lfpykit forward models.
        if dipole_locations is None:
            if model_key == "FourSphereVolumeConductor":
                dipole_locations = np.array([0.0, 0.0, 78000.0])
            elif model_key == "SphericallySymmetricVolCondMEG":
                dipole_locations = np.array([0.0, 0.0, 90000.0])
            else:
                dipole_locations = np.zeros(3)

        p_list, loc_list = self._normalize_cdm_and_locations(CDM, dipole_locations)

        if model_key != "NYHeadModel":
            if sensor_locations is None:
                raise ValueError("sensor_locations must be provided for this model.")
            sensor_locations = np.asarray(sensor_locations, dtype=float)
            if sensor_locations.ndim != 2 or sensor_locations.shape[1] != 3:
                raise ValueError("sensor_locations must have shape (n_sensors, 3).")
        else:
            if sensor_locations is not None:
                sensor_locations = None
                print("Warning: sensor_locations is ignored for NYHeadModel.")

        total = None
        for p_i, loc_i in zip(p_list, loc_list):
            part = self._compute_eegmeg_from_cdm(
                p_i,
                dipole_location=loc_i,
                model=model_key,
                sensor_locations=sensor_locations,
                model_kwargs=model_kwargs,
                align_to_surface=align_to_surface,
                return_all_electrodes=True,
            )
            total = part if total is None else total + part

        return total


    def _normalize_cdm_and_locations(self, CDM, dipole_locations):
        """Normalize (possibly multi-dipole) CDM + locations into lists.

        Parameters
        ----------
        CDM: np.ndarray
            Current dipole moment time series shaped (3, n_times) or (n_dipoles, 3, n_times).
        dipole_locations: np.ndarray or None
            Dipole locations shaped (3,) or (n_dipoles, 3). If None, locations are set to zeros.

        Returns
        -------
        p_list: list[np.ndarray]
            List of arrays shaped (3, n_times)
        loc_list: list
            List of per-dipole locations (each a (3,) array_like).
        """
        cdm = np.asarray(CDM)
        if cdm.dtype == object:
            raise ValueError("CDM must be a numeric array shaped (3, n_times) or (n_dipoles, 3, n_times).")

        if cdm.ndim == 2:
            if 3 not in cdm.shape:
                raise ValueError("CDM must be shaped (3, n_times) for a single dipole.")
            if cdm.shape[0] != 3:
                cdm = cdm.T
            p_list = [cdm]
            n_dip = 1
        elif cdm.ndim == 3:
            n_dip = cdm.shape[0]
            if cdm.shape[1] == 3:
                p_list = [cdm[i] for i in range(n_dip)]
            elif cdm.shape[2] == 3:
                p_list = [cdm[i].T for i in range(n_dip)]
            else:
                raise ValueError("CDM must be shaped (n_dipoles, 3, n_times) or (n_dipoles, n_times, 3).")
        else:
            raise ValueError("CDM must be shaped (3, n_times) or (n_dipoles, 3, n_times).")

        # Normalize dipole locations into list
        if dipole_locations is None:
            dipole_locations = np.zeros((n_dip, 3)) if n_dip > 1 else np.zeros(3)

        dip_arr = np.asarray(dipole_locations, dtype=float)
        if dip_arr.dtype == object:
            raise ValueError("dipole_locations must be a numeric array with shape (3,) or (n_dipoles, 3).")
        if n_dip == 1:
            if dip_arr.shape == (3,):
                loc_list = [dip_arr]
            elif dip_arr.shape == (1, 3):
                loc_list = [dip_arr[0]]
            else:
                raise ValueError("dipole_locations must have shape (3,) for a single dipole.")
        else:
            if dip_arr.shape != (n_dip, 3):
                raise ValueError("dipole_locations must have shape (n_dipoles, 3) to match CDM.")
            loc_list = [dip_arr[i] for i in range(n_dip)]

        p_list = [self._format_cdm(p_i) for p_i in p_list]
        return p_list, loc_list


    def _compute_eegmeg_from_cdm(
            self,
            CDM,
            dipole_location=None,
            model="NYHeadModel",
            sensor_locations=None,
            model_kwargs=None,
            align_to_surface=True,
            return_all_electrodes=False
    ):
        """
        Compute EEG/MEG from a single current dipole moment across all sensors using the selected forward model.

        Notes
        -----
        - All forward models are linear in the dipole moment: output = M(dipole_location, sensors) @ p
        - For MEG models in lfpykit, the output is a vector field with shape (n_sensors, 3, n_times)
        - For NYHeadModel, sensor locations are fixed to the built-in 231 electrodes.

        Parameters
        ----------
        CDM: array_like
            Current dipole moment, accepted by :meth:`_format_cdm` (nA·µm).
        dipole_location: array_like
            Dipole location. Units: mm for NYHeadModel, µm for other models.
        model: str
            Forward model name.
        sensor_locations: array_like
            Required for all models except NYHeadModel. Shape (n_sensors, 3), units in µm.
        model_kwargs: dict
            Optional kwargs forwarded to model constructors.
        align_to_surface: bool
            NYHeadModel only. If True, rotate dipole moment to be aligned with the local surface normal of the
            head model at the dipole location.
        return_all_electrodes: bool
            NYHeadModel only. If True, return the signal for all 231 built-in electrodes. If False, return only
            the signal for the single electrode closest to the dipole location.

        Returns
        -------
        meeg: np.ndarray
            EEG/MEG signal at each sensor/electrode. Shape depends on model:
            - EEG models and NYHeadModel: (n_sensors, n_times) or (n_times,) if return_all_electrodes=False
            for NYHeadModel.
            - MEG models: (n_sensors, 3, n_times)
        """
        p = self._format_cdm(CDM)
        model_kwargs = model_kwargs or {}
        if not isinstance(model, str):
            raise TypeError("model must be a string name for the forward model.")
        model_key = model

        # --- NYHeadModel EEG ---
        if model_key == "NYHeadModel":
            if self.nyhead is None:
                nyhead_model = self._load_eegmegcalc_model("NYHeadModel")
                self.nyhead = nyhead_model(**model_kwargs) if model_kwargs else nyhead_model()
            self.nyhead.set_dipole_pos(dipole_location)
            M = self.nyhead.get_transformation_matrix()  # (231, 3)
            p_use = self.nyhead.rotate_dipole_to_surface_normal(p) if align_to_surface else p
            eeg = M @ p_use  # (231, n_times)
            # Get the closest electrode idx to dipole location
            if return_all_electrodes:
                return eeg
            else:
                dist, closest_elec_idx = self.nyhead.find_closest_electrode()
                return eeg[closest_elec_idx, :]

        # --- All other models need numeric dipole location ---
        if dipole_location is None:
            raise ValueError("dipole_location must be provided for this model.")
        dipole_location = np.asarray(dipole_location, dtype=float)

        # Normalize sensor locations for non-NYHead models
        if sensor_locations is None:
            raise ValueError("sensor_locations must be provided for this model.")
        sensor_locations = np.atleast_2d(np.asarray(sensor_locations, dtype=float))
        if sensor_locations.ndim != 2 or sensor_locations.shape[1] != 3:
            raise ValueError("sensor_locations must have shape (n_sensors, 3).")

        # --- EEG models ---
        if model_key == "FourSphereVolumeConductor":
            FourSphere = self._load_eegmegcalc_model("FourSphereVolumeConductor")
            model_obj = FourSphere(sensor_locations, **model_kwargs)
            M = model_obj.get_transformation_matrix(dipole_location)  # (n_sensors, 3)
            return M @ p  # (n_sensors, n_times)

        if model_key == "InfiniteVolumeConductor":
            InfiniteVol = self._load_eegmegcalc_model("InfiniteVolumeConductor")
            model_obj = InfiniteVol(**model_kwargs)
            r = sensor_locations - dipole_location  # displacement vectors (n_sensors, 3)
            M = model_obj.get_transformation_matrix(r)  # (n_sensors, 3)
            return M @ p  # (n_sensors, n_times)

        # --- MEG models ---
        if model_key == "InfiniteHomogeneousVolCondMEG":
            IHVCMEG = self._load_eegmegcalc_model("InfiniteHomogeneousVolCondMEG")
            model_obj = IHVCMEG(sensor_locations, **model_kwargs)
            M = model_obj.get_transformation_matrix(dipole_location)  # (n_sensors, 3, 3)
            return M @ p  # (n_sensors, 3, n_times)

        if model_key == "SphericallySymmetricVolCondMEG":
            SSVMEG = self._load_eegmegcalc_model("SphericallySymmetricVolCondMEG")
            model_obj = SSVMEG(sensor_locations, **model_kwargs)
            M = model_obj.get_transformation_matrix(dipole_location)  # (n_sensors, 3, 3)
            return M @ p  # (n_sensors, 3, n_times)

        raise ValueError(f"Unknown model '{model}'.")


    ############################################################
    ## Proxy-based methods for field potential estimation ######
    ############################################################

    def compute_proxy(self, method, sim_data, sim_step, *, excitatory_only=None):
        """
        Compute a proxy for the extracellular signal by combining variables directly measured from network simulations.
        Note: some methods require a valid sim_step to compute delays.

        Parameters
        ----------
        method: str
            Method to compute the proxy. Options are:
            - 'FR': firing rate averaged over all neurons in the population.
            - 'AMPA': sum of AMPA currents.
            - 'GABA': sum of GABA currents.
            - 'Vm': sum of membrane potentials.
            - 'I': sum of synaptic currents.
            - 'I_abs': sum of their absolute values.
            - 'LRWS': LFP reference weighted sum.
            - 'ERWS1': EEG reference weighted sum 1 (non-causal)
            - 'ERWS2': EEG reference weighted sum 2 (non-causal)

        sim_data: dict
            Dictionary containing the simulation data. Expected keys depend on `method`:
            - "FR": requires FR data (n_units, n_times)
            - "AMPA": requires AMPA data (n_units, n_times)
            - "GABA": requires GABA data (n_units, n_times)
            - "Vm": requires Vm data (n_units, n_times)
            - "I", "I_abs": requires AMPA and GABA data
            - "LRWS", "ERWS1": requires AMPA and GABA data
            - "ERWS2": requires AMPA, GABA, and nu_ext data

            If `excitatory_only` is set, the method will prefer keys with suffixes:
            - excitatory_only=True: "<KEY>_exc"
            - excitatory_only=False: "<KEY>_all" or "<KEY>_total"
            and fall back to "<KEY>" if the suffixed keys are not provided.

        sim_step: float or None
            Simulation time step in ms. Required for delay-based methods ("LRWS", "ERWS1", "ERWS2").
        excitatory_only: bool or None
            If provided, indicates whether inputs should be treated as excitatory-only. If None, the method will
            use the base keys without any suffixes. This flag is used to select data keys, if present.

        Returns
        -------
        proxy: np.array
            Proxy for the extracellular signal.
        """

        def _select(key):
            if excitatory_only is True:
                candidates = [f"{key}_exc", key]
            elif excitatory_only is False:
                candidates = [f"{key}_all", f"{key}_total", key]
            else:
                candidates = [key]
            for cand in candidates:
                if cand in sim_data:
                    return sim_data[cand]
            raise KeyError(f"Missing data for '{key}' (tried {candidates}).")

        if method in {"LRWS", "ERWS1", "ERWS2"}:
            if sim_step is None:
                raise ValueError(f"sim_step must be provided for proxy method '{method}'.")
            if not isinstance(sim_step, (int, float)) or sim_step <= 0:
                raise ValueError("sim_step must be a positive number (ms).")

        if method == 'FR':
            proxy = np.mean(_select('FR'), axis=0)
        elif method == 'AMPA':
            proxy = np.sum(_select('AMPA'), axis=0)
        elif method == 'GABA':
            proxy = -np.sum(_select('GABA'), axis=0)
        elif method == 'Vm':
            proxy = np.sum(_select('Vm'), axis=0)
        elif method == 'I':
            proxy = np.sum(_select('AMPA') + _select('GABA'), axis=0)
        elif method == 'I_abs':
            proxy = np.sum(np.abs(_select('AMPA')) + np.abs(_select('GABA')), axis=0)
        elif method == 'LRWS':
            delay = int(6.0 / sim_step)
            ampa = _select('AMPA')
            gaba = _select('GABA')
            AMPA_delayed = np.array([self.roll_with_zeros(ampa[i], delay) for i in range(ampa.shape[0])])
            proxy = np.sum(AMPA_delayed - 1.65 * gaba, axis=0)
        elif method == 'ERWS1':
            AMPA_delay = -int(0.9 / sim_step)
            GABA_delay = int(2.3 / sim_step)
            ampa = _select('AMPA')
            gaba = _select('GABA')
            AMPA_delayed = np.array([self.roll_with_zeros(ampa[i], AMPA_delay) for i in range(ampa.shape[0])])
            GABA_delayed = np.array([self.roll_with_zeros(gaba[i], GABA_delay) for i in range(gaba.shape[0])])
            proxy = np.sum(AMPA_delayed - 0.3 * GABA_delayed, axis=0)
        elif method == 'ERWS2':
            coeff = [-0.6, 0.1, -0.4, -1.9, 0.6, 3.0, 1.4, 1.7, 0.2]
            nu_ext = _select('nu_ext')
            AMPA_delay = int(coeff[0] * np.power(nu_ext, -coeff[1]) + coeff[2])
            GABA_delay = int(coeff[3] * np.power(nu_ext, -coeff[4]) + coeff[5])
            alpha = coeff[6] * np.power(nu_ext, -coeff[7]) + coeff[8]
            ampa = _select('AMPA')
            gaba = _select('GABA')
            AMPA_delayed = np.array([self.roll_with_zeros(ampa[i], AMPA_delay) for i in range(ampa.shape[0])])
            GABA_delayed = np.array([self.roll_with_zeros(gaba[i], GABA_delay) for i in range(gaba.shape[0])])
            proxy = np.sum(AMPA_delayed - alpha * GABA_delayed, axis=0)
        else:
            raise ValueError(f"Method {method} not recognized.")

        return proxy
