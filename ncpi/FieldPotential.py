import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from ncpi import tools


def roll_with_zeros(arr, shift):
    """
    Roll an array with zeros.
    Parameters
    ----------
    arr: np.array
        Array to be rolled.
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


class FieldPotential:
    def __init__(self, kernel=True, nyhead=False, MEEG=None, CDM_shape=15500):
        """
        Initialize the FieldPotential object.

        Parameters
        ----------
        kernel: bool
            Choose to compute kernels.

        nyhead: bool
            Use the NYHeadModel for computing EEG.

        MEEG: str
            Prepare data structures for computing EEG or MEG signals. Options are:
                'EEG': Compute EEG.
                'MEG': Compute MEG.

        CDM_shape: int
            Number of time points in the CDM.
        """

        if kernel:
            # Check that the required modules are installed
            if not tools.ensure_module("lfpykernels"):
                raise ImportError("lfpykernels is required for computing kernels but is not installed.")
            self.KernelApprox = tools.dynamic_import("lfpykernels",
                                                     "KernelApprox")
            self.GaussCylinderPotential = tools.dynamic_import("lfpykernels",
                                                               "GaussCylinderPotential")
            self.KernelApproxCurrentDipoleMoment = tools.dynamic_import("lfpykernels",
                                                                        "KernelApproxCurrentDipoleMoment")

            # Is LFPy import really needed?
            if not tools.ensure_module("LFPy"):
                raise ImportError("LFPy is required for computing kernels but is not installed.")

            if not tools.ensure_module("neuron"):
                raise ImportError("neuron is required for computing kernels but is not installed.")
            self.neuron = tools.dynamic_import("neuron")

            if not tools.ensure_module("h5py"):
                raise ImportError("h5py is required for computing kernels but is not installed.")
            self.h5py = tools.dynamic_import("h5py")

            # Initialize dictionary for storing kernels
            self.H_YX = dict()

        if nyhead:
            # Check that the required module is installed
            if not tools.ensure_module("lfpykit"):
                raise ImportError("lfpykit is required for computing the NYHeadModel but is not installed.")
            self.NYHeadModel = tools.dynamic_import("lfpykit.eegmegcalc",
                                                    "NYHeadModel")

            # Initialize the head model
            self.nyhead = self.NYHeadModel()

        # Import InfiniteHomogeneousVolCondMEG
        if MEEG is not None:
            self.IHVCMEG = tools.dynamic_import("LFPy", "InfiniteHomogeneousVolCondMEG")

        # Allocate memory for the transformation matrix and MEEG
        if MEEG == 'EEG':
            self.M = np.zeros((231,3))
            self.MEEG = np.zeros((231,CDM_shape))
        elif MEEG == 'MEG':
            self.M = np.zeros((1,3,3))
            self.MEEG = np.zeros((1,3,CDM_shape))


    def create_kernel(self, MC_folder, output_path, params, biophys, dt, tstop, electrodeParameters=None, CDM=True):
        """
        Create kernels from multicompartment neuron network descriptions.

        Parameters
        ----------
        MC_folder: str
            Path to the folder containing the multicompartment neuron network descriptions.
        output_path: str
            Path to the output folder.
        params: module
            `<module 'example_network_parameters'>`
        biophys: list
            List of biophysical membrane properties.
        dt: float
            Time step.
        tstop: float
            Simulation time.
        electrodeParameters: dict
            Electrode parameters. If None, no LFP is computed.
        CDM: bool
            Compute the current dipole moment.

        Returns
        -------
        H_YX: dict
            Dictionary containing the kernels.
        """

        # Check that folder exists
        if not os.path.exists(MC_folder):
            raise FileNotFoundError(f"{MC_folder} not found.")

        # Check that the output path exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{output_path} not found.")

        # Add some checks for the parameters?

        # Check that biophys is a list
        if not isinstance(biophys, list):
            raise TypeError("biophys must be a list.")

        # Check that dt is > 0
        if dt <= 0:
            raise ValueError("dt must be > 0.")

        # Check that tstop is > 0
        if tstop <= 0:
            raise ValueError("tstop must be > 0.")

        # Check that electrodeParameters is a dictionary
        if electrodeParameters is not None and not isinstance(electrodeParameters, dict):
            raise TypeError("electrodeParameters must be a dictionary.")

        # Check that CDM is a boolean
        if not isinstance(CDM, bool):
            raise TypeError("CDM must be a boolean.")

        # Update paths of cellParameters and morphologies
        params.cellParameters['templatefile'] = os.path.join(MC_folder, params.cellParameters['templatefile'])
        params.morphologies = [os.path.join(MC_folder, m) for m in params.morphologies]

        # Recompile mod files if needed
        mech_loaded = self.neuron.load_mechanisms(os.path.join(MC_folder, 'mod'))
        if not mech_loaded:
            os.system(f'cd {os.path.join(MC_folder, "mod")} && nrnivmodl && cd -')
            self.neuron.load_mechanisms(os.path.join(MC_folder, "mod"))

        # Presynaptic activation time
        t_X = params.transient

        # Define biophysical membrane properties
        set_biophys = [getattr(self, f"{b}") for b in biophys]

        # Synapse max. conductance (function, mean, st.dev., min.):
        weights = [[params.MC_params['weight_EE'], params.MC_params['weight_IE']],
                   [params.MC_params['weight_EI'], params.MC_params['weight_II']]]

        # Class RecExtElectrode/PointSourcePotential parameters:
        if electrodeParameters is not None:
            for key in ['r', 'n', 'N', 'method']:
                del electrodeParameters[key]

        # Predictor assuming planar disk source elements convolved with Gaussian
        # along z-axis
        probes = []
        if electrodeParameters is not None:
            gauss_cyl_potential = self.GaussCylinderPotential(
                cell=None,
                z=electrodeParameters['z'],
                sigma=electrodeParameters['sigma'],
                R=params.populationParameters['pop_args']['radius'],
                sigma_z=params.populationParameters['pop_args']['scale'],
            )
            probes.append(gauss_cyl_potential)


        # Set up recording of current dipole moments.
        if CDM:
            current_dipole_moment = self.KernelApproxCurrentDipoleMoment(cell=None)
            probes.append(current_dipole_moment)

        # Compute average firing rate of presynaptic populations X
        mean_nu_X = self.compute_mean_nu_X(params, output_path, params.transient)

        # Compute kernels
        for i, (X, N_X) in enumerate(zip(params.population_names,
                                         params.population_sizes)):
            for j, (Y, N_Y, morphology) in enumerate(zip(params.population_names,
                                                         params.population_sizes,
                                                         params.morphologies)):
                # Extract median soma voltages from actual network simulation and
                # assume this value corresponds to Vrest.
                with self.h5py.File(os.path.join(output_path, 'somav.h5'
                                            ), 'r') as f:
                    Vrest = np.median(f[Y][()][:, 200:])

                cellParameters = deepcopy(params.cellParameters)
                cellParameters.update(dict(
                    morphology=morphology,
                    custom_fun=set_biophys,
                    custom_fun_args=[dict(Vrest=Vrest), dict(Vrest=Vrest)],
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
                kernel = self.KernelApprox(
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
                    n_ext=params.MC_params['n_ext'][j],
                    nu_X=mean_nu_X,
                )

                # make kernel predictions
                self.H_YX['{}:{}'.format(Y, X)] = kernel.get_kernel(
                    probes=probes,
                    Vrest=Vrest, dt=dt, X=X, t_X=t_X, tau=params.tau,
                    g_eff=params.MC_params['g_eff'],
                )

        return self.H_YX


    def compute_MEEG(self, CDM, location=None, cMEG = False):
        """
        Compute EEG/MEG from a current dipole moment using the NYHeadModel (EEG) or a homogeneous, isotropic, linear
        volume conductor model (MEG).

        Warning: The MEG calculation has not yet been tested and is currently limited to use with a single
        sensor location.

        Parameters
        ----------
        CDM: np.array
            Current dipole moment
        location: np.array
            Location of the dipole moment
        cMEG: bool
            Compute MEG instead of EEG.

        Returns
        -------
        all_MEEG: list of np.array
            MEEG at the electrode/sensor locations
        """

        debug = False

        # Reformat the CDM to be a 3DxN array, where N is the number of time points
        p = np.zeros((3, len(CDM)))
        p[2, :] = CDM

        # If location is provided, compute MEEG at that location
        if location is not None:
            if cMEG == False:
                # Set the dipole location
                self.nyhead.set_dipole_pos(location)

                # Get the transformation matrix
                self.M = self.nyhead.get_transformation_matrix()

                # Rotate current dipole moment to be oriented along the normal vector of cortex
                p = self.nyhead.rotate_dipole_to_surface_normal(p)

                # Compute EEG
                self.MEEG = self.M @ p  # (mV)

                # Get the closest electrode idx to dipole location
                dist, closest_elec_idx = self.nyhead.find_closest_electrode()
                all_MEEG = self.MEEG[closest_elec_idx, :]

            # Warning: this code snippet has not been tested yet!
            else:
                # Instantiate MEG object
                meg = self.IHVCMEG(location)

                # Get the transformation matrix
                self.M = meg.get_transformation_matrix(location)

                # Compute the magnetic signal in a single sensor location
                self.MEEG = self.M @ p
                all_MEEG = self.MEEG[0]

        # If location is not provided, place the dipole at the different locations of the 10-20 EEG setup. We assume
        # that each dipole is working independently, i.e., the EEG at each electrode is computed separately.
        elif location is None and cMEG == False:
            all_MEEG = np.zeros((20,self.MEEG.shape[1]))
            locations = [np.array([-25, 65, 0]), # Fp1
                        np.array([25, 65, 0]),  # Fp2
                        np.array([-50, 36, -10]), # F7
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
                        np.array([32, -90, 18])]  # O2

            labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                      'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']

            # Plot electrode positions
            if debug:
                x_lim = [-100, 100]
                y_lim = [-130, 100]
                z_lim = [-160, 120]

                fig = plt.figure(figsize=(7.5, 3.), dpi=300)
                ax1 = fig.add_axes([0.15, 0.15, 0.2, 0.7],
                                   xlabel="x (mm)", ylabel='y (mm)', xlim=x_lim, ylim=y_lim)
                ax2 = fig.add_axes([0.45, 0.15, 0.2, 0.7],
                                   xlabel="x (mm)", ylabel='z (mm)', xlim=x_lim, ylim=z_lim)
                ax3 = fig.add_axes([0.75, 0.15, 0.2, 0.7],
                                   xlabel="y (mm)", ylabel='z (mm)', xlim=y_lim, ylim=z_lim)

                for i, location in enumerate(locations):
                    # Set the dipole location
                    self.nyhead.set_dipole_pos(location)

                    # Plot the head model and the dipole location
                    if i == 0:
                        threshold = 2
                        xz_plane_idxs = np.where(np.abs(self.nyhead.cortex[1, :] - 0) < threshold)[0]
                        xy_plane_idxs = np.where(np.abs(self.nyhead.cortex[2, :] - 0) < threshold)[0]
                        yz_plane_idxs = np.where(np.abs(self.nyhead.cortex[0, :] - 25) < threshold)[0]

                        ax1.scatter(self.nyhead.cortex[0, xy_plane_idxs], self.nyhead.cortex[1, xy_plane_idxs], s=0.05, color = 'r')
                        ax2.scatter(self.nyhead.cortex[0, xz_plane_idxs], self.nyhead.cortex[2, xz_plane_idxs], s=0.05, color = 'r')
                        ax3.scatter(self.nyhead.cortex[1, yz_plane_idxs], self.nyhead.cortex[2, yz_plane_idxs], s=0.05, color = 'r')

                    ax1.plot(self.nyhead.dipole_pos[0], self.nyhead.dipole_pos[1], 'o', ms=6, color='orange', zorder=1000)
                    ax2.plot(self.nyhead.dipole_pos[0], self.nyhead.dipole_pos[2], 'o', ms=6, color='orange', zorder=1000)
                    ax3.plot(self.nyhead.dipole_pos[1], self.nyhead.dipole_pos[2], 'o', ms=6, color='orange', zorder=1000)

                # Add labels
                for i, txt in enumerate(labels):
                    ax1.annotate(txt, (locations[i][0]+6, locations[i][1]), fontsize=6, color='k')
                    ax2.annotate(txt, (locations[i][0]+6, locations[i][2]), fontsize=6, color='k')
                    ax3.annotate(txt, (locations[i][1]+6, locations[i][2]), fontsize=6, color='k')

                plt.show()

            for ii,location in enumerate(locations):
                # Set the dipole location
                self.nyhead.set_dipole_pos(location)

                # Get the transformation matrix
                self.M = self.nyhead.get_transformation_matrix()

                # Rotate current dipole moment to be oriented along the normal vector of cortex
                p = self.nyhead.rotate_dipole_to_surface_normal(p)

                # Compute EEG
                self.MEEG = self.M @ p  # (mV)

                # Get the closest electrode idx to dipole location
                dist, closest_elec_idx = self.nyhead.find_closest_electrode()
                all_MEEG[ii,:] = self.MEEG[closest_elec_idx, :]

            # Delete variables
            # del p, dist, closest_elec_idx

        else:
            raise ValueError("Configuration (location/cMEG) not recognized.")


        return all_MEEG


    def compute_proxy(self, method, sim_data, sim_step):
        """
        Compute a proxy for the extracellular signal by combining variables directly measured from network simulations.
        Warning: not tested yet!

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
            Dictionary containing the simulation data. The keys are 'FR', 'AMPA', 'GABA', 'Vm', 'nu_ext'. Each key
            contains a 2D array with the data for each neuron in the population.

        sim_step: float
            Simulation time step in ms.

        Returns
        -------
        proxy: np.array
            Proxy for the extracellular signal.
        """

        if method == 'FR':
            proxy = np.mean(sim_data['FR'], axis = 0)
        elif method == 'AMPA':
            proxy = np.sum(sim_data['AMPA'], axis = 0)
        elif method == 'GABA':
            proxy = -np.sum(sim_data['GABA'], axis = 0)
        elif method == 'Vm':
            proxy = np.sum(sim_data['Vm'], axis = 0)
        elif method == 'I':
            proxy = np.sum(sim_data['AMPA'] + sim_data['GABA'], axis = 0)
        elif method == 'I_abs':
            proxy = np.sum(np.abs(sim_data['AMPA']) + np.abs(sim_data['GABA']), axis = 0)
        elif method == 'LRWS':
            delay = int(6.0 / sim_step)
            AMPA_delayed = np.array([roll_with_zeros(sim_data['AMPA'][i],delay) for i in range(sim_data['AMPA'].shape[0])])
            proxy = np.sum(AMPA_delayed - 1.65 * sim_data['GABA'], axis=0)
        elif method == 'ERWS1':
            AMPA_delay = -int(0.9 / sim_step)
            GABA_delay = int(2.3 / sim_step)
            AMPA_delayed = np.array([roll_with_zeros(sim_data['AMPA'][i],
                                                     AMPA_delay) for i in range(sim_data['AMPA'].shape[0])])
            GABA_delayed = np.array([roll_with_zeros(sim_data['GABA'][i],
                                                     GABA_delay) for i in range(sim_data['GABA'].shape[0])])
            proxy = np.sum(AMPA_delayed - 0.3 * GABA_delayed, axis=0)
        elif method == 'ERWS2':
            coeff = [-0.6,  0.1, -0.4, -1.9,  0.6,  3.0 ,  1.4,  1.7,  0.2]
            AMPA_delay = int(coeff[0] * np.power(sim_data['nu_ext'], -coeff[1]) + coeff[2])
            GABA_delay = int(coeff[3] * np.power(sim_data['nu_ext'], -coeff[4]) + coeff[5])
            alpha = coeff[6] * np.power(sim_data['nu_ext'], -coeff[7]) + coeff[8]
            AMPA_delayed = np.array([roll_with_zeros(sim_data['AMPA'][i],
                                                     AMPA_delay) for i in range(sim_data['AMPA'].shape[0])])
            GABA_delayed = np.array([roll_with_zeros(sim_data['GABA'][i],
                                                     GABA_delay) for i in range(sim_data['GABA'].shape[0])])
            proxy = np.sum(AMPA_delayed - alpha * GABA_delayed, axis=0)
        else:
            raise ValueError(f"Method {method} not recognized.")

        return proxy


    """
    The following methods for setting up the neuron model were downloaded from the LFPykernels repository: 
    https://github.com/LFPy/LFPykernels/blob/main/examples/example_network_methods.py
    Copyright (C) 2021 https://github.com/espenhgn
    """

    def set_active(self, cell, Vrest):
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


    def set_passive(self, cell, Vrest):
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


    def set_Ih(self, cell, Vrest):
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


    def set_Ih_linearized(self, cell, Vrest):
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


    def set_pas_hay2011(self, cell, Vrest):
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


    def set_active_hay2011(self, cell, Vrest):
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


    def set_frozen_hay2011(self, cell, Vrest):
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


    def set_frozen_hay2011_no_Ih(self, cell, Vrest):
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


    def set_Ih_hay2011(self, cell, Vrest):
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


    def set_Ih_linearized_hay2011(self, cell, Vrest):
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


    def set_V_R_Ih_linearized_v2(self, cell, Vrest):
        """
        Parameters
        ----------
        cell: object
            LFPy.NetworkCell like object
        Vrest: float
            Steady state potential
        """
        for sec in cell.template.all:
            if self.neuron.h.ismembrane("Ih_linearized_v2", sec=sec):
                sec.V_R_Ih_linearized_v2 = Vrest
            elif self.neuron.h.ismembrane("Ih_linearized_v2_frozen", sec=sec):
                sec.V_R_Ih_linearized_v2_frozen = Vrest


    def set_V_R(self, cell, Vrest):
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
                if self.neuron.h.ismembrane(ion, sec=sec):
                    setattr(sec, f'V_R_{ion}', Vrest)


    def make_cell_uniform(self, cell, Vrest=-65):
        """
        Adjusts e_pas to enforce a uniform resting membrane potential at Vrest

        Parameters
        ----------
        cell: object
            LFPy.NetworkCell like object
        Vrest: float
            Steady state potential
        """
        self.neuron.h.t = 0
        self.neuron.h.finitialize(Vrest)
        self.neuron.h.fcurrent()
        for sec in cell.allseclist:
            for seg in sec:
                seg.e_pas = seg.v
                if self.neuron.h.ismembrane("na_ion", sec=sec):
                    seg.e_pas += seg.ina / seg.g_pas
                if self.neuron.h.ismembrane("k_ion", sec=sec):
                    seg.e_pas += seg.ik / seg.g_pas
                if self.neuron.h.ismembrane("ca_ion", sec=sec):
                    seg.e_pas += seg.ica / seg.g_pas
                if self.neuron.h.ismembrane("Ih", sec=sec):
                    seg.e_pas += seg.ihcn_Ih / seg.g_pas
                if self.neuron.h.ismembrane("Ih_z", sec=sec):
                    seg.e_pas += seg.ih_Ih_z / seg.g_pas
                if self.neuron.h.ismembrane("Ih_linearized_v2_frozen", sec=sec):
                    seg.e_pas += seg.ihcn_Ih_linearized_v2_frozen / seg.g_pas
                if self.neuron.h.ismembrane("Ih_linearized_v2", sec=sec):
                    seg.e_pas += seg.ihcn_Ih_linearized_v2 / seg.g_pas
                if self.neuron.h.ismembrane("Ih_frozen", sec=sec):
                    seg.e_pas += seg.ihcn_Ih_frozen / seg.g_pas

    def compute_mean_nu_X(self, params, OUTPUTPATH, TRANSIENT=200.):
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
            with self.h5py.File(os.path.join(OUTPUTPATH, 'spikes.h5'), 'r') as f:
                times = np.concatenate(f[X]['times'][()])
                times = times[times >= TRANSIENT]
                nu_X[X] = (times.size / N_X
                           / (params.networkParameters['tstop'] - TRANSIENT)
                           * 1000)
        return nu_X

