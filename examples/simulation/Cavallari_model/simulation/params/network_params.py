# Cavallari et al. (2014) conductance-based recurrent network parameters.

Network_params = {
    "N_exc": 4000,
    "N_inh": 1000,
    "P": 0.2,
    "extent": 1.0,
    "g_EE": 0.178,
    "g_IE": 0.233,
    "g_EI": -2.01,
    "g_II": -2.70,
    "g_th_exc_external": 0.234,
    "g_th_inh_external": 0.317,
    "g_cc_exc_external": 0.187,
    "g_cc_inh_external": 0.254,
    "v_0": 1.5,
    "A_ext": 0.0,
    "f_ext": 0.0,
    "OU_sigma": 0.4,
    "OU_tau": 16.0,
}

excitatory_cell_params = {
    "V_th": -52.0,
    "V_reset": -59.0,
    "t_ref": 2.0,
    "g_L": 25.0,
    "C_m": 500.0,
    "E_ex": 0.0,
    "E_in": -80.0,
    "E_L": -70.0,
    "tau_rise_AMPA": 0.4,
    "tau_decay_AMPA": 2.0,
    "tau_rise_GABA_A": 0.25,
    "tau_decay_GABA_A": 5.0,
    "tau_m": 20.0,
    "I_e": 0.0,
}

inhibitory_cell_params = {
    "V_th": -52.0,
    "V_reset": -59.0,
    "t_ref": 1.0,
    "g_L": 20.0,
    "C_m": 200.0,
    "E_ex": 0.0,
    "E_in": -80.0,
    "E_L": -70.0,
    "tau_rise_AMPA": 0.2,
    "tau_decay_AMPA": 1.0,
    "tau_rise_GABA_A": 0.25,
    "tau_decay_GABA_A": 5.0,
    "tau_m": 10.0,
    "I_e": 0.0,
}

Neuron_params = [excitatory_cell_params, inhibitory_cell_params]
