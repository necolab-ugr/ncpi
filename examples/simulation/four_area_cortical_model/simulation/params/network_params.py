# Parameters defining a four-area cortical network model in which the Hagen et al. local LIF microcircuit is
# replicated and coupled across four cortical areas. Local network parameters match the Hagen model; only
# J_ext is adjusted.

areas = ['frontal', 'parietal', 'temporal', 'occipital']

# Base local parameters (Hagen model)
J_EE = 1.589
J_IE = 2.020
J_EI = -23.84
J_II = -8.441

# Inter-area (long-range) excitatory connectivity is defined according to the following general rules:
# - weaker than local recurrent excitation (here 15% of the local J_EE and J_IE)
# - sparse (here 2% connection probability)
# - longer delays (here 10 ms, which is longer than the longest local delay of 2.52 ms)
inter_area_scale = 0.15
inter_area_p     = 0.02
inter_area_delay = 10.0

LIF_params = dict(
    areas=areas,
    X=['E', 'I'],
    N_X=[8192, 1024],
    C_m_X=[289.1, 110.7],
    tau_m_X=[10., 10.],
    E_L_X=[-65., -65.],
    C_YX=[[0.2, 0.2], [0.2, 0.2]],
    J_YX=[[J_EE, J_IE], [J_EI, J_II]],
    delay_YX=[[2.520, 1.714], [1.585, 1.149]],
    tau_syn_YX=[[0.5, 0.5], [0.5, 0.5]],
    n_ext=[465, 160],
    nu_ext=40.,
    # The external drives reflects inputs from other brain areas, subcortical structures and background noise
    J_ext=29.89,
    model='iaf_psc_exp',
    # Inter-area excitatory-only connections (E->E and E->I); no inhibitory cortico-cortical connections
    inter_area=dict(
        C_YX=[[inter_area_p, inter_area_p], [0.0, 0.0]],
        J_YX=[[J_EE * inter_area_scale, J_IE * inter_area_scale], [0.0, 0.0]],
        delay_YX=[[inter_area_delay, inter_area_delay], [0.0, 0.0]],
    ),
)
