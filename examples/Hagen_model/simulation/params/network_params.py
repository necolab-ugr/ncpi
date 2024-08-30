# Best fit params of the LIF network model
LIF_params = dict(
    X=['E', 'I'],
    N_X=[8192, 1024],
    C_m_X=[289.1, 110.7],
    tau_m_X=[10., 10.],
    E_L_X=[-65., -65.],
    C_YX=[[0.2, 0.2], [0.2, 0.2]],
    J_YX=[[1.589, 2.020], [-23.84, -8.441]],
    delay_YX=[[2.520, 1.714], [1.585, 1.149]],
    tau_syn_YX=[[0.5, 0.5], [0.5, 0.5]],
    n_ext=[465, 160],
    nu_ext=40.,
    J_ext=29.89,
    model='iaf_psc_exp')