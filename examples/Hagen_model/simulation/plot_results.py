import pickle
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import LIF_network

def zscore(data, ch, time):
    """
    Compute the z score using the the maximum value of all channels instead of
    the standard deviation of the sample.

    Parameters
    ----------
    data : list
        List of data arrays.
    ch : int
        Channel to normalize.
    time : list
        Time array to normalize.
    """
    tr_data = np.array(data)[:,time]
    tr_data -= np.mean(tr_data,axis = 1).reshape(-1,1)
    return np.max(np.abs(tr_data)),tr_data[ch] /np.max(np.abs(tr_data))

if __name__ == "__main__":
    # Create a LIF_network object
    LIF_net = LIF_network.LIF_network()

    # Path to simulation data (replace with the path to the simulation data)
    path = 'LIF_simulations/bbc9be11172f39006422adfb9370cd07'

    # Load simulation data
    LIF_net.LIF_params = pickle.load(open(path+'/LIF_params','rb'))
    LIF_net.TRANSIENT = pickle.load(open(path+'/TRANSIENT','rb'))
    LIF_net.dt = pickle.load(open(path+'/dt','rb'))
    LIF_net.tstop = pickle.load(open(path+'/tstop','rb'))
    LIF_net.tau = pickle.load(open(path+'/tau','rb'))

    LIF_net.H_YX = pickle.load(open('LIF_simulations/H_YX','rb')) # replace with the path to the H_YX file
    LFP_data = pickle.load(open(path+'/LFP_data','rb'))
    CDM_data = pickle.load(open(path+'/CDM_data_all','rb'))
    lif_mean_nu_X = pickle.load(open(path+'/lif_mean_nu_X','rb'))
    [bins, lif_nu_X] = pickle.load(open(path+'/lif_nu_X','rb'))

    # Plot spikes and firing rates
    fig = plt.figure(figsize=[6,5], dpi=150)
    ax1 = fig.add_axes([0.15,0.45,0.75,0.5])
    ax2 = fig.add_axes([0.15,0.08,0.75,0.3])
    T = [4000, 4100]

    # Spikes
    for i, Y in enumerate(LIF_net.LIF_params['X']):
        times = pickle.load(open(path+'/times_'+Y,'rb'))
        gids = pickle.load(open(path+'/gids_'+Y,'rb'))
        gids = gids[times >= LIF_net.TRANSIENT]
        times = times[times >= LIF_net.TRANSIENT]

        ii = (times >= T[0]) & (times <= T[1])
        ax1.plot(times[ii], gids[ii], '.',
                mfc='C{}'.format(i),
                mec='w',
                label=r'$\langle \nu_\mathrm{%s} \rangle =%.2f$ s$^{-1}$' % (
                    Y, lif_mean_nu_X[Y] / LIF_net.LIF_params['N_X'][i])
               )
    ax1.legend(loc=1)
    ax1.axis('tight')
    ax1.set_xticklabels([])
    ax1.set_ylabel('gid', labelpad=0)

    # Rates
    Delta_t = LIF_net.dt
    binsr = np.linspace(T[0], T[1], int(np.diff(T) / Delta_t + 1))

    for i, Y in enumerate(LIF_net.LIF_params['X']):
        times = pickle.load(open(path+'/times_'+Y,'rb'))
        ii = (times >= T[0]) & (times <= T[1])
        ax2.hist(times[ii], bins=binsr, histtype='step')

    ax2.axis('tight')
    ax2.set_xlabel('t (ms)', labelpad=0)
    ax2.set_ylabel(r'$\nu_X$ (spikes/$\Delta t$)', labelpad=0)

    # Plot kernels and LFP/CDM data
    # Create figure and panels
    fig1 = plt.figure(figsize=[7,6], dpi=150)
    fig2 = plt.figure(figsize=[7,6], dpi=150)
    ax1 = []
    ax2 = []

    # First row
    for k in range(4):
        ax1.append(fig1.add_axes([0.1 + k*0.22,0.4,0.18,0.5]))
        ax2.append(fig2.add_axes([0.1 + k*0.22,0.4,0.18,0.5]))
    # Second row
    for k in range(4):
        ax1.append(fig1.add_axes([0.1 + k*0.22,0.1,0.18,0.2]))
        ax2.append(fig2.add_axes([0.1 + k*0.22,0.1,0.18,0.2]))

    # Time arrays
    LIF_net.dt*= 10 # take into account the decimate ratio
    bins = bins[::10] # take into account the decimate ratio
    time = np.arange(-LIF_net.tau,LIF_net.tau+LIF_net.dt,LIF_net.dt)
    T = [4000,4100]
    ii = (bins >= T[0]) & (bins <= T[1])
    iii = np.where(bins >= T[0] + np.diff(T)[0]/2)[0][0]

    # LFP probe
    probe = 'GaussCylinderPotential'
    k = 0
    for X in LIF_net.LIF_params['X']:
        for Y in LIF_net.LIF_params['X']:
            n_ch = LIF_net.H_YX[f'{X}:{Y}'][probe].shape[0]
            for ch in range(n_ch):
                # Decimate first
                dec_kernel = ss.decimate(LIF_net.H_YX[f'{X}:{Y}'][probe],
                                         q=10,zero_phase=True)
                # Z-scored kernel from 0 to 1/2 of tau
                maxk, norm_ker = zscore(dec_kernel,ch,
                                        np.arange(int(time.shape[0]/2),
                                                  int(3*time.shape[0]/4)))
                # Z-scored LFP signal from T[0] to T[1]
                maxs,norm_sig = zscore(LFP_data[f'{X}{Y}'],ch,ii[:-1])
                # Plot data stacked in the Z-axis
                ax1[k].plot(time[int(time.shape[0]/2):int(3*time.shape[0]/4)],
                            norm_ker - ch)
                ax2[k].plot(bins[ii],norm_sig - ch)
            ax1[k].set_title(f'H_{X}:{Y}')
            ax2[k].set_title(f'H_{X}:{Y}')
            if k == 0:
                ax1[k].set_yticks(np.arange(0,-n_ch,-1))
                ax1[k].set_yticklabels(['ch. '+str(ch) for ch in np.arange(1,n_ch+1)])
                ax2[k].set_yticks(np.arange(0,-n_ch,-1))
                ax2[k].set_yticklabels(['ch. '+str(ch) for ch in np.arange(1,n_ch+1)])
            else:
                ax1[k].set_yticks([])
                ax1[k].set_yticklabels([])
                ax2[k].set_yticks([])
                ax2[k].set_yticklabels([])
            ax1[k].set_xlabel(r'$tau_{ms}$')
            ax2[k].set_xlabel('t (ms)')

            # Add scales
            ax1[k].plot([time[int(0.59*time.shape[0])],time[int(0.59*time.shape[0])]],
                         [0,-1],linewidth = 2., color = 'k')
            sexp = np.round(np.log2(maxk))
            ax1[k].text(time[int(0.6*time.shape[0])],-0.5,r'$2^{%s}mV$' % sexp)
            # ax2[k].plot([bins[iii],bins[iii]],
            #              [0,-1],linewidth = 2., color = 'k')
            # sexp = np.round(np.log2(maxs))
            # ax2[k].text(bins[iii],-0.5,r'$2^{%s}mV$' % sexp)
            k+=1

    # Current dipole moment
    probe = 'KernelApproxCurrentDipoleMoment'
    k = 0
    for X in LIF_net.LIF_params['X']:
        for Y in LIF_net.LIF_params['X']:
            # Pick only the z-component of the CDM kernel.
            # (* 1E-4 : nAum --> nAcm unit conversion)
            dec_kernel = ss.decimate([1E-4*LIF_net.H_YX[f'{X}:{Y}'][probe][2]],
                                     q=10,zero_phase=True)
            maxk, norm_ker = zscore(dec_kernel,0,
                                     np.arange(int(time.shape[0]/2),
                                               int(3*time.shape[0]/4)))
            # Z-scored CDM signal
            maxs,norm_sig = zscore([1E-4*CDM_data[f'{X}{Y}']],0,ii[:-1])
            # Plot data
            ax1[k+4].plot(time[int(time.shape[0]/2):int(3*time.shape[0]/4)],norm_ker)
            ax2[k+4].plot(bins[ii],norm_sig)

            if k == 0:
                ax1[k+4].set_yticks([0])
                ax1[k+4].set_yticklabels([r'$P_z$'])
                ax2[k+4].set_yticks([0])
                ax2[k+4].set_yticklabels([r'$P_z$'])
            else:
                ax1[k+4].set_yticks([])
                ax1[k+4].set_yticklabels([])
                ax2[k+4].set_yticks([])
                ax2[k+4].set_yticklabels([])
            ax1[k+4].set_xlabel(r'$tau_{ms}$')
            ax2[k+4].set_xlabel('t (ms)')

            # Add scales
            ax1[k+4].plot([time[int(0.59*time.shape[0])],time[int(0.59*time.shape[0])]],
                         [0,-1],linewidth = 2., color = 'k')
            sexp = np.round(np.log2(maxk))
            ax1[k+4].text(time[int(0.6*time.shape[0])],-0.5,r'$2^{%s}nAcm$' % sexp)
            ax2[k+4].plot([bins[iii+5],bins[iii+5]],
                         [0,-1],linewidth = 2., color = 'k')
            sexp = np.round(np.log2(maxs))
            ax2[k+4].text(bins[iii],-0.5,r'$2^{%s}nAcm$' % sexp)
            k+=1

    plt.show()
