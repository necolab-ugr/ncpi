# # Plot some statistics of the simulation data
        # plt.figure(dpi = 300)
        # plt.rc('font', size=8)
        # plt.rc('font', family='Arial')
        #
        # for param in range(7):
        #     print(f'Parameter {theta["parameters"][param]}')
        #     plt.subplot(2,4,param+1)
        #     ax = sns.histplot(theta['data'][:,param], kde=True, bins=50, color='blue')
        #     ax.set_title(f'Parameter {theta["parameters"][param]}')
        #     ax.set_xlabel('')
        #     ax.set_ylabel('')
        #     plt.tight_layout()
        #
        # plt.figure(figsize=(15, 15))
        # plt.rc('font', size=8)
        # plt.rc('font', family='Arial')
        #
        # # Iterate over pairs of columns in theta['data']
        # for i in range(7):
        #     for j in range(i + 1, 7):
        #         print(f'Parameter {theta["parameters"][i]} vs Parameter {theta["parameters"][j]}')
        #         plt.subplot(7, 7, i * 7 + j + 1)
        #         hist, xedges, yedges = np.histogram2d(theta['data'][:, i], theta['data'][:, j], bins=50)
        #         plt.imshow(hist.T, origin='lower', interpolation='bilinear', cmap='viridis', aspect='auto',
        #                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        #         plt.colorbar()
        #         plt.xlabel(f'{theta["parameters"][i]}')
        #         plt.ylabel(f'{theta["parameters"][j]}')
        #         plt.title(f'{theta["parameters"][i]} vs {theta["parameters"][j]}')
        #         plt.tight_layout()
        #
        # plt.show()