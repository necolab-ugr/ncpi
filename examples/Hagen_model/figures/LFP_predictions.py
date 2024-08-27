import numpy as np
from matplotlib import pyplot as plt

# Create a figure
fig = plt.figure(figsize=(7.5, 4.5), dpi=300)
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

# Titles for the subplots
titles = [r'$E/I$', r'$\tau_{exc}^{syn}$', r'$\tau_{inh}^{syn}$', r'$J_{ext}^{syn}$', r'$fr$ (spikes/s)',]

for row in range(3):
    for col in range(5):
        ax = fig.add_axes([0.1 + col * 0.18, 0.68 - row * 0.29, 0.15, 0.25])

        # Titles
        if row == 0:
            ax.set_title(titles[col])

        # X-axis labels
        if row == 2:
            ax.set_xticks(np.arange(1,6))
            ax.set_xticklabels([f'{str(i)}' for i in np.arange(1,6)])
            ax.set_xlabel('Postnatal days')
        else:
            ax.set_xticks([])

        # Y-axis labels
        if col == 0:
            if row == 0:
                ax.set_ylabel(r'$catch22$')
            elif row == 1:
                ax.set_ylabel(r'$1/f$'+' '+r'$slope$')
            else:
                ax.set_ylabel(r'$fE/I$')
        else:
            ax.set_yticks([])

plt.show()