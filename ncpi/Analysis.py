# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    """ This class is used to perform analysis on the data and results. """
    def __init__(self, data, **kwargs):
        self.data = data
        print('Analysis object created.')


    def EEG_topographic_plot(self, group='AD', system=19, **kwargs):
        '''
        This function creates a topographic plot of the EEG data.

        Parameters
        ----------
        group: (str)
            Name of the group to plot.
        system: (int)
            Number of electrodes in the EEG system.
        **kwargs: Additional keyword arguments like:
            - radius: (float)
                Radius of the head circumference.
            - pos: (float)
                Position of the head on the x-axis.
            - figsize: (tuple)
                Size of the figure.
            - p_value: (float)
                P-value threshold for plotting.
            - electrode_size: (float)
                Size of the electrodes.
            - label: (bool)
                Show the colorbar label.
        '''
        default_parameters = {
            'p_value': 0.05,
            'figsize': (8, 8),
            'radius': 0.6,
            'pos': 0.0,
            'electrode_size': 0.9,
            'label': True,
            'ax': None,
            'fig': None,
            'vmin': None,
            'vmax': None,
            'sensors': None
        }

        for key in kwargs.keys():
            if key not in default_parameters.keys():
                raise ValueError(f'Invalid parameter: {key}')

        p_value = kwargs.get('p_value', default_parameters['p_value'])
        figsize = kwargs.get('figsize', default_parameters['figsize'])
        radius = kwargs.get('radius', default_parameters['radius'])
        pos = kwargs.get('pos', default_parameters['pos'])
        electrode_size = kwargs.get('electrode_size', default_parameters['electrode_size'])
        label = kwargs.get('label', default_parameters['label'])
        ax = kwargs.get('ax', default_parameters['ax'])
        fig = kwargs.get('fig', default_parameters['fig'])
        vmin = kwargs.get('vmin', default_parameters['vmin'])
        vmax = kwargs.get('vmax', default_parameters['vmax'])
        sensors = kwargs.get('sensors', default_parameters['sensors'])

        if not isinstance(group, str):
            raise ValueError('The group parameter must be a string.')
        if not isinstance(system, int):
            raise ValueError('The system parameter must be an integer.')
        if not isinstance(figsize, tuple):
            raise ValueError('The figsize parameter must be a tuple.')
        if not isinstance(radius, float):
            raise ValueError('The radius parameter must be a float.')
        if not isinstance(pos, float):
            raise ValueError('The pos parameter must be a float.')
        if not isinstance(electrode_size, float):
            raise ValueError('The electrode_size parameter must be a float.')
        if not isinstance(label, bool):
            raise ValueError('The label parameter must be a boolean.')
        if not isinstance(p_value, float) or not (0.0 <= p_value <= 1.0):
            raise ValueError('The p_value parameter must be a float between 0 and 1.')
        if not isinstance(ax, plt.Axes):
            raise ValueError('The ax parameter must be a matplotlib Axes object.')
        if not isinstance(fig, plt.Figure):
            raise ValueError('The fig parameter must be a matplotlib Figure object.')
        if not isinstance(vmin, float):
            raise ValueError('The vmin parameter must be a float.')
        if not isinstance(vmax, float):
            raise ValueError('The vmax parameter must be a float.')
        
              
        def plot_simple_head_feature(ax, radius=0.6, pos=0):
            '''
            Plot a simple head feature for adding results of the EEG data analysis later.

            Parameters
            ----------
            ax: matplotlib Axes object
            radius: float,
                radius of the head circumference.
            pos: float
                Position of the head on the x-axis.
            '''

            import matplotlib.patches as mpatches
            from matplotlib.collections import PatchCollection

            # Adjust the aspect ratio of the plot
            ax.set_aspect('equal')

            # Head
            head_circle = mpatches.Circle((pos, 0), radius+0.02, edgecolor='k', facecolor='none', linewidth=0.5)
            ax.add_patch(head_circle)

            # Ears
            right_ear = mpatches.FancyBboxPatch([pos + radius + radius / 20, -radius / 10],
                                                radius / 50, radius / 5,
                                                boxstyle=mpatches.BoxStyle("Round", pad=radius / 20),
                                                linewidth=0.5)
            ax.add_patch(right_ear)

            left_ear = mpatches.FancyBboxPatch([pos - radius - radius / 20 - radius / 50, -radius / 10],
                                            radius / 50, radius / 5,
                                            boxstyle=mpatches.BoxStyle("Round", pad=radius / 20),
                                            linewidth=0.5)
            ax.add_patch(left_ear)

            # Nose
            ax.plot([pos - radius / 10, pos, pos + radius / 10], 
                    [radius + 0.02, radius + radius / 10 + 0.02,0.02 + radius], 
                    'k', linewidth=0.5)

        def plot_EEG(fig, Vax, data, radius, pos, vmin, vmax, label, electrode_size):
            '''
            Plot slopes or E/I predictions on EEG electrodes (20 EEG montage) as a
            contour plot.

            Parameters
            ----------
            fig: matplotlib figure
            Vax: matplotlib Axes object
            data: list
                Data containing slopes or E/I predictions.
            radius: float,
                radius of the head circumference.
            pos: float
                Position of the head on the x-axis.
            vmin, vmax: float
                Min and max values used for plotting.
            '''

            import scipy.interpolate
            from matplotlib.cm import ScalarMappable
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # Some parameters
            N = 100             # number of points for interpolation
            xy_center = [pos,0]   # center of the plot

            # Coordinates of the EEG electrodes in the 20 montage
            koord_dict = {
                'Fp1': [pos - 0.25 * radius, 0.8 * radius],
                'Fp2': [pos + 0.25 * radius, 0.8 * radius],
                'F3': [pos - 0.3 * radius, 0.35 * radius],
                'F4': [pos + 0.3 * radius, 0.35 * radius],
                'C3': [pos - 0.35 * radius, 0.0],
                'C4': [pos + 0.35 * radius, 0.0],
                'P3': [pos - 0.3 * radius, -0.4 * radius],
                'P4': [pos + 0.3 * radius, -0.4 * radius],
                'O1': [pos - 0.35 * radius, -0.8 * radius],
                'O2': [pos + 0.35 * radius, -0.8 * radius],
                'F7': [pos - 0.6 * radius, 0.45 * radius],
                'F8': [pos + 0.6 * radius, 0.45 * radius],
                'T3': [pos - 0.8 * radius, 0.0],
                'T4': [pos + 0.8 * radius, 0.0],
                'T5': [pos - 0.6 * radius, -0.2],
                'T6': [pos + 0.6 * radius, -0.2],
                'Fz': [pos, 0.35 * radius],
                'Cz': [pos, 0.0],
                'Pz': [pos, -0.4 * radius],
                'Oz': [pos, -0.8 * radius]
            }
            
            if system == 19:
                del koord_dict['Oz']
                koord = list(koord_dict.values())
            else:
                koord_keys = list(koord_dict.keys())
                available_sensors = [sensor for sensor in sensors if sensor in koord_keys]
                koord_sensors = [sensor for sensor in koord_keys if sensor in available_sensors]
                koord = [koord_dict[sensor] for sensor in koord_sensors]

            # External fake electrodes for completing interpolation
            for xx in np.linspace(pos-radius,pos+radius,50):
                koord.append([xx,np.sqrt(radius**2 - (xx)**2)])
                koord.append([xx,-np.sqrt(radius**2 - (xx)**2)])
                data.append(0)
                data.append(0)

            # Interpolate data points
            x,y = [],[]
            for i in koord:
                x.append(i[0])
                y.append(i[1])
            z = data

            xi = np.linspace(-radius, radius, N)
            yi = np.linspace(-radius, radius, N)
            zi = scipy.interpolate.griddata((np.array(x), np.array(y)), z,
                                            (xi[None,:], yi[:,None]), method='cubic')


            # Use different number of levels for the fill and the lines
            CS = Vax.contourf(xi, yi, zi, 30, cmap = plt.cm.bwr, zorder = 1,
                            vmin = vmin,vmax = vmax)
            Vax.contour(xi, yi, zi, 5, colors = "grey", zorder = 2,linewidths = 0.4,
                        vmin = vmin,vmax = vmax)

            # Make a color bar
            # cbar = fig.colorbar(CS, ax=Vax)
            divider = make_axes_locatable(Vax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            if np.sum(np.abs(data)) > 2: 
                colorbar = fig.colorbar(ScalarMappable(norm=CS.norm, cmap=CS.cmap), cax=cax)
                colorbar.ax.tick_params(labelsize=8)
                if label == True:
                    colorbar.ax.xaxis.set_label_position('bottom')
                    # bbox = colorbar.ax.get_position()
                    # print(bbox)
                    colorbar.set_label('z-ratio', size=5, labelpad=-15, rotation=0, y=0.)
                    
            else:
                # Hide the colorbar if the data is not significant
                cax.axis('off')

            # Add the EEG electrode positions
            Vax.scatter(x[:system], y[:system], marker = 'o', c = 'k', s = electrode_size, zorder = 3)
        

        if type(self.data) == list or type(self.data) == np.ndarray:
            results = self.data
        else:
            if 'p.value' in self.data.columns:
                results = self.data['Ratio'].where((self.data['Group'] == group) & (self.data['p.value'] < p_value)).to_list()
            else:
                results = self.data['Ratio'].where(self.data['Group'] == group).to_list()
     
        plot_simple_head_feature(ax, radius, pos)

        plot_EEG(fig, ax, results, radius, pos, vmin, vmax, label, electrode_size)

