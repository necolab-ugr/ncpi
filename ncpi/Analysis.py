import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy.interpolate
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro

class Analysis:
    """ The Analysis class is designed to facilitate statistical analysis and data visualization.

    Parameters
    ----------
    data: (list, np.ndarray, pd.DataFrame)
        Data to be analyzed.
    """
    def __init__(self, data):
        self.data = data


    def lmer(self, control_group = 'HC', data_col = 'Y', data_index = -1, sensors = False):

        # Activate pandas2ri
        pandas2ri.activate()

        # Load R libraries
        r('''
        library(dplyr)
        library(lme4)
        library(emmeans)
        library(ggplot2)
        library(repr)
        library(mgcv)
        ''')

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError('The data parameter must be a pandas DataFrame.')

        # Copy the dataframe
        df = self.data.copy()

        # Remove all columns except 'ID', 'Group', 'Epoch', 'Sensor' and data_col
        df = df[['ID', 'Group', 'Epoch', 'Sensor', data_col]]

        # If data_index is not empty, select the element from the data_col column
        if data_index >= 0:
            df[data_col] = df[data_col].apply(lambda x: x[data_index])

        # Filter out control_group from the list of unique groups
        groups = df['Group'].unique()
        groups = [group for group in groups if group != control_group]

        # Create a list with the different group comparisons
        groups_comp = [f'{group}vs{control_group}' for group in groups]

        # Remove rows where the data_col is zero
        df = df[df[data_col] != 0]

        # Rename data_col column to Y
        df.rename(columns={data_col: 'Y'}, inplace=True)

        results = {}
        for label, label_comp in zip(groups, groups_comp):
            print(f'\n\n--- Group: {label}')
            # Filter the DataFrame to obtain the desired groups
            df_pair = df[(df['Group'] == control_group) | (df['Group'] == label)]
            ro.globalenv['df_pair'] = pandas2ri.py2rpy(df_pair)
            ro.globalenv['label'] = label
            ro.globalenv['control_group'] = control_group

            # Convert columns to factors
            r('''
            df_pair$ID = as.factor(df_pair$ID)
            df_pair$Group = factor(df_pair$Group, levels = c(label, control_group))
            df_pair$Epoch = as.factor(df_pair$Epoch)
            df_pair$Sensor = as.factor(df_pair$Sensor)
            print(table(df_pair$Group))
            ''')

            # if table in R is empty for any group, skip the analysis
            if r('table(df_pair$Group)')[0] == 0 or r('table(df_pair$Group)')[1] == 0:
                results[label_comp] = pd.DataFrame({'p.value': [1], 'z.ratio': [0]})
            else:
                # Fit the linear mixed-effects models:
                # - mod00: full model with random intercepts for each subject
                # - mod01: reduced model without random intercepts
                # - mod02: full model with random intercepts for each subject and sensor
                # - mod03: reduced model without random intercepts for each sensor
                if sensors == False:
                    r('''
                    mod00 = Y ~ Group  + (1 | ID)
                    mod01 = Y ~ Group
                    m00 <- lmer(mod00, data=df_pair)
                    m01 <- lm(mod01, data=df_pair)
                    print(summary(m00))
                    print(summary(m01))
                    ''')
                else:
                    r('''
                    mod00 = Y ~ Group * Sensor + (1 | ID)
                    mod01 = Y ~ Group * Sensor
                    mod02 = Y ~ Group + Sensor + (1 | ID)
                    mod03 = Y ~ Group + Sensor
                    m00 <- lmer(mod00, data=df_pair)
                    m01 <- lm(mod01, data=df_pair)
                    m02 <- lmer(mod02, data=df_pair)
                    m03 <- lm(mod03, data=df_pair)
                    print(summary(m00))
                    print(summary(m01))
                    print(summary(m02))
                    print(summary(m03))
                    ''')

                # BIC model selection
                r('''
                all_models <- c('m00', 'm01')
                bics <- c(BIC(m00), BIC(m01))
                print(bics)
                index <- which.min(bics)
                mod_sel <- all_models[index]

                if (mod_sel == 'm00') {
                    m_sel <- m00
                }
                if (mod_sel == 'm01') {
                    m_sel <- m01
                }
                ''')

                # ANOVA test
                if sensors == True:
                    r('''
                    if (mod_sel == 'm00') {
                        anova_result = capture.output(anova(m02, m00))
                        val <- strsplit(anova_result[7], " ")[[1]]
                        val <- val[val != ""]
                        p_value = as.numeric(val[length(val)])
                        p_value = ifelse(is.na(p_value), 0, p_value)
                        if (p_value >= 0.05) {
                            m_sel <- m02
                        }
                    }
                    if (mod_sel == 'm01') {
                        anova_result = capture.output(anova(m03, m01))
                        val <- strsplit(anova_result[7], " ")[[1]]
                        val <- val[val != ""]
                        p_value = as.numeric(val[length(val)])
                        p_value = ifelse(is.na(p_value), 0, p_value)
                        if (p_value >= 0.05) {
                            m_sel <- m03
                        }
                    }
                    ''')

                # Compute the pairwise comparisons between groups
                if sensors == False:
                    r('''
                    emm <- suppressMessages(emmeans(m_sel, specs=~Group))
                    ''')
                else:
                    r('''
                    emm <- suppressMessages(emmeans(m_sel, specs=~Group | Sensor))
                    ''')

                r('''
                res <- pairs(emm, adjust='holm')
                df_res <- as.data.frame(res)
                print(df_res)
                ''')

                df_res_r = ro.r['df_res']

                # Convert the R DataFrame to a pandas DataFrame
                with (pandas2ri.converter + pandas2ri.converter).context():
                    df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)

                results[label_comp] = df_res_pd

        return results

    def EEG_topographic_plot(self, **kwargs):
        '''
        This function generates a topographical plot of EEG data using the 10-20 electrode placement system,
        visualizing activity from 19 or 20 electrodes.

        Parameters
        ----------
        **kwargs: keyword arguments:
            - radius: (float)
                Radius of the head circumference.
            - pos: (float)
                Position of the head on the x-axis.
            - electrode_size: (float)
                Size of the electrodes.
            - label: (bool)
                Show the colorbar label.
            - ax: (matplotlib Axes object)
                Axes object to plot the data.
            - fig: (matplotlib Figure object)
                Figure object to plot the data.
            - vmin: (float)
                Min value used for plotting.
            - vmax: (float)
                Max value used for plotting.
        '''

        default_parameters = {
            'radius': 0.6,
            'pos': 0.0,
            'electrode_size': 0.9,
            'label': True,
            'ax': None,
            'fig': None,
            'vmin': None,
            'vmax': None
        }

        for key in kwargs.keys():
            if key not in default_parameters.keys():
                raise ValueError(f'Invalid parameter: {key}')

        radius = kwargs.get('radius', default_parameters['radius'])
        pos = kwargs.get('pos', default_parameters['pos'])
        electrode_size = kwargs.get('electrode_size', default_parameters['electrode_size'])
        label = kwargs.get('label', default_parameters['label'])
        ax = kwargs.get('ax', default_parameters['ax'])
        fig = kwargs.get('fig', default_parameters['fig'])
        vmin = kwargs.get('vmin', default_parameters['vmin'])
        vmax = kwargs.get('vmax', default_parameters['vmax'])

        if not isinstance(radius, float):
            raise ValueError('The radius parameter must be a float.')
        if not isinstance(pos, float):
            raise ValueError('The pos parameter must be a float.')
        if not isinstance(electrode_size, float):
            raise ValueError('The electrode_size parameter must be a float.')
        if not isinstance(label, bool):
            raise ValueError('The label parameter must be a boolean.')
        if not isinstance(ax, plt.Axes):
            raise ValueError('The ax parameter must be a matplotlib Axes object.')
        if not isinstance(fig, plt.Figure):
            raise ValueError('The fig parameter must be a matplotlib Figure object.')
        if not isinstance(vmin, float):
            raise ValueError('The vmin parameter must be a float.')
        if not isinstance(vmax, float):
            raise ValueError('The vmax parameter must be a float.')
        if not isinstance(self.data, (list, np.ndarray)):
            raise ValueError('The data parameter must be a list or numpy array.')
        if len(self.data) not in [19, 20]:
            raise ValueError('The data parameter must contain 19 or 20 elements.')
        
              
        def plot_simple_head(ax, radius=0.6, pos=0):
            '''
            Plot a simple head model with ears and nose.

            Parameters
            ----------
            ax: matplotlib Axes object
            radius: float,
                radius of the head circumference.
            pos: float
                Position of the head on the x-axis.
            '''

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


        def plot_EEG(data, radius, pos, electrode_size, label, ax, fig, vmin, vmax):
            '''
            Plot the EEG data on the head model as a topographic map.

            Parameters
            ----------
            data: list or np.ndarray of size (19,) or (20,)
                EEG data.
            radius: float
                Radius of the head circumference.
            pos: float
                Position of the head on the x-axis.
            electrode_size: float
                Size of the electrodes.
            label: bool
                Show the colorbar label.
            ax: matplotlib Axes object
                Axes object to plot the data.
            fig: matplotlib Figure object
                Figure object to plot the data.
            vmin: float
                Min value used for plotting.
            vmax: float
                Max value used for plotting.
            '''

            # Coordinates of the EEG electrodes
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
            
            if len(data) == 19:
                del koord_dict['Oz']
            koord = list(koord_dict.values())

            # Number of points used for interpolation
            N = 100

            # External fake electrodes used for interpolation
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
            CS = ax.contourf(xi, yi, zi, 30, cmap = plt.cm.bwr, zorder = 1,
                             vmin = vmin, vmax = vmax)
            ax.contour(xi, yi, zi, 5, colors ="grey", zorder = 2, linewidths = 0.4,
                       vmin = vmin, vmax = vmax)

            # Make a color bar
            # cbar = fig.colorbar(CS, ax=Vax)
            divider = make_axes_locatable(ax)
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
            ax.scatter(x[:len(koord_dict)], y[:len(koord_dict)], marker ='o', c ='k', s = electrode_size, zorder = 3)


        plot_simple_head(ax, radius, pos)
        plot_EEG(self.data, radius, pos, electrode_size, label, ax, fig, vmin, vmax)

