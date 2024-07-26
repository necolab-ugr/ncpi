# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
# from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt

class Analysis:
    """ This class is used to perform analysis on the results and input data."""
    def __init__(self, data, **kwargs):
        """
        Initialize the Analysis class with input data.

        Parameters
        ----------
        data: pd.DataFrame
            Input data to be analyzed.
        **kwargs: Additional keyword arguments
            Any additional parameters for the analysis.
        """
        self.data = data
        self.kwargs = kwargs

    def lmer(self):

        # Activate pandas2ri to enable conversion between pandas DataFrames and R
        pandas2ri.activate()

        # Load R libraries directly in R
        r('''
        library(dplyr)
        library(lme4)
        library(emmeans)
        library(ggplot2)
        library(repr)
        library(mgcv)
        library(mixedup)
        ''')

        # Load the DataFrame (pkl)
        # df = pd.read_pickle(df_path)

        self.analyze_variable(self.data)

    def analyze_variable(self, df):

        df = df[df['Data'] != 0]  # Remove rows where the variable is zero
        # print(df)
        groups = df['Group'].unique()
        sensors = df['Sensor'].unique()
        print(sensors)

        # IMPORTANT: The  control group must be named ''HC''
        # Filter out 'HC' from the list of unique groups
        groups = [group for group in groups if group != 'HC']

        # Create a list with the different group comparisons (Group0 vs Group1, Group0 vs Group2, etc.)
        groups_comp = [f'{group}vsHC' for group in groups]

        multi_df = pd.DataFrame()

        for label, label_comp in zip(groups, groups_comp):

            # Filter DataFrame to obtain the desired groups
            df_red = df[(df['Group'] == 'HC') | (df['Group'] == label)]
            print(df_red)
            ro.globalenv['df_red'] = pandas2ri.py2rpy(df_red)
            ro.globalenv['label'] = label

            # Create a df for each group comparison, with columns 'sensor, 'ratio' and 'p_value'
            df_label = pd.DataFrame(columns=['Sensor', 'Ratio', 'p.value'])

            r(''' 
            df_red$ID = as.factor(df_red$ID)
            df_red$Group = factor(df_red$Group, levels = c(label, 'HC'))
            df_red$Epoch = as.factor(df_red$Epoch)
            df_red$Sensor = as.factor(df_red$Sensor)
            ''')

            r('''
            mod00 = Data ~ Group * Sensor + (1 | ID)
            mod03 = Data ~ Group * Sensor
            mod0 = Data ~ Group + Sensor + (1 | ID)
            mod3 = Data ~ Group + Sensor
        
            m00 <- lmer(mod00, data=df_red)
            m03 <- lm(mod03, data=df_red)
            mix_models <- c('m00', 'm03')
            bics <- c(BIC(m00), BIC(m03))

            index <- which.min(bics)
            print(paste("Models:", mix_models))
            print(paste("BICs:", bics))
            print(paste("Minimum BIC:", min(bics)))
            print(paste("Model with lowest BIC:", mix_models[index]))
            mod_sel <- mix_models[index]
            print(mod_sel)
            ''')

            r('''
            if (mod_sel == 'm00') {
                m0 = lmer(mod0, data=df_red)
                anova_result = capture.output(anova(m0, m00))
                print(anova_result)
                valores <- strsplit(anova_result[7], " ")[[1]]
                valores <- valores[valores != ""]
                p_value = as.numeric(valores[length(valores)])
                p_value = ifelse(is.na(p_value), 0, p_value)
                print(p_value)
                if (p_value < 0.05) {
                    mod = m00
                    mod_sel = "m00"
                    cat("mod selected:", mod_sel)
                } else {
                    mod = m0
                    mod_sel <- "m0"
                    cat("mod selected:", mod_sel)
                }
            }
            ''')

            r('''
            if (mod_sel == 'm03') {
                m3 = lm(mod3, data=df_red)
                anova_result = capture.output(anova(m3, m03))
                print(anova_result)
                valores <- strsplit(anova_result[7], " ")[[1]]
                valores <- valores[valores != ""]
                p_value <- as.numeric(valores[length(valores)])
                p_value <- ifelse(is.na(p_value), 0, p_value)
                print(p_value)
                if (p_value < 0.05) {
                    mod = m03
                    mod_sel = "m03"
                    cat("mod selected:", mod_sel)
                } else {
                    mod = m3
                    mod_sel = "m3"
                    cat("mod selected:", mod_sel)
                }
            }
            ''')

            r('''
            print(mod)
            if (p_value < 0.05) {
                emm <- suppressMessages(emmeans(mod, specs=~Group | Sensor))
            } else {
                emm <- suppressMessages(emmeans(mod, specs=~Group))
            }
            res <- pairs(emm, adjust='holm')
            df_res <- as.data.frame(res)
            print(df_res)
            filtered_df <- df_res[df_res$p.value < 0.05, ]
            ''')

            df_res_r = ro.r['df_res']
            mod_sel = ro.r['mod_sel']
            # Convert the R DataFrame to a pandas DataFrame
            with (pandas2ri.converter + pandas2ri.converter).context():
                df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)
            print(df_res_pd)
            print(mod_sel)

            # We extract p-values from the results DataFrame
            df_label['p.value'] = df_res_pd['p.value']

            if mod_sel == 'm00' or mod_sel == 'm03':

                df_label['Sensor'] = sensors
                df_label['Ratio'] = df_res_pd['z.ratio']

            elif mod_sel == 'm0' or mod_sel == 'm3':

                df_label['Sensor'] = 0
                df_label['Ratio'] = df_res_pd['t.ratio']

            print(df_label)

            # Asigna un superíndice a df_label
            df_label.index = pd.MultiIndex.from_product([[label_comp], df_label.index])
            # Concatena df_label a multi_df
            multi_df = pd.concat([multi_df, df_label])
            print(multi_df)

            # ruta = os.path.join(f'/home/juanmiguel/GitHub/hctsa/R/{db}/trials',
            #                     f'Results_{group_sel}_fooof_{fc}.pkl')


    def EEG_topographic_plot(self, group='AD', p_value=0.05, system=19, figsize=(5, 5), radius=0.6, pos=0):
        '''
        This function creates a topographic plot of the EEG data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the EEG data.
        group: str
            Name of the group to plot.
        p.value: float
            P-value threshold for the analysis.
        system: int
            Number of electrodes in the EEG system.
        '''
        if type(self.data) is not pd.DataFrame:
            raise ValueError('The df parameter must be a pandas DataFrame.')
        if type(system) is not int:
            raise ValueError('The system parameter must be an integer.')
        
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
                                                boxstyle=mpatches.BoxStyle("Round", pad=radius / 20))
            ax.add_patch(right_ear)

            left_ear = mpatches.FancyBboxPatch([pos - radius - radius / 20 - radius / 50, -radius / 10],
                                            radius / 50, radius / 5,
                                            boxstyle=mpatches.BoxStyle("Round", pad=radius / 20))
            ax.add_patch(left_ear)

            # Nose
            ax.plot([pos - radius / 10, pos, pos + radius / 10], [radius + 0.02, radius + radius / 10 + 0.02,0.02 + radius], 'k')

        def plot_EEG(fig, Vax, data, radius, pos, vmin, vmax, label, **kwargs):
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
            elif system == 20:
                koord = list(koord_dict.values())

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
                    colorbar.ax.xaxis.set_label_position('top')
                    bbox = colorbar.ax.get_position()
                    print(bbox)
                    colorbar.set_label('Ratio', size=8, labelpad=0, rotation=0, va='center')
                    
            else:
                # Hide the colorbar if the data is not significant
                cax.axis('off')


            # Add the EEG electrode positions
            print(len(koord))
            Vax.scatter(x[:system], y[:system], marker = 'o', c = 'k', s = 0.9, zorder = 3)
        
        results = self.data['Ratio'].where((self.data['Group'] == group) & (self.data['p.value'] < p_value))
        
        data = results.fillna(0.0).tolist()

        max_Ratio = self.data[self.data['Group'] == group]['Ratio'].max()

        vmin = -max_Ratio
        vmax = max_Ratio

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        print("Aqui")
        plot_simple_head_feature(ax, radius, pos)

        plot_EEG(fig, ax, data, radius, pos, vmin, vmax, True)

        plt.show()
