import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from matplotlib.cm import ScalarMappable
from ncpi import tools


def extract_variables(formula):
    tokens = re.split(r'[\s\+\~\|\(\)\*\/\-]+', formula)
    return set(t for t in tokens if t and not t.replace('.', '', 1).isdigit())


class Analysis:
    """ The Analysis class is designed to facilitate statistical analysis and data visualization.

    Parameters
    ----------
    data: (list, np.ndarray, pd.DataFrame)
        Data to be analyzed.
    """
    def __init__(self, data):
        self.data = data


    def lmer(self, control_group = 'HC', data_col = 'Y', other_col = ['ID', 'Group', 'Epoch', 'Sensor'],
             data_index = -1, models = None, bic_models = None, anova_tests = None, specs = None):
        """
        Perform linear mixed-effects model (lmer) or linear model (lm) comparisons using R's `lme4` and `emmeans`
        packages. We assume that, at least, a 'Group' column is present in the dataframe.

        Parameters
        ----------
        control_group: str
            The control group to be used for comparisons.
        data_col: str
            The name of the data column to be analyzed.
        other_col: list
            The names of the other columns to be included in the analysis.
            The default is ['ID', 'Group', 'Epoch', 'Sensor'].
        data_index: int
            The index of the data column to be analyzed. If -1, the entire column is used.
        models: dict
            A dictionary of models to be used for analysis. The keys are model names and the values are model formulas.
            if models is None, the default models are used:
                - mod00: Y ~ Group + (1 | ID)
                - mod01: Y ~ Group
            The best model is selected based on BIC (Bayesian Information Criterion), unless bic_models is None.
        bic_models: list
            A  list of models to be evaluated using BIC. If bic_models is None, the first model is selected. All models
            have to be defined in the models dictionary.
            Example:
                bic_models = ["mod01", "mod02"] # Compare mod01 vs. mod02
        anova_tests: dict
            A dictionary that specifies which models should undergo an ANOVA test after BIC selection. If anova_tests
            is None, no ANOVA tests are performed. Each test must contain two models to be compared. All models have
            to be defined in the models dictionary.
            Example:
                anova_tests = {
                    "test1": ["mod00", "mod02"],  # Compare mod00 vs. mod02
                    "test2": ["mod01", "mod03"]   # Compare mod01 vs. mod03
                }
        specs: string
            The specifications for the emmeans function in R. If specs is None, the default specs are used:
                - ~Group

        Returns
        -------
        results: dict
            A dictionary containing the results of the analysis. The keys are the names of the groups being compared
            and the values are DataFrames containing the results of the analysis.

        """

        # Check if rpy2 is installed
        if not tools.ensure_module("rpy2"):
            raise ImportError("rpy2 is required for lmer but is not installed.")
        pandas2ri = tools.dynamic_import("rpy2.robjects.pandas2ri")
        r = tools.dynamic_import("rpy2.robjects","r")
        ListVector = tools.dynamic_import("rpy2.robjects","ListVector")
        ro = tools.dynamic_import("rpy2","robjects")

        # Activate pandas2ri
        pandas2ri.activate()

        # Import R packages
        ro.r('''
        # Function to check and load packages
        load_packages <- function(packages) {
            for (pkg in packages) {
                if (!require(pkg, character.only = TRUE)) {
                    stop("R package '", pkg, "' is not installed.")
                }
            }
        }

        # Load required packages
        load_packages(c("lme4", "emmeans"))
        ''')

        # Check if the data is a pandas DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError('The data must be a pandas DataFrame.')

        # Check if the data_col is in the DataFrame
        if data_col not in self.data.columns:
            raise ValueError(f'The data_col "{data_col}" is not in the DataFrame columns.')

        # Check if the other_col columns are in the DataFrame
        for col in other_col:
            if col not in self.data.columns:
                raise ValueError(f'The column "{col}" is not in the DataFrame.')

        # Copy the dataframe
        df = self.data.copy()

        # Remove all columns except data_col and other_col
        df = df[other_col + [data_col]]

        # If data_index is not -1, select the data_index value from the data_col
        if data_index >= 0:
            df[data_col] = df[data_col].apply(lambda x: x[data_index])

        # Check if 'Group' is in the DataFrame
        if 'Group' not in df.columns:
            raise ValueError('The column "Group" is not in the DataFrame.')

        # Filter out control_group from the list of unique groups
        groups = df['Group'].unique()
        groups = [group for group in groups if group != control_group]

        # Create a list with the different group comparisons
        groups_comp = [f'{group}vs{control_group}' for group in groups]

        # Remove rows where the data_col is zero
        df = df[df[data_col] != 0]

        # Rename data_col column to Y
        df.rename(columns={data_col: 'Y'}, inplace=True)

        # Force categorical data type for Sensor
        if 'Sensor' in df.columns:
            df["Sensor"] = df["Sensor"].astype(str).astype('category')

        # Default models if none are provided
        if models is None:
            models = {
                'mod00': 'Y ~ Group + (1 | ID)',
                'mod01': 'Y ~ Group'
            }

        # Default specs if none are provided
        if specs is None:
            specs = '~Group'

        # Check that all models defined in bic_models have been also included in the models dictionary
        if bic_models is not None:
            for model in bic_models:
                if model not in models.keys():
                    raise ValueError(f'bic_models: the model "{model}" is not defined in the models dictionary.')

        # Check that all models defined in anova_tests have been also included in the models dictionary
        if anova_tests is not None:
            for test in anova_tests.values():
                for model in test:
                    if model not in models.keys():
                        raise ValueError(f'anova_tests: the model "{model}" is not defined in the models dictionary.')

        # Check that variables included in models are also in the dataframe
        df_columns = set(df.columns)
        missing_vars = {}
        for name, formula in models.items():
            vars_in_formula = extract_variables(formula)
            missing = vars_in_formula - df_columns
            if missing:
                missing_vars[name] = missing

        if missing_vars:
            raise ValueError(f"Some models have missing variables: {missing_vars}")

        results = {}
        for label, label_comp in zip(groups, groups_comp):
            print(f'\n\n--- Group: {label}')
            r('rm(list = ls())')
            df_pair = df[df['Group'].isin([control_group, label])]
            ro.globalenv['df_pair'] = pandas2ri.py2rpy(df_pair)
            ro.globalenv['label'] = label
            ro.globalenv['control_group'] = control_group

            # # Convert to factors
            # r('''
            # df_pair$ID = as.factor(df_pair$ID)
            # df_pair$Group = factor(df_pair$Group, levels = c(label, control_group))
            # df_pair$Epoch = as.factor(df_pair$Epoch)
            # df_pair$Sensor = as.factor(df_pair$Sensor)
            # print(table(df_pair$Group))
            # ''')

            # Convert to factors
            r_code = []
            for col in other_col:
                if col == 'Group':
                    r_code.append(f'df_pair${col} = factor(df_pair${col}, levels = c("{label}", "{control_group}"))')
                else:
                    r_code.append(f'df_pair${col} = as.factor(df_pair${col})')

            # Print table for 'Group'
            if 'Group' in other_col:
                r_code.append('print(table(df_pair$Group))')

            # Join all lines into one R script string
            full_r_script = '\n'.join(r_code)

            # Pass to R
            r(full_r_script)

            # if table in R is empty for any group, skip the analysis
            if r('table(df_pair$Group)')[0] == 0 or r('table(df_pair$Group)')[1] == 0:
                results[label_comp] = pd.DataFrame({'p.value': [1], 'z.ratio': [0]})
            # Fit the linear (mixed-effects) models
            else:
                for ii, (model_name, formula) in enumerate(models.items()):
                    if (bic_models is None and ii == 0) or (bic_models is not None and model_name in bic_models):
                        # print(f'--- BIC test: fitting model: {model_name}')
                        ro.globalenv[model_name] = formula
                        r(f"{model_name} <- {'lmer' if '(1 | ID)' in formula else 'lm'}({model_name}, data=df_pair)")

                # BIC test: handle single and multiple model cases properly
                r('''
                all_models <- names(which(sapply(ls(), function(x) inherits(get(x), "merMod") || inherits(get(x), "lm"))))

                if (length(all_models) == 1) {
                    m_sel <- all_models[1]  # Use the only model available
                } else {
                    bics <- sapply(all_models, function(m) BIC(get(m)))
                    index <- which.min(bics)
                    m_sel <- all_models[index]
                }
                
                final_model <- get(m_sel)

                ''')
                print(f'--- BIC test. Selected model: {r("m_sel")}')

                # Perform ANOVA tests only for user-specified comparisons
                if anova_tests is not None:
                    # Fit the remaining models
                    for ii, (model_name, formula) in enumerate(models.items()):
                        # Check if model already exists in the R environment
                        if model_name not in r.ls():
                            # print(f'--- ANOVA test: fitting model: {model_name}')
                            ro.globalenv[model_name] = formula
                            r(f"{model_name} <- {'lmer' if '(1 | ID)' in formula else 'lm'}({model_name}, data=df_pair)")

                    # Convert to R list
                    r_anova_tests = ListVector(anova_tests)

                    # Assign to R global environment
                    ro.globalenv['anova_tests'] = r_anova_tests


                    r('''
                    m_name <- m_sel
                    for (comparison in names(anova_tests)) {
                        models <- anova_tests[[comparison]]
                        
                        # Check if the selected model is in the list of models to compare
                        if (m_sel %in% models) {
                            anova_result <- capture.output(anova(get(as.character(models[2])), get(as.character(models[1]))))
                            
                            # Extract p-value from the ANOVA result.
                            # Case 1: Standard ANOVA table with Pr(>F)
                            if (any(grepl("Pr\\\\(>F\\\\)", anova_result))) {
                                p_line <- anova_result[grep("Pr\\\\(>F\\\\)", anova_result)+2]
                            }
                            # Case 2: Mixed effects output with Pr(>Chisq)
                            else if (any(grepl("Pr\\\\(>Chisq\\\\)", anova_result))) {
                                p_line <- anova_result[grep("Pr\\\\(>Chisq\\\\)", anova_result) + 2]
                            }
                            # are there any other cases?
                            else {
                            next
                            }
                            matches <- regmatches(p_line, gregexpr("[0-9]+\\\\.?[0-9]*(e[+-]?[0-9]+)?", p_line))[[1]]
                            p_value <- as.numeric(tail(matches, 1))
                             
                            if (!is.na(p_value) && p_value >= 0.05) {
                                # Determine which model is simpler (counts all fixed-effect terms, including interactions)
                                formula1 <- formula(get(as.character(models[1])))
                                formula2 <- formula(get(as.character(models[2])))
                                
                                # Count terms (excluding random effects after '|')
                                terms1 <- length(attr(terms(formula1), "term.labels"))  # Handles *, :, etc.
                                terms2 <- length(attr(terms(formula2), "term.labels"))

                                # Select the model with less complexity
                                if (terms1 <= terms2) {
                                    final_model <- get(as.character(models[1]))
                                    m_name <- models[1]
                                } else {
                                    final_model <- get(as.character(models[2]))
                                    m_name <- models[2]
                                }
                                break
                            }
                        }
                    }
                    ''')
                    print(f'--- ANOVA test. Selected model: {r("m_name")}')

                # Compute pairwise comparisons
                ro.globalenv['specs'] = specs

                r('''
                emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                res <- pairs(emm, adjust='holm')
                df_res <- as.data.frame(res)
                ''')

                # Ensure Sensor remains as a character column
                if 'Sensor' in r('names(df_res)'):
                    r('''
                        df_res$Sensor <- as.character(df_res$Sensor)
                    ''')

                df_res_r = ro.r['df_res']
                with (pandas2ri.converter + pandas2ri.converter).context():
                    df_res_pd = pandas2ri.conversion.get_conversion().rpy2py(df_res_r)

                results[label_comp] = df_res_pd

        return results

    def cohend(self, control_group='HC', data_col='Y', data_index=-1):
        '''
        Compute Cohen's d for all pairwise group comparisons across sensors.

        Parameters
        ----------
        control_group: str
            The control group to be used for comparisons.
        data_col: str
            The name of the data column to be analyzed.
        data_index: int
            The index of the data column to be analyzed. If -1, the entire column is used.

        Returns
        -------
        results: dict
            A dictionary containing the results of the analysis. The keys are the names of the groups being compared
            and the values are lists containing the Cohen's d values for each sensor.
        '''

        # Check if the data is a pandas DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError('The data must be a pandas DataFrame.')

        # Check if the data_col is in the DataFrame
        if data_col not in self.data.columns:
            raise ValueError(f'The data_col "{data_col}" is not in the DataFrame columns.')

        # Check if 'Group' and 'Sensor' are in the DataFrame
        for col in ['Group', 'Sensor']:
            if col not in self.data.columns:
                raise ValueError(f'The column "{col}" is not in the DataFrame.')

        # Copy the dataframe
        df = self.data.copy()

        # Remove all columns except 'Group', 'Sensor' and data_col
        df = df[['Group', 'Sensor', data_col]]

        # If data_index is not -1, select the data_index value from the data_col
        if data_index >= 0:
            df[data_col] = df[data_col].apply(lambda x: x[data_index])

        # Filter out control_group from the list of unique groups
        groups = df['Group'].unique()
        groups = [group for group in groups if group != control_group]

        # Create a list with the different group comparisons
        groups_comp = [f'{group}vs{control_group}' for group in groups]

        # Remove rows where the data_col is zero
        df = df[df[data_col] != 0]

        results = {}
        for label, label_comp in zip(groups, groups_comp):
            print(f'\n\n--- Group: {label}')

            # filter out control_group and the current group
            df_pair = df[df['Group'].isin([control_group, label])]

            all_d = []
            all_sensors = []
            for sensor in df_pair['Sensor'].unique():
                df_sensor = df_pair[df_pair['Sensor'] == sensor]

                group1 = np.array(df_sensor[df_sensor['Group'] == label][data_col])
                group2 = np.array(df_sensor[df_sensor['Group'] == control_group][data_col])

                # Check if both groups have more than 2 elements
                if len(group1) > 2 and len(group2) > 2:
                    # Calculate Cohen's d
                    n1, n2 = len(group1), len(group2)
                    mean1, mean2 = np.mean(group1), np.mean(group2)
                    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

                    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
                    d = (mean1 - mean2) / pooled_std
                    all_d.append(d)
                else:
                    all_d.append(np.nan)
                all_sensors.append(sensor)

            results[label_comp] = pd.DataFrame({'d': all_d, 'Sensor': all_sensors})

        return results

    def EEG_topographic_plot(self, **kwargs):
        '''
        Generate a topographical plot of EEG data using the 10-20 electrode placement system,
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

        # Check if mpl_toolkits is installed
        if not tools.ensure_module("mpl_toolkits"):
            raise ImportError("mpl_toolkits is required for EEG_topographic_plot but is not installed.")
        make_axes_locatable = tools.dynamic_import("mpl_toolkits.axes_grid1",
                                                   "make_axes_locatable")

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

            # Check data type
            if not isinstance(data, (list, np.ndarray)):
                raise ValueError('The data must be a list or numpy array.')

            # Check data length
            if len(data) not in [19, 20]:
                raise ValueError('The data must contain 19 or 20 elements.')

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