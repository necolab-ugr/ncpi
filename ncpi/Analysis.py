import numpy as np
import pandas as pd
import scipy.interpolate
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from ncpi import tools


def extract_variables(formula):
    """Extract variable names from a formula string."""
    tokens = re.split(r'[\s\+\~\|\(\)\*\/\-\:]+', formula)
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

    def lmer_tests(self, models=None,  group_col=None, control_group=None, numeric=[], specs=None, print_info=True):
        """
        Perform linear mixed-effects model (lmer) or linear model (lm) fitting and post-hoc tests using R's lme4 and
        emmeans packages.

        Parameters
        ----------
        models: str or list
            A model formula, or list of formulas, to be used for analysis. If more than one formula is provided,
            the best model is selected based on BIC (Bayesian Information Criterion).
            Example formulas:
                - Y ~ {group_col}
                - Y ~ {group_col} + (1 | ID)
        group_col: str
            The name of the column containing the group variable. It has to be specified if no specs are given (see
            `specs` argument).
        control_group: str
            The name of the control group to be used for comparisons. If None, all groups will be compared against each
            other (see `specs` argument).
        numeric: list
            Variables that are to be considered as numeric. The others will be converted to factors. If numeric=[],
            all variables in the model are considered as factors.
        specs: str or list
            A term, or list of terms, to be used for post-hoc tests.

            - Scenario 1: If all variables contained in a term are factors (i.e., not given to `numeric` argument), the
                          term is used as specifications for pairwise comparisons via the emmeans function in R.
            Example:
            specs='Group'  # pairwise differences of all group against control group (or between all groups if
                            `control_group` is None)
            specs='Sensor|Group'  # pairwise differences between sensors within each group

            - Scenario 2: If the term contains a numeric variable (i.e., given to numeric argument), then its effect on
                          the dependent variable is tested as a slope via emtrends function in R, with all remaining
                          terms as specifications.
            Example:
            specs='Epoch'  # slope of data_col w.r.t. Epoch is tested (specs=~1, var='Epoch')
            specs='Epoch:Group'  # slope of data_col w.r.t. Epoch is tested separately for each group (specs=~Group,
                                   var='Epoch')

            - Scenario 3: If `specs` is None, `group_col` has to be defined and will be used as default specs.

        print_info: bool
            Whether to print info (True) or not (False).

        Returns
        -------
        results: dict
            A dictionary containing the results of the analyses. The keys are the specs and the values are DataFrames
            containing the results of the analysis.

        """
        # Check if rpy2 is installed
        if not tools.ensure_module("rpy2"):
            raise ImportError("rpy2 is required for lmer_tests but is not installed.")
        pandas2ri = tools.dynamic_import("rpy2.robjects.pandas2ri")
        r = tools.dynamic_import("rpy2.robjects","r")
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

        ##################################
        #### Check and prepare models ####
        ##################################

        # Check  if models is not None
        if models is None:
            raise ValueError('No models provided. Please provide a model or a list of models.')

        # If only one model is provided, make it into a list of length 1
        if isinstance(models, str):
            models = [models]

        # Remove blanks from formulas
        models = [formula.replace(' ', '') for formula in models]

        # Check if all models have the same dependent variable
        Y = models[0].split('~')[0]
        if any([not formula.startswith(f'{Y}~') for formula in models]):
            raise ValueError(f'All models must have the same dependent variable.')

        # Automatically treat Y as numeric
        if Y not in numeric:
            numeric = numeric + [Y]

        # Check if every variable has a corresponding column in the DataFrame
        all_vars = set()
        for formula in models:
            all_vars = all_vars.union(extract_variables(formula))
        for var in all_vars:
            if var not in self.data.columns:
                raise ValueError(f'Variable "{var}" is not in the DataFrame.')

        ################################
        #### Check and prepare data ####
        ################################

        # Check if group_col is in the DataFrame and add it to all_vars if not already present
        if group_col:
            if group_col not in self.data.columns:
                raise ValueError(f'Column "{group_col}" (group_col) is not in the DataFrame.')
            all_vars.add(group_col)
        elif specs is None:
            raise ValueError('If no specs are given, group_col must be specified.')

        # Check if control_group is in the list of unique values of group_col
        if control_group is not None and control_group not in self.data[group_col].unique():
            raise ValueError(f'Control group "{control_group}" is not in the group_col "{group_col}" values.')

        # Copy the dataframe
        df = self.data.copy()

        # Remove all columns except the variables to analyse
        df = df[list(all_vars)]

        # [***] Remove rows where the data_col is zero (is it necessary?)
        # df = df[df[data_col] != 0]

        #################################
        #### Check and prepare specs ####
        #################################

        # Default specs if none are provided
        if specs is None:
            if group_col is None:
                raise ValueError('If no specs are given, group_col must be specified.')
            specs = [group_col]

        # If only one spec is provided, make it into a list of length 1
        if isinstance(specs, str):
            specs = [specs]
        # Remove blanks from specs
        specs = [x.replace(' ', '') for x in specs]

        # Check if all specs are valid and contain variables present in the DataFrame
        posthoc = []
        for ph in specs:
            vars = extract_variables(ph)
            if any([v not in df.columns for v in vars]):
                raise ValueError(f'specs "{ph}" contains variable not present in data columns.')
            if all([v not in numeric for v in vars]):
                posthoc.append('~' + ph)
            else:
                var_tmp = []
                for v in vars:
                    if len(var_tmp)>1:
                        raise ValueError(f'specs "{ph}": max 1 numeric variable per test.')
                    if v in numeric:
                        var_tmp.append(v)
                sp = '1' if len(vars)==1 else ':'.join([v for v in vars if v!=var_tmp[0]])
                posthoc.append(('~' + sp, var_tmp[0]))

        ################
        #### R code ####
        ################

        # Pass dataframe to R
        ro.globalenv['df'] = pandas2ri.py2rpy(df)
        if control_group is not None:
            ro.globalenv['control_group'] = control_group

        # Convert to factors
        factors = all_vars - set(numeric)
        r_code = []
        for col in factors:
            r_code.append(f'df${col} = as.factor(df${col})')

        # Join all lines into one R script string
        full_r_script = '\n'.join(r_code)
        r(full_r_script)

        # Pass models to R
        for ii, formula in enumerate(models):
            r(f"m{ii} <- {'lmer' if '|' in formula else 'lm'}({formula}, data=df)")

        ro.globalenv['fitted_models'] = [f'm{ii}' for ii in range(len(models))]

        # BIC test: handle single and multiple model cases properly
        if len(models) == 1:
            r(f"final_model <- m0")
        else:
            r('''
            fitted_models <- unlist(fitted_models)
            bics <- sapply(fitted_models, function(m) BIC(get(m)))
            index <- which.min(bics)
            final_model <- get(fitted_models[index])
            ''')
            if print_info:
                print(f'--- BIC model selection')
        if print_info:
            print(f"Model: {r('formula(final_model)')}")

        # Post-hoc analyses
        if print_info:
            print('--- Post-hoc tests:')
        results = {}

        for sp, ph in zip(specs, posthoc):
            if isinstance(ph, str):
                ro.globalenv['specs'] = ph
                if ph == '~'+group_col and control_group is not None:
                    # Compare all groups against control group
                    r('''
                    emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                    res <- contrast(emm, method='trt.vs.ctrl', ref=control_group, adjust = "holm")
                    df_res <- as.data.frame(res)
                    # Extract readable contrast names (e.g., "ADMIL - HC") 
                    df_res$contrast <- as.character(res@grid$contrast)
                    ''')
                else:
                    # All pairwise comparisons
                    r('''
                    emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                    res <- pairs(emm, adjust='holm')
                    df_res <- as.data.frame(res)
                    # Extract readable contrast names (e.g., "ADMIL - HC") 
                    df_res$contrast <- as.character(res@grid$contrast)
                    ''')
            else:
                ro.globalenv['specs'] = ph[0]
                ro.globalenv['var'] = ph[1]
                # Test slope(s)
                r('''
                emt <- suppressMessages(emtrends(final_model, specs=as.formula(specs), var=var))
                res <- test(emt, adjust='holm')
                df_res <- as.data.frame(res)
                ''')

            # Ensure Sensor remains as a character column (is this necessary?)
            if 'Sensor' in r('names(df_res)'):
                r('''
                df_res$Sensor <- as.character(df_res$Sensor)
                ''')

            df_res_r = ro.r['df_res']
            with pandas2ri.converter.context():
                df_res_pd = pandas2ri.rpy2py(df_res_r)

            if print_info:
                print('\n' + sp)
                print(df_res_pd)

            results[sp] = df_res_pd

        return results

    def lmer_selection(self, full_model=None,  group_col=None, numeric=[], crit=None, random_crit='BIC',
                       fixed_crit='LRT', include=None, print_info=True):
        """
        Perform linear mixed-effects model (lmer) or linear model (lm) model selection using R's lme4 and buildmer
        packages.

        Parameters
        ----------
        full_model: str
            The full model formula to be used as a starting point for backward selection.
            Example full model formulas:
            - Y ~ {group_col}
            - Y ~ {group_col} + (1 | ID)
        group_col: str
            The name of the column containing the group variable (only used if `full_model` is not specified).
        numeric: list
            Variables that are to be considered as numeric. The others will be converted to factors. If numeric=[],
            all variables in the model are considered as factors.
        crit: str
            The method to be used for model selection via the `buildmer` function in R.
            Possible options are:
            - 'LRT' (likelihood-ratio test based on chi-square mixtures per Stram & Lee 1994 for random effects),
            - 'LL' (use the raw -2 log likelihood)
            - 'AIC' (Akaike Information Criterion)
            - 'BIC' (Bayesian Information Criterion)
            - 'deviance' (explained deviance – note that this is not a formal test)
            - None (no selection)
            If the `crit` argument is not None, `random_crit` and `fixed_crit` are ignored.
        random_crit: str
            The method to be used for random effect selection via the `buildmer` function in R.
            Possible options are the same listed for the `crit` argument.
            Models are compared by keeping the fixed-effect structure the same and only changing random effects.
            `random_crit` is ignored if `crit` is not None.
        fixed_crit: str
            The method to be used for fixed effect selection via the `buildmer` function in R.
            Possible options are the same listed for the `crit` argument. 
            Models are compared by keeping the random-effect structure the same and only changing fixed effects.
            If both `fixed_crit` and `random_crit` are provided, random effect selection is performed first; 
            fixed effect selection is performed starting from the model with the optimal random effect structure.
            `fixed_crit` is ignored if `crit` is not None.
        include: str or list
            A term, or list of terms, to always include in the model - won't be tested during model selection.
        print_info: bool
            Whether to print info (True) or not (False).

        Returns
        -------
        results: str
            Formula for the selected model.

        """
        
        # Check if rpy2 is installed
        if not tools.ensure_module("rpy2"):
            raise ImportError("rpy2 is required for lmer but is not installed.")
        pandas2ri = tools.dynamic_import("rpy2.robjects.pandas2ri")
        r = tools.dynamic_import("rpy2.robjects", "r")
        ListVector = tools.dynamic_import("rpy2.robjects", "ListVector")
        ro = tools.dynamic_import("rpy2", "robjects")

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
        load_packages(c("lme4", "buildmer"))
        ''')

        # Check if the data is a pandas DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError('The data must be a pandas DataFrame.')

        ######################################
        #### Check and prepare full_model ####
        ######################################

        # Check  if full_model is not None
        if full_model is None:
            raise ValueError('No full_model provided. Please provide a full model.')

        # Remove blanks from formula
        full_model = full_model.replace(' ', '')

        # Automatically treat Y as numeric
        Y = full_model.split('~')[0]
        if Y not in numeric:
            numeric = numeric + [Y]

        # Check if every variable has a corresponding column in the DataFrame
        all_vars = extract_variables(full_model)
        for var in all_vars:
            if var not in self.data.columns:
                raise ValueError(f'Variable "{var}" is not in the DataFrame.')

        ################################
        #### Check and prepare data ####
        ################################

        # Check if group_col is in the DataFrame and add it to all_vars if not already present
        if group_col:
            if group_col not in self.data.columns:
                raise ValueError(f'Column "{group_col}" (group_col) is not in the DataFrame.')
            all_vars.add(group_col)
        elif full_model is None:
            raise ValueError('If no full_model is given, group_col must be specified.')

        # Copy the dataframe
        df = self.data.copy()

        # Remove all columns except the variables to analyse
        df = df[list(all_vars)]

        # [***] Remove rows where the data_col is zero (is it necessary?)
        # df = df[df[data_col] != 0]

        ############################################
        #### Check and prepare crit and include ####
        ############################################

        if crit is not None:
            random_crit, fixed_crit = None, None

        if include is None:
            include = []

        if isinstance(include, str):
            include = [include]

        ################
        #### R code ####
        ################

        # Pass dataframe to R
        ro.globalenv['df'] = pandas2ri.py2rpy(df)

        # Convert to factors
        factors = all_vars - set(numeric)
        r_code = []
        for col in factors:
            r_code.append(f'df${col} = as.factor(df${col})')

        # Join all lines into one R script string
        full_r_script = '\n'.join(r_code)

        # Pass to R
        r(full_r_script)

        # Pass full model to R
        ro.globalenv['full_model'] = full_model

        # Terms to always include in the model
        ro.globalenv['include'] = include

        # Extract fixed and random effects
        r('''
        ff <- as.formula(full_model)
        # Extract fixed and random effects
        fixed_terms <- unique(c(attr(terms(nobars(ff)), "term.labels"), unlist(include)))
        fixed <- as.formula(paste("~", paste(fixed_terms, collapse = " + ")))
        random_terms <- unique(c(sapply(findbars(ff), function(term) paste0("(", deparse(term), ")")), unlist(include)))
        random <- as.formula(paste("~", paste(random_terms, collapse = " + ")))
        ''')

        # Model selection
        if random_crit is not None:
            ro.globalenv['random_crit'] = random_crit
            r('''
            selmod <- buildmer(ff, data=df, buildmerControl=list(direction="backward", crit=random_crit, include=fixed, quiet=T))
            ff <- formula(selmod)
            random_terms <- sapply(findbars(ff), function(term) paste0("(", deparse(term), ")"))
            random_formula <- as.formula(paste("~", paste(random_terms, collapse = " + ")))
            ''')
            if print_info:
                print(f"Random effect structure selected with {random_crit}: {r('random_formula')}")
        if fixed_crit is not None:
            ro.globalenv['fixed_crit'] = fixed_crit
            r('''
            selmod <- buildmer(ff, data=df, buildmerControl=list(direction="backward", crit=fixed_crit, include=random, 
            quiet=T))
            ff <- formula(selmod)
            fixed_formula <- reformulate(attr(terms(nobars(ff)), "term.labels"))
            ''')
            if print_info:
                print(f"Fixed effect structure selected with {fixed_crit}: {r('fixed_formula')}")
        if crit is not None:
            ro.globalenv['crit'] = crit
            r('''
            selmod <- buildmer(ff, data=df, buildmerControl=list(direction="backward", crit=crit, 
            include=paste0('~', paste(include, collapse='+')), quiet=T))
            ff <- formula(selmod)
            ''')
            if print_info:
                print(f'Selection method: {crit}')
        if print_info:
            print(f"Selected model: {r('ff')}")
            if len(include) > 0:
                print(f"(selected model forced to include: {', '.join(include)})")

        opt_f = str(r('ff')).replace('\n', '')

        return opt_f

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