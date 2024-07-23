# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
# from rpy2.robjects.conversion import localconverter

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


    def EEG_topographic_plot(self):

        pass
