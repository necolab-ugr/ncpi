
import numpy as np
import pandas as pd

from ncpi import Analysis

####################################################################
# Generate toy data

nsubj, nsensors, nepochs = 10, 3, 5
id = np.repeat(np.arange(nsubj), 3*5)
sensor = np.tile(np.repeat(np.arange(3), 5), 10)
epoch = np.tile(np.arange(5), 3*10)
data = pd.DataFrame([id, epoch, sensor], index=['id', 'epoch', 'sensor']).T
data['gr'] = 'HC'
data.loc[data['id'].isin((4, 5, 6)), 'gr'] = 'g1'
data.loc[data['id'].isin((7, 8, 9)), 'gr'] = 'g2'
data['Y'] = np.random.normal(size=10*3*5)
# Make Y vary with epoch for HC and g1 but not for g2
data.loc[data['gr']=='g1','Y'] = data.loc[data['gr']=='g1','Y'] + np.random.normal(
                    loc=1, scale=.5, size = len(data.loc[data['gr']=='g1','Y']))
data.loc[data['gr']=='g2','Y'] = data.loc[data['gr']=='g2','Y'] + np.random.normal(
                    loc=3, scale=.5, size = len(data.loc[data['gr']=='g2','Y']))
data.loc[data['gr']!='g2', 'Y'] = data.loc[data['gr']!='g2', 'Y'] + np.random.normal(
                                loc=data.loc[data['gr']!='g2', 'epoch'], scale=.5)
# Include subject-level deviations
raneff = pd.DataFrame(np.random.normal(loc=0, scale=3, size=nsubj), index=range(nsubj), columns=['raneff'])
data = data.join(raneff, on='id')
data['Y'] = data['Y'] + np.random.normal(loc=data['raneff'], scale=.1)
data.drop(columns='raneff', inplace=True)

print(data.head())

####################################################################
# Run analyses

# Initialise class
analysis = Analysis(data)

# 1.
# Reduce random effect structure with BIC
opt_f = analysis.lmer_selection(full_model='Y ~ gr*epoch + sensor + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit=None)
# Run post-hoc analyses using selected model
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['gr', 'gr:epoch'])


# 2.
# Reduce random effect structure with BIC and fixed effect structure with LRT (anova)
opt_f = analysis.lmer_selection(full_model='Y~gr*epoch + sensor + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit='LRT')
# Run post-hoc analyses using selected model
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['gr', 'gr:epoch'])


# 3.
# Reduce random effect structure with BIC and fixed effect structure with LRT (anova),
# but always keeping the sensor term in the model
opt_f = analysis.lmer_selection(full_model='Y~gr*epoch + sensor + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit='LRT', include=['sensor'])
# Run post-hoc analyses using selected model
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['gr', 'gr:epoch'])


# 4.
# Use BIC to compare only a specified set of possible models
# (i.e., without exploring all their subsets via backward selection),
# then run post-hoc analyses
results = analysis.lmer_tests(models=['Y~ gr*epoch + (1|sensor) + (1|id)',
                                      'Y~ gr*epoch + sensor + (1|id)'],  # Best model is selected using BIC
                group_col='gr', control_group='HC', numeric=['epoch'],
                specs=['gr', 'gr:epoch'])