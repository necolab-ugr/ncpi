
import numpy as np
import pandas as pd

from ncpi import Analysis

####################################################################
# Generate toy data

np.random.seed(0)

nsubj, nsensors, nepochs = 10, 3, 5
id = np.repeat(np.arange(nsubj), nsensors*nepochs)
sensor = np.tile(np.repeat(np.arange(nsensors), nepochs), nsubj)
epoch = np.tile(np.arange(nepochs), nsensors*nsubj)
data = pd.DataFrame([id, epoch, sensor], index=['id', 'epoch', 'sensor']).T
group_size = nsubj//3
data['gr'] = 'HC'
data.loc[data['id'].isin(range(group_size)), 'gr'] = 'g1'
data.loc[data['id'].isin(range(group_size, group_size * 2)), 'gr'] = 'g2'
data['Y'] = np.random.normal(size=nsubj*nsensors*nepochs)
# Group*Sensor effects
halfsensor = nsensors//2
data.loc[(data['gr'] == 'g1') & (data['sensor'] <= halfsensor), 'Y'] = data.loc[
                    (data['gr'] == 'g1') & (data['sensor'] <= halfsensor), 'Y'] + np.random.normal(
                    loc=10, scale=2, size=len((data.loc[(data['gr'] == 'g1') & (data['sensor'] <= halfsensor), 'Y'])))
data.loc[data['gr'] == 'g2', 'Y'] = data.loc[data['gr'] == 'g2', 'Y'] + np.random.normal(
                    loc=7, scale=.5, size = len(data.loc[data['gr'] == 'g2', 'Y']))
# Make Y vary with epoch for HC and g2 but not for g1
data.loc[data['gr'] != 'g1', 'Y'] = data.loc[data['gr'] != 'g1', 'Y'] + np.random.normal(
                                loc=data.loc[data['gr'] != 'g1', 'epoch'], scale=.5)
# Include subject-level deviations
raneff = pd.DataFrame(np.random.normal(loc=0, scale=3, size=nsubj), index=range(nsubj), columns=['raneff'])
data = data.join(raneff, on='id')
data['Y'] = data['Y'] + np.random.normal(loc=data['raneff'], scale=.1)
data.drop(columns='raneff', inplace=True)
# Include covariates
data['sex'] = np.random.choice(['F', 'M'], size=len(data))

print('\nData:')
print(data.head())

####################################################################
# Run analyses

# Initialise class
analysis = Analysis(data)

# 1.
print(f'\n\n{"Example 1":=^30}')
# Reduce random effect structure with BIC
print(f'\n{"Model selection":-^30}\n')
opt_f = analysis.lmer_selection(full_model='Y ~ gr * epoch * sensor + sex + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit=None)
# Run post-hoc analyses using selected model
print(f'\n{"Testing (1)":-^30}\n')
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['epoch:gr', 'gr|sensor', 'x'])
# Note: if a variable 'x' which is not in the models is given to specs, it is simply skipped.
print(f'\n{"Testing (2)":-^30}\n')
results = analysis.lmer_tests(models=opt_f, numeric=['epoch'], specs=['gr|sensor'])
# Note: here we're not specifying the control group, so group comparisons include *all* combinations!

# 2.
print(f'\n\n{"Example 2":=^30}')
# Reduce random effect structure with BIC and fixed effect structure with LRT (anova)
print(f'\n{"Model selection":-^30}\n')
opt_f = analysis.lmer_selection(full_model='Y ~ gr * epoch * sensor + sex + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit='LRT')
# Run post-hoc analyses using selected model
print(f'\n{"Testing":-^30}\n')
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['gr|sensor', 'epoch:gr', 'sex'])
# Note: as sex is not present in the model anymore, the test is automatically skipped.

# 3.
print(f'\n\n{"Example 3":=^30}')
# Reduce random effect structure with BIC and fixed effect structure with LRT (anova),
# but always keeping the sensor term in the model
print(f'\n{"Model selection":-^30}\n')
opt_f = analysis.lmer_selection(full_model='Y ~ gr * epoch * sensor + sex + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit='LRT', include=['sensor'])
# Run post-hoc analyses using selected model
print(f'\n{"Testing":-^30}\n')
results = analysis.lmer_tests(models=opt_f, group_col='gr',
                              control_group='HC', numeric=['epoch'])
# Note: no specs given, so group_col is used.

# 4.
print(f'\n\n{"Example 4":=^30}')
# Use BIC to compare only a specified set of possible models
# (i.e., without exploring all their subsets via backward selection),
# then run post-hoc analyses
results = analysis.lmer_tests(models=['Y ~ gr * epoch * sensor + sex + (1|sensor) + (1|id)',
                                      'Y ~ gr * epoch * sensor + (1|sex) + (1|id)'],  # Best model is selected using BIC
                group_col='gr', control_group='HC', numeric=['epoch'],
                specs=['gr|sensor', 'epoch:gr'])

# 5.
print(f'\n\n{"Example 5":=^30}')
# Reduce random effect structure with BIC and fixed effect structure with LRT (anova)
print(f'\n{"Model selection":-^30}\n')
opt_f = analysis.lmer_selection(full_model='Y ~ gr * epoch * sensor + sex + (1|id)',
                                numeric=['epoch'],
                                random_crit='BIC', fixed_crit='LRT')
# Run post-hoc analyses using selected model
print(f'\n{"Testing":-^30}\n')
results = analysis.lmer_tests(models=opt_f, group_col='gr', control_group='HC',
                              numeric=['epoch'], specs=['sex'])
# Note: as sex is not present in the model, the test is automatically skipped.
# No specs left: empty output and warning message.
