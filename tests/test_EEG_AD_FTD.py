# First try using create_dataframe() with POCTEP database

import os
import sys
import pandas as pd

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ccpi

root_path = '/home/juanmiguel/ccpi/POCTEP/'
data_path = os.path.join(root_path, 'processed')
df_path = os.path.join(root_path, 'df_poctep.pkl')
feat_path = os.path.join(root_path, 'feat_poctep.pkl')

recording_type = 'EEG'
data_format = 'mat'
epoch_l = 5

ccpi_feat = ccpi.Features()

# ##### Load Data and Create DataFrame #####

# data = ccpi_feat.load_data(data_path, recording_type, data_format, epoch_l)
# Save the data
# data.to_pickle(df_path)

################################

##### Calculate Features #####

# Read df_path
df_data = pd.read_pickle(df_path)
feat_data = ccpi_feat.compute_features(df_data)
# feat_data.to_pickle(feat_path)

################################

##### LMER Features Analysis #####

# # Calculate the mean of the 'Data' column and overwrite the 'Data' column with the mean
# data['Data'] = data['Data'].apply(lambda x: x.mean())

# ccpi_analysis = ccpi.Analysis(data)
# ccpi_analysis.lmer()

################################
