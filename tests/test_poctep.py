# First try using create_dataframe() with POCTEP database

import os
import sys

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ccpi

ccpi_feat = ccpi.Features()

root_path = '/home/juanmiguel/Databases/POCTEP/SENSORS'
data_path = os.path.join(root_path, 'processed')
recording_type = 'EEG'
data_format = 'mat'
epoch_l = 5

output_path = os.path.join(root_path, 'df_poctep.pkl')
data = ccpi_feat.create_dataframe(data_path, recording_type, data_format, epoch_l)

# Save the data
# data.to_pickle(output_path)