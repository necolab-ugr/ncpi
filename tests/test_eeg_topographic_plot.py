import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ncpi.Analysis import Analysis

import pandas as pd
import numpy as np

z_ratio = np.random.uniform(-10, 10, 19)
p_value = np.random.uniform(0.03, 0.16, 19)
group_arr = np.random.choice(['HCvsADMIL', 'HCvsADMOD'], 19)
df = pd.DataFrame({'Group': group_arr, 'Ratio': z_ratio, 'p.value': p_value})

def test_plot_df(df):
    
    ccpi_analysis = Analysis(df)
    ccpi_analysis.EEG_topographic_plot(
        group='HCvsADMIL', 
        system=19,
        p_value=0.05,
        electrode_size=0.9
        )
    
    # assert os.path.exists('EEG_tomography_plot.png')
    # os.remove('EEG_tomography_plot.png')

test_plot_df(df)