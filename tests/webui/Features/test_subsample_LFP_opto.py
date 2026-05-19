
from pathlib import Path
import pytest
import pandas as pd
import math

### ------------- FEATURE COMPUTATION TEST WITH 10% SUBSAMPLING of OPTO LFP DATA --------------------

# Inherit common code for all feature tests
from common_features_test_utils import *

class Test_features_subsample_LFP_opto(BaseFeatureTest):
    # Keep Path objects for test_tools (it expects Path-like API)
    DATA_DIR = Path('/DATOS/pablomc/empirical_datasets/development_EI_decorrelation/opto/LFP')

    def load_subsample_LFP_opto(self, data_dir):
        """Load empirical LFP data from the data folder and parse it into canonical schema."""
        files = sorted(data_dir.iterdir())
        # Calculate % of number of files to SUBSAMPLE
        original_count = len(files)
        keep_count = max(1, math.ceil(original_count * sampling_percentage))
        # Select SUBSAMPLING % of data
        files_subsampling = files[:keep_count]

        mat_dict = self._load_mat(self.DATA_DIR)

        # Convert mat_dict['OptoRampsLFP']['LFP'] to a flat list of arrays
        # if isinstance(mat_dict['OptoRampsLFP']['LFP'], np.ndarray):
        #     mat_dict['OptoRampsLFP']['LFP'] = mat_dict['OptoRampsLFP']['LFP'].flatten().tolist()   # list of arrays

        rows = []
   
        for subject_id, file_path in enumerate(files_subsampling):
            print(f"\r Progress: {subject_id + 1} of {len(files)} files loaded", end="", flush=True)
            config = ParseConfig(
                fields=CanonicalFields(
                    data='OptoRampsLFP.LFP',
                    fs='OptoRampsLFP.fs',  
                    ch_names=[f"ch{i}" for i in range(mat_dict['OptoRampsLFP']['LFP'][0].shape[0])], 
                    array_axes={
                        'channels': 0,
                        'samples': 2,
                        'epochs': 1,
                    },
                    metadata={
                        "subject_id": subject_id,
                        "group": 'OptoRampsLFP.age', # age → group
                        "species": "mouse",
                        "recording_type": "LFP",
                    },
                ),
                # 🔑 epoching (3 seconds, non-overlapping)
                epoch_length_s=3.0,
                epoch_step_s=3.0,
                zscore=False,
            )

            parser = EphysDatasetParser(config)
            df = parser.parse(file_path)
            rows.append(df)

        print(f"\nFiles loaded: {len(rows)}")
        out = pd.concat(rows, ignore_index=True)
        return out

    # Assign loader of this test to common class loader
    loader = load_subsample_LFP_opto

    def test_features_subsample_LFP_opto(self):
        assert self.compute_mean("catch22") == pytest.approx(15.00100261095558, abs=tolerance)
        assert self.compute_mean("power_spectrum_parameterization") == pytest.approx(
            1.571771863912633, abs=tolerance
    )