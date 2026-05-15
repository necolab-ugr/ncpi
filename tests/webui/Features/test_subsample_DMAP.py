from pathlib import Path
import pytest
import pandas as pd
import math

### ------------- FEATURE COMPUTATION TEST WITH 10% SUBSAMPLING of DMAP DATA --------------------
# Inherit common code for all feature tests
from common_features_test_utils import *

class Test_features_subsample_DMAP(BaseFeatureTest):
    # Keep Path objects for test_tools (it expects Path-like API)
    DATA_DIR = Path('/DATOS/pablomc/empirical_datasets/DMAP_data')

    def load_subsample_DMAP(self, data_dir):
        """Load empirical data from the data folder and parse it into canonical schema."""
        files = sorted(data_dir.iterdir())

        # Calculate % of number of files to SUBSAMPLE
        original_count = len(files)
        keep_count = max(1, math.ceil(original_count * sampling_percentage))
        # Select SUBSAMPLING % of data
        files_subsampling = files[:keep_count]

        mat_dict = self._load_mat(f'{self.DATA_DIR}/DMAP_pcai_export_2024.mat')

        rows = []
        for subject_id, file_path in enumerate(files_subsampling):
            print(f"\r Progress: {subject_id + 1} of {len(files)} files loaded", end="", flush=True)
            config = ParseConfig(
                fields=CanonicalFields(
                    data= 'pcai_matrix',
                    fs= None, # In the UI it is a 'None' value. But 'None' only works with 'catch22', not 'power_spectrum_parameterization'
                    ch_names=[f"ch{i}" for i in range(mat_dict['pcai_matrix'].shape[1])],
                    array_axes={
                        'channels': 1,
                        'samples': 0,
                        'ids': 2,
                        'epochs': 3,
                    },
                    metadata={
                        "subject_id": 'sub_IDs',
                        "recording_type": 'EEG',
                        "group": 'label',
                    },
                ),
                epoch_length_s=5.0,
                epoch_step_s=5.0,
                zscore=False,
            )
            parser = EphysDatasetParser(config)
            df = parser.parse(file_path)
            rows.append(df)

        print(f"\nFiles loaded: {len(rows)}")
        out = pd.concat(rows, ignore_index=True)
        return out

    # Assign loader of this test to common class loader
    loader = load_subsample_DMAP

    def test_features_subsample_DMAP(self):
        assert self.compute_mean("catch22") == pytest.approx(4.405096379251722, abs=tolerance)
        # Specparam method requires `df['fs']` (sampling frequency) to be convertible to float. 'None' is not a valid option for this method
        # assert self.compute_mean("power_spectrum_parameterization") == pytest.approx(
        #     4.405096379251722, abs=tolerance
        # )