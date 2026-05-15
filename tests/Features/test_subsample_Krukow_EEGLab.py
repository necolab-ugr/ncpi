from pathlib import Path
import pytest
import pandas as pd
import math

### ------------- FEATURE COMPUTATION TEST WITH 10% SUBSAMPLING of Krukow/EEGLab DATA --------------------
# Inherit common code for all feature tests
from common_features_test_utils import *

class Test_features_subsample_Krukow_EEGLab(BaseFeatureTest):
    # Keep Path objects for test_tools (it expects Path-like API)
    DATA_DIR = Path('/DATOS/pablomc/empirical_datasets/Krukow_data/Task-Related signals/EEGLab_Band_1-70/Band_1-70')

    def load_subsample_Krukow_EEGLab(self, data_dir):
        """Load empirical data from the data folder and parse it into canonical schema."""
        files = sorted(data_dir.iterdir())

        # Calculate % of number of files to SUBSAMPLE
        original_count = len(files)
        keep_count = max(1, math.ceil(original_count * sampling_percentage))
        # Select SUBSAMPLING % of data
        files_subsampling = files[:keep_count]

        rows = []
        for subject_id, file_path in enumerate(files_subsampling):
            print(f"\r Progress: {subject_id + 1} of {len(files)} files loaded", end="", flush=True)
            config = ParseConfig(
                fields=CanonicalFields(
                    data= 'get_data',
                    fs='info.sfreq',
                    ch_names='ch_names',
                    array_axes={
                        'samples': 0,
                    },
                    metadata={
                        "subject_id": subject_id,
                        "species": "human",
                        "recording_type": 'EEG',
                    },
                ),
                zscore=False,
            )
            parser = EphysDatasetParser(config)
            df = parser.parse(file_path)
            rows.append(df)

        print(f"\nFiles loaded: {len(rows)}")
        out = pd.concat(rows, ignore_index=True)
        return out

    # Assign loader of this test to common class loader
    loader = load_subsample_Krukow_EEGLab

    def test_features_subsample_Krukow_EEGLab(self):
        assert self.compute_mean("catch22") == pytest.approx(8.923003681346412, abs=tolerance)
        assert self.compute_mean("power_spectrum_parameterization") == pytest.approx(
            1.5674661226357849, abs=tolerance
        )