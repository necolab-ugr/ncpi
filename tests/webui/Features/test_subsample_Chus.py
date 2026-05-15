from pathlib import Path
import pytest
import pandas as pd
import math

### ------------- FEATURE COMPUTATION TEST WITH 10% SUBSAMPLING of Chus_Mayores DATA --------------------
# Inherit common code for all feature tests
from common_features_test_utils import *

class Test_features_subsample_Chus_Mayores(BaseFeatureTest):
    # Keep Path objects for test_tools (it expects Path-like API)
    DATA_DIR = Path('/DATOS/pablomc/empirical_datasets/Chus_Mayores_Memoria/data')

    def load_subsample_Chus_Mayores(self, data_dir):
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
                    data= 'data_clean.trial',
                    fs='data_clean.fsample',
                    ch_names='data_clean.label',
                    array_axes={
                        'samples': 2,
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
    loader = load_subsample_Chus_Mayores

    def test_features_subsample_Chus_Mayores(self):
        assert self.compute_mean("catch22") == pytest.approx(13.117046201076034, abs=tolerance)
        assert self.compute_mean("power_spectrum_parameterization") == pytest.approx(
            6.0514125700115, abs=tolerance
        )