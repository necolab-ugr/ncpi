from pathlib import Path
import pytest
import pandas as pd
import math

### ------------- FEATURE COMPUTATION TEST WITH 10% SUBSAMPLING of POCTEP/CLEAN/SOURCES/dSPM/DK DATA --------------------
# Inherit common code for all feature tests
from common_features_test_utils import *

class Test_features_subsample_POCTEP_SOURCES(BaseFeatureTest):
    # Keep Path objects for test_tools (it expects Path-like API)
    DATA_DIR = Path('/DATOS/pablomc/empirical_datasets/POCTEP_data/CLEAN/SOURCES/dSPM/DK')

    def load_subsample_POCTEP_SOURCES(self, data_dir):
        """Load empirical data from the data folder and parse it into canonical schema."""
        files = sorted(data_dir.iterdir())

        # Calculate % of number of files to SUBSAMPLE
        original_count = len(files)
        keep_count = max(1, math.ceil(original_count * sampling_percentage))
        # Select SUBSAMPLING % of data
        files_subsampling = files[:keep_count]

        rows = []
        for subject_id, file_path in enumerate(files_subsampling):
            # Extract group metadata information from filenames
            fname = file_path.name
            group = fname.split("_")[0]

            print(f"\r Progress: {subject_id + 1} of {len(files)} files loaded", end="", flush=True)
            config = ParseConfig(
                fields=CanonicalFields(
                    data= 'data.signal',
                    fs='data.cfg.fs',
                    ch_names='data.cfg.channels',
                    array_axes={
                        'channels': 1,
                        'samples': 0,
                    },
                    metadata={
                        "subject_id": subject_id,
                        "group": group, 
                        "species": "human",
                        "recording_type": 'EEG',
                        "condition": "resting-state"
                    },
                ),
                zscore=True,
                # epoching (5 seconds, non-overlapping)
                epoch_length_s=5.0,
                epoch_step_s=5.0,
            )
            parser = EphysDatasetParser(config)
            df = parser.parse(file_path)
            rows.append(df)

        print(f"\nFiles loaded: {len(rows)}")
        out = pd.concat(rows, ignore_index=True)
        return out

    # Assign loader of this test to common class loader
    loader = load_subsample_POCTEP_SOURCES

    def test_features_subsample_POCTEP_SOURCES(self):
        assert self.compute_mean("catch22") == pytest.approx(12.882350179449922, abs=tolerance)
        assert self.compute_mean("power_spectrum_parameterization") == pytest.approx(
            2.1968496182453037, abs=tolerance
        )