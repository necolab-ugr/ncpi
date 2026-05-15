from pathlib import Path
import pytest
import os
import numpy as np
import importlib.util
import pandas as pd
import math
import scipy.io
import h5py  

# ----------------------------
# Paths
# ----------------------------
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")

# ----------------------------
# Import shared tools explicitly (examples/tools.py) as shared_tools
# ----------------------------
_spec_shared = importlib.util.spec_from_file_location("shared_tools", os.path.join(EXAMPLES_DIR, "tools.py"))
shared_tools = importlib.util.module_from_spec(_spec_shared)
_spec_shared.loader.exec_module(shared_tools)

# (!!) To use the latest version of the parser: pip install .
from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields

# Sampling percentage
sampling_percentage = float("10")/100

# Tolerance error range
tolerance = 1e-1

# ----------------------------
# Abstract class (base class for Feature tests)
# ----------------------------
class BaseFeatureTest:
    DATA_DIR: str = None
    loader = None  # ← each test replaces this loader with their own load_data function

    @classmethod
    def setup_class(cls):
        if cls.DATA_DIR is None:
            raise ValueError(f"Please, define DATA_DIR in {cls.__name__}")
        if not os.path.isdir(cls.DATA_DIR):
            raise FileNotFoundError(f"DATA_DIR doesn't exist: {cls.DATA_DIR}")
        if cls.loader is None:
            raise ValueError(f"Please, define 'loader' in {cls.__name__}")

    def _load_mat(self, directory: str):
        """Find the first .mat file in the path and return the dict of loadmat."""
        # Fast file version detection
        if not h5py.is_hdf5(directory):
            try:
                mat_files = sorted([f for f in os.listdir(directory) if f.endswith('.mat')])
                if not mat_files:
                    raise FileNotFoundError(f"No .mat file found in {directory}")
                filepath = os.path.join(directory, mat_files[0])
                return scipy.io.loadmat(filepath) 
            except Exception as exc:
                raise ValueError(f"Failed to load legacy MATLAB file '{directory}' with scipy: {exc}")
        else:
            try:
                with h5py.File(directory, "r") as h5f:
                    data = {}
                    for key in h5f.keys():
                        obj = h5f[key]
                        if isinstance(obj, h5py.Dataset):
                            # Load the whole dataset
                            data[key] = obj[()]
                    return data
            except Exception as exc:
                raise ValueError(f"Failed to load v7.3 MATLAB file '{directory}' with h5py: {exc}")

    # Common code for all feature tests
    def compute_mean(self, method: str) -> float:
        print(f"\n\n--- Method: {method}")
        # 1) Load subsample of data with EphysDatasetParser
        df_emp = self.loader(self.DATA_DIR)   # It uses the loader function of each test

        # 2) Feature extraction
        emp_data = shared_tools.compute_features(method, df_emp)

        # 3) Calculate features' average
        features = emp_data["Features"].tolist()
        avg = np.nanmean([np.nanmean(np.asarray(x)) for x in features])
        return float(avg)