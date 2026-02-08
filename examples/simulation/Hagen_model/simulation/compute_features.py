import os
import pickle
import shutil
import numpy as np
import ncpi
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Path to the folder containing the processed data
sim_file_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'data', 'Hagen_model_v1')

# Path to the folder where the features will be saved
features_path = os.path.join(os.sep, 'DATOS', 'pablomc', 'data')

# Set to True if features should be computed for the EEG data instead of the CDM data
compute_EEG = False
# NYHeadModel dipole alignment (FieldPotential.compute_MEEG)
align_to_surface = True
# Number of CDM files to process (use "all" for every file, or int for a count).
cdm_files_to_process = 1
# Number of CDM samples to process per file (use "all" for every sample, or int for a count).
cdm_samples_to_process = "all"

# Feature methods to compute
methods = [
    'catch22',
    'power_spectrum_parameterization_1',
]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _list_files(folder):
    return sorted([f for f in os.listdir(folder)])


def _load_cdm_samples(path):
    data = pickle.load(open(path, 'rb'))
    if isinstance(data, dict):
        cdm_sum = data.get('EE', 0) + data.get('EI', 0) + data.get('IE', 0) + data.get('II', 0)
        return [np.asarray(cdm_sum, dtype=float)]
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported CDM data shape in {path}: {arr.shape}")


def _cdm_to_3xn(cdm_1d_or_2d):
    arr = np.asarray(cdm_1d_or_2d, dtype=float)
    if arr.ndim == 1:
        zeros = np.zeros_like(arr)
        return np.stack([zeros, zeros, arr], axis=0)
    if arr.ndim == 2:
        if arr.shape[0] == 3:
            return arr
        if arr.shape[1] == 3:
            return arr.T
    raise ValueError("CDM must be a 1D array or 2D array with 3 components.")


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _resolve_sample_limit(value):
    if value in (None, "all"):
        return None
    if isinstance(value, int):
        if value < 1:
            raise ValueError("cdm_samples_to_process must be >= 1 or 'all'.")
        return value
    raise ValueError("cdm_samples_to_process must be an int >= 1 or 'all'.")


def _resolve_file_limit(value):
    if value in (None, "all"):
        return None
    if isinstance(value, int):
        if value < 1:
            raise ValueError("cdm_files_to_process must be >= 1 or 'all'.")
        return value
    raise ValueError("cdm_files_to_process must be an int >= 1 or 'all'.")


def _limit_samples(samples, limit):
    if limit is None:
        return samples
    return samples[:limit]


def _limit_files(files, limit):
    if limit is None:
        return files
    return files[:limit]


def _chunk_count(n_items, chunk_size):
    if n_items == 0:
        return 0
    return (n_items + chunk_size - 1) // chunk_size


def _progress_chunks(chunks, total, desc):
    if tqdm is None:
        return chunks
    return tqdm(chunks, total=total, desc=desc)


def _compute_eeg_for_locations(potential, cdm, locations, align_to_surface):
    eeg = []
    for loc in locations:
        sig = potential._compute_eegmeg_from_cdm(
            cdm,
            dipole_location=loc,
            model="NYHeadModel",
            align_to_surface=align_to_surface,
            return_all_electrodes=False,
        )
        eeg.append(sig)
    return np.stack(eeg, axis=0)


def _compute_features_for_series(series_list, method, fs):
    if method == "catch22":
        features = ncpi.Features(method="catch22", params={"normalize": True})
        return features.compute_features(series_list)

    if method in {"power_spectrum_parameterization_1", "power_spectrum_parameterization_2"}:
        fooof_setup_sim = {
            "peak_threshold": 1.0,
            "min_peak_height": 0.0,
            "max_n_peaks": 5,
            "peak_width_limits": (10.0, 50.0),
        }
        features = ncpi.Features(
            method="specparam",
            params={
                "fs": fs,
                "freq_range": (5.0, 200.0),
                "specparam_model": fooof_setup_sim,
                "r_squared_th": 0.9,
            },
        )
        feats = features.compute_features(series_list)
        if method == "power_spectrum_parameterization_1":
            return [float(np.asarray(d["aperiodic_params"])[1]) for d in feats]

        def _pack(d):
            ap = np.asarray(d.get("aperiodic_params", [np.nan, np.nan]), dtype=float)
            exponent = float(ap[1]) if ap.size > 1 else np.nan
            peak_cf = float(d.get("peak_cf", np.nan))
            peak_pw = float(d.get("peak_pw", np.nan))
            knee = np.nan
            mean_power = np.nan
            return np.array([exponent, peak_cf, peak_pw, knee, mean_power], dtype=float)

        return [_pack(d) for d in feats]

    raise ValueError(f"Unknown method: {method}")


if __name__ == '__main__':
    file_limit = _resolve_file_limit(cdm_files_to_process)
    sample_limit = _resolve_sample_limit(cdm_samples_to_process)
    cdm_files = [f for f in _list_files(sim_file_path) if f.startswith('CDM')]
    cdm_files = _limit_files(cdm_files, file_limit)
    theta_files = {f.split('_')[-1]: f for f in _list_files(sim_file_path) if f.startswith('theta')}

    potential = ncpi.FieldPotential() if compute_EEG else None
    if compute_EEG:
        eeg_locations, _ = potential._get_eeg_1020_locations()
        n_electrodes = eeg_locations.shape[0]

    for method in methods:
        print(f"\n--- Computing features: {method}")
        eeg_dir = None
        if compute_EEG:
            eeg_dir = os.path.join(features_path, method, 'EEG')
            if os.path.isfile(os.path.join(eeg_dir, 'sim_X_0')) and os.path.isfile(
                os.path.join(features_path, method, 'sim_theta')
            ):
                print(f"Features already exist for {method} (EEG). Skipping.")
                continue
        else:
            if os.path.isfile(os.path.join(features_path, method, 'sim_X')) and os.path.isfile(
                os.path.join(features_path, method, 'sim_theta')
            ):
                print(f"Features already exist for {method}. Skipping.")
                continue

        tmp_dir = os.path.join(features_path, method, 'tmp')
        _ensure_dir(tmp_dir)
        if compute_EEG:
            _ensure_dir(eeg_dir)

        for cdm_file in cdm_files:
            print(f"Processing {cdm_file}...")
            suffix = cdm_file.split('_')[-1]
            cdm_path = os.path.join(sim_file_path, cdm_file)
            samples = _limit_samples(_load_cdm_samples(cdm_path), sample_limit)

            if compute_EEG:
                chunk_size = 10
                feats_per_elec = [[] for _ in range(n_electrodes)]
                total_chunks = _chunk_count(len(samples), chunk_size)
                chunks = _progress_chunks(_chunked(samples, chunk_size), total_chunks, "EEG chunks")
                for iii,chunk in enumerate(chunks):
                    print(f"\nComputing features for chunk {iii+1}/{total_chunks}...\n")
                    eeg_chunk = []
                    for sample in chunk:
                        cdm_3 = _cdm_to_3xn(sample)
                        eeg = _compute_eeg_for_locations(
                            potential,
                            cdm_3,
                            eeg_locations,
                            align_to_surface,
                        )
                        eeg_chunk.append(eeg)

                    for elec in range(n_electrodes):
                        series_list = [eeg_sample[elec] for eeg_sample in eeg_chunk]
                        feats = _compute_features_for_series(series_list, method, fs=1000.0 / 0.625)
                        feats_per_elec[elec].extend(feats)

                for elec in range(n_electrodes):
                    out_path = os.path.join(tmp_dir, f"sim_X_{suffix}_{elec}")
                    pickle.dump(np.asarray(feats_per_elec[elec]), open(out_path, 'wb'))
            else:
                chunk_size = 5000
                feats_all = []
                total_chunks = _chunk_count(len(samples), chunk_size)
                chunks = _progress_chunks(_chunked(samples, chunk_size), total_chunks, "CDM chunks")
                for iii,chunk in enumerate(chunks):
                    print(f"\nComputing features for chunk {iii+1}/{total_chunks}...\n")
                    feats = _compute_features_for_series(chunk, method, fs=1000.0 / 0.625)
                    feats_all.extend(feats)
                out_path = os.path.join(tmp_dir, f"sim_X_{suffix}")
                pickle.dump(np.asarray(feats_all), open(out_path, 'wb'))

            theta_file = theta_files.get(suffix)
            if theta_file:
                theta = pickle.load(open(os.path.join(sim_file_path, theta_file), 'rb'))
                if sample_limit is not None:
                    if isinstance(theta, dict) and "data" in theta:
                        theta = dict(theta)
                        theta["data"] = theta["data"][:sample_limit]
                    elif isinstance(theta, (list, np.ndarray)):
                        theta = theta[:sample_limit]
                pickle.dump(theta, open(os.path.join(tmp_dir, f"sim_theta_{suffix}"), 'wb'))

        # Merge features
        tmp_files = _list_files(tmp_dir)
        theta_paths = [f for f in tmp_files if f.startswith('sim_theta_')]
        theta_data = []
        theta_params = None
        for fname in theta_paths:
            data = pickle.load(open(os.path.join(tmp_dir, fname), 'rb'))
            theta_data.append(data['data'])
            if theta_params is None:
                theta_params = data['parameters']

        if theta_data and theta_params is not None:
            th = {'data': np.concatenate(theta_data), 'parameters': theta_params}
            pickle.dump(th, open(os.path.join(features_path, method, 'sim_theta'), 'wb'))
            print(f"Features computed for {len(th['data'])} samples.")

        if compute_EEG:
            for elec in range(n_electrodes):
                series = []
                for fname in tmp_files:
                    if fname.startswith('sim_X_') and fname.endswith(f"_{elec}"):
                        series.append(pickle.load(open(os.path.join(tmp_dir, fname), 'rb')))
                if series:
                    out = np.concatenate(series)
                    pickle.dump(out, open(os.path.join(eeg_dir, f"sim_X_{elec}"), 'wb'))
        else:
            series = []
            for fname in tmp_files:
                if fname.startswith('sim_X_') and len(fname.split('_')) == 3:
                    series.append(pickle.load(open(os.path.join(tmp_dir, fname), 'rb')))
            if series:
                out = np.concatenate(series)
                pickle.dump(out, open(os.path.join(features_path, method, 'sim_X'), 'wb'))

        shutil.rmtree(tmp_dir)
