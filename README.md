> **The current code is quite mature and we are on the road to launch ncpi version 1.0 by June 1, 2026.**

<div align="center">

# ncpi: neural circuit parameter inference
___

<img src="https://raw.githubusercontent.com/necolab-ugr/ncpi/main/img/ncpi_logo.png" alt="ncpi logo" width="150">

</div>

[Documentation](https://necolab-ugr.github.io/ncpi/)

`ncpi` is a Python package for model-based inference of neural circuit parameters from population-level
electrophysiological recordings, such as LFP, ECoG, MEG, and EEG. `ncpi` provides a rapid, reproducible, and robust
framework for estimating the most probable neural circuit parameters associated with an empirical observation,
streamlining traditionally complex workflows into a minimal amount of code.

# Key Features of `ncpi`
- **All-in-one solution**: a unified package for forward and inverse modeling of extracellular signals from neural
  circuit simulations.
- **Biophysically grounded analysis**: practical workflows to bridge electrophysiology and neural circuit parameters.
- **Flexible and extensible**: use individual modules independently or run complete end-to-end pipelines.

# Installation

`ncpi` requires **Python 3.10+**. We strongly recommend using a dedicated **Conda** environment.
If you need to install Anaconda first, download it from the official page:
https://www.anaconda.com/download

## 1) Unix (Linux/macOS)

### Base installation
```bash
# Create environment
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env

# Install ncpi
pip install ncpi
```

### NEST (for LIF simulation examples)
If you run examples that depend on NEST (for example in `examples/simulation`), install it in the same environment:

```bash
conda install -c conda-forge nest-simulator=3.8
```

If `nest-simulator` is not available for your platform/channel combination, follow the official NEST build/install
instructions: https://nest-simulator.readthedocs.io/

## 2) Windows

### Base installation (native Windows Python)
```powershell
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
```

### NEST on Windows
NEST-based examples are recommended via **WSL2 (Ubuntu)**, not native Windows.

```powershell
# One-time WSL2 setup
wsl --install -d Ubuntu
```

Then, inside the Ubuntu/WSL shell:
```bash
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
conda install -c conda-forge nest-simulator=3.8
```

## 3) Optional Dependencies

`ncpi` supports optional extras. Install only what your workflow needs.

### Extras shortcuts
```bash
pip install "ncpi[hctsa]"           # hctsa backend support
pip install "ncpi[parser]"          # extended parser backends
pip install "ncpi[fieldpotential]"  # kernel/CDM/LFP + M/EEG forward models
pip install "ncpi[analysis]"        # statistics + EEG/MEG analysis helpers
pip install "ncpi[webui]"           # WebUI runtime backends
pip install "ncpi[examples]"        # dependencies for example scripts
pip install "ncpi[tests]"           # test stack dependencies
pip install "ncpi[all]"             # all optional extras
```

### Optional extras map
- **`hctsa`**: `h5py`, `matlabengine`.
- **`parser`**: `h5py`, `mne`, `pyEDFlib`, `pyarrow` (with base `pandas` for tabular/parquet I/O).
- **`fieldpotential`**: `LFPy`, `lfpykernels`, `lfpykit`, `h5py`, `neuron`, `mne`, `nibabel`.
- **`analysis`**: `rpy2`, `mne`, `nibabel`, `pyvistaqt`, `PyQt5`.
- **`webui`**: `Pillow`, `h5py`, `mne`, `nibabel`, `pyEDFlib`, `pyarrow`, `openpyxl`, `xlrd`.
- **`examples`**: `h5py`, `LFPy`, `mpi4py`, `nest-simulator`, `neuron`, `pdf2image`, `seaborn`, `statsmodels`.
- **`tests`**: `LFPy`, `h5py`, `lfpykernels`, `lfpykit`, `matlabengine`, `mne`, `nibabel`, `neuron`, `playwright`, `Pillow`, `pyEDFlib`, `pytest`, `rpy2`.

## 4) WebUI: installation and usage

The WebUI is available from the repository source (`webui/app.py`).

### Install WebUI dependencies
After activating your Conda environment:

```bash
git clone https://github.com/necolab-ugr/ncpi.git
cd ncpi
pip install -e ".[webui]"
```

### Start WebUI
From the repository root:

```bash
python webui/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Windows note
You can run the same command from Anaconda Prompt or PowerShell.
If your workflow needs NEST, run the WebUI from your WSL environment where NEST is installed.

### Optional backends note
Optional backends are listed in **Section 3 (Optional Dependencies)**. Install only the extras required by your workflow.

### hctsa note
For hctsa-based features, install hctsa first: https://github.com/benfulcher/hctsa

Then install MATLAB Engine for Python (for example from `<MATLAB_ROOT>/extern/engines/python`, or a compatible
`matlabengine` pip package), and pass the hctsa repository path as `hctsa_folder`.

# Folder Structure

- `ncpi/`: core library modules (`Simulation`, `Features`, `FieldPotential`, `Inference`, `Analysis`, parser utilities).
- `examples/`: reproducible scripts and simulation/inference examples.
- `docs/`: active documentation pages (`index.html`, `installation.html`, `tutorials.html`, `api.html`, `faq.html`, `contributing.html`, `citation.html`, `credits.html`).
- `img/`: shared visual assets (including the project logo).
- `webui/`: ncpi web interface.
- `tests/`: unit/integration/webui test suites.

# Example Usage

The example below follows a Gao-style synthetic setup: two Poisson generators are convolved with synaptic kernels to
obtain AMPA and GABA currents, and their sum is used as a synthetic LFP
(Gao et al., 2017: https://doi.org/10.1016/j.neuroimage.2017.06.078).

```python
import inspect
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import ncpi


RNG = np.random.default_rng(42)
FS = 1000.0  # Hz
DT = 1.0 / FS # s
DURATION_S = 12.0 # s
N_SAMPLES_SIGNAL = int(FS * DURATION_S) # number of time samples in the signal
N_SAMPLES_DATASET = 1000  # number of synthetic LFP samples for training/testing


def synaptic_kernel(tau_rise_ms, tau_decay_ms, fs_hz, support_ms=200.0):
    # Exponential decay kernel with rise and decay times.
    t = np.arange(0.0, support_ms / 1000.0, 1.0 / fs_hz)
    k = np.exp(-t / (tau_decay_ms / 1000.0)) - np.exp(-t / (tau_rise_ms / 1000.0))
    k[k < 0.0] = 0.0
    k = k / (np.sum(k) + 1e-12)
    return k


K_AMPA = synaptic_kernel(tau_rise_ms=0.5, tau_decay_ms=2.0, fs_hz=FS)
K_GABA = synaptic_kernel(tau_rise_ms=0.5, tau_decay_ms=7.0, fs_hz=FS)


def poisson_spike_train(rate_hz, n_samples, rng):
    # Discrete-time Poisson counts per sample bin (dt).
    return rng.poisson(rate_hz * DT, size=n_samples).astype(float)


def simulate_lfp(rate_exc_hz, rate_inh_hz, w_exc=1.0, w_inh=1.4):
    # Simulate LFP with excitatory and inhibitory Poisson spike trains.
    spikes_e = poisson_spike_train(rate_exc_hz, N_SAMPLES_SIGNAL, RNG)
    spikes_i = poisson_spike_train(rate_inh_hz, N_SAMPLES_SIGNAL, RNG)

    i_ampa = w_exc * np.convolve(spikes_e, K_AMPA, mode="same")
    i_gaba = -w_inh * np.convolve(spikes_i, K_GABA, mode="same")
    lfp = i_ampa + i_gaba

    return lfp, i_ampa, i_gaba


if __name__ == "__main__":
    print("[1/7] Initializing configuration and feature engine...")
    lfp_samples = []
    ei_ratios = np.zeros(N_SAMPLES_DATASET, dtype=float)
    feature_engine = ncpi.Features(method="catch22", params={"normalize": True})

    print("[2/7] Generating synthetic dataset...")
    traces = None
    for i in range(N_SAMPLES_DATASET):
        # Sample a target E/I ratio and a total drive, then derive rates.
        # This reduces ambiguity compared with drawing excitatory/inhibitory rates independently.
        target_ei = RNG.uniform(0.20, 1.60)
        total_rate = RNG.uniform(10.0, 24.0)
        rate_inh = total_rate / (1.0 + target_ei)
        rate_exc = total_rate - rate_inh

        lfp, i_ampa, i_gaba = simulate_lfp(rate_exc, rate_inh)
        lfp_samples.append(lfp)
        ei_ratios[i] = target_ei

        if i == 0:
            traces = (lfp, i_ampa, i_gaba)
        if (i + 1) % 50 == 0 or (i + 1) == N_SAMPLES_DATASET:
            pct = 100.0 * (i + 1) / N_SAMPLES_DATASET
            print(f"  -> Dataset progress: {i + 1}/{N_SAMPLES_DATASET} ({pct:.1f}%)")

    print("  -> Computing catch22 features in parallel...")
    def feature_progress(completed, total, percent):
        if completed > 0 and (percent % 10 == 0 or completed == total):
            print(f"  -> Feature progress: {completed}/{total} ({percent}%)")

    catch22_features = np.asarray(
        feature_engine.compute_features(
            samples=lfp_samples,
            n_jobs=None,
            progress_callback=feature_progress,
        ),
        dtype=float,
    )

    print("[3/7] Splitting train/test data...")
    idx = np.arange(N_SAMPLES_DATASET)
    RNG.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    X_train, X_test = catch22_features[tr], catch22_features[te]
    y_train, y_test = ei_ratios[tr], ei_ratios[te]

    print("[4/7] Initializing inference model...")
    model = ncpi.Inference(
        model="RandomForestRegressor",
        hyperparams={
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
        },
    )
    model.add_simulation_data(X_train, y_train)

    print("[5/7] Training model...")
    model.train(param_grid=None, scaler=None, seed=42)
    print("[6/7] Computing predictions...")
    y_pred = model.predict(X_test, scaler=None, n_jobs=1)
 
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE (E/I ratio): {mse:.5f}")

    print("[7/7] Plotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(y_test, y_pred, s=18, alpha=0.7, label="Predictions")
    lo, hi = float(min(np.min(y_test), np.min(y_pred))), float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1, label="Ideal fit")
    ax.set_xlabel("Real E/I ratio")
    ax.set_ylabel("Predicted E/I ratio")
    ax.set_title("Predicted vs real E/I ratio")
    ax.legend(frameon=False)
    ax.text(
        0.03,
        0.95,
        f"MSE = {mse:.5f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    plt.tight_layout()
    plt.show()
```

# Tutorials
Explore step-by-step tutorials and complete workflows at:
https://necolab-ugr.github.io/ncpi/tutorials.html

# Citation
If you use `ncpi` in your research, please consider citing our work:

**[1] Alejandro Orozco Valero, Victor Rodriguez-Gonzalez, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie,
Francisco Pelayo, Christian Morillas, Jesus Poza, Carlos Gomez & Pablo Martinez-Canada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9

# Acknowledgements
This work was supported by grants PID2022-139055OA-I00 and PID2022-137461NB-C31, funded by MCIN/AEI/10.13039/501100011033
and by ERDF "A way of making Europe"; and by Junta de Andalucia - Postdoctoral Fellowship Programme PAIDI 2021.
