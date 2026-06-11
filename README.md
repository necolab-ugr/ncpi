> **The current code is quite mature and we are on the road to launch ncpi version 1.0 by June 16, 2026.**

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

<div align="center">
  <img src="docs/tutorials/automation/videos/compute-field-potentials.gif" alt="Field-potential computation workflow in the ncpi WebUI" width="900">
</div>

# Key Features of `ncpi`
- **All-in-one solution**: a unified package for forward and inverse modeling of extracellular signals from neural
  circuit simulations.
- **Biophysically grounded analysis**: practical workflows to bridge electrophysiology and neural circuit parameters.
- **Graphical interface for simulation and empirical workflows**: includes a GUI that can run both simulation and
  empirical pipelines, and can load many different file formats and dataset structures.
- **Flexible and extensible**: use individual modules independently or run complete end-to-end pipelines.

# Installation

`ncpi` requires **Python 3.10+**.

## 1) Create and activate a Conda environment (Windows + Unix)

We strongly recommend installing `ncpi` in a dedicated **Conda** environment.
If you need to install Anaconda first, download it from the official page:
https://www.anaconda.com/download

On Windows, start from **Anaconda Prompt** (recommended).

## 2) Install ncpi on Unix (Linux/macOS)

```bash
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
```

### NEST on Unix/macOS (for LIF simulation examples)
If you run examples that depend on NEST (for example in `examples/simulation`), install it in the same environment:

```bash
conda install -c conda-forge nest-simulator=3.8
```

If `nest-simulator` is not available for your platform/channel combination, follow the official NEST build/install
instructions: https://nest-simulator.readthedocs.io/

## 3) Install ncpi on Windows

In **Anaconda Prompt**, run:

```powershell
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
```

### NEST on Windows
NEST-based examples are recommended via **WSL2 (Ubuntu)**, not native Windows.

```powershell
# One-time WSL2 setup
wsl --install
```

Then, inside the Ubuntu/WSL shell:
```bash
conda install -c conda-forge nest-simulator=3.8
```

## 4) Optional Dependencies

`ncpi` supports optional extras. Install only what your workflow needs.

### Extras shortcuts
```bash
pip install "ncpi[parser]"          # extended parser backends
pip install "ncpi[fieldpotential]"  # kernel/CDM/LFP + M/EEG forward models
pip install "ncpi[webui]"           # WebUI runtime backends
pip install "ncpi[examples]"        # dependencies for example scripts
# Note: the dependencies listed below refer to Section 6 (Optional backends notes).
pip install "ncpi[tests]"           # test stack dependencies
pip install "ncpi[analysis]"        # statistics + EEG/MEG analysis helpers
pip install "ncpi[hctsa]"           # hctsa backend support
```

## 5) WebUI: installation and usage

The WebUI is available from the repository source (webui/app.py).

### Install WebUI dependencies

After activating your Conda environment:

```bash
pip install "ncpi[webui]"
```

### Start WebUI

From the repository root, with the Conda environment activated:

```bash
python webui/launcher.py local
```

The launcher starts Flask and opens the default browser at:

```text
http://127.0.0.1:5000
```

The following compatibility command also starts the local WebUI:

```bash
python webui/app.py
```

For best results, we recommend running the WebUI in Chrome, as our tests are most stable there and we have observed a couple of issues in other browsers.

### Run on a remote server over SSH

#### Remote command

Run the following command from the `ncpi` repository on your local machine:

```bash
python webui/launcher.py remote <user>@<server> \
  --ssh-port <P> \
  --local-port <L> \
  --remote-port <R> \
  --remote-dir <path/to/ncpi> \
  --python <path/to/python>
```

Replace the placeholders:
- `<user>@<server>`: SSH destination (e.g., `username@example.org`)
- `<P>`: SSH port (default: 22)
- `<L>`: Local port on your machine for the browser (default: 5000)
- `<R>`: Remote port on the server where Flask runs (default: 5000)
- `<path/to/ncpi>`: Absolute path to the `ncpi` repository on the server
- `<path/to/python>`: Absolute path to Python executable on the server (from `which python`)

The browser opens `http://127.0.0.1:<L>`. Keep the launcher terminal open while using the WebUI. Press `Ctrl+C` to close the SSH tunnel and stop the remote Flask process.

#### Manual SSH alternative

The launcher is recommended, but the equivalent workflow can be performed manually:

```bash
ssh -L <L>:127.0.0.1:<R> -p <P> <user>@<server>
```

Then, inside the SSH session:

```bash
cd <path/to/ncpi>
<path/to/python> webui/app.py \
  --host 127.0.0.1 \
  --port <R> \
  --no-browser
```

Finally, open `http://127.0.0.1:<L>` on the local machine.

### Windows note

You can run the same command from Anaconda Prompt or PowerShell. If your workflow needs NEST, run the WebUI from your WSL environment where NEST is installed.

## 6) Optional backends notes
Optional backends are listed in **Section 4 (Optional Dependencies)**. Install only the extras required by your workflow.

### analysis/tests (R) note
Both the `analysis` and `tests` extras include `rpy2` (R-backed dependency). Before installing `ncpi[analysis]` or
`ncpi[tests]`, make sure R is installed on your system (e.g. from https://cran.r-project.org/ or your package
manager). Installing `rpy2` via pip or conda can fail if a suitable R installation is not present. If you use Conda,
you can install R with: `conda install -c conda-forge r-base`.

R packages required by `Analysis` class methods:
- `lmer_tests(...)`: `lme4`, `emmeans`
- `lmer_selection(...)`: `lme4`, `buildmer`

Install options for the R backend:
- In R:
  `install.packages(c("lme4", "emmeans", "buildmer"), repos="https://cloud.r-project.org")`
- With conda-forge:
  `conda install -c conda-forge r-base rpy2 r-lme4 r-emmeans r-buildmer`

If you want to run tests without setting up R, avoid `ncpi[tests]` and install only the specific test dependencies you
need. R-dependent tests (e.g. `tests/Analysis/test_lmer.py`) are skipped automatically when `rpy2`/R is unavailable.

### tests/playwright note
After installing test dependencies, install Playwright browsers with:

```bash
python -m playwright install
```

### hctsa note
For hctsa-based features, install hctsa first: https://github.com/benfulcher/hctsa

The `hctsa` extra depends on the hctsa MATLAB toolbox and the MATLAB Engine for Python. Attempting to install a
`matlabengine` pip package without MATLAB installed (or without a matching MATLAB Engine distribution) can fail or
raise errors. The recommended approach is to install MATLAB first, then install the MATLAB Engine for Python from the
MATLAB installation directory (see MathWorks docs: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html),
or otherwise ensure the engine distribution you install matches your MATLAB version. After that, install hctsa and
pass the hctsa repository path as `hctsa_folder` when invoking hctsa-backed features in ncpi.

Note: the `tests` extra does not install `matlabengine`; hctsa tests are skipped unless MATLAB Engine and an hctsa
folder are available.


# Folder Structure

- `ncpi/`: core library modules (`Simulation`, `Features`, `FieldPotential`, `Inference`, `Analysis`, parser utilities).
- `examples/`: example scripts and runnable workflows.
- `docs/`: project documentation sources and published pages.
- `img/`: static visual assets.
- `webui/`: web application and related runtime code.
- `tests/`: automated test suite.
- `.github/`: GitHub workflows and repository automation.

# Tutorials
The documentation includes installation guides, API references, and end-to-end tutorials for simulation and empirical
pipelines. Browse it at: https://necolab-ugr.github.io/ncpi/

# Example Usage

The example below follows Gao et al. (2017: https://doi.org/10.1016/j.neuroimage.2017.06.078): Poisson spike trains are generated by integrating ISIs
drawn from exponential distributions, convolved with AMPA/GABAA difference-of-exponential conductance kernels, then
converted to currents using reversal potentials and summed to form the LFP. E:I is set by scaling inhibition so
mean gI is 2-6x mean gE, and each LFP is power-normalized to unity.

After simulation, each LFP is transformed into a compact catch22 feature vector, then split into train/test sets to fit a
RandomForest regressor that maps features to the ground-truth E:I ratio. Predictions on held-out samples are evaluated
with MSE and visualized in a predicted-versus-real scatter plot with a diagonal reference line for ideal agreement.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import ncpi


RNG = np.random.default_rng(42)
FS = 1000.0  # Hz
DT = 1.0 / FS  # s
DURATION_S = 12.0  # s
N_SAMPLES_SIGNAL = int(FS * DURATION_S)
N_SAMPLES_DATASET = 1000  # number of synthetic LFP samples for training/testing

# Gao et al. (2017) Table 1 parameters
RATE_E_HZ = 2.0
RATE_I_HZ = 5.0
N_E = 8000
N_I = 2000
V_REST_MV = -65.0
E_AMPA_MV = 0.0
E_GABAA_MV = -80.0
TAU_RISE_AMPA_MS = 0.1
TAU_DECAY_AMPA_MS = 2.0
TAU_RISE_GABAA_MS = 0.5
TAU_DECAY_GABAA_MS = 10.0
EPS = 1e-12


def conductance_kernel(tau_rise_ms, tau_decay_ms, fs_hz, support_ms=200.0):
    # Difference-of-exponentials with area normalization constant C.
    t = np.arange(0.0, support_ms / 1000.0, 1.0 / fs_hz)
    tau_r = tau_rise_ms / 1000.0
    tau_d = tau_decay_ms / 1000.0
    k = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    k[k < 0.0] = 0.0
    k = k / (np.sum(k) + EPS)
    return k


K_AMPA = conductance_kernel(TAU_RISE_AMPA_MS, TAU_DECAY_AMPA_MS, FS)
K_GABAA = conductance_kernel(TAU_RISE_GABAA_MS, TAU_DECAY_GABAA_MS, FS)


def poisson_counts_from_isi(rate_hz, duration_s, dt_s, n_samples, rng):
    # Poisson process by ISI integration (ISI ~ Exponential(rate_hz)).
    expected_spikes = max(1, int(rate_hz * duration_s))
    n_draws = max(32, int(expected_spikes + 8.0 * np.sqrt(expected_spikes) + 64))
    isi = rng.exponential(scale=1.0 / rate_hz, size=n_draws)
    spike_times = np.cumsum(isi)

    # Extend if this draw did not cover full duration.
    while spike_times[-1] < duration_s:
        extra = rng.exponential(scale=1.0 / rate_hz, size=n_draws)
        spike_times = np.concatenate([spike_times, spike_times[-1] + np.cumsum(extra)])

    spike_times = spike_times[spike_times < duration_s]
    spike_bins = (spike_times / dt_s).astype(int)
    return np.bincount(spike_bins, minlength=n_samples).astype(float)


def simulate_lfp(target_inh_over_exc):
    # Superposition of Poisson neurons in each population is Poisson with summed rate.
    spikes_e = poisson_counts_from_isi(
        rate_hz=RATE_E_HZ * N_E,
        duration_s=DURATION_S,
        dt_s=DT,
        n_samples=N_SAMPLES_SIGNAL,
        rng=RNG,
    )
    spikes_i = poisson_counts_from_isi(
        rate_hz=RATE_I_HZ * N_I,
        duration_s=DURATION_S,
        dt_s=DT,
        n_samples=N_SAMPLES_SIGNAL,
        rng=RNG,
    )

    g_e = np.convolve(spikes_e, K_AMPA, mode="same")
    g_i = np.convolve(spikes_i, K_GABAA, mode="same")

    # Set mean gI to 2x-6x mean gE (Gao et al. Table 1 E:I range 1:2 to 1:6).
    g_i *= (target_inh_over_exc * np.mean(g_e)) / (np.mean(g_i) + EPS)
    ei_ratio = np.mean(g_e) / (np.mean(g_i) + EPS)

    i_e = g_e * (V_REST_MV - E_AMPA_MV)
    i_i = g_i * (V_REST_MV - E_GABAA_MV)
    lfp = i_e + i_i

    # Normalize total LFP power to unity for each E:I ratio.
    total_power = np.sum(np.abs(np.fft.rfft(lfp)) ** 2)
    norm = np.sqrt(total_power + EPS)
    return lfp / norm, i_e / norm, i_i / norm, ei_ratio


if __name__ == "__main__":
    print("[1/7] Initializing configuration and feature engine...")
    lfp_samples = []
    ei_ratios = np.zeros(N_SAMPLES_DATASET, dtype=float)
    feature_engine = ncpi.Features(method="catch22", params={"normalize": True})

    print("[2/7] Generating synthetic dataset...")
    traces = None
    for i in range(N_SAMPLES_DATASET):
        target_inh_over_exc = RNG.uniform(2.0, 6.0)  # gI/gE
        lfp, i_e, i_i, ei_ratio = simulate_lfp(target_inh_over_exc)
        lfp_samples.append(lfp)
        ei_ratios[i] = ei_ratio

        if i == 0:
            traces = (lfp, i_e, i_i)
        if (i + 1) % 50 == 0 or (i + 1) == N_SAMPLES_DATASET:
            pct = 100.0 * (i + 1) / N_SAMPLES_DATASET
            print(f"  -> Dataset progress: {i + 1}/{N_SAMPLES_DATASET} ({pct:.1f}%)")

    print("  -> Computing catch22 features...")
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
    model.train(param_grid=None, scaler=False, seed=42)
    print("[6/7] Computing predictions...")
    y_pred = model.predict(X_test, scaler=False, n_jobs=1)
 
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

# Citation
If you use `ncpi` in your research, please consider citing our work:

**[1] Alejandro Orozco Valero, Victor Rodriguez-Gonzalez, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie,
Francisco Pelayo, Christian Morillas, Jesus Poza, Carlos Gomez & Pablo Martinez-Cañada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9

# Acknowledgements
Supported by grants PID2022-139055OA-I00 and PID2022-137461NB-C31 (MCIN/AEI/10.13039/501100011033, ERDF), by grant RYC2024-049595-I (MCIN/AEI/10.13039/501100011033,  FSE+) and by Junta de Andalucia Postdoctoral Fellowship Programme PAIDI 2021.
