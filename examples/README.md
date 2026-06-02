# Example Scripts

This folder contains example scripts used to generate results for the publication below, plus additional simulation scripts
(e.g., Cavallari and four-area cortical models) that were added later and are not part of the original publication:

**[1] Alejandro Orozco Valero, Victor Rodriguez-Gonzalez, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie,**
**Francisco Pelayo, Christian Morillas, Jesus Poza, Carlos Gomez & Pablo Martinez-Cañada**
*A Python toolbox for neural circuit parameter inference.*
npj Syst Biol Appl 11, 45 (2025).
https://doi.org/10.1038/s41540-025-00527-9

## Requirements for these examples

Use the same base setup as the repository README:

```bash
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
pip install "ncpi[examples]"
```

### NEST-dependent examples

Some simulation examples require NEST (for example `examples/simulation/Hagen_model/` and
`examples/simulation/four_area_cortical_model/`):

```bash
conda install -c conda-forge nest-simulator=3.8
```

If `nest-simulator` is not available for your platform/channel combination, follow:
https://nest-simulator.readthedocs.io/

On Windows, NEST-based examples are recommended via WSL2 (Ubuntu), not native Windows. One-time setup:

```powershell
wsl --install
```

Then inside Ubuntu/WSL:

```bash
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
pip install "ncpi[examples]"
conda install -c conda-forge nest-simulator=3.8
```

### R-dependent analysis in figure scripts

`examples/LFP_developing_brain/figures/LFP_predictions.py` and
`examples/EEG_AD/figures/EEG_predictions.py` can run LME analysis through `Analysis.lmer_tests`, which requires R,
`rpy2`, and specific R packages.

The `analysis` extra includes `rpy2` (R-backed dependency). Before installing `ncpi[analysis]`, make sure R is installed
on your system (e.g. from https://cran.r-project.org/ or your package manager). Installing `rpy2` via pip or conda can
fail if a suitable R installation is not present. If you use Conda, you can install R with:

```bash
conda install -c conda-forge r-base
```

Install Python-side analysis dependencies (includes `rpy2`):

```bash
pip install "ncpi[analysis]"
```

R packages required by `Analysis` class methods:

- `lmer_tests(...)`: `lme4`, `emmeans`
- `lmer_selection(...)`: `lme4`, `buildmer`

Install options for the R backend:

In R:

```r
install.packages(c("lme4", "emmeans", "buildmer"), repos="https://cloud.r-project.org")
```

With conda-forge:

```bash
conda install -c conda-forge r-base rpy2 r-lme4 r-emmeans r-buildmer
```

If you do not want to use R for these figure scripts, set their statistical mode to Cohen's d.

## Folder structure

- `tools.py`: shared utilities used by multiple example scripts.
- `EEG_AD/`: EEG-based empirical pipeline and figure scripts.
  - `EEG_AD.py`
  - `figures/EEG_predictions.py`
- `LFP_developing_brain/`: developmental LFP empirical pipeline and figure scripts.
  - `LFP_developing_brain.py`
  - `figures/LFP_predictions.py`
  - `figures/emp_features.py`
- `simulation/`: simulation workflows and model-specific scripts.
  - `Hagen_model/`
    - `figures/SBI_results.py`
    - `figures/example_full_pipeline.py`
    - `figures/save_code_as_image.py`
    - `figures/sim_features_v1.py`
    - `figures/sim_features_v2.py`
    - `figures/sim_predictions.py`
    - `simulation/example_model_simulation.py`
    - `simulation/massive_model_simulation.py`
    - `simulation/merge_massive_model_simulation_batches.py`
    - `simulation/run_massive_model_simulation_slurm_array.sh`
    - `simulation/params/analysis_params.py`
    - `simulation/params/network_params.py`
    - `simulation/params/simulation_params.py`
    - `simulation/python/analysis.py`
    - `simulation/python/network.py`
    - `simulation/python/simulation.py`
    - `train/RepeatedKFold.py`
  - `four_area_cortical_model/`
    - `simulation/example_model_simulation.py`
    - `simulation/params/analysis_params.py`
    - `simulation/params/network_params.py`
    - `simulation/params/simulation_params.py`
    - `simulation/python/analysis.py`
    - `simulation/python/network.py`
    - `simulation/python/simulation.py`
  - `Cavallari_model/`
    - `LIF_simulation/example_model_simulation.py`
    - `LIF_simulation/massive_model_simulation.py`
    - `LIF_simulation/merge_massive_model_simulation_batches.py`
    - `LIF_simulation/run_massive_model_simulation_slurm_array.sh`
    - `LIF_simulation/params/network_params.py`
    - `LIF_simulation/params/simulation_params.py`
    - `LIF_simulation/python/simulation.py`
    - `MC_simulation/analysis_params.py`
    - `MC_simulation/example_model_simulation.py`
    - `neuron_model/README.md`
    - `neuron_model/install.sh`
    - `neuron_model/CMakeLists.txt`
    - `neuron_model/src/CMakeLists.txt`
    - `neuron_model/src/cavallari_module.cpp`
    - `neuron_model/src/iaf_bw_2003.cpp`
    - `neuron_model/src/iaf_bw_2003.h`

## Custom NEST extension (Cavallari model)

`examples/simulation/Cavallari_model/neuron_model/` provides a custom NEST extension (`iaf_bw_2003`) used by the
Cavallari LIF simulation scripts. Build it before running those scripts:

```bash
cd examples/simulation/Cavallari_model/neuron_model
./install.sh
```
