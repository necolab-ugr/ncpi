# Example Scripts

This folder contains example scripts used to generate results for the publication below, plus additional simulation scripts
(e.g., Cavallari and four-area cortical models) that were added later and are not part of the original publication:

**[1] Alejandro Orozco Valero, Victor Rodriguez-Gonzalez, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie,**
**Francisco Pelayo, Christian Morillas, Jesus Poza, Carlos Gomez & Pablo Martinez-Cañada**
*A Python toolbox for neural circuit parameter inference.*
npj Syst Biol Appl 11, 45 (2025).
https://doi.org/10.1038/s41540-025-00527-9

## Requirements for these examples

Use the same base setup as the repository README. `ncpi` requires Python 3.10+, and a dedicated Conda environment is
recommended:

```bash
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env
pip install ncpi
pip install "ncpi[examples]"
```

The `examples` extra includes dependencies for scripts that use NEURON-related workflows, so use WSL2 on Windows when
running examples that require NEST, NEURON, field-potential kernels, or the full simulation stack. Native Windows is
suitable only for example workflows that do not require those Linux-oriented backends, matching the repository README
guidance for optional extras such as `examples`, `fieldpotential`, `tests`, and `all`.

### NEST-dependent examples

Some simulation examples require NEST (for example `examples/simulation/Hagen_model/`,
`examples/simulation/four_area_cortical_model/`, and the Cavallari LIF simulation):

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

- `tools.py`: Shared utilities for downloading Zenodo datasets, loading simulation assets, mapping model folders, and
  preparing feature/model inputs across multiple example scripts.
- `EEG_AD/`: EEG Alzheimer's disease example workflow and associated figure script.
  - `EEG_AD/EEG_AD.py`: Runs the EEG Alzheimer's disease empirical pipeline, including data loading, feature extraction,
    inference, and result preparation.
  - `EEG_AD/figures/EEG_predictions.py`: Generates EEG prediction figures and optional statistical summaries for the
    Alzheimer's disease workflow.
- `LFP_developing_brain/`: Developmental LFP empirical workflow and figure scripts.
  - `LFP_developing_brain/LFP_developing_brain.py`: Runs the LFP developing-brain pipeline from empirical data loading through prediction.
  - `LFP_developing_brain/figures/LFP_predictions.py`: Generates prediction plots and optional statistical summaries for the LFP workflow.
  - `LFP_developing_brain/figures/emp_features.py`: Plots empirical feature distributions for the developing-brain LFP dataset.
- `simulation/`: Simulation workflows, model-specific simulation code, training scripts, and figure-generation scripts.
  - `simulation/run_massive_training.py`: Trains inverse models across the massive simulation datasets and feature sets.
  - `Hagen_model/`: Hagen recurrent LIF model simulations, figures, and inverse-model training utilities.
    - `simulation/Hagen_model/figures/SBI_results.py`: Recreates SBI result figures from Hagen simulation data and trained models.
    - `simulation/Hagen_model/figures/example_full_pipeline.py`: Demonstrates the full Hagen simulation-to-analysis pipeline.
    - `simulation/Hagen_model/figures/save_code_as_image.py`: Utility script for rendering code snippets as figure-ready images.
    - `simulation/Hagen_model/figures/sim_features_v1.py`: Generates feature-space visualizations for the original Hagen simulation dataset.
    - `simulation/Hagen_model/figures/sim_features_v2.py`: Generates feature-space visualizations for the merged Hagen v2 simulation dataset.
    - `simulation/Hagen_model/figures/sim_predictions.py`: Generates prediction figures for Hagen simulation examples.
    - `simulation/Hagen_model/simulation/example_model_simulation.py`: Runs a smaller Hagen model simulation suitable for example workflows.
    - `simulation/Hagen_model/simulation/massive_model_simulation.py`: Runs batches of the larger Hagen parameter-sweep simulation dataset.
    - `simulation/Hagen_model/simulation/merge_massive_model_simulation_batches.py`: Merges batch outputs from Hagen massive simulations.
    - `simulation/Hagen_model/simulation/run_massive_model_simulation_slurm_array.sh`: SLURM array launcher for Hagen massive simulations.
    - `simulation/Hagen_model/simulation/params/analysis_params.py`: Defines analysis-stage parameters for Hagen simulation post-processing.
    - `simulation/Hagen_model/simulation/params/network_params.py`: Defines Hagen network architecture and population parameters.
    - `simulation/Hagen_model/simulation/params/simulation_params.py`: Defines Hagen simulation runtime and stimulation parameters.
    - `simulation/Hagen_model/simulation/python/analysis.py`: Implements Hagen simulation analysis and output extraction.
    - `simulation/Hagen_model/simulation/python/network.py`: Builds the Hagen NEST network from the parameter files.
    - `simulation/Hagen_model/simulation/python/simulation.py`: Executes the Hagen network simulation stage.
    - `simulation/Hagen_model/train/RepeatedKFold.py`: Trains and evaluates inverse models with repeated K-fold cross-validation.
  - `four_area_cortical_model/`: Four-area cortical LIF model simulation workflow.
    - `simulation/four_area_cortical_model/simulation/example_model_simulation.py`: Runs the four-area cortical model example simulation.
    - `simulation/four_area_cortical_model/simulation/params/analysis_params.py`: Defines analysis-stage parameters for the four-area simulation.
    - `simulation/four_area_cortical_model/simulation/params/network_params.py`: Defines populations, connectivity, and architecture for the four-area model.
    - `simulation/four_area_cortical_model/simulation/params/simulation_params.py`: Defines runtime and stimulation settings for the four-area model.
    - `simulation/four_area_cortical_model/simulation/python/analysis.py`: Extracts and stores analysis outputs from four-area simulations.
    - `simulation/four_area_cortical_model/simulation/python/network.py`: Builds the four-area NEST network from model parameters.
    - `simulation/four_area_cortical_model/simulation/python/simulation.py`: Executes the four-area simulation stage.
  - `Cavallari_model/`: Cavallari model examples, including LIF simulation, multicompartment setup, and custom NEST code.
    - `simulation/Cavallari_model/LIF_simulation/example_model_simulation.py`: Runs a smaller Cavallari LIF simulation example.
    - `simulation/Cavallari_model/LIF_simulation/massive_model_simulation.py`: Runs batches of the larger Cavallari LIF parameter-sweep dataset.
    - `simulation/Cavallari_model/LIF_simulation/merge_massive_model_simulation_batches.py`: Merges batch outputs from Cavallari massive simulations.
    - `simulation/Cavallari_model/LIF_simulation/run_massive_model_simulation_slurm_array.sh`: SLURM array launcher for Cavallari massive simulations.
    - `simulation/Cavallari_model/LIF_simulation/params/network_params.py`: Defines Cavallari LIF network structure and population parameters.
    - `simulation/Cavallari_model/LIF_simulation/params/simulation_params.py`: Defines Cavallari LIF simulation runtime and stimulation parameters.
    - `simulation/Cavallari_model/LIF_simulation/python/simulation.py`: Executes the Cavallari LIF simulation stage.
    - `simulation/Cavallari_model/MC_simulation/analysis_params.py`: Defines analysis parameters for Cavallari multicompartment simulations.
    - `simulation/Cavallari_model/MC_simulation/example_model_simulation.py`: Runs the Cavallari multicompartment simulation example.
    - `simulation/Cavallari_model/neuron_model/README.md`: Documents the custom Cavallari NEST extension build and usage.
    - `simulation/Cavallari_model/neuron_model/install.sh`: Builds and installs the custom Cavallari NEST extension.
    - `simulation/Cavallari_model/neuron_model/CMakeLists.txt`: Top-level CMake configuration for the custom Cavallari NEST module.
    - `simulation/Cavallari_model/neuron_model/src/CMakeLists.txt`: Source-level CMake configuration for the Cavallari module library.
    - `simulation/Cavallari_model/neuron_model/src/cavallari_module.cpp`: Registers the custom Cavallari module with NEST.
    - `simulation/Cavallari_model/neuron_model/src/iaf_bw_2003.cpp`: Implements the custom `iaf_bw_2003` neuron model.
    - `simulation/Cavallari_model/neuron_model/src/iaf_bw_2003.h`: Declares the custom `iaf_bw_2003` neuron model interface and state.

## Custom NEST extension (Cavallari model)

`examples/simulation/Cavallari_model/neuron_model/` provides a custom NEST extension (`iaf_bw_2003`) used by the
Cavallari LIF simulation scripts. Build it before running those scripts:

```bash
cd examples/simulation/Cavallari_model/neuron_model
./install.sh
```
