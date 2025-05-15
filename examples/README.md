# Example Scripts

This folder contains example scripts to generate results published in:

**[1] Alejandro Orozco Valero, Víctor Rodríguez-González, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie, 
Francisco Pelayo, Christian Morillas, Jesús Poza, Carlos Gómez & Pablo Martínez-Cañada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9  

---

## 📂 Folder Structure

### 🔵 `EEG_AD_FTD/`  
Scripts to generate results from applying our inverse models to the EEG dataset used in the study. This dataset 
includes both healthy controls (HCs) and patients clinically diagnosed with Alzheimer’s Disease (AD) at varying stages: 
mild (ADMIL), moderate (ADMOD), and severe (ADSEV).

**Note**: This EEG dataset is not publicly available but can be provided by the authors upon reasonable 
  request.

- **`EEG_AD_FTD.py`**:  
  Extracts empirical features from EEG data and computes predictions of changes in cortical circuit parameters in 
patients with dementia due to AD
- **`figures/EEG_predictions.py`**:  
    Generates **Figure 7**: *"Predicted circuit parameter imbalances in AD based on EEG data"*

### 🔵 `LFP_developing_brain/`  
Scripts to generate results of applying our inverse models to resting-state LFP recordings from the prefrontal 
cortex (PFC) of unanesthetized mice during early postnatal development.  

- **`LFP_developing_brain.py`**:  
  Extracts empirical features from developmental LFP data and computes predictions of changes in cortical circuit 
parameters during the early postnatal period.

- **`figures/LFP_predictions.py`**:  
  Generates **Figure 6**: *"Predictions of changes in cortical circuit parameters derived from developmental 
LFP data."*

- **`figures/emp_features.py`**: 
    Plots features extracted from LFP data as a function of postnatal days.

**Note**: `figures/LFP_predictions.py` performs a Linear Mixed-Effects (LME) analysis, which requires both R and the Python `rpy2` 
packages to be installed beforehand. In the R environment, the following packages must also be installed:


  - `lme4`  
  - `emmeans`

  You can install them with conda using:
  ```bash
  conda install -c conda-forge r-base rpy2 r-lme4 r-emmeans
  ```

  To just install the R packages, use:

  ```r
  install.packages("lme4", dependencies = TRUE)
  install.packages("emmeans", dependencies = TRUE)
  ```

  If you prefer not to use the LME analysis, you can opt to compute Cohen's d statistic instead, which does not 
  require R.


### 🔵 `simulation/`

> **Note:** To run examples that include simulations of the LIF network model (e.g., in `example_full_pipeline.py`), 
> the [NEST simulator](https://nest-simulator.readthedocs.io/) must be installed.  
> If you're using Conda or Docker, you can install a pre-built NEST package with:
>
> ```bash
> conda install -c conda-forge nest-simulator=3.8
> ```
>
> Similarly, to compute field potentials, the [`LFPykernels`](https://github.com/LFPy/LFPykernels) package must be installed via pip:
>
> ```bash
> pip install LFPykernels
> ```


#### Figures Generation (`Hagen_model/figures/`)

- **`example_full_pipeline.py`**  
  Generates plots for **Figure 3**: *"Representative simulations illustrating the dynamics of the LIF network model in 
response to varying external input values."*

- **`save_code_as_image.py`**  
  Generates **Figure 2**: *"A Python code snippet demonstrating the usage of the `ncpi` library and its core classes."*

- **`SBI_results.py`**  
  Generates plots using SBI-based models (**Supplementary Material**).

- **`sim_features.py`**  
  Generates **Figure 4**: *"Comparison of features extracted from simulated neural signals."*

- **`sim_predictions.py`**  
  Generates **Figure 5**: *"Predicted model parameters based on simulation data under various feature configurations."*

---

#### Simulation Code (`Hagen_model/simulation/`)

- **`params/`**  
  Contains parameter sets used to run simulations of Hagen's model.

- **`python/`**  
  Contains the main Python script to execute Hagen’s model simulations.

- **`compute_features.py`**  
  Extracts features from simulated data.

- **`example_model_simulation.py`**  
  Example script demonstrating how to simulate Hagen’s model.

- **`preprocessing.py`**  
  Preprocesses simulated data and structures it appropriately before feature extraction.

- **`sim_statistics.py`**  
  Plots statistical relationships between model parameters and extracted features.

---

#### Model Training (`Hagen_model/train/`)

- **`RepeatedKFold.py`**  
  Trains inverse models on simulation data using repeated K-fold cross-validation.


---
 