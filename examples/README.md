# Example Scripts

This folder contains example scripts to generate results published in:

**[1] Alejandro Orozco Valero, Víctor Rodríguez-González, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie, 
Francisco Pelayo, Christian Morillas, Jesús Poza, Carlos Gómez & Pablo Martínez-Cañada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9  

---

**Note**: `LFP_developing_brain/figures/LFP_predictions.py` and `EEG_AD/figures/EEG_predictions.py`  perform a Linear 
Mixed-Effects (LME) analysis, which requires both R and the Python `rpy2` packages to be installed beforehand 
(e.g., ```conda install -c conda-forge r-base rpy2```). In the R environment, the following packages must also be 
installed:


  - `lme4`  
  - `emmeans`

  To install these packages in R, use:

  ```r
  install.packages(c("lme4", "emmeans"))
  ```

  Or alternatively (if R packages fail to install), with conda:
  ```bash
  conda install -c conda-forge r-lme4 r-emmeans
  ```


  If you prefer not to use the LME analysis, you can opt to compute Cohen's d statistic instead, which does not 
  require R.

## 📂 Folder Structure

- **`tools.py`**:  
  Contains shared utility functions used across the different example scripts.

### 🔵 `EEG_AD/`  
Scripts to generate results from applying our inverse models to the EEG dataset used in the study. This dataset 
includes both healthy controls (HCs) and patients clinically diagnosed with Alzheimer’s Disease (AD) at varying stages: 
mild (ADMIL), moderate (ADMOD), and severe (ADSEV).

**Note**: This EEG dataset is not publicly available but can be provided by the authors upon reasonable 
  request.

- **`EEG_AD.py`**:  
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

### 🔵 `simulation/`

> **Note:** To run examples that include simulations of the LIF network model (e.g., in `example_full_pipeline.py`), 
> the [NEST simulator](https://nest-simulator.readthedocs.io/) must be installed. You can install a pre-built NEST 
> package with:
>
> ```bash
> conda install -c conda-forge nest-simulator=3.8
> ```
>
> Similarly, to compute field potentials, the [`LFPykernels`](https://github.com/LFPy/LFPykernels) package must be 
> installed via pip:
>
> ```bash
> pip install LFPykernels
> ```
>  If you encounter a binary incompatibility between your installed NumPy and scikit-learn packages after installing NEST—
> for example, an error like *numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, 
> got 88 from PyObject*—you can resolve it by force-reinstalling both packages:
>
> ```bash
> pip uninstall scikit-learn numpy -y
> pip install scikit-learn==1.3.2 numpy
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
 