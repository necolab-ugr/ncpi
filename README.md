> 🚧 **This repository is under active development, and features or documentation may still evolve. That said, version 
> v0.2.6 marks a maturing stage of the project: it is functionally stable and has been thoroughly tested with all 
> included examples, though some aspects may still change ahead of a full stable release** 🚧

<div align="center">

# ncpi: neural circuit parameter inference
___

<img src="https://raw.githubusercontent.com/necolab-ugr/ncpi/main/img/ncpi_logo.png" alt="ncpi logo" width="150">

</div>

[Getting Started](https://necolab-ugr.github.io/ncpi/tutorials/getting_started.html) | 
[Documentation](https://necolab-ugr.github.io/ncpi/)

`ncpi` is a Python package for model-based inference of neural circuit parameters from population-level 
electrophysiological recordings, such as LFP, ECoG, MEG, and EEG. `ncpi` provides a rapid, reproducible, and robust 
framework for estimating the most probable neural circuit parameters associated with an empirical observation, 
streamlining traditionally complex workflows into a minimal amount of code. Check out the [example usage](#example-usage) 
to see how `ncpi` can accelerate your research and unlock the full potential of **model-based neural inference**.

# Key Features of `ncpi`
- **All-in-one solution**: Streamline your workflow with a unified package that integrates state-of-the-art methods for
both forward and inverse modeling of extracellular signals based on single-neuron network model simulations. `ncpi` 
integrates tools to simulate realistic population-level neural signals, extract key neurophysiological features, 
train inverse modeling approaches, predict circuit parameters, and ultimately benchmark candidate biomarkers, or 
sets of biomarkers, as proxy markers of neural circuit dynamics.

- **Biophysically grounded analysis**: Effortlessly decode experimental electrophysiological recordings using robust, 
reproducible, and streamlined code—unlocking model-based biophysical insights in just a few lines.

- **Flexible and extensible**: Seamlessly integrate ncpi into your workflow—use individual functionalities (e.g.,
parameter inference techniques based on [sbi](https://sbi-dev.github.io/sbi)) with custom functions or deploy the full 
inverse modelling pipeline. Designed for modularity, it adapts to your needs without constraints.


# Installation

`ncpi` requires **Python 3.10 or higher**. `ncpi` can be run on both **Linux** and **Windows** when installed in an 
**Anaconda** or **Miniconda** Python environment. To install `ncpi`, you can use `pip`; the package is available on 
PyPI and can be installed directly from there. 

```bash
# Step 1: Create and activate a conda environment (recommended)
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env

# Step 2: Install ncpi using pip
pip install ncpi
```

**Note:** To run examples that include simulations of the LIF network model (e.g., in `example_full_pipeline.py`), 
the [NEST simulator](https://nest-simulator.readthedocs.io/) must be installed (note also that to run **NEST** on Windows you must first install the 
**Windows Subsystem for Linux (WSL)**). You can install a pre-built NEST package in Linux with:

```bash
conda install -c conda-forge nest-simulator=3.8
 ```

Optional dependencies by class:
- `Features`: `pycatch22` (catch22), `specparam` (spectral features), `mne` (band/spectrum DFA/fEI), MATLAB Engine for Python + `h5py` + the hctsa repo (hctsa feature set; set `hctsa_folder` to the repo path).
- `FieldPotential`: `LFPy`, `lfpykernels`, `neuron`, `h5py` (kernel workflow) and `lfpykit` (EEG/MEG forward models).
- `Analysis`: `rpy2` plus R packages `lme4`, `emmeans`, and `buildmer` (depending on analysis methods used).
- `Simulation` examples: `nest-simulator` for LIF network simulations.

Install extras via pip (as needed):
```bash
pip install "ncpi[meeg]"
pip install "ncpi[fieldpotential]"
pip install "ncpi[analysis]"
pip install "ncpi[all]"
```

For hctsa, follow the hctsa installation instructions (https://github.com/benfulcher/hctsa), and install the MATLAB
Engine for Python either from your MATLAB distribution (`<MATLAB_ROOT>/extern/engines/python`) or via a compatible
`matlabengine` pip package (e.g., `pip install "matlabengine==24.2.2"`). Then pass the hctsa repo path as
`hctsa_folder`.

If you encounter a binary incompatibility between your installed NumPy and scikit-learn packages after installing NEST—
for example, an error like *numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, 
got 88 from PyObject*—you can resolve it by force-reinstalling both packages:

```bash
pip uninstall scikit-learn numpy -y
pip install scikit-learn==1.5.0 numpy
```

## Installation of web GUI
You'll need activate the virtual environment with conda to run the application.
```bash
conda activate ncpi-env
```

**Run the web application**
Open a terminal window. Go to `webui/` folder. Then, run the application with:
```bash
flask run
```

To access the frontend in the same computer you are running this web application, visit http://localhost:5000/ in your browser (or the URL flask indicates in the terminal window).

# Folder Structure

- `ncpi/`: Contains the source code for the library, organized into modules and classes.
- `examples/`: Includes scripts used to reproduce the results presented in the reference papers (see [Citation](#Citation)).
- `docs/`: Contains documentation for the library, including usage instructions, API references, and guides.
- `img/`: Stores image assets for the project, such as the library logo.
- `webui/`: Contains the web application of ncpi


# Example Usage
```python
import ncpi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import inspect

# Parameters for generating simulated neural data
n_neurons = 50           # Number of neurons in each simulated recording
sim_time = 2000          # Total number of time points per sample 
sampling_rate = 100      # Sampling rate in Hz
n_samples = 1000        # Total number of samples (combined training + test set)

def simulator(θ):
    """
    Simulates spike trains modulated by a shared sinusoidal signal and independent noise.

    Parameters
    ----------
    θ : float
        Controls the exponential decay of the synaptic kernel and the influence of the 
        shared signal.

    Returns
    -------
    spikes : ndarray of shape (n_neurons, sim_time)
        Binary array representing spikes (1s) and no spikes (0s) for each neuron over time.
    """
    spikes = np.zeros((n_neurons, sim_time))

    # Synaptic kernel
    tau = sampling_rate * (θ + 0.01)
    t_kernel = np.arange(int(sampling_rate * 4))
    exp_kernel = np.exp(-t_kernel / tau)

    # Sinusoidal shared signal
    freq = 2.0  # Hz
    time = np.arange(sim_time) / sampling_rate
    shared_input = 0.5 * (1 + np.sin(2 * np.pi * freq * time))  # Values in [0, 1]

    for neuron in range(n_neurons):
        # Independent signal for each neuron
        private_input = np.random.rand(sim_time)
        private_modulated = np.convolve(private_input, exp_kernel, mode='same')

        # Combine shared and private inputs
        k = 0.5 * θ # Mixing coefficient based on θ
        modulated = private_modulated + k * np.convolve(shared_input, exp_kernel, mode='same')

        # Normalize combined modulation
        modulated -= modulated.min()
        modulated /= modulated.max()

        # Generate spikes
        spike_probs = modulated - 0.9
        spikes[neuron] = np.random.rand(sim_time) < spike_probs

    return spikes



if __name__ == "__main__":
    # Define the bin size and number of bins for firing rate computation
    bin_size = 100  # ms
    bin_size = int(bin_size * sampling_rate / 1000)  # convert to time steps
    n_bins = int(sim_time / bin_size) # Number of bins
    
    # Preallocate arrays for storing simulation output
    sim_data = {
        'X': np.zeros((n_samples, n_bins)),
        'θ': np.zeros((n_samples, 1))
    }
    
    # Create the simulation dataset
    for sample in range(n_samples):
        print(f'Creating sample {sample + 1} of {n_samples}', end='\r', flush=True)
        # Generate a random parameter θ
        θ = np.random.uniform(0, 0.5)
        
        # Simulate the spike train
        spikes = simulator(θ)
        
        # Compute firing rates
        fr = [
            [np.sum(spikes[ii, jj * bin_size:(jj + 1) * bin_size])
             for jj in range(n_bins)]
             for ii in range(spikes.shape[0])
        ]
    
        # Create a FieldPotential object
        fp = ncpi.FieldPotential(kernel = False)    
        
        # Get the field potential proxy
        proxy = fp.compute_proxy(method = 'FR', sim_data = {'FR': fr}, sim_step = None)
        
        # Save simulation data 
        sim_data['X'][sample, :] = proxy
        sim_data['θ'][sample, 0] = θ
    
    # If sim_data['θ'] is a 2D array with one column, reshape it to a 1D array
    if sim_data['θ'].shape[1] == 1:
        sim_data['θ'] = np.reshape(sim_data['θ'], (-1,))
        
    # Compute features
    df = pd.DataFrame({'Data': sim_data['X'].tolist()})
    features = ncpi.Features(method='catch22')
    df = features.compute_features(df)
    
    # Split simulation data into 90% training and 10% test data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    X_train = np.array(df.iloc[train_indices].drop(columns=['Data'])['Features'].tolist())
    X_test = np.array(df.iloc[test_indices].drop(columns=['Data'])['Features'].tolist())
    θ_train = np.array(sim_data['θ'][train_indices])
    θ_test = np.array(sim_data['θ'][test_indices])
    
    # Create the inference object and add simulation data
    inference = ncpi.Inference(model='RandomForestRegressor',
                               hyperparams={'n_estimators': 100,
                                            'max_depth': 10,
                                            'min_samples_split': 2})
    inference.add_simulation_data(X_train, θ_train)
    
    # Create a scaler for the features
    scaler = StandardScaler()
    
    # Train the model
    sig = inspect.signature(inference.train)
    if "scaler" in sig.parameters:
        inference.train(param_grid=None, scaler=scaler)
        # Evaluate the model using the test data
        predictions = inference.predict(X_test, scaler=scaler)
    else:
        # Old package version: train() internally creates/uses a scaler already
        # so just call without scaler.
        inference.train(param_grid=None)
        predictions = inference.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(θ_test, predictions)

    
    # Plot real vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(θ_test, predictions, alpha=0.5, label=f'MSE = {mse:.3f}')
    plt.plot([θ_test.min(), θ_test.max()], [θ_test.min(), θ_test.max()], 'r--', label='Ideal Fit')
    plt.xlabel('True θ')
    plt.ylabel('Predicted θ')
    plt.legend()
    plt.show()

```

# Tutorials
If you're new to `ncpi`, we recommend starting with our 
[Getting Started](https://necolab-ugr.github.io/ncpi/tutorials/getting_started.html)
tutorial.

# Citation
If you use `ncpi` in your research, please consider citing our work:

**[1] Alejandro Orozco Valero, Víctor Rodríguez-González, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie, 
Francisco Pelayo, Christian Morillas, Jesús Poza, Carlos Gómez & Pablo Martínez-Cañada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9  

# Acknowledgements
This work was supported by grants PID2022-139055OA-I00 and PID2022-137461NB-C31,funded by MCIN/AEI/10.13039/501100011033 
and by “ERDF A way of making Europe”; and by “Junta de Andalucía” - Postdoctoral Fellowship Programme PAIDI 2021.
