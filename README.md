> 🚧 **This repository is under active development. Features and docs may change.** 🚧

<div align="center">

# ncpi: neural circuit parameter inference
___

<img src="https://raw.githubusercontent.com/necolab-ugr/ncpi/main/img/ncpi_logo.png" alt="ncpi logo" width="150">

</div>

[Getting Started](https://necolab-ugr.github.io/ncpi/tutorials/getting_started.html) | [Documentation](https://necolab-ugr.github.io/ncpi/)

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

`ncpi` requires Python 3.10 or higher. To install `ncpi`, you can use pip. The package is available on PyPI, so you can 
install it directly from there.

```bash
# Step 1: Create and activate a conda environment (recommended)
conda create -n ncpi-env python=3.10 -y
conda activate ncpi-env

# Step 2: Install ncpi using pip
pip install ncpi
```

# Folder Structure

- `ncpi/`: Contains the source code for the library, organized into modules and classes.
- `examples/`: Includes scripts used to reproduce the results presented in the reference papers (see [Citation](#Citation)).
- `docs/`: Contains documentation for the library, including usage instructions, API references, and guides.
- `img/`: Stores image assets for the project, such as the library logo.


# Example Usage
```python
import ncpi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for generating simulated neural data
n_neurons = 50           # Number of neurons in each simulated recording
sim_time = 2000          # Total number of time points per sample 
sampling_rate = 100      # Sampling rate in Hz
n_samples = 1000        # Total number of samples (combined training + test set)

# Preallocate arrays for storing simulation output
sim_data = {
    'X': np.zeros((n_samples, int((sim_time + sampling_rate - 1) / sampling_rate))), 
    'θ': np.zeros((n_samples, 1))
}

def simulator(θ):
    """
    Simulates spike trains modulated by a synaptic-like process.
    
    This function generates synthetic neural spike trains by first creating
    a random input signal for each neuron and convolving it with an exponential
    kernel that mimics synaptic dynamics. The result is a smooth, temporally 
    correlated signal interpreted as the probability of spiking at each time step.

    Parameters
    ----------
    θ : float
        Controls the exponential decay of the synaptic kernel.

    Returns
    -------
    spikes : ndarray of shape (n_neurons, sim_time)
        Binary array representing spikes (1s) and no spikes (0s) for each neuron over time.
    """
    spikes = np.zeros((n_neurons, sim_time))

    # Define exponential kernel to model synaptic integration
    tau = sampling_rate * (θ + 0.1)   
    t_kernel = np.arange(int(sampling_rate * 4))  # Kernel length of 4 seconds
    exp_kernel = np.exp(-t_kernel / tau)

    for neuron in range(n_neurons):
        # Generate random input signal for this neuron
        raw_input = np.random.rand(sim_time)

        # Convolve input with synaptic kernel to create smooth modulated signal
        modulated = np.convolve(raw_input, exp_kernel, mode='same')

        # Normalize modulated signal to [0, 1] to use as spike probability
        modulated -= modulated.min()
        modulated /= modulated.max()
        spike_probs = modulated

        # Sample binary spikes based on spike probabilities
        spikes[neuron] = np.random.rand(sim_time) < spike_probs

    return spikes

if __name__ == "__main__":
    # Create the simulation dataset
    for sample in range(n_samples):
        print(f'Creating sample {sample + 1} of {n_samples}', end='\r', flush=True)
        # Generate a random parameter θ
        θ = np.random.uniform(0, 1)
        
        # Simulate the spike train
        spikes = simulator(θ)
        
        # Compute firing rates
        fr = [[np.sum(spikes[ii, jj * sampling_rate:(jj + 1) * sampling_rate]) 
               for jj in range(sim_data['X'].shape[1])]
               for ii in range(spikes.shape[0])]
    
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
    inference = ncpi.Inference(model='MLPRegressor', 
                               hyperparams={'hidden_layer_sizes': (50,50),
                                            'max_iter': 100,
                                            'tol': 1e-1,
                                            'n_iter_no_change': 5})
    inference.add_simulation_data(X_train, θ_train)
    
    # Train the model
    inference.train(param_grid=None)
    
    # Evaluate the model using the test data
    predictions = inference.predict(X_test)
    
    # Plot real vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(θ_test, predictions, alpha=0.5)
    plt.plot([θ_test.min(), θ_test.max()], [θ_test.min(), θ_test.max()], 'r--')
    plt.xlabel('True θ')
    plt.ylabel('Predicted θ')
    plt.show()

```

# Tutorials
If you're new to `ncpi`, we recommend starting with our [Getting Started](https://necolab-ugr.github.io/ncpi/)
tutorial.

# Citation
If you use `ncpi` in your research, please consider citing our work:

**[1] Alejandro Orozco Valero, Víctor Rodríguez-González, Noemi Montobbio, Miguel A. Casal, Alejandro Tlaie, 
Francisco Pelayo, Christian Morillas, Jesús Poza, Carlos Gómez & Pablo Martínez-Cañada**  
*A Python toolbox for neural circuit parameter inference.*  
npj Syst Biol Appl 11, 45 (2025).  
https://doi.org/10.1038/s41540-025-00527-9  

# Acknowledgements
This study was supported by grants PID2022-139055OA-I00, PID2022-137461NB-C31, and PID2022-138286NB-I00, 
funded by MCIN/AEI/10.13039/501100011033 and by “ERDF A way of making Europe”; by “Junta de Andalucía” - 
Postdoctoral Fellowship Programme PAIDI 2021; and by “CIBER en Bioingeniería, Biomateriales y Nanomedicina 
(CIBER-BBN), Spain” through “Instituto de Salud Carlos III” co-funded with ERDF funds.
