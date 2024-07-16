"""
To do.
"""
import importlib
import os
import sys
import pickle
import json
import scipy
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt

# ccpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ccpi

# simulation
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

def load_data(file_path):
    """Load data from a file path. The file should be a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_data_chunk(file_list_chunk, folder_path):
    """Load and process a chunk of files."""
    emp_data_chunk = {'LFP': [], 'fs': [], 'age': []}
    for file_name in file_list_chunk:
        structure = scipy.io.loadmat(os.path.join(folder_path, file_name))
        age = structure['LFP']['age'][0][0][0][0]
        fs = structure['LFP']['fs'][0][0][0][0]
        LFP = structure['LFP']['LFP'][0][0]
        sum_LFP = np.sum(LFP, axis=0)  # sum LFP across channels
        emp_data_chunk['LFP'].append(sum_LFP)
        emp_data_chunk['fs'].append(fs)
        emp_data_chunk['age'].append(age)
    return emp_data_chunk

def chunkify(lst, n):
    """Split lst into n chunks."""
    return [lst[i::n] for i in range(n)]

def load_empirical_data(folder_path):
    """Load empirical data in parallel."""
    file_list = os.listdir(folder_path)
    num_processes = multiprocessing.cpu_count()
    file_list_chunks = chunkify(file_list, num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(load_data_chunk, [(chunk, folder_path) for chunk in file_list_chunks])

    # Combine the results
    combined_emp_data = {'LFP': [], 'fs': [], 'age': []}
    for result in results:
        combined_emp_data['LFP'].extend(result['LFP'])
        combined_emp_data['fs'].extend(result['fs'])
        combined_emp_data['age'].extend(result['age'])

    return combined_emp_data


def compute_feature_for_chunk(data_chunk):
    """Helper function to compute features for a single chunk."""
    ccpi_feat = ccpi.Features(method=data_chunk['method'])
    chunk = data_chunk['chunk']
    age = data_chunk['age']
    mouse = data_chunk['mouse']
    # Normalize the chunk
    chunk = (chunk - np.mean(chunk)) / np.std(chunk)
    # Compute features
    catch22 = ccpi_feat.catch22(chunk)
    return catch22, age, mouse

def compute_features(data, chunk_size=500, method='catch22'):
    """Compute features from empirical data using multiprocessing."""
    if os.path.exists('data/emp_features.pkl'):
        with open('data/emp_features.pkl', 'rb') as file:
            return pickle.load(file)
    else:
        print('No features found. Computing features from scratch.')
        features = {'feature': [], 'age': [], 'mouse': []}

        # Prepare data chunks for parallel processing
        data_chunks = [{'chunk': data['LFP'][i][j:j + chunk_size], 'age': data['age'][i], 'mouse': i, 'method': method}
                       for i in range(len(data['LFP']))
                       for j in range(0, len(data['LFP'][i]) - chunk_size, chunk_size)]

        # Create a pool of workers
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        # Process data chunks in parallel
        results = pool.map(compute_feature_for_chunk, data_chunks)

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # Collect the results
        for catch22, age, mouse in results:
            features['feature'].append(catch22)
            features['age'].append(age)
            features['mouse'].append(mouse)

        # save and return the features
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/emp_features.pkl', 'wb') as file:
            pickle.dump(features, file)
        return features


def predict_parameters(estimator, features):
    """Predict parameters for empirical data."""
    predictions = {'prediction': [], 'age': [], 'mouse': []}

    for i in range(len(features['feature'])):
        print(f'\rPredicting parameters for chunk {i} / {len(features["feature"]) - 1}',
              end='', flush=True)
        feature = features['feature'][i]
        age = features['age'][i]
        mouse = features['mouse'][i]
        prediction = estimator.predict([feature])
        predictions['prediction'].append(prediction)
        predictions['age'].append(age)
        predictions['mouse'].append(mouse)

    return predictions


def compute_firing_rates(params, multi_compartment_model_path):
    """Run a simulation with the given parameters and return the firing rates of the excitatory population."""
    print(f'Computing firing rates for parameters: {params}')
    # create an object of the Simulation class
    Simulation = importlib.import_module('Simulation')
    sim = Simulation.Simulation(params, multi_compartment_model_path)
    # run a simulation
    folder = sim.run()
    # retrieve results of firing rates
    with open('LIF_simulations/'+folder+'/lif_mean_nu_X', 'r') as fr_file:
        fr = pickle.load(fr_file)
    # return the mean firing rate of the excitatory population
    return fr['E']


def plot_results(predictions, multi_compartment_model_path = ''):
    """Plot the results of the predictions."""
    # rearrange predictions for inserting E/I ratio
    new_predictions = np.array([
        [
            (p[0][0] / p[0][2]) / (p[0][1] / p[0][3]),  # E/I ratio calculation
            p[0][4],  # tau_exc
            p[0][5],  # tau_inh
            p[0][6]  # J_ext
        ] for p in predictions['prediction']
    ])

    # properties of the plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    dpi = 300

    # create the figure
    fig1 = plt.figure(figsize=[6, 4], dpi=dpi)
    titles = [r'$(E/I)_{net}$', r'$\tau_{exc}^{syn}$', r'$\tau_{inh}^{syn}$', r'$J_{ext}^{syn}$']

    # predictions plot
    for row in range(2):
        for col in range(2):
            ax = fig1.add_axes([0.12 + col * 0.45, 1.0 - (row + 1) * 0.42, 0.35, 0.35])
            # filter by age (for ages greater than 3 days)
            for age in np.unique(predictions['age']):
                idx = [i for i in range(len(predictions['age'])) if predictions['age'][i] == age and age >= 4]
                # boxplots of predictions
                if age >= 4:
                    ax.boxplot(new_predictions[idx, row * 2 + col], positions=[age], widths=0.9,
                               showfliers=False, boxprops=dict(facecolor="steelblue", alpha=0.35),
                               medianprops=dict(color="darkviolet", linewidth=1.), patch_artist=True)

            if row==1:
                ax.set_xlabel('Postnatal days')
            ax.set_ylabel(titles[row * 2 + col])
            ax.set_xticks(np.arange(4, 13, 2))
            ax.set_xticklabels([f'P{str(i)}' for i in np.arange(4, 13, 2)])

    # firing rates plot
    if multi_compartment_model_path != '':
        firing_rates = []
        # change dir
        cwd = os. getcwd()
        os.chdir('simulation')
        # filter by age (for ages greater than 3 days)
        for age in np.unique(predictions['age']):
            idx = [i for i in range(len(predictions['age'])) if predictions['age'][i] == age and age >= 4]
            # firing rates of median values
            if age >= 4:
                fr = compute_firing_rates(np.median(np.array(predictions['prediction'])[idx, :, :], axis=0).flatten(),
                                          multi_compartment_model_path)
                firing_rates.append(fr)
        # restore working directory
        os.chdir(cwd)
        # plot firing rates
        fig2 = plt.figure(figsize=[4, 3], dpi=dpi)
        ax = fig2.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.plot(np.unique(predictions['age'])[np.unique(predictions['age']) >= 4],
                firing_rates, 'o-', color='black')
        ax.set_xlabel('Postnatal days')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_xticks(np.arange(4, 13, 2))
        ax.set_xticklabels([f'P{str(i)}' for i in np.arange(4, 13, 2)])

    plt.show()


def main():
    # Load the configuration file that stores all file paths used in the script
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_features_path']
    emp_data_path = config['LFP_development_data_path']
    # multi_comp_path = config['multi_compartment_model_path']

    # Load parameters of the model (theta) and features from simulation data (X)
    print('\n--- Loading simulation data.')
    start_time = time.time()
    theta = load_data(os.path.join(sim_file_path, 'sim_theta'))
    X = load_data(os.path.join(sim_file_path, 'sim_X'))
    end_time = time.time()
    print(f'Samples loaded: {len(theta["data"])}')
    print(f'Done in {end_time - start_time} s')

    # Train a regression model (surrogate model) to predict the parameters of the model
    print('\n--- Training the regression model.')
    start_time = time.time()
    inference = ccpi.Inference(X, theta['data'])
    best_estimator = inference.train_model()
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Load empirical data
    print('\n--- Loading empirical data.')
    start_time = time.time()
    emp_data = load_empirical_data(emp_data_path)
    end_time = time.time()
    print(f'Data files loaded: {len(emp_data["LFP"])}')
    print(f'Done in {end_time - start_time} s')

    # Compute features from empirical data using multiprocessing
    print('\n--- Computing features from empirical data.')
    start_time = time.time()
    emp_features = compute_features(emp_data, chunk_size=500, method='catch22')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    # Compute parameter predictions for empirical data
    print('\n--- Predicting parameters for empirical data.')
    start_time = time.time()
    predictions = predict_parameters(best_estimator, emp_features)
    end_time = time.time()
    print(f'\nDone in {end_time - start_time} s')

    # Plot the results
    print('\n--- Plotting the results.')
    start_time = time.time()
    plot_results(predictions, '')
    end_time = time.time()
    print(f'Done in {end_time - start_time} s')

    return theta, X, best_estimator, emp_data, emp_features, predictions


if __name__ == "__main__":
    theta, X, best_estimator, emp_data, emp_features, predictions = main()
