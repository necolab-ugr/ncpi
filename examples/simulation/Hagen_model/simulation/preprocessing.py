import json
import os
import pickle
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


def process_batch(ldir):
    """
    Load simulation data and create the data structures of normalized CDMs and
    LIF network model parameters.

    Parameters
    ----------
    ldir: list of string
        Absolute paths of the folders that contain the simulation data.

    Returns
    -------
    CDM_data: nested list
        Normalized CDMs.
    theta_data: nested list
        Synapse parameters of the LIF network model.
    """
    theta_data = []
    CDM_data = []

    for folder in ldir:
        # Load and sum CDMs for all combinations of populations (EE, EI, IE, II)
        # if CDM_data file is a dict
        try:
            cdm = pickle.load(open(os.path.join(folder,"CDM_data"),'rb'))
            if isinstance(cdm, dict):
                cdm_sum = cdm['EE'] + cdm['EI'] + cdm['IE'] + cdm['II']
            else:
                cdm_sum = cdm

            # Dismiss CDMs that are constant over time
            if np.std(cdm_sum) > 10 ** (-10):
                # Remove the first 500 samples containing the transient response (not really necessary as some
                # transient time is already removed when computing the CDMs)
                cdm_sum = cdm_sum[500:]
                # Normalization
                CDM_data.append((cdm_sum - np.mean(cdm_sum)) / np.std(cdm_sum))
                # Collect synapse parameters of recurrent connections and
                # external input
                try:
                    LIF_params = pickle.load(open(os.path.join(
                        folder, "LIF_params"), 'rb'))
                    theta_data.append([LIF_params['J_YX'][0][0],
                                       LIF_params['J_YX'][0][1],
                                       LIF_params['J_YX'][1][0],
                                       LIF_params['J_YX'][1][1],
                                       LIF_params['tau_syn_YX'][0][0],
                                       LIF_params['tau_syn_YX'][0][1],
                                       LIF_params['J_ext']
                                       ])

                except (FileNotFoundError, IOError):
                    print(f'File LIF_params not found in {folder}')

        except (FileNotFoundError, IOError):
            print(f'File CDM_data not found in {folder}')

    return CDM_data, theta_data


if __name__ == '__main__':
    # Path to the folders containing the simulation data
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    sim_file_path = config['simulation_raw_data_path']

    # List of all the folders containing the simulation data (there are three folders that correspond
    # to the different computing environments used to run the simulations)
    folder1 = os.path.join(sim_file_path, 'LIF_simulations')
    folder2 = os.path.join(sim_file_path, 'LIF_simulations_hpmoon','LIF_simulations')
    folder3 = os.path.join(sim_file_path, 'LIF_simulations_hpc','LIF_simulations')

    ldir = [os.path.join(folder1, f) for f in os.listdir(folder1)] + \
           [os.path.join(folder2, f) for f in os.listdir(folder2)] + \
           [os.path.join(folder3, f) for f in os.listdir(folder3)]

    # Dictionary to store parameters of the LIF network model
    theta_data = {'parameters':['J_EE',
                                'J_IE',
                                'J_EI',
                                'J_II',
                                'tau_syn_E',
                                'tau_syn_I',
                                'J_ext'],
                  'data': []}
    # Current Dipole Moment (CDM) data
    CDM_data = []

    # Split the list of folders into batches
    batch_size_1 = len(ldir) // 5 # 5 is a factor to avoid memory issues
                                  # (set it to a smaller size if your system has sufficient memory)
    batches_1 = [ldir[i:i + batch_size_1] for i in range(0, len(ldir), batch_size_1)]

    # Preprocess data in parallel using all available CPUs
    num_cpus = os.cpu_count()
    for ii, batch in enumerate(batches_1):
        print(f"Processing batch {ii+1}/{len(batches_1)}")
        batch_size_2 = len(batch) // (num_cpus*10) # 10 is a factor to update the progress bar more frequently
        if batch_size_2 < 1:
            batch_size_2 = 1
        batches_2 = [batch[i:i + batch_size_2] for i in range(0, len(batch), batch_size_2)]

        with Pool(num_cpus) as pool:
            results = list(tqdm(pool.imap(process_batch, batches_2),
                                total=len(batches_2), desc="Processing data"))

        # Collect the results
        for result in results:
            CDM_data.extend(result[0])
            theta_data['data'].extend(result[1])

        # Transform to numpy arrays
        theta_data['data'] = np.array(theta_data['data'],dtype="float32")
        CDM_data = np.array(CDM_data,dtype="float32")

        print(f"Number of simulations in the batch: {len(batch)}")
        print(f"Number of samples processed: {CDM_data.shape[0]}\n")

        # Path to the folder containing the processed data
        sim_file_path = config['simulation_processed_data_path']

        # Create folders if they do not exist
        splits = os.path.split(sim_file_path)
        if not os.path.isdir(splits[0]):
            os.mkdir(splits[0])
        if not os.path.isdir(sim_file_path):
            os.mkdir(sim_file_path)

        # Save numpy arrays to file
        pickle.dump(theta_data,open(os.path.join(sim_file_path,f'theta_data_{ii}'),'wb'))
        pickle.dump(CDM_data,open(os.path.join(sim_file_path,f'CDM_data_{ii}'),'wb'))

        # Clear memory
        theta_data['data'] = []
        CDM_data = []
        results = []
