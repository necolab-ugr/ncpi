import os
import pandas as pd
import pickle
import numpy as np
import ncpi
import shutil

sim_data_path = 'zenodo_sim_files/data/'
model_scaler_path = 'zenodo_sim_files/ML_models/4_param/MLP'

# Dataframe file upload format check
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'parquet', 'feather', 'pkl', 'pickle'}

# Check if the dataframe has an allowed extension
def allowed_file(filename):
    if not filename or '.' not in filename:
        return False

    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension[1:] in ALLOWED_EXTENSIONS # file_extension without the dot


def read_file_preprocessing(file_path):
     # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if file has allowed extension
    if not allowed_file(file_path):
        raise ValueError(
            f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower() # .csv

    return file_extension


def read_df_file(file_path):
    """ Read file as pandas dataframe """
    file_extension = read_file_preprocessing(file_path)

    try:
        # Read file as pandas dataframe based on extension
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_extension == '.feather':
            df = pd.read_feather(file_path)
        elif file_extension in ['.pkl', '.pickle']:
            df = pd.read_pickle(file_path)
        else:
            # This shouldn't happen if allowed_file() works correctly
            raise ValueError(f"Unsupported file format: {file_extension}")

        return df

    except Exception as e:
        # Re-raise with more context
        print(f"Error occurred: {type(e).__name__}: {e}")
        raise Exception(f"Failed to read file {file_path}: {type(e).__name__}: {str(e)}")


def read_file(file_path):
    """ Read file as file object """
    file_extension = read_file_preprocessing(file_path)

    try:
        # Read file as file object based on extension
        if file_extension in ['.pkl', '.pickle']:
            with open(os.path.join(file_path), 'rb') as file:
                file_object = pickle.load(file)
        elif file_extension == '.csv':
            # Load as numpy array
            file_object = np.loadtxt(os.path.join(file_path),  delimiter=',', skiprows=1)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return file_object

    except Exception as e:
        # Re-raise with more context
        print(f"Error occurred: {type(e).__name__}: {e}")
        raise Exception(f"Failed to read file {file_path}: {type(e).__name__}: {str(e)}")


def save_df(job_id, output_df, temp_uploaded_files):
    """ Saves the output dataframe to a pickle file and returns its name """
    output_df_name = f"{job_id}_output.pkl"
    output_path = f"{temp_uploaded_files}/{output_df_name}"
    output_df.to_pickle(output_path)

    return output_path


def cleanup_temp_files(file_paths):
    """Delete all temporary files in file_paths (params['file_paths']) silently."""    
    for file_path in file_paths.values():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Silently ignore errors




#############################################################
##########        COMPUTATION FUNCTIONS           ###########
#############################################################


def features_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        # Read the file path into a dataframe
        df = read_df_file(params['file_paths']['data_file'])

        # If select-method == 'power_spectrum_parameterization', prepare its parameters
        if params['select-method'] == 'power_spectrum_parameterization':
            fooof_setup_emp = {'peak_threshold': float(params['peak-threshold-foof']),
                            'min_peak_height': float(params['min-peak-height-foof']),
                            'max_n_peaks': int(params['max-peak-number-foof']),
                            'peak_width_limits': (float(params['peak-width-min-foof']), float(params['peak-width-max-foof']))}
            params_features ={'fs': int(df['fs'].iloc[0]),
                'fmin': float(params['min-freq-power']),
                'fmax': float(params['max-freq-power']),
                'fooof_setup': fooof_setup_emp,
                'r_squared_th': float(params['threshold-r-power'])}                
            df.Recording = 'LFP'
            df.fs = int(df['fs'].iloc[0])

        # Compute features from the dataframe
        features = ncpi.Features(method=params['select-method']) if params['select-method'] == 'catch22' else ncpi.Features(method=params['select-method'], params=params_features)
        output_df = features.compute_features(df)

        # Keep only the aperiodic exponent (1/f slope)
        if params['select-method'] == 'power_spectrum_parameterization':
            output_df['Features'] = output_df['Features'].apply(lambda x: x[1])

        # Save the output dataframe to a file
        output_df_path = save_df(job_id, output_df, temp_uploaded_files)

        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": output_df_path, # Return to the client the output filepath
                "error": False
            })

    except Exception as e:
        print(e)
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    # Remove the file after using it
    cleanup_temp_files(params['file_paths'])



def inference_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        # Read the files
        # if sim_X file wasn't uploaded, use the one in the server
        features_sim_path = params['file_paths']['features_sim']
        if not (allowed_file(features_sim_path)): 
            features_sim_path = os.path.join(sim_data_path, params['method'], 'sim_X.pkl')
        array_features_sim = read_file(features_sim_path) # sim_X.pkl

        # if sim_theta file wasn't uploaded, use the one in the server
        parameters_path = params['file_paths']['parameters']
        if not (allowed_file(parameters_path)): 
            parameters_path = os.path.join(sim_data_path, params['method'], 'sim_theta.pkl')
        array_parameters = read_file(parameters_path) # sim_theta.pkl

        df_features_predict = read_df_file(params['file_paths']['features_predict']) # features_results_lfp_catch22.pkl
        
        # Rename model and scaler files, make them readable for inference object
        model_path = params['file_paths']['model-file']
        if not (allowed_file(model_path)):
            model_path = os.path.join(model_scaler_path, params['method'], 'model.pkl')
        
        scaler_path = params['file_paths']['scaler-file']
        if not (allowed_file(scaler_path)):
            scaler_path = os.path.join(model_scaler_path, params['method'], 'scaler.pkl')
        
        shutil.copy(
            os.path.join(model_path),
            os.path.join(temp_uploaded_files, 'model.pkl')
        )
        shutil.copy(
            os.path.join(scaler_path),
            os.path.join(temp_uploaded_files, 'scaler.pkl')
        )
        
        # Column transformation to list (for parquet files)
        if isinstance(df_features_predict['Features'].iloc[0], np.ndarray):
            df_features_predict['Features'] = df_features_predict['Features'].apply(lambda x: x.tolist())

        # Compute inference depending on the example computation
        if params['example'] == 'lfp':
            emp_data = inference_lfp(params['model'], array_features_sim, array_parameters, df_features_predict)
        else: # eeg
            # Estimated waiting time will take much longer than the rest of the tasks
            job_status[job_id]["estimated_time_remaining"] = time.time() + 330 # 5:35 min power_spectrum, 10:30 min catch22
            emp_data = inference_eeg(params['model'], array_features_sim, array_parameters, df_features_predict)

        # Replace parameters of recurrent synaptic conductances with the ratio (E/I)_net
        E_I_net = emp_data['Predictions'].apply(lambda x: (x[0]/x[2]) / (x[1]/x[3]))
        others = emp_data['Predictions'].apply(lambda x: x[4:])
        emp_data['Predictions'] = (np.concatenate((E_I_net.values.reshape(-1,1),
                                                    np.array(others.tolist())), axis=1)).tolist()
        # Load inference (if EEG and LFP examples were the same)
        # inference = ncpi.Inference(model=params['model'])
        # inference.add_simulation_data(array_features_sim, array_parameters['data'])
        # # if (params['train-option'] == 'load'):
        # #     # My custom training code for inference
        # predictions = inference.predict(np.array(df_features_predict['Features'].to_list()), result_dir=temp_uploaded_files)

        # Save the output dataframe to a file
        output_df_path = save_df(job_id, emp_data)

        job_status[job_id].update({
                    "status": "finished",
                    "progress": 100,
                    "estimated_time_remaining": 0,
                    "results": output_df_path, # Return to the client the output filename
                    "error": False
                })
        
    except Exception as e:
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    # Remove the files after using them
    cleanup_temp_files(params['file_paths'])



def inference_lfp(ML_model, array_features_sim, array_parameters, df_features_predict, temp_uploaded_files):
    # Compute inference the way lfp does
    inference = ncpi.Inference(model=ML_model)
    inference.add_simulation_data(array_features_sim, array_parameters['data'])

    # Predict the parameters from the features of the empirical data. Model and scaler are searched in RESULT_DIR
    predictions = inference.predict(np.array(df_features_predict['Features'].tolist()), result_dir=temp_uploaded_files)

    # Append the predictions to the DataFrame
    pd_preds = pd.DataFrame({'Predictions': predictions})
    df_features_predict = pd.concat([df_features_predict, pd_preds], axis=1)
    return df_features_predict



def inference_eeg(ML_model, array_features_sim, array_parameters, df_features_predict, temp_uploaded_files):
    # Add "Predictions" column to later store the parameters infered
    df_features_predict['Predictions'] = np.nan

    # List of sensors
    sensor_list = [
        'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
        'O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

    # Create inference object
    inference = ncpi.Inference(model=ML_model)
    inference.add_simulation_data(array_features_sim, array_parameters['data'])

    for s, sensor in enumerate(sensor_list):
        print(f'--- Sensor: {sensor}')
        sensor_df = df_features_predict[df_features_predict['Sensor'].isin([sensor, s])]
        predictions = inference.predict(np.array(sensor_df['Features'].to_list()), result_dir=temp_uploaded_files)
        sensor_df['Predictions'] = [list(pred) for pred in predictions]
        df_features_predict.update(sensor_df['Predictions'])

    return df_features_predict



def analysis_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        # Save the image in temp_uploaded_files/LFP_predictions.png
        # LFP_predictions_webversion.run_full_pipeline([params['method-plot']], params['method'])
        
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": f'{temp_uploaded_files}/LFP_predictions.png', # Return to the client the output filepath
                "error": False
            })

    except Exception as e:
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    # Remove the file after using it
    cleanup_temp_files(params['file_paths'])