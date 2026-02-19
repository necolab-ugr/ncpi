import os
import glob
import pandas as pd
import pickle
import numpy as np
import ncpi
import shutil
import ast
import importlib.util

sim_data_path = 'zenodo_sim_files/data/'
model_scaler_path = 'zenodo_sim_files/ML_models/4_param/MLP'
DEFAULT_SIM_DATA_DIR = '/tmp/simulation_data'
MAX_OUTPUT_LINES = 200

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


def _resolve_sim_file(file_paths, key, default_name, required=True):
    uploaded_path = file_paths.get(key)
    if uploaded_path and allowed_file(uploaded_path) and os.path.exists(uploaded_path):
        return uploaded_path

    if default_name:
        default_path = os.path.join(DEFAULT_SIM_DATA_DIR, default_name)
        if os.path.exists(default_path):
            return default_path

    if required:
        raise FileNotFoundError(
            f"Missing required input '{key}'. Upload a file or place {default_name} in {DEFAULT_SIM_DATA_DIR}."
        )
    return None


def _append_job_output(job_status, job_id, message):
    if job_id not in job_status:
        return
    if not message.endswith("\n"):
        message += "\n"
    current = job_status[job_id].get("output", "")
    if current and not current.endswith("\n"):
        current += "\n"
    combined = current + message
    lines = combined.splitlines()[-MAX_OUTPUT_LINES:]
    job_status[job_id]["output"] = "\n".join(lines)


def _load_module_from_path(path, name="kernel_params"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_literal_value(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list, tuple, float, int, bool)):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default if default is not None else value


def _parse_bool(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _compute_mean_firing_rate(times, gids, bin_size_ms):
    times = _flatten_spike_data(times)
    gids = _flatten_spike_data(gids)
    if times.size == 0:
        return np.zeros((1, 0))
    if bin_size_ms <= 0:
        raise ValueError("bin_size must be a positive number (ms).")

    n_units = max(int(np.unique(gids).size), 1) if gids.size > 0 else 1
    t_min = float(np.min(times))
    t_max = float(np.max(times))
    bins = np.arange(t_min, t_max + bin_size_ms, bin_size_ms)
    hist, _ = np.histogram(times, bins=bins)
    mean_rate = hist.astype(float) / float(n_units)
    return mean_rate.reshape(1, -1)


def _flatten_spike_data(data):
    if data is None:
        return np.asarray([])
    if isinstance(data, dict):
        values = [np.asarray(v).ravel() for v in data.values()]
        return np.concatenate(values) if values else np.asarray([])
    if isinstance(data, (list, tuple)):
        if not data:
            return np.asarray([])
        if all(np.isscalar(x) for x in data):
            return np.asarray(data).ravel()
        values = [np.asarray(v).ravel() for v in data]
        return np.concatenate(values) if values else np.asarray([])
    return np.asarray(data).ravel()




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


def field_potential_proxy_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting field potential proxy computation.")
        method = params.get('proxy_method', 'FR')
        _append_job_output(job_status, job_id, f"Proxy method: {method}")
        sim_step_value = params.get('sim_step')
        sim_step = float(sim_step_value) if sim_step_value not in (None, '') else None
        bin_size_value = params.get('bin_size')
        bin_size = float(bin_size_value) if bin_size_value not in (None, '') else 1.0

        excitatory_only_value = params.get('excitatory_only', 'default').lower()
        if excitatory_only_value == 'true':
            excitatory_only = True
        elif excitatory_only_value == 'false':
            excitatory_only = False
        else:
            excitatory_only = None

        file_paths = params.get('file_paths', {})
        sim_data = {}

        if method == 'FR':
            _append_job_output(job_status, job_id, "Loading spike times and gids...")
            times_path = _resolve_sim_file(file_paths, 'times_file', 'times.pkl', required=True)
            gids_path = _resolve_sim_file(file_paths, 'gids_file', 'gids.pkl', required=True)
            times = read_file(times_path)
            gids = read_file(gids_path)
            sim_data['FR'] = _compute_mean_firing_rate(times, gids, bin_size)

        elif method == 'AMPA':
            _append_job_output(job_status, job_id, "Loading AMPA currents...")
            ampa_path = _resolve_sim_file(file_paths, 'ampa_file', 'ampa.pkl', required=True)
            sim_data['AMPA'] = read_file(ampa_path)

        elif method == 'GABA':
            _append_job_output(job_status, job_id, "Loading GABA currents...")
            gaba_path = _resolve_sim_file(file_paths, 'gaba_file', 'gaba.pkl', required=True)
            sim_data['GABA'] = read_file(gaba_path)

        elif method == 'Vm':
            _append_job_output(job_status, job_id, "Loading membrane potentials...")
            vm_path = _resolve_sim_file(file_paths, 'vm_file', 'vm.pkl', required=True)
            sim_data['Vm'] = read_file(vm_path)

        elif method in {'I', 'I_abs', 'LRWS', 'ERWS1', 'ERWS2'}:
            _append_job_output(job_status, job_id, "Loading AMPA and GABA currents...")
            ampa_path = _resolve_sim_file(file_paths, 'ampa_file', 'ampa.pkl', required=True)
            gaba_path = _resolve_sim_file(file_paths, 'gaba_file', 'gaba.pkl', required=True)
            sim_data['AMPA'] = read_file(ampa_path)
            sim_data['GABA'] = read_file(gaba_path)

            if method == 'ERWS2':
                _append_job_output(job_status, job_id, "Loading nu_ext...")
                nu_ext_path = _resolve_sim_file(file_paths, 'nu_ext_file', 'nu_ext.pkl', required=False)
                nu_ext_value = params.get('nu_ext_value')
                if nu_ext_path:
                    sim_data['nu_ext'] = read_file(nu_ext_path)
                elif nu_ext_value not in (None, ''):
                    sim_data['nu_ext'] = float(nu_ext_value)
                else:
                    raise FileNotFoundError(
                        f"Missing nu_ext. Upload a file or provide a value, or place nu_ext.pkl in {DEFAULT_SIM_DATA_DIR}."
                    )
        else:
            raise ValueError(f"Unknown proxy method '{method}'.")

        _append_job_output(job_status, job_id, "Computing proxy with ncpi.FieldPotential.compute_proxy...")
        potential = ncpi.FieldPotential()
        proxy = potential.compute_proxy(method, sim_data, sim_step, excitatory_only=excitatory_only)

        output_root = '/tmp/field_potential_proxy'
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)

        sim_data_path = os.path.join(run_dir, 'sim_data.pkl')
        proxy_path = os.path.join(run_dir, 'proxy.pkl')

        with open(sim_data_path, 'wb') as f:
            pickle.dump(sim_data, f)
        with open(proxy_path, 'wb') as f:
            pickle.dump(proxy, f)

        _append_job_output(job_status, job_id, f"Saved sim_data.pkl and proxy.pkl to {run_dir}")
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": proxy_path,
                "error": False
            })

    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })

    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_kernel_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting kernel computation.")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_mc_folder = os.path.expandvars(
            os.path.expanduser(os.path.join("$HOME", "multicompartment_neuron_network"))
        )
        default_output_sim = os.path.join(
            default_mc_folder, "output", "adb947bfb931a5a8d09ad078a6d256b0"
        )
        default_params_path = os.path.join(
            repo_root,
            "examples",
            "simulation",
            "Hagen_model",
            "simulation",
            "params",
            "analysis_params.py",
        )

        mc_folder = params.get("mc_folder") or default_mc_folder
        output_sim_path = params.get("output_sim_path") or default_output_sim
        params_path = params.get("kernel_params_module") or default_params_path

        _append_job_output(job_status, job_id, f"MC folder: {mc_folder}")
        _append_job_output(job_status, job_id, f"Params module: {params_path}")
        _append_job_output(job_status, job_id, f"Output simulation path: {output_sim_path}")

        module = _load_module_from_path(params_path, name="kernel_params")
        KernelParams = getattr(module, "KernelParams", None)
        if KernelParams is None:
            raise AttributeError("KernelParams not found in params module.")
        pop_names = getattr(KernelParams, "population_names", None)
        pop_count = len(pop_names) if pop_names is not None else None

        dt = params.get("dt")
        dt = float(dt) if dt not in (None, "") else KernelParams.networkParameters.get("dt", 0.0625)
        tstop = params.get("tstop")
        tstop = float(tstop) if tstop not in (None, "") else KernelParams.networkParameters.get("tstop", 12000.0)
        t_X = params.get("t_x")
        t_X = float(t_X) if t_X not in (None, "") else getattr(KernelParams, "transient", None)
        tau = params.get("tau")
        tau = float(tau) if tau not in (None, "") else getattr(KernelParams, "tau", None)

        biophys = _parse_literal_value(params.get("biophys"), None)
        if biophys is None:
            biophys = ["set_Ih_linearized_hay2011", "make_cell_uniform"]

        g_eff = _parse_bool(params.get("g_eff"), KernelParams.MC_params.get("g_eff", None))
        n_ext = _parse_literal_value(params.get("n_ext"), KernelParams.MC_params.get("n_ext", None))
        if isinstance(n_ext, (int, float, np.integer, np.floating)) and pop_count:
            n_ext = [int(n_ext)] * pop_count

        weights = _parse_literal_value(params.get("weights"), None)
        if weights is None:
            weight_vals = [
                params.get("weight_ee"),
                params.get("weight_ie"),
                params.get("weight_ei"),
                params.get("weight_ii"),
            ]
            if all(v not in (None, "") for v in weight_vals):
                weights = [
                    [float(weight_vals[0]), float(weight_vals[1])],
                    [float(weight_vals[2]), float(weight_vals[3])],
                ]

        file_paths = params.get("file_paths", {})
        probe_selection_present = params.get("probe_selection_present") not in (None, "")
        probe_kernel_approx = _parse_bool(params.get("probe_kernel_approx"), False)
        probe_current_dipole = _parse_bool(params.get("probe_current_dipole"), False)
        probe_gauss_cylinder = _parse_bool(params.get("probe_gauss_cylinder"), False)
        if probe_kernel_approx and probe_current_dipole:
            raise ValueError("Select only one of KernelApproxCurrentDipoleMoment or CurrentDipoleMoment.")

        electrodeParameters = None
        electrode_path = file_paths.get("electrode_parameters_file")
        if electrode_path:
            electrodeParameters = read_file(electrode_path)
        else:
            electrode_value = params.get("electrode_parameters")
            if electrode_value not in (None, ""):
                if electrode_value.strip() == "KernelParams.electrodeParameters":
                    electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
                else:
                    electrodeParameters = _parse_literal_value(electrode_value, None)
            else:
                electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
        if probe_selection_present and not probe_gauss_cylinder:
            electrodeParameters = None
        if probe_selection_present and probe_gauss_cylinder and electrodeParameters is None:
            electrodeParameters = getattr(KernelParams, "electrodeParameters", None)
            if electrodeParameters is None:
                raise ValueError("electrodeParameters must be provided when GaussCylinderPotential is selected.")

        if probe_selection_present:
            cdm = probe_kernel_approx or probe_current_dipole
            if probe_kernel_approx:
                cdm_probe = "KernelApproxCurrentDipoleMoment"
            elif probe_current_dipole:
                cdm_probe = "CurrentDipoleMoment"
            else:
                cdm_probe = None
        else:
            cdm_probe = params.get("cdm_probe") or "KernelApproxCurrentDipoleMoment"
            cdm = _parse_bool(params.get("cdm"), True)
        probes = _parse_literal_value(params.get("probes"), None)
        selected_probe_names = []
        if probe_selection_present:
            if probe_current_dipole:
                selected_probe_names.append("CurrentDipoleMoment")
            if probe_kernel_approx:
                selected_probe_names.append("KernelApproxCurrentDipoleMoment")
            if probe_gauss_cylinder:
                selected_probe_names.append("GaussCylinderPotential")
            if not selected_probe_names:
                raise ValueError("Select at least one probe for CDM/LFP computation.")
        else:
            fallback_probe = params.get("cdm_probe_name") or params.get("cdm_probe")
            if fallback_probe:
                selected_probe_names = [fallback_probe]

        mean_nu_x = None
        vrest = None
        mean_nu_value = params.get("mean_nu_x")
        if mean_nu_value in (None, ""):
            mean_nu_value = params.get("mean_nu_x_value")
        if mean_nu_value not in (None, ""):
            parsed = _parse_literal_value(mean_nu_value, None)
            if isinstance(parsed, dict):
                mean_nu_x = parsed
            elif isinstance(parsed, (list, tuple, np.ndarray)):
                if pop_names and len(parsed) == len(pop_names):
                    mean_nu_x = {name: float(parsed[i]) for i, name in enumerate(pop_names)}
                else:
                    mean_nu_x = parsed
            elif isinstance(parsed, (int, float, np.integer, np.floating)):
                if pop_names:
                    mean_nu_x = {name: float(parsed) for name in pop_names}
                else:
                    mean_nu_x = float(parsed)
        else:
            mean_nu_path = params.get("mean_nu_x_path")
            if mean_nu_path:
                mean_nu_x = read_file(mean_nu_path)
        vrest_path = params.get("vrest_path")
        if vrest_path:
            vrest = read_file(vrest_path)
        vrest_value = params.get("vrest_value")
        if vrest is None and vrest_value not in (None, ""):
            vrest = float(vrest_value)
        if mean_nu_x is None and vrest is None:
            mean_nu_x = None
            vrest = None

        if not output_sim_path or not os.path.exists(output_sim_path):
            if mean_nu_x is None or vrest is None:
                raise FileNotFoundError(
                    "Output simulation path not found. Provide mean_nu_X and Vrest or a valid output_sim_path."
                )
            output_sim_path = None

        _append_job_output(job_status, job_id, "Computing kernels with ncpi.FieldPotential.create_kernel...")
        potential = ncpi.FieldPotential()
        kernels = potential.create_kernel(
            mc_folder,
            KernelParams,
            biophys,
            dt,
            tstop,
            output_sim_path=output_sim_path,
            electrodeParameters=electrodeParameters,
            CDM=cdm,
            probes=probes,
            cdm_probe=cdm_probe,
            mean_nu_X=mean_nu_x,
            Vrest=vrest,
            t_X=t_X,
            tau=tau,
            g_eff=g_eff,
            n_ext=n_ext,
            weights=weights,
        )

        output_root = "/tmp/field_potential_kernel"
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)
        kernels_path = os.path.join(run_dir, "kernels.pkl")
        with open(kernels_path, "wb") as f:
            pickle.dump(kernels, f)

        _append_job_output(job_status, job_id, f"Saved kernels to {kernels_path}")

        _append_job_output(job_status, job_id, "Computing CDM/LFP from kernels...")
        spike_times_path = _resolve_sim_file(
            file_paths, "kernel_spike_times_file", "times.pkl", required=True
        )
        spike_times = read_file(spike_times_path)

        population_sizes = None
        pop_path = _resolve_sim_file(
            file_paths, "kernel_population_sizes_file", "population_sizes.pkl", required=False
        )
        if pop_path:
            population_sizes = read_file(pop_path)

        cdm_dt = params.get("cdm_dt")
        cdm_dt = float(cdm_dt) if cdm_dt not in (None, "") else dt
        cdm_tstop = params.get("cdm_tstop")
        cdm_tstop = float(cdm_tstop) if cdm_tstop not in (None, "") else tstop
        cdm_transient = params.get("cdm_transient")
        if cdm_transient not in (None, ""):
            transient = float(cdm_transient)
        elif t_X not in (None, ""):
            transient = float(t_X)
        else:
            transient = 0.0
        probe_names = selected_probe_names or ["KernelApproxCurrentDipoleMoment"]
        component_val = params.get("cdm_component")
        component = None if component_val in (None, "", "None") else int(float(component_val))
        mode = params.get("cdm_mode") or "same"
        scale_val = params.get("cdm_scale")
        scale = float(scale_val) if scale_val not in (None, "") else 1.0
        aggregate_val = params.get("cdm_aggregate")
        aggregate = aggregate_val if aggregate_val else None
        probe_output_map = {
            "KernelApproxCurrentDipoleMoment": "kernel_approx_cdm.pkl",
            "CurrentDipoleMoment": "current_dipole_moment.pkl",
            "GaussCylinderPotential": "gauss_cylinder_potential.pkl",
        }

        probe_outputs = {}
        output_paths = {}
        for probe_name in probe_names:
            cdm_signals = potential.compute_cdm_lfp_from_kernels(
                kernels,
                spike_times,
                cdm_dt,
                cdm_tstop,
                population_sizes=population_sizes,
                transient=transient,
                probe=probe_name,
                component=component,
                mode=mode,
                scale=scale,
                aggregate=aggregate,
            )
            probe_outputs[probe_name] = cdm_signals

            output_name = probe_output_map.get(probe_name)
            if not output_name:
                safe_probe = "".join(ch.lower() if ch.isalnum() else "_" for ch in probe_name).strip("_")
                output_name = f"{safe_probe or 'probe_output'}.pkl"
            output_path = os.path.join(run_dir, output_name)
            with open(output_path, "wb") as f:
                pickle.dump(cdm_signals, f)
            output_paths[probe_name] = output_path
            _append_job_output(job_status, job_id, f"Saved probe output to {output_path}")

        results_path = None
        if len(probe_outputs) == 1:
            results_path = next(iter(output_paths.values()))
        else:
            combined_path = os.path.join(run_dir, "probe_outputs.pkl")
            with open(combined_path, "wb") as f:
                pickle.dump(probe_outputs, f)
            results_path = combined_path
            _append_job_output(job_status, job_id, f"Saved combined probe outputs to {combined_path}")

        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": results_path,
                "error": False
            })
    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })
    cleanup_temp_files(params.get('file_paths', {}))


def field_potential_meeg_computation(job_id, job_status, params, temp_uploaded_files):
    try:
        _append_job_output(job_status, job_id, "Starting M/EEG computation.")
        job_status[job_id]["progress"] = 5
        file_paths = params.get("file_paths", {})

        cdm_path = file_paths.get("meeg_cdm_file")
        if not cdm_path or not os.path.exists(cdm_path):
            preferred = glob.glob(os.path.join("/tmp/field_potential_kernel", "*", "current_dipole_moment.pkl"))
            if preferred:
                cdm_path = max(preferred, key=os.path.getmtime)
            else:
                fallback = glob.glob(os.path.join("/tmp/field_potential_kernel", "*", "kernel_approx_cdm.pkl"))
                if fallback:
                    cdm_path = max(fallback, key=os.path.getmtime)
        if not cdm_path or not os.path.exists(cdm_path):
            raise FileNotFoundError("CDM input is required (upload .pkl or compute kernels first).")
        CDM = read_file(cdm_path)
        job_status[job_id]["progress"] = 20
        if isinstance(CDM, dict):
            if "sum" in CDM:
                _append_job_output(job_status, job_id, "CDM input is a dict; using 'sum' entry.")
                CDM = CDM["sum"]
            elif len(CDM) == 1:
                key = next(iter(CDM.keys()))
                _append_job_output(job_status, job_id, f"CDM input is a dict; using '{key}' entry.")
                CDM = CDM[key]
            else:
                _append_job_output(job_status, job_id, "CDM input has multiple entries; summing all keys for M/EEG.")
                total = None
                for value in CDM.values():
                    arr = np.asarray(value)
                    if total is None:
                        total = np.array(arr, copy=True)
                    else:
                        total = total + arr
                CDM = total
        CDM = np.asarray(CDM)
        if CDM.ndim == 2 and CDM.shape[0] != 3 and CDM.shape[1] == 3:
            CDM = CDM.T
        if CDM.ndim == 3 and CDM.shape[1] != 3 and CDM.shape[2] == 3:
            CDM = np.transpose(CDM, (0, 2, 1))
        if CDM.ndim == 1:
            _append_job_output(job_status, job_id, "CDM is 1D; assuming z-axis dipole (x=y=0).")
            CDM = np.vstack([np.zeros_like(CDM), np.zeros_like(CDM), CDM])
        elif CDM.ndim == 2 and CDM.shape[0] != 3:
            _append_job_output(job_status, job_id, "CDM has 1 component per dipole; assuming z-axis dipoles (x=y=0).")
            zeros = np.zeros_like(CDM)
            CDM = np.stack([zeros, zeros, CDM], axis=1)
        if not ((CDM.ndim == 2 and CDM.shape[0] == 3) or (CDM.ndim == 3 and CDM.shape[1] == 3)):
            raise ValueError(
                "CDM must have 3 components. Ensure you computed a dipole-moment probe (e.g., "
                "KernelApproxCurrentDipoleMoment or CurrentDipoleMoment), not a scalar potential."
            )

        dipole_locations = None
        dipole_path = file_paths.get("meeg_dipole_file")
        if dipole_path and os.path.exists(dipole_path):
            dipole_locations = read_file(dipole_path)
        else:
            dipole_text = params.get("meeg_dipole_locations")
            dipole_locations = _parse_literal_value(dipole_text, None)

        sensor_locations = None
        sensor_path = file_paths.get("meeg_sensor_file")
        if sensor_path and os.path.exists(sensor_path):
            sensor_locations = read_file(sensor_path)
        else:
            sensor_text = params.get("meeg_sensor_locations")
            sensor_locations = _parse_literal_value(sensor_text, None)

        model = params.get("meeg_model") or "NYHeadModel"
        model_kwargs = _parse_literal_value(params.get("meeg_model_kwargs"), None)
        align_to_surface = _parse_bool(params.get("meeg_align_to_surface"), True)
        auto_1020 = _parse_bool(params.get("meeg_auto_1020"), False)

        _append_job_output(job_status, job_id, f"Model: {model}")
        job_status[job_id]["progress"] = 45
        potential = ncpi.FieldPotential()
        if auto_1020 and model == "NYHeadModel":
            dipole_locations, _ = potential._get_eeg_1020_locations()
            dipole_locations = np.asarray(dipole_locations, dtype=float)
            n_dip = 1 if CDM.ndim == 2 else int(CDM.shape[0])
            if n_dip == 1:
                dipole_locations = dipole_locations[0]
                _append_job_output(job_status, job_id, "Using first EEG 10–20 dipole location for NYHeadModel.")
            else:
                if dipole_locations.shape[0] < n_dip:
                    needed = n_dip - dipole_locations.shape[0]
                    tail = np.repeat(dipole_locations[-1:], needed, axis=0)
                    dipole_locations = np.vstack([dipole_locations, tail])
                    _append_job_output(
                        job_status,
                        job_id,
                        "EEG 10–20 locations fewer than dipoles; repeating last location to match count.",
                    )
                dipole_locations = dipole_locations[:n_dip]
                _append_job_output(job_status, job_id, "Using EEG 10–20 dipole locations for NYHeadModel.")
            sensor_locations = None
        elif model == "NYHeadModel":
            sensor_locations = None
        else:
            if sensor_locations is None:
                if model in {"FourSphereVolumeConductor", "InfiniteVolumeConductor"}:
                    sensor_locations = np.array([[0.0, 0.0, 90000.0]])
                elif model == "InfiniteHomogeneousVolCondMEG":
                    sensor_locations = np.array([[10000.0, 0.0, 0.0]])
                elif model == "SphericallySymmetricVolCondMEG":
                    sensor_locations = np.array([[0.0, 0.0, 92000.0]])

        if dipole_locations is None:
            if model == "FourSphereVolumeConductor":
                dipole_locations = np.array([0.0, 0.0, 78000.0])
            elif model == "SphericallySymmetricVolCondMEG":
                dipole_locations = np.array([0.0, 0.0, 90000.0])
            else:
                dipole_locations = np.zeros(3)

        p_list, loc_list = potential._normalize_cdm_and_locations(CDM, dipole_locations)
        model_kwargs = model_kwargs or {}
        is_meg = model in {"InfiniteHomogeneousVolCondMEG", "SphericallySymmetricVolCondMEG"}

        matrices = []
        p_use_list = []
        n_sensors = 0
        if model == "NYHeadModel":
            if potential.nyhead is None:
                nyhead_model = potential._load_eegmegcalc_model("NYHeadModel")
                potential.nyhead = nyhead_model(**model_kwargs) if model_kwargs else nyhead_model()
            for p_i, loc_i in zip(p_list, loc_list):
                potential.nyhead.set_dipole_pos(loc_i)
                M = potential.nyhead.get_transformation_matrix()
                p_use = potential.nyhead.rotate_dipole_to_surface_normal(p_i) if align_to_surface else p_i
                matrices.append(M)
                p_use_list.append(p_use)
            if matrices:
                n_sensors = matrices[0].shape[0]
        else:
            if sensor_locations is None:
                raise ValueError("sensor_locations must be provided for this model.")
            sensor_locations = np.asarray(sensor_locations, dtype=float)
            if sensor_locations.ndim != 2 or sensor_locations.shape[1] != 3:
                raise ValueError("sensor_locations must have shape (n_sensors, 3).")
            if model == "FourSphereVolumeConductor":
                FourSphere = potential._load_eegmegcalc_model("FourSphereVolumeConductor")
                model_obj = FourSphere(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            elif model == "InfiniteVolumeConductor":
                InfiniteVol = potential._load_eegmegcalc_model("InfiniteVolumeConductor")
                model_obj = InfiniteVol(**model_kwargs)
                def get_M(loc):
                    r = sensor_locations - loc
                    return model_obj.get_transformation_matrix(r)
            elif model == "InfiniteHomogeneousVolCondMEG":
                IHVCMEG = potential._load_eegmegcalc_model("InfiniteHomogeneousVolCondMEG")
                model_obj = IHVCMEG(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            elif model == "SphericallySymmetricVolCondMEG":
                SSVMEG = potential._load_eegmegcalc_model("SphericallySymmetricVolCondMEG")
                model_obj = SSVMEG(sensor_locations, **model_kwargs)
                def get_M(loc):
                    return model_obj.get_transformation_matrix(loc)
            else:
                raise ValueError(f"Unknown model '{model}'.")

            for p_i, loc_i in zip(p_list, loc_list):
                matrices.append(get_M(loc_i))
                p_use_list.append(p_i)
            if matrices:
                n_sensors = matrices[0].shape[0]

        if not matrices or n_sensors <= 0:
            raise ValueError("Unable to determine number of sensors/electrodes.")

        n_times = p_use_list[0].shape[1]
        if is_meg:
            meeg = np.zeros((n_sensors, 3, n_times))
        else:
            meeg = np.zeros((n_sensors, n_times))

        log_every = max(1, n_sensors // 50)
        progress_start = 50
        progress_end = 90
        for idx in range(n_sensors):
            if is_meg:
                acc = np.zeros((3, n_times))
                for M, p_i in zip(matrices, p_use_list):
                    acc = acc + (M[idx] @ p_i)
                meeg[idx] = acc
            else:
                acc = np.zeros((n_times,))
                for M, p_i in zip(matrices, p_use_list):
                    acc = acc + (M[idx] @ p_i)
                meeg[idx] = acc

            if (idx + 1) % log_every == 0 or (idx + 1) == n_sensors:
                _append_job_output(
                    job_status,
                    job_id,
                    f"Computed electrodes: {idx + 1}/{n_sensors}",
                )
            progress = progress_start + int((idx + 1) / n_sensors * (progress_end - progress_start))
            job_status[job_id]["progress"] = progress

        output_root = "/tmp/field_potential_meeg"
        run_dir = os.path.join(output_root, job_id)
        os.makedirs(run_dir, exist_ok=True)
        meeg_path = os.path.join(run_dir, "meeg.pkl")
        with open(meeg_path, "wb") as f:
            pickle.dump(meeg, f)

        _append_job_output(job_status, job_id, f"Saved M/EEG to {meeg_path}")
        job_status[job_id].update({
                "status": "finished",
                "progress": 100,
                "estimated_time_remaining": 0,
                "results": meeg_path,
                "error": False
            })
    except Exception as e:
        _append_job_output(job_status, job_id, f"Error: {e}")
        job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "progress": job_status[job_id].get("progress", 0)
            })
    cleanup_temp_files(params.get('file_paths', {}))
