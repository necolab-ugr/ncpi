import os
import pickle
import subprocess
import sys

SIMULATION_BUNDLE_FILE = "simulation.pkl"
SIMULATION_CORE_FILES = ("times.pkl", "gids.pkl", "dt.pkl", "tstop.pkl", "network.pkl")
SIMULATION_OPTIONAL_FILES = ("population_sizes.pkl", "vm.pkl", "ampa.pkl", "gaba.pkl", "exc_state_events.pkl")
SIMULATION_GRID_METADATA_FILES = ("grid_metadata.pkl", "simulation_grid_metadata.pkl")
SIMULATION_FIELD_BY_FILENAME = {
    "times.pkl": "times",
    "gids.pkl": "gids",
    "dt.pkl": "dt",
    "tstop.pkl": "tstop",
    "network.pkl": "network",
    "population_sizes.pkl": "population_sizes",
    "vm.pkl": "vm",
    "ampa.pkl": "ampa",
    "gaba.pkl": "gaba",
    "exc_state_events.pkl": "exc_state_events",
}


def _headless_subprocess_env():
    """Return a child-process environment that avoids GUI backends in tests/headless runs."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    for key in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(key, None)
    return env


def run_script(script_path, param_path, output_folder):
    """
    Run a python script with the given parameters.

    Parameters
    ----------
    script_path : str
        Path to the python script to run.
    param_path : str
        Path to the parameter file to use.
    output_folder : str
        Path to the folder where the output files will be saved.
    """
    # Check if scripts exist
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script '{script_path}' does not exist.")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Parameter file '{param_path}' does not exist.")

    # Check if scripts are python scripts
    if not script_path.endswith('.py'):
        raise ValueError(f"Script '{script_path}' is not a python script.")
    if not param_path.endswith('.py'):
        raise ValueError(f"Parameter file '{param_path}' is not a python script.")

    # Run the script and propagate failures to callers.
    # This ensures orchestrators (e.g. WebUI) can mark jobs as failed.
    cmd = [sys.executable, script_path, param_path, output_folder]
    result = subprocess.run(cmd, check=False, env=_headless_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"Script '{os.path.basename(script_path)}' failed with exit code {result.returncode}."
        )


class Simulation:
    def __init__(self, param_folder, python_folder, output_folder):

        # Check if param and python folders exist
        if not os.path.exists(param_folder):
            raise FileNotFoundError(f"Parameter folder '{param_folder}' does not exist.")
        if not os.path.exists(python_folder):
            raise FileNotFoundError(f"Python folder '{python_folder}' does not exist.")

        # Check if output folder exists, if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.param_folder = param_folder
        self.python_folder = python_folder
        self.output_folder = output_folder


    def network(self, script_path, param_path):
        run_script(os.path.join(self.python_folder, script_path),
                   os.path.join(self.param_folder, param_path),
                   self.output_folder)

    def simulate(self, script_path, param_path):
        run_script(os.path.join(self.python_folder, script_path),
                   os.path.join(self.param_folder, param_path),
                   self.output_folder)


    def analysis(self, script_path, param_path):
        run_script(os.path.join(self.python_folder, script_path),
                   os.path.join(self.param_folder, param_path),
                   self.output_folder)

    @staticmethod
    def bundle_output_folder(output_folder, output_filename=SIMULATION_BUNDLE_FILE, remove_source_files=False):
        if not os.path.isdir(output_folder):
            raise FileNotFoundError(f"Output folder '{output_folder}' does not exist.")

        def _load_pickle(name):
            path = os.path.join(output_folder, name)
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as handle:
                return pickle.load(handle)

        def _as_trial_list(value):
            return value if isinstance(value, list) else [value]

        missing = []
        payload = {}
        loaded_names = []
        for file_name in SIMULATION_CORE_FILES:
            obj = _load_pickle(file_name)
            if obj is None:
                missing.append(file_name)
                continue
            payload[SIMULATION_FIELD_BY_FILENAME[file_name]] = _as_trial_list(obj)
            loaded_names.append(file_name)
        if missing:
            raise ValueError(
                "Cannot create simulation bundle; missing required file(s): "
                + ", ".join(missing)
            )

        for file_name in SIMULATION_OPTIONAL_FILES:
            obj = _load_pickle(file_name)
            if obj is None:
                continue
            payload[SIMULATION_FIELD_BY_FILENAME[file_name]] = _as_trial_list(obj)
            loaded_names.append(file_name)

        for meta_name in SIMULATION_GRID_METADATA_FILES:
            obj = _load_pickle(meta_name)
            if isinstance(obj, dict):
                payload["grid_metadata"] = obj
                loaded_names.append(meta_name)
                break

        bundle_path = os.path.join(output_folder, output_filename)
        with open(bundle_path, "wb") as handle:
            pickle.dump(payload, handle)

        if remove_source_files:
            removable = set(loaded_names)
            removable.update({"sim_data.pkl"})
            removable.discard(output_filename)
            for name in removable:
                path = os.path.join(output_folder, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        return bundle_path

    def bundle_outputs(self, output_filename=SIMULATION_BUNDLE_FILE, remove_source_files=False):
        return self.bundle_output_folder(
            self.output_folder,
            output_filename=output_filename,
            remove_source_files=remove_source_files,
        )
