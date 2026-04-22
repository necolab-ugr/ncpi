import os
import subprocess
import sys


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
