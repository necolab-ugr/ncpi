import importlib
import os
import pickle
import sys


if __name__ == "__main__":
    # Read the script file path from sys.argv[1]
    script_path = sys.argv[1]

    # Add the directory containing the script to the Python path
    script_dir = os.path.dirname(script_path)
    sys.path.append(script_dir)

    # Import the script as a module
    module_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # LIF network parameters
    LIF_params = module.LIF_params

    # Save data to a pickle file
    with open(os.path.join(sys.argv[2],'network.pkl'), 'wb') as f:
        pickle.dump(LIF_params, f)

