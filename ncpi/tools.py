import importlib.util
import subprocess
import sys


def ensure_module(module_name, package_name=None):
    """
    Check if a module is installed; if not, try to install it.

    Args:
        module_name (str): The module to check (e.g., 'rpy2').
        package_name (str, optional): The package name to install (if different from module_name).

    Returns:
        bool: True if the module is installed successfully, False otherwise.
    """
    try:
        if importlib.util.find_spec(module_name) is None:
            print(f"Module '{module_name}' not found. Attempting to install...")
            package = package_name if package_name else module_name
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Module '{module_name}' installed successfully.")

            # Check again after installation
            if importlib.util.find_spec(module_name) is None:
                raise ImportError(f"Installation failed: '{module_name}' not found after installation.")

        return True  # Module is available
    except subprocess.CalledProcessError:
        print(f"Error: Failed to install '{module_name}'. Please install it manually.")
    except ImportError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return False  # Module is not available