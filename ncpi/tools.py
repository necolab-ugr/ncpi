import importlib.util
import importlib
from typing import Any, Optional
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


def dynamic_import(
        module_path: str,
        attribute_name: Optional[str] = None,
        package: Optional[str] = None
) -> Any:
    """
    Dynamically import a module or module attribute.

    Args:
        module_path: Full path to the module (e.g., 'package.subpackage.module')
        attribute_name: Optional name of attribute to import from the module
        package: Optional package name for relative imports

    Returns:
        The imported module or attribute

    Raises:
        ImportError: If the module or attribute cannot be imported
    """
    try:
        module = importlib.import_module(module_path, package=package)

        if attribute_name is not None:
            return getattr(module, attribute_name)
        return module

    except ImportError as e:
        raise ImportError(f"Failed to import {module_path}" +
                          (f".{attribute_name}" if attribute_name else "")) from e