import importlib.util
import importlib
from typing import Any, Optional
import subprocess
import sys
import os
from functools import wraps
import time

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
        
        # Check if installed version matches required version from package_name
        if package_name and '==' in package_name:
            required_version = package_name.split('==')[1]
            module = importlib.import_module(module_name)
            installed_version = getattr(module, '__version__', None)
            
            if installed_version != required_version:
                print(f"Module '{module_name}' version {installed_version} found, but {required_version} required. Reinstalling...")
                # Uninstall current version
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", module_name])
                # Install required version
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                
                # Clear cached modules
                for mod_name in list(sys.modules.keys()):
                    if mod_name == module_name or mod_name.startswith(f"{module_name}."):
                        del sys.modules[mod_name]
                
                print(f"Module '{module_name}' reinstalled successfully with version {required_version}.")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error: {e} Failed to install '{module_name}'. Please install it manually.")
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


def download_zenodo_record(api_url, download_dir="zenodo_files", extract_tar=True, delete_tar=True):
    """
    Download files from a Zenodo record.

    Args:
        api_url (str): The API URL of the Zenodo record.
        download_dir (str): Directory to save downloaded files. Default is 'zenodo_files'.
        extract_tar (bool): Whether to extract tar files after downloading. Default is True.
        delete_tar (bool): Whether to delete tar files after extraction. Default is True.
    """

    # Check if the download directory exists, if not create it
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)

        # Check if the required modules are installed
        if not ensure_module("requests"):
            raise ImportError("requests is required for downloading data. ")
        requests = dynamic_import("requests")

        if not ensure_module("tqdm"):
            raise ImportError("tqdm is required for progress bars. ")
        tqdm = dynamic_import("tqdm", "tqdm")

        if not ensure_module("tarfile"):
            raise ImportError("tarfile is required for extracting tar files. ")
        tarfile = dynamic_import("tarfile")

        # Download the Zenodo record
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        record = response.json()

        for file_info in record.get("files", []):
            file_url = file_info["links"]["self"]
            file_name = file_info["key"]
            file_path = os.path.join(download_dir, file_name)

            print(f"Downloading {file_name}...")
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(file_path, "wb") as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=file_name,
                    initial=0
                ) as progress:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
            print(f"Saved to {file_path}")

        # After all files are downloaded, untar and delete tar files
        if extract_tar:
            for file_name in os.listdir(download_dir):
                file_path = os.path.join(download_dir, file_name)

                if tarfile.is_tarfile(file_path):
                    print(f"Extracting {file_name}...")
                    with tarfile.open(file_path, "r:*") as tar:
                        tar.extractall(path=download_dir)
                    print(f"Extracted to {download_dir}")

                    if delete_tar:
                        os.remove(file_path)
                        print(f"Deleted {file_name}")

    else:
        print(f"Directory {download_dir} already exists. Skipping download.")
        print("If you want to re-download, please delete the directory.")


def timer(description=None):
    """
    Decorator that measures and prints execution time of a function.
    
    Args:
        description (str): A custom message describing what the function does.
                         This will be printed before the function executes.
    
    Returns:
        function: A decorator that can be applied to any function to add timing.
    
    Example:
        @timer("Downloading simulation data and ML models.")
        def download_data():
            # function content
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper function that adds timing before and after function execution.

            Args:
                *args: Variable length argument list passed to the original function.
                **kwargs: Arbitrary keyword arguments passed to the original function.
            """
            # Print the custom description message before function execution
            print(f'\n--- {description}')

            start_time = time.time()
            
            # Execute the original function with all its arguments
            result = func(*args, **kwargs)
            
            end_time = time.time()

            # Calculate and print the execution time in minutes
            print(f'Done in {(end_time - start_time):.2f} seconds.{(f", which is equivalent to {int((end_time - start_time) // 60)}:{int((end_time - start_time) % 60):02d} min" if (end_time - start_time) >= 60 else "")}')
            return result
        return wrapper
    return decorator
