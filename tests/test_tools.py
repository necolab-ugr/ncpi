import os
import ncpi
from ncpi import tools



def download_data_if_needed(zenodo_URL_test, zenodo_dir):
    """Download test data only when running locally and data is missing"""    
    # Skip if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("Running in CI - assuming test data is cached")
        return
    
    # Download if zenodo sim data files don't exist locally
    if not os.path.exists(zenodo_dir):
        tools.timer("Downloading data for local execution...")(
            tools.download_zenodo_record
        )(zenodo_URL_test, download_dir=zenodo_dir)
    else:
        print("Zenodo test files already exist locally")

