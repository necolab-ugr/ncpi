from pathlib import Path
import os
from ncpi import tools

# Detect CI
IN_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))

if IN_GITHUB:
    # GitHub Actions: use cached test data inside the repo
    BASE_DIR = Path(__file__).resolve().parent / "data" / "zenodo_test_files_LFP"
else:
    # Local runs: use HOME
    BASE_DIR = Path.home()

ZENODO_SIM_DIR = BASE_DIR / "zenodo_sim_files"
ZENODO_EMP_DIR = BASE_DIR / "zenodo_emp_files"


def download_data_if_needed(zenodo_URL_test, zenodo_dir):
    if os.getenv("GITHUB_ACTIONS"):
        print("Running in CI - assuming test data is cached")
        return

    if zenodo_dir.is_dir() and any(zenodo_dir.iterdir()):
        print(f"Zenodo test files already exist locally in {zenodo_dir}")
        return

    zenodo_dir.mkdir(parents=True, exist_ok=True)
    tools.timer("Downloading data for local execution...")(tools.download_zenodo_record)(
        zenodo_URL_test, download_dir=str(zenodo_dir)
    )

