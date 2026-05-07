import re
from playwright.sync_api import Playwright, sync_playwright, expect
# import importlib.util
from pathlib import Path
# import math
# import os
# import sys
# import pandas as pd
import time
import pickle
import numpy as np
import pytest

# ----------------------------
# Paths
# ----------------------------
# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_file_dir, "..", "..", ".."))

# tests_dir = os.path.join(project_root, "tests")
# examples_dir = os.path.join(project_root, "examples")

# ----------------------------
# Import test helpers explicitly (tests/tools.py) as test_tools
# ----------------------------
# _test_tools_path = os.path.join(tests_dir, "tools.py")
# _spec_test = importlib.util.spec_from_file_location("test_tools", _test_tools_path)
# test_tools = importlib.util.module_from_spec(_spec_test)
# _spec_test.loader.exec_module(test_tools)

# These are Paths in your repo (PosixPath)
# ZENODO_EMP_DIR = test_tools.ZENODO_EMP_DIR

# Keep Path objects for test_tools (it expects Path-like API)
# zenodo_dir_emp_path = Path(ZENODO_EMP_DIR)

# # But pass strings into shared_tools (it validates `str`)
# zenodo_dir_emp = Path(str(zenodo_dir_emp_path)) / 'development_EI_decorrelation' / 'baseline' / 'LFP'

# ----------------------------
# Import shared tools explicitly (examples/tools.py) as shared_tools
# ----------------------------
# _examples_tools_path = os.path.join(examples_dir, "tools.py")
# _spec_shared = importlib.util.spec_from_file_location("shared_tools", _examples_tools_path)
# shared_tools = importlib.util.module_from_spec(_spec_shared)
# _spec_shared.loader.exec_module(shared_tools)


# pip install . required!
#from ncpi.EphysDatasetParser import EphysDatasetParser, ParseConfig, CanonicalFields

# Sampling percentage
sampling_percentage = "10"


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False, slow_mo=150)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://localhost:5000/features/methods?entry=compute")
    page.get_by_role("button", name="add_circle New data Upload or").click()
    page.get_by_role("button", name="Add server folder").click()
    page.get_by_role("button", name="Up").click()
    page.get_by_role("button", name="Up").click()
    page.get_by_role("button", name="DATOS").click()
    page.get_by_role("button", name="pablomc").click()
    page.get_by_role("button", name="empirical_datasets").click()
    page.get_by_role("button", name="development_EI_decorrelation").click()
    page.get_by_role("button", name="baseline").click()
    page.get_by_role("button", name="LFP").click()
    page.get_by_role("button", name="Add this folder").click()
    page.get_by_label("Data extension used as source").select_option("__ext__:.mat")
    page.get_by_label("Data locator * Preferred:").first.select_option("LFP.LFP")
    page.get_by_label("Sampling frequency source *").first.select_option("LFP.fs")
    page.get_by_label("Subject ID source None Custom").select_option("__file_extracted_chain_0")
    page.get_by_label("Group source None Custom").select_option("LFP.age")
    page.get_by_label("Species source None Custom").select_option("__value__")
    page.get_by_role("textbox", name="Species value").click()
    page.get_by_role("textbox", name="Species value").fill("mouse")
    page.get_by_role("checkbox", name="Enable epoching").check()
    page.get_by_role("checkbox", name="Enable aggregation").check()
    page.get_by_role("button", name="Next: Select method").click()
    page.get_by_role("radio", name="Catch22 22 canonical time-").check()
    page.get_by_role("button", name="Next step arrow_forward").click()
    page.get_by_role("checkbox", name="Subsampling of data").check()
    page.get_by_role("slider").fill(sampling_percentage)
    expect(page.get_by_role("button", name="Compute features arrow_forward")).to_be_visible(timeout= 900000)
    page.get_by_role("button", name="Compute features arrow_forward").click()
    
    # Search output path in the terminal text
    log_selector = "#output-terminal"
    found_line = None
    timeout = 120
    start = time.time()
    while not found_line and (time.time() - start) < timeout:
        # Get the whole text of the terminal element
        log_text = page.text_content(log_selector) if page.locator(log_selector).count() else ""
        # Search the line with the text "Persisted dashboard features file"
        lines = log_text.splitlines()
        for line in lines:
            if "Persisted dashboard features file:" in line:
                found_line = line
                break
        if not found_line:
            time.sleep(2)

    if not found_line:
        raise Exception("The output filepath could not be found in the UI")

    # Extract UUID with regex
    match = re.search(r"/tmp/ncpi_webui_session_([a-f0-9]+)/", found_line)
    if not match:
        raise Exception("Session UUID could not be found in the terminal element")
    session_uuid = match.group(1)
    print(f"Session UUID extracted from the terminal: {session_uuid}")

    try:
        expect(page.get_by_role("main")).to_contain_text("Features Computed", timeout=1500000)
        print("Computation finished (UI indicates 'Features Computed')")
    except Exception:
        print("No UI indicator of 'completed' could be detected. Trying to read the file anyway...")
    expect(page.get_by_role("link", name="Continue to Inference")).to_be_visible(timeout=1500000)



    # ----- BROWSER INDEPENDENT FILE CHECK --------
    # Load the pickle file
    output_file_path = Path(f'/tmp/ncpi_webui_session_{session_uuid}/features/data/catch22_features.pkl')

    timeout = 60
    start = time.time()
    while not output_file_path.exists() and (time.time() - start) < timeout:
        time.sleep(1)
        print('Output file doesnt exist yet, reloading... (60 seconds maximum timeout)')
    if not output_file_path.exists():
        raise FileNotFoundError(f"Pickle file couldnt be found in the expected path after a timeout of {timeout}s")
    print(f"File found in: {output_file_path}")

    with open(output_file_path, 'rb') as f:
        emp_data = pickle.load(f)

    # Select just the column with the data "Features"
    emp_data_features = emp_data["Features"].tolist()

    # Calculate the average of all data (ignoring NaN)
    avg = np.nanmean(tuple(np.nanmean(x) for x in emp_data_features))

    expected_avg = 14.422486513600091
    tolerance = 1e-1
    assert avg == pytest.approx(expected_avg, abs=tolerance), f"The calculated average {avg} doesnt coincide with the expected average {expected_avg}"

    # ---------------------
    context.close()
    browser.close()


# def load_empirical_data_filtered(file_names, base_dir):
#     """
#     Load the SPECIFIED FILES in file_names in `base_dir`.
#     """ 
#     base_path = Path(base_dir)
#     rows = []

#     for subject_id, fname in enumerate(file_names):
#         file_path = base_path / fname
#         print(f"\r Progress: {subject_id + 1} of {len(file_names)} files loaded", end="", flush=True)

#         config = ParseConfig(
#             fields=CanonicalFields(
#                 data=lambda d: d["LFP"].LFP,
#                 fs=lambda d: float(d["LFP"].fs),
#                 ch_names=lambda d: [f"ch{i}" for i in range(d["LFP"].LFP.shape[0])],
#                 metadata={
#                     "subject_id": subject_id,
#                     "group": lambda d: int(d["LFP"].age),   # age → group
#                     "species": "mouse",
#                     "recording_type": "LFP",
#                 },
#             ),
#             epoch_length_s=5.0,
#             epoch_step_s=5.0,
#             aggregate_over=("sensor",),
#             aggregate_method="sum",
#         )

#         parser = EphysDatasetParser(config)
#         df = parser.parse(file_path)
#         rows.append(df)

#     print(f"\nFiles loaded: {len(rows)}")
#     out = pd.concat(rows, ignore_index=True)
#     return out


# def subsampling_computation():
#     # 1) Subsampling of data files 
#     sampling_percentage = float(sampling_percentage)/100

#     # 2) List datafiles of path
#     datafiles = []
#     for element in os.listdir(zenodo_dir_emp):
#         if os.path.isfile(os.path.join(zenodo_dir_emp, element)):  # only files. Ignore subfolders
#             datafiles.append(element)

#     datafiles_sorted = sorted(datafiles)

#     # 3) Calculate % of number of files
#     original_count = len(datafiles_sorted)
#     keep_count = max(1, math.ceil(original_count * sampling_percentage))

#     # Select only 10% of files
#     datafiles_sorted = datafiles_sorted[:keep_count] 

#     # 4) Empirical LFP (loader accepts str fine)
#     df_emp = load_empirical_data_filtered(datafiles_sorted, zenodo_dir_emp)

#     # 5) Feature extraction
#     method = "catch22"
#     emp_data = shared_tools.compute_features(method, df_emp)

#     emp_data = emp_data.tolist()

#     avg = np.nanmean(tuple(np.nanmean(x) for x in emp_data))
#     print('The average of the first 10 percent of files is: ', avg)


with sync_playwright() as playwright:
    run(playwright)

