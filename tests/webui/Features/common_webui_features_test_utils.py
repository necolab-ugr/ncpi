import re
import os
import time
import pickle
from pathlib import Path
import numpy as np
from playwright.sync_api import Page, expect
import pytest
from playwright.sync_api import Playwright, sync_playwright, expect


# Sampling percentage
sampling_percentage = "10"

# Tolerance error range
tolerance = 1e-1

# Parallel workers (n_jobs)
parallel_workers = "8"


def navigate_and_select(page, target_path, folder=True):
    """
    Go to any target path dynamically, deciding the next button to press to achieve the target path
    """
    # Normalize the path to avoid errors with final / (e.g. /DATOS/ vs /DATOS) 
    target_path = os.path.normpath(target_path)

    # If it's a file, we need to reach the parent directory first
    if folder:
        destination_folder = target_path
        item_name = None
    else:
        destination_folder = os.path.dirname(target_path)
        item_name = os.path.basename(target_path)

    while True:
        # 1. Get and clean the current path
        # This picks the active path display in the modal
        current_folder_locator = page.get_by_text("Current folder:").filter(visible=True).last
        
        # Get the full text (e.g., "Current folder: /home/user")
        full_text = current_folder_locator.inner_text()

        # Clean the string to get just the path
        current_path = os.path.normpath(full_text.replace("Current folder:", "").strip())

        # CASE A: We are already in the target path
        if current_path == destination_folder:
            if folder:
                break
            else:
                # If it's a file, click it and we are done
                page.get_by_role("button", name=item_name, exact=True).click()
                return

        # CASE B: The current path is "father" or part of the path to the target path
        # E.g.: We're in "/DATOS" and the target is "/DATOS/empirical"
        if destination_folder.startswith(current_path + os.sep) or current_path == "/":
            # Decide the next folder to click
            path_suffix = destination_folder[len(current_path):].lstrip(os.sep)
            next_step = path_suffix.split(os.sep)[0]
            
            page.get_by_role("button", name=next_step, exact=True).click()
        
        # CASE C: We're in a totally different path (e.g: /home/user)
        # or we're deeper than we should (e.g: /DATOS/empirical/local/data/sub-001)
        else:
            page.get_by_role("button", name="Up", exact=True).click()

        # Small pause to wait until the DOM is updated after the click
        page.wait_for_load_state("networkidle")

    if folder:
        # Confirm the folder
        page.get_by_role("button", name="Add this folder").click()


def wait_and_get_feature_average(
    page: Page,
    method_name: str = "catch22",
    timeout_terminal: int = 240,
    timeout_file: int = 60,
    timeout_completion: int = 1_500_000  # 25 min
) -> float:

    # --- 1. Search output path in the terminal text ---
    log_selector = "#output-terminal"
    found_line = None
    start = time.time()
    while not found_line and (time.time() - start) < timeout_terminal:
        # Get the whole text of the terminal element
        log_text = page.text_content(log_selector) if page.locator(log_selector).count() else ""
        # Search the line with the text "Persisted dashboard features file"
        for line in log_text.splitlines():
            if "Persisted dashboard features file:" in line:
                found_line = line
                break
        if not found_line:
            time.sleep(2)

    if not found_line:
        raise Exception("The output filepath could not be found in the UI or the timeout was too low for this data")

    # Extract UUID with regex
    match = re.search(r"/tmp/ncpi_webui_session_([a-f0-9]+)/", found_line)
    if not match:
        raise Exception("Session UUID could not be found in the terminal element")
    session_uuid = match.group(1)
    print(f"Session UUID extracted from the terminal: {session_uuid}")

    # --- 2. Expect the end of the computation in the UI ---
    try:
        expect(page.get_by_role("main")).to_contain_text(
            "Features Computed", timeout=timeout_completion
        )
        print("Computation finished (UI indicates 'Features Computed')")
    except Exception:
        print("No UI indicator of 'completed' could be detected. Trying to read the file anyway...")

    expect(page.get_by_role("link", name="Continue to Inference")).to_be_visible(
        timeout=timeout_completion
    )

    # --- 3. Wait and load the pickle file ---
    output_file_path = Path(f"/tmp/ncpi_webui_session_{session_uuid}/features/data/{method_name}_features.pkl")

    start = time.time()
    while not output_file_path.exists() and (time.time() - start) < timeout_file:
        time.sleep(1)
        print("Output file doesnt exist yet, reloading... (60 seconds maximum timeout)")
    if not output_file_path.exists():
        raise FileNotFoundError(
            f"Pickle file {output_file_path} couldnt be found in the expected path after a timeout of {timeout_file}s"
        )
    print(f"File found in: {output_file_path}")

    with open(output_file_path, "rb") as f:
        emp_data = pickle.load(f)

    # --- 4. Calculate the average of all data (ignoring NaN) ---
    features = emp_data["Features"].tolist()
    avg = np.nanmean([np.nanmean(np.asarray(x)) for x in features])
    return float(avg)