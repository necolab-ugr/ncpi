# Import average computation function
from common_webui_features_test_utils import *

# Expected average 
expected_avg = 8.923003681346412

# Feature extraction of this data may take up to 70 minutes if 8 parallel workers are used.


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False, slow_mo=100)
    context = browser.new_context()
    page = context.new_page()

    page.goto("http://localhost:5000/features/methods?entry=compute")
    page.get_by_role("button", name="add_circle New data Upload or").click()
    page.get_by_role("button", name="Server folder", exact=True).click()
    page.get_by_role("button", name="Add server folder").click()
    navigate_and_select(page, "/DATOS/pablomc/empirical_datasets/Krukow_data/Task-Related signals/EEGLab_Band_1-70/Band_1-70")
    page.get_by_label("Data extension used as source").select_option("__ext__:.set")
    page.get_by_label("Sampling frequency source *").first.select_option("info.sfreq")
    page.get_by_label("Data locator * Preferred:").first.select_option("get_data")
    page.get_by_label("Channel names source *").first.select_option("ch_names")
    page.get_by_label("Channels axis None Dim 0 Dim").first.select_option("-1")
    page.get_by_label("Samples axis Dim 0 Dim 1 Dim").first.select_option("0")
    page.get_by_label("IDs axis None Dim 0 Dim 1 Dim").first.select_option("-1")
    page.get_by_label("Trials/Epochs axis None Dim 0").first.select_option("-1")
    page.get_by_label("Recording type value LFP CDM").first.select_option("EEG")
    page.get_by_label("Subject ID source None Custom").select_option("__file_extracted_chain_0")
    page.get_by_label("Species source None Custom").select_option("__value__")
    page.get_by_role("textbox", name="Species value").click()
    page.get_by_role("textbox", name="Species value").fill("human")
    page.get_by_role("button", name="Next: Select method").click()

    page.get_by_role("radio", name="Catch22 22 canonical time-").check()
    page.get_by_role("button", name="Next step arrow_forward").click()

    page.get_by_role("checkbox", name="Subsampling of data").check()
    page.get_by_role("slider").fill(sampling_percentage)
    page.get_by_role("spinbutton", name="Parallel workers (n_jobs)").click()
    page.get_by_role("spinbutton", name="Parallel workers (n_jobs)").fill(parallel_workers)
    expect(page.get_by_role("button", name="Compute features arrow_forward")).to_be_visible(timeout= 900000)
    page.get_by_role("button", name="Compute features arrow_forward").click()

    avg = wait_and_get_feature_average(
        page,
        method_name = "catch22",
        timeout_terminal=5000,
        timeout_file=500,
        timeout_completion=5000000,
    )

    assert avg == pytest.approx(expected_avg, abs=tolerance), f"The calculated average {avg} doesnt coincide with the expected average {expected_avg}"

    # ---------------------
    context.close()
    browser.close()

@pytest.mark.playwright
def test_features_subsample_Krukow_EEGLab():
    with sync_playwright() as playwright:
        run(playwright)

