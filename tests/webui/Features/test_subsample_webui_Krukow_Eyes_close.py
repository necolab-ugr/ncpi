# Import average computation function
from common_webui_features_test_utils import *

# Expected average 
expected_avg = 8.060330761195495

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("http://localhost:5000/features/methods?entry=compute")
    page.get_by_role("button", name="add_circle New data Upload or").click()
    page.get_by_role("button", name="Server folder", exact=True).click()
    page.get_by_role("button", name="Add server folder").click()
    navigate_and_select(page, "/DATOS/pablomc/empirical_datasets/Krukow_data/Eyes close/Band_1-70")
    page.get_by_label("Data locator * Preferred:").first.select_option("data.signal")
    page.get_by_label("Sampling frequency source *").first.select_option("data.cfg.fs")
    page.get_by_label("Channel names source *").first.select_option("data.cfg.channels")
    page.get_by_label("Channels axis None Dim 0 Dim").first.select_option("-1")
    page.get_by_label("Samples axis Dim 0 Dim 1 Dim").first.select_option("1")
    page.get_by_label("IDs axis None Dim 0 Dim 1 Dim").first.select_option("-1")
    page.get_by_label("Trials/Epochs axis None Dim 0").first.select_option("0")
    page.get_by_label("Recording type source * Custom (Select Recording type value)").first.select_option("__value__")
    page.get_by_label("Recording type value LFP CDM").first.select_option("EEG")
    page.get_by_label("Subject ID source None Custom").select_option("__file_extracted_sep__underscore__2")
    page.get_by_label("Group source None Custom").select_option("__file_extracted_sep__underscore__0")
    page.get_by_label("Species source None Custom").select_option("__value__")
    page.get_by_role("textbox", name="Species value").click()
    page.get_by_role("textbox", name="Species value").fill("human")
    page.locator("div:nth-child(12) > div:nth-child(4)").click()
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
        timeout_terminal=460,
        timeout_file=60,
    )

    assert avg == pytest.approx(expected_avg, abs=tolerance), f"The calculated average {avg} doesnt coincide with the expected average {expected_avg}"

    # ---------------------
    context.close()
    browser.close()

@pytest.mark.playwright
def test_features_subsample_Krukow_Eyes_close():
    with sync_playwright() as playwright:
        run(playwright)
