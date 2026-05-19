# Import average computation function and common imports
from common_webui_features_test_utils import *

# Expected average 
expected_avg = 14.422486513600091

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False, slow_mo=100)
    context = browser.new_context()
    page = context.new_page()

    page.goto("http://localhost:5000/features/methods?entry=compute")
    page.get_by_role("button", name="add_circle New data Upload or").click()
    page.get_by_role("button", name="Server folder", exact=True).click()
    page.get_by_role("button", name="Add server folder").click()
    navigate_and_select(page, "/DATOS/pablomc/empirical_datasets/development_EI_decorrelation/baseline/LFP")
    page.get_by_label("Data extension used as source").select_option("__ext__:.mat")
    page.get_by_label("Data locator * Preferred:").first.select_option("LFP.LFP")
    page.get_by_label("Sampling frequency source *").first.select_option("LFP.fs")
    page.get_by_label("Channel names source *").first.select_option("__autocomplete__")
    page.get_by_label("Channels axis None Dim 0 Dim").first.select_option("0")
    page.get_by_label("Samples axis Dim 0 Dim 1 Dim").first.select_option("1")
    page.get_by_role("radio", name="Empirical").check()
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
    page.get_by_role("spinbutton", name="Parallel workers (n_jobs)").click()
    page.get_by_role("spinbutton", name="Parallel workers (n_jobs)").fill(parallel_workers)
    expect(page.get_by_role("button", name="Compute features arrow_forward")).to_be_visible(timeout= 900000)
    page.get_by_role("button", name="Compute features arrow_forward").click()

    avg = wait_and_get_feature_average(
        page,
        method_name = "catch22",
        timeout_terminal=120,
        timeout_file=60,
        timeout_completion=3000000,
    )
    print('AVG is : ', avg)

    assert avg == pytest.approx(expected_avg, abs=tolerance), f"The calculated average {avg} doesnt coincide with the expected average {expected_avg}"
    # ---------------------
    context.close()
    browser.close()

@pytest.mark.playwright
def test_features_subsample_LFP_baseline():
    with sync_playwright() as playwright:
        run(playwright)

