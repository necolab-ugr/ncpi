# Import average computation function
from common_webui_features_test_utils import *

# Expected average 
expected_avg = 13.117046201076034

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(
        headless=False, 
        args=[f"--window-size=1600,900"],
        slow_mo=300) # start_server=True, job_timeout_sec=3600,
    context = browser.new_context(
        viewport={"width": 1600, "height": 900},
        screen={"width": 1600, "height": 900},
        record_video_dir='.',
        record_video_size={"width": 1600, "height": 900},
        device_scale_factor=1,)

    page = context.new_page()
    video = page.video
    page.set_default_timeout(60 * 1000)

    page.goto("http://localhost:5000")
    click_with_demo_cursor(page, page.get_by_role("link", name="Feature extraction Features"))
    click_with_demo_cursor(page, page.get_by_role("link", name="add_circle Compute new"))
    click_with_demo_cursor(page, page.get_by_role("button", name="add_circle New data Upload or"))
    click_with_demo_cursor(page, page.get_by_role("button", name="Server folder", exact=True))
    click_with_demo_cursor(page, page.get_by_role("button", name="Add server folder"))
    navigate_and_select(page, "/DATOS/pablomc/empirical_datasets/Chus_Mayores_Memoria/data")
    select_option_with_cursor(page, page.get_by_label("Data extension used as source"), "")
    select_option_with_cursor(page, page.get_by_label("Data extension used as source"), "__ext__:.mat")
    select_option_with_cursor(page, page.get_by_label("Data locator * Preferred:").first, "data_clean.trial")
    select_option_with_cursor(page, page.get_by_label("Sampling frequency source *").first, "data_clean.fsample")
    select_option_with_cursor(page, page.get_by_label("Channel names source *").first, "data_clean.label")
    select_option_with_cursor(page, page.get_by_label("Channels axis None Dim 0 Dim").first, "-1")
    select_option_with_cursor(page, page.get_by_label("Samples axis Dim 0 Dim 1 Dim").first, "2")
    select_option_with_cursor(page, page.get_by_label("IDs axis None Dim 0 Dim 1 Dim").first, "-1")
    select_option_with_cursor(page, page.get_by_label("Trials/Epochs axis None Dim 0").first, "-1")
    select_option_with_cursor(page, page.get_by_label("Recording type value LFP CDM").first, "EEG")
    select_option_with_cursor(page, page.get_by_label("Subject ID source None Custom"), "__file_extracted_sep__underscore__0")
    select_option_with_cursor(page, page.get_by_label("Species source None Custom"), "__value__")
    click_with_demo_cursor(page, page.get_by_role("textbox", name="Species value"))
    fill_with_cursor(page, page.get_by_role("textbox", name="Species value"), "human")
    click_with_demo_cursor(page, page.get_by_role("button", name="Next: Select method"))

    check_with_cursor(page, page.get_by_role("radio", name="Catch22 22 canonical time-"))
    click_with_demo_cursor(page, page.get_by_role("button", name="Next step arrow_forward"))

    #check_with_cursor(page, page.get_by_role("checkbox", name="Subsampling of data"))
    #fill_with_cursor(page, page.get_by_role("slider"), sampling_percentage)
    click_with_demo_cursor(page, page.get_by_role("spinbutton", name="Parallel workers (n_jobs)"))
    fill_with_cursor(page, page.get_by_role("spinbutton", name="Parallel workers (n_jobs)"), parallel_workers)
    expect(page.get_by_role("button", name="Compute features arrow_forward")).to_be_visible(timeout= 9000)
    click_with_demo_cursor(page, page.get_by_role("button", name="Compute features arrow_forward"), 900000)

    avg = wait_and_get_feature_average(
        page,
        method_name = "catch22",
        timeout_terminal=900,
        timeout_file=120,
    )

    assert avg == pytest.approx(expected_avg, abs=tolerance), f"The calculated average {avg} doesnt coincide with the expected average {expected_avg}"
    # ---------------------
    context.close()
    print("[automation] saving video...", flush=True)
    video.save_as(str("./Chus.webm"))
    browser.close()

@pytest.mark.playwright
def test_features_subsample_Chus_Mayores():
    with sync_playwright() as playwright:
        run(playwright)

