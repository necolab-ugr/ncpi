import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
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
    page.get_by_role("button", name="Krukow_data").click()
    page.get_by_role("button", name="Eyes close").click()
    page.get_by_role("button", name="Band_1-").click()
    page.get_by_role("button", name="Add this folder").click()
    page.get_by_label("Data locator * Preferred:").first.select_option("data.signal")
    page.get_by_label("Sampling frequency source *").first.select_option("data.cfg.fs")
    page.get_by_label("Channel names source *").first.select_option("data.cfg.channels")
    page.get_by_label("Recording type source * Value").first.select_option("data")
    page.get_by_label("Recording type source * Value").first.press("ArrowUp")
    page.get_by_label("Recording type source * Value").first.select_option("__value__")
    page.get_by_label("Recording type value LFP CDM").first.select_option("EEG")
    page.get_by_label("Subject ID source None Custom").select_option("__file_extracted_chain_2")
    page.get_by_label("Group source None Custom").select_option("__file_extracted_chain_0")
    page.locator("div:nth-child(12) > div:nth-child(4)").click()
    page.get_by_role("button", name="Next: Select method").click()
    page.get_by_role("radio", name="Catch22 22 canonical time-").check()
    page.get_by_role("button", name="Next step arrow_forward").click()
    page.get_by_role("checkbox", name="Subsampling of data").check()
    page.get_by_role("slider").fill("5")
    expect(page.get_by_role("button", name="Compute features arrow_forward")).to_be_visible(timeout= 900000)
    page.get_by_role("button", name="Compute features arrow_forward").click()
    expect(page.get_by_role("main")).to_contain_text("Features Computed", timeout=1500000)
    expect(page.get_by_role("link", name="Continue to Inference")).to_be_visible(timeout=1500000)

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
