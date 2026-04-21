import importlib.util
import pickle
import threading
import uuid
from pathlib import Path

import pytest
from werkzeug.serving import make_server


_SHARED_PATH = Path(__file__).resolve().with_name("test_hagen.py")
_SHARED_SPEC = importlib.util.spec_from_file_location("test_hagen_shared", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_SHARED_SPEC)
assert _SHARED_SPEC.loader is not None
_SHARED_SPEC.loader.exec_module(_shared)


sync_playwright = _shared.sync_playwright
PlaywrightError = _shared.PlaywrightError


def _log_test_progress(message):
    text = f"[test_sim_upload] {message}"
    if _shared._CAPTURE_MANAGER is not None:
        with _shared._CAPTURE_MANAGER.global_and_fixture_disabled():
            _shared._write_progress_line(text)
        return
    _shared._write_progress_line(text)


def _build_pickle_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def _navigate_to_upload_sim_page(page, live_webui_server):
    page.goto(f"{live_webui_server}/simulation/upload_sim", wait_until="domcontentloaded")
    page.locator("#simulationUploadForm").wait_for(state="visible")
    page.locator("#spikeFileInput").wait_for(state="attached")


def _set_local_simulation_upload_files(page, paths):
    page.locator("#spikeFileInput").set_input_files([str(path) for path in paths])
    page.evaluate(
        """() => {
            const modeInput = document.getElementById("simulationSourceModeInput");
            const serverPathInput = document.getElementById("simulationServerFilePathInput");
            if (modeInput) {
                modeInput.value = "upload";
            }
            if (serverPathInput) {
                serverPathInput.value = "";
            }
        }"""
    )


def _submit_local_simulation_upload(page):
    with page.expect_navigation(url="**/simulation/upload_sim", wait_until="domcontentloaded"):
        page.locator("#simulationUploadForm").evaluate("(form) => form.submit()")


@pytest.fixture(scope="module")
def webui_app_module():
    if sync_playwright is None:
        pytest.skip("Playwright is required for the web UI simulation upload tests.")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("NCPI_WEBUI_SESSION_ID", f"pytest_webui_sim_upload_{uuid.uuid4().hex}")
    monkeypatch.delenv("NCPI_WEBUI_SESSION_ROOT", raising=False)
    module = _shared._reload_webui_app()
    module._clear_simulation_output_folder_all_files()
    yield module
    module._clear_simulation_output_folder_all_files()
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _attach_test_upload_terminal_reporter(pytestconfig):
    previous_reporter = _shared._TERMINAL_REPORTER
    previous_capture_manager = _shared._CAPTURE_MANAGER
    _shared._TERMINAL_REPORTER = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    _shared._CAPTURE_MANAGER = pytestconfig.pluginmanager.get_plugin("capturemanager")
    try:
        yield
    finally:
        _shared._TERMINAL_REPORTER = previous_reporter
        _shared._CAPTURE_MANAGER = previous_capture_manager


@pytest.fixture()
def live_webui_server(webui_app_module, pytestconfig):
    server_port = int(pytestconfig.getoption("webui_port") or 0)
    server = make_server("127.0.0.1", server_port, webui_app_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_simulation_upload_local_pickle_files_updates_module_state(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting local simulation-data upload scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    local_files = [
        _build_pickle_file(tmp_path / "times.pkl", {"E": [1.0]}),
        _build_pickle_file(tmp_path / "network.pkl", {"X": ["E"], "N_X": [1]}),
    ]
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=300 if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                _log_test_progress("uploading local pickle files")
                _set_local_simulation_upload_files(page, local_files)
                _submit_local_simulation_upload(page)
                page.locator("text=Status: Files detected").wait_for(state="visible")

                page_text = page.locator("body").inner_text()
                assert "Status: Files detected" in page_text
                assert "times.pkl" in page_text
                assert "network.pkl" in page_text
                continue_link = page.locator('a[href="/field_potential"]').first
                continue_link.wait_for(state="visible")
            finally:
                browser.close()

    stored_files = sorted(path.name for path in Path(webui_app_module.SIMULATION_DATA_DIR).glob("*.pkl"))
    assert stored_files == ["network.pkl", "times.pkl"]


def test_simulation_upload_server_path_pickle_files_updates_module_state(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting server-path simulation-data upload scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    server_files = [
        _build_pickle_file(tmp_path / "gids.pkl", {"E": [1]}),
        _build_pickle_file(tmp_path / "tstop.pkl", 1000.0),
    ]
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=300 if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                _log_test_progress("submitting server-path pickle files")
                page.evaluate(
                    """(paths) => {
                        document.querySelector("#simulationSourceModeInput").value = "server-path";
                        document.querySelector("#simulationServerFilePathInput").value = paths.join("\\n");
                    }""",
                    [str(path) for path in server_files],
                )
                page.locator("#simulationUploadForm").evaluate("(form) => form.submit()")
                page.wait_for_load_state("domcontentloaded")

                page_text = page.locator("body").inner_text()
                assert "Status: Files detected" in page_text
                assert "gids.pkl" in page_text
                assert "tstop.pkl" in page_text
            finally:
                browser.close()

    stored_files = sorted(path.name for path in Path(webui_app_module.SIMULATION_DATA_DIR).glob("*.pkl"))
    assert stored_files == ["gids.pkl", "tstop.pkl"]


def test_simulation_upload_invalid_local_extension_shows_client_side_error(
    live_webui_server, tmp_path, pytestconfig
):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("bad", encoding="utf-8")
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=300 if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                start_url = page.url
                page.locator("#spikeFileInput").set_input_files(str(bad_file))
                error_box = page.locator("#simulationUploadValidationError")
                error_box.wait_for(state="visible")
                assert "Unsupported format:" in error_box.inner_text()
                assert page.url == start_url
            finally:
                browser.close()


def test_simulation_upload_invalid_server_path_extension_rejected_with_flash(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting invalid server-path simulation-data upload scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    bad_file = tmp_path / "not_pickle.txt"
    bad_file.write_text("bad", encoding="utf-8")
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=300 if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                page.evaluate(
                    """(badPath) => {
                        document.querySelector("#simulationSourceModeInput").value = "server-path";
                        document.querySelector("#simulationServerFilePathInput").value = badPath;
                    }""",
                    str(bad_file),
                )
                page.locator("#simulationUploadForm").evaluate("(form) => form.submit()")
                page.wait_for_load_state("domcontentloaded")
                flash = page.locator(".flash-container").first
                flash.wait_for(state="visible")
                flash_text = flash.inner_text()
                assert "Simulation server file must have one of these extensions" in flash_text
            finally:
                browser.close()


def test_simulation_upload_remove_file_updates_module_state(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting simulation-data removal scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    local_file = _build_pickle_file(tmp_path / "dt.pkl", 0.0625)
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=300 if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                _set_local_simulation_upload_files(page, [local_file])
                _submit_local_simulation_upload(page)
                page.locator("text=Status: Files detected").wait_for(state="visible")
                remove_button = page.get_by_role("button", name="Remove").first
                remove_button.wait_for(state="visible")
                remove_button.click()
                page.wait_for_load_state("domcontentloaded")
                page_text = page.locator("body").inner_text()
                assert "Status: Awaiting files" in page_text
                assert page.locator('a[href="/field_potential"]').count() == 0
                assert page.get_by_role("button", name="Remove").count() == 0
            finally:
                browser.close()

    assert not any(Path(webui_app_module.SIMULATION_DATA_DIR).glob("*.pkl"))
