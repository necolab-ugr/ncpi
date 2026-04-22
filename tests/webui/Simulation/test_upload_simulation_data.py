import importlib.util
import json
import pickle
import threading
import urllib.parse
import urllib.request
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
    """Emit a namespaced progress message for this test module."""
    text = f"[test_sim_upload] {message}"
    if _shared._CAPTURE_MANAGER is not None:
        with _shared._CAPTURE_MANAGER.global_and_fixture_disabled():
            _shared._write_progress_line(text)
        return
    _shared._write_progress_line(text)


def _build_pickle_file(path, payload):
    """Create a pickle file with the provided payload for upload-form tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def _build_simulation_output_files(root_dir, payload_by_name):
    """Create a set of simulation output pickle files under the provided directory."""
    return {
        name: _build_pickle_file(root_dir / name, payload)
        for name, payload in payload_by_name.items()
    }


def _post_proxy_trial_inference(base_url, file_key):
    """Call the proxy trial-count inference endpoint using the uploaded default simulation data."""
    body = urllib.parse.urlencode({
        "file_key": file_key,
        "use_default": "1",
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/field_potential/proxy/infer_trials",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _navigate_to_upload_sim_page(page, live_webui_server):
    """Open the simulation-data upload page and wait until it is ready for interaction."""
    page.goto(f"{live_webui_server}/simulation/upload_sim", wait_until="domcontentloaded")
    page.locator("#simulationUploadForm").wait_for(state="visible")
    page.locator("#spikeFileInput").wait_for(state="attached")


def _set_local_simulation_upload_files(page, paths):
    """Populate the local simulation-data upload controls with pickle files."""
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
    """Submit the local simulation-data upload form."""
    with page.expect_navigation(url="**/simulation/upload_sim", wait_until="domcontentloaded"):
        page.locator("#simulationUploadForm").evaluate("(form) => form.submit()")


@pytest.fixture(scope="module")
def webui_app_module(_webui_test_session):
    """Provide a freshly reloaded WebUI app module for the current test module."""
    if sync_playwright is None:
        pytest.skip("Playwright is required for the web UI simulation upload tests.")

    module = _shared._reload_webui_app()
    _shared._set_fast_simulation_defaults(module)
    module._clear_simulation_output_folder_all_files()
    yield module
    module._clear_simulation_output_folder_all_files()


@pytest.fixture(autouse=True)
def _attach_test_upload_terminal_reporter(pytestconfig):
    """Attach pytest terminal-reporting plugins so progress logs remain visible."""
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
    """Serve the WebUI app on a temporary local HTTP server for a single test."""
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
    """Verify that simulation upload local pickle files updates module state."""
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
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
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
    """Verify that simulation upload server path pickle files updates module state."""
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
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
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
    """Verify that simulation upload invalid local extension shows client side error."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("bad", encoding="utf-8")
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
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
    """Verify that simulation upload invalid server path extension rejected with flash."""
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
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
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
    """Verify that simulation upload remove file updates module state."""
    _log_test_progress("starting simulation-data removal scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    local_file = _build_pickle_file(tmp_path / "dt.pkl", 0.0625)
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
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


def test_simulation_upload_complete_hagen_payload_is_semantically_loadable(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    """Verify that a complete uploaded Hagen payload can be loaded and interpreted semantically."""
    _log_test_progress("starting semantic Hagen simulation-data upload scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    payloads = {
        "times.pkl": {"E": [1.0, 2.5], "I": [3.0]},
        "gids.pkl": {"E": [0, 1], "I": [2]},
        "dt.pkl": 0.2,
        "tstop.pkl": 5.0,
        "network.pkl": {"X": ["E", "I"], "N_X": [2, 1], "model": "iaf_psc_exp"},
    }
    local_files = _build_simulation_output_files(tmp_path / "complete_hagen_payload", payloads)
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                _set_local_simulation_upload_files(page, list(local_files.values()))
                _submit_local_simulation_upload(page)
                page.locator("text=Status: Files detected").wait_for(state="visible")
            finally:
                browser.close()

    loaded = webui_app_module._load_simulation_outputs()
    assert loaded["times"] == payloads["times.pkl"]
    assert loaded["gids"] == payloads["gids.pkl"]
    assert loaded["dt"] == payloads["dt.pkl"]
    assert loaded["tstop"] == payloads["tstop.pkl"]
    assert loaded["network"] == payloads["network.pkl"]
    assert webui_app_module._simulation_trial_count(loaded) == 1
    assert webui_app_module._simulation_model_type(loaded) == "hagen"

    infer_payload = _post_proxy_trial_inference(live_webui_server, "times_file")
    assert infer_payload["trial_count"] == 1
    assert infer_payload["source_kind"] == "default-simulation"
    assert infer_payload["file_key"] == "times_file"


def test_simulation_upload_complete_four_area_repeated_payload_is_semantically_loadable(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    """Verify that a repeated four-area payload can be loaded and interpreted semantically."""
    _log_test_progress("starting semantic four-area repeated simulation-data upload scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    payloads = {
        "times.pkl": [
            {
                "frontal": {"E": [1.0], "I": [2.0]},
                "parietal": {"E": [1.5], "I": [2.5]},
            },
            {
                "frontal": {"E": [3.0], "I": [4.0]},
                "parietal": {"E": [3.5], "I": [4.5]},
            },
        ],
        "gids.pkl": [
            {
                "frontal": {"E": [0], "I": [1]},
                "parietal": {"E": [2], "I": [3]},
            },
            {
                "frontal": {"E": [4], "I": [5]},
                "parietal": {"E": [6], "I": [7]},
            },
        ],
        "dt.pkl": [0.2, 0.2],
        "tstop.pkl": [5.0, 5.0],
        "network.pkl": [
            {"areas": ["frontal", "parietal"], "X": ["E", "I"], "N_X": [1, 1], "model": "iaf_psc_exp"},
            {"areas": ["frontal", "parietal"], "X": ["E", "I"], "N_X": [1, 1], "model": "iaf_psc_exp"},
        ],
    }
    server_files = _build_simulation_output_files(tmp_path / "complete_four_area_payload", payloads)
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _shared._playwright_temp_environment():
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=not show_browser,
                    slow_mo=_shared.PLAYWRIGHT_SHOW_BROWSER_SLOW_MO_MS if show_browser else 0,
                )
            except PlaywrightError as exc:
                pytest.skip(f"Playwright Chromium is not available: {exc}")

            try:
                page = browser.new_page()
                _navigate_to_upload_sim_page(page, live_webui_server)
                page.evaluate(
                    """(paths) => {
                        document.querySelector("#simulationSourceModeInput").value = "server-path";
                        document.querySelector("#simulationServerFilePathInput").value = paths.join("\\n");
                    }""",
                    [str(path) for path in server_files.values()],
                )
                page.locator("#simulationUploadForm").evaluate("(form) => form.submit()")
                page.wait_for_load_state("domcontentloaded")
                page.locator("text=Status: Files detected").wait_for(state="visible")
            finally:
                browser.close()

    loaded = webui_app_module._load_simulation_outputs()
    assert loaded["times"] == payloads["times.pkl"]
    assert loaded["gids"] == payloads["gids.pkl"]
    assert loaded["dt"] == payloads["dt.pkl"]
    assert loaded["tstop"] == payloads["tstop.pkl"]
    assert loaded["network"] == payloads["network.pkl"]
    assert webui_app_module._simulation_trial_count(loaded) == 2
    assert webui_app_module._simulation_model_type(loaded) == "four_area"
    assert webui_app_module._simulation_area_names(loaded) == ["frontal", "parietal"]

    infer_payload = _post_proxy_trial_inference(live_webui_server, "times_file")
    assert infer_payload["trial_count"] == 2
    assert infer_payload["source_kind"] == "default-simulation"
    assert infer_payload["file_key"] == "times_file"
