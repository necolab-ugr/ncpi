import importlib.util
import shutil
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest


_SHARED_PATH = Path(__file__).resolve().with_name("test_hagen.py")
_SHARED_SPEC = importlib.util.spec_from_file_location("test_hagen_shared", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_SHARED_SPEC)
assert _SHARED_SPEC.loader is not None
_SHARED_SPEC.loader.exec_module(_shared)


REPO_ROOT = _shared.REPO_ROOT
HAGEN_EXAMPLE_DIR = REPO_ROOT / "examples" / "simulation" / "Hagen_model" / "simulation"

sync_playwright = _shared.sync_playwright
PlaywrightError = _shared.PlaywrightError
webui_app_module = _shared.webui_app_module
live_webui_server = _shared.live_webui_server
_attach_test_custom_terminal_reporter = _shared._attach_test_hagen_terminal_reporter


def _log_test_progress(message):
    text = f"[test_custom_upload] {message}"
    if _shared._CAPTURE_MANAGER is not None:
        with _shared._CAPTURE_MANAGER.global_and_fixture_disabled():
            _shared._write_progress_line(text)
        return
    _shared._write_progress_line(text)


def _hagen_example_input_paths():
    return {
        "network_params": HAGEN_EXAMPLE_DIR / "params" / "network_params.py",
        "network_py": HAGEN_EXAMPLE_DIR / "python" / "network.py",
        "simulation_params": HAGEN_EXAMPLE_DIR / "params" / "simulation_params.py",
        "simulation_py": HAGEN_EXAMPLE_DIR / "python" / "simulation.py",
    }


def _prepare_deterministic_hagen_custom_inputs(root_dir, numpy_seed=123456):
    root_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _hagen_example_input_paths()
    input_paths = {}
    for key, source_path in source_paths.items():
        target = root_dir / source_path.name
        shutil.copy2(source_path, target)
        input_paths[key] = target

    simulation_params = input_paths["simulation_params"]
    simulation_params.write_text(
        simulation_params.read_text(encoding="utf-8")
        + f"\n# Added by WebUI tests for deterministic custom-upload validation.\n"
        + f"numpy_seed = {int(numpy_seed)}\n",
        encoding="utf-8",
    )
    return input_paths


def _navigate_to_custom_simulation_form(page, live_webui_server):
    page.goto(f"{live_webui_server}/simulation/new_sim/custom", wait_until="domcontentloaded")
    page.locator("#custom-simulation-form").wait_for(state="visible")
    page.locator("#network_params_file_input").wait_for(state="attached")


def _set_custom_local_uploads(page, input_paths):
    page.evaluate(
        """() => {
            [
                "network_params_file",
                "network_py_file",
                "simulation_params_file",
                "simulation_py_file",
            ].forEach((fieldName) => {
                const modeInput = document.getElementById(`${fieldName}_source_mode`);
                const serverInput = document.getElementById(`${fieldName}_server_path`);
                if (modeInput) {
                    modeInput.value = "upload";
                }
                if (serverInput) {
                    serverInput.value = "";
                }
            });
        }"""
    )
    page.locator("#network_params_file_input").set_input_files(str(input_paths["network_params"]))
    page.locator("#network_py_file_input").set_input_files(str(input_paths["network_py"]))
    page.locator("#simulation_params_file_input").set_input_files(str(input_paths["simulation_params"]))
    page.locator("#simulation_py_file_input").set_input_files(str(input_paths["simulation_py"]))


def _select_custom_server_file_via_modal(page, field_name, file_path):
    card = page.locator(f'.custom-file-card[data-field="{field_name}"]').first
    card.locator(".custom-mode-server-btn").click()
    page.evaluate(
        """([modeSelector, pathSelector, value]) => {
            document.querySelector(modeSelector).value = "server-path";
            document.querySelector(pathSelector).value = value;
        }""",
        [
            f"#{field_name}_source_mode",
            f"#{field_name}_server_path",
            str(file_path),
        ],
    )
    card.click()
    modal = page.locator("#custom-sim-server-file-modal")
    modal.wait_for(state="visible")
    modal.locator(f'input[name="custom-sim-server-file-selected"][value="{str(file_path)}"]').wait_for(
        state="attached"
    )
    modal.locator("#custom-sim-server-file-select").click()
    modal.wait_for(state="hidden")
    server_path_value = page.locator(f"#{field_name}_server_path").input_value()
    assert server_path_value == str(file_path)


def _set_custom_server_paths(page, input_paths):
    field_map = {
        "network_params_file": input_paths["network_params"],
        "network_py_file": input_paths["network_py"],
        "simulation_params_file": input_paths["simulation_params"],
        "simulation_py_file": input_paths["simulation_py"],
    }
    for field_name, file_path in field_map.items():
        card = page.locator(f'.custom-file-card[data-field="{field_name}"]').first
        card.locator(".custom-mode-server-btn").click()
        page.evaluate(
            """([modeSelector, pathSelector, value]) => {
                document.querySelector(modeSelector).value = "server-path";
                document.querySelector(pathSelector).value = value;
            }""",
            [
                f"#{field_name}_source_mode",
                f"#{field_name}_server_path",
                str(file_path),
            ],
        )


def _submit_custom_simulation(page):
    with page.expect_navigation(url="**/job_status/*", wait_until="domcontentloaded"):
        page.locator("#custom-simulation-form").evaluate("(form) => form.submit()")
    return Path(urllib.parse.urlparse(page.url).path).name


def _run_custom_webui_job_with_local_uploads(live_webui_server, pytestconfig, input_paths):
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
                _log_test_progress(f"opening custom simulation form at {live_webui_server}")
                _navigate_to_custom_simulation_form(page, live_webui_server)
                _log_test_progress("uploading local custom simulation files")
                _set_custom_local_uploads(page, input_paths)
                _log_test_progress("submitting custom simulation")
                job_id = _submit_custom_simulation(page)
                _log_test_progress(f"submitted custom simulation as job {job_id}")
                status_payload = _shared._wait_for_job_completion(live_webui_server, job_id)
                assert status_payload["status"] == "finished", status_payload
                _log_test_progress(f"custom simulation job {job_id} finished successfully")
            finally:
                browser.close()

    return job_id


def _run_custom_webui_job_with_server_paths(live_webui_server, pytestconfig, input_paths):
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
                _log_test_progress(f"opening custom simulation form at {live_webui_server}")
                _navigate_to_custom_simulation_form(page, live_webui_server)
                _log_test_progress("setting server-path custom simulation files")
                _set_custom_server_paths(page, input_paths)
                _log_test_progress("submitting custom simulation")
                job_id = _submit_custom_simulation(page)
                _log_test_progress(f"submitted custom simulation as job {job_id}")
                status_payload = _shared._wait_for_job_completion(live_webui_server, job_id)
                assert status_payload["status"] == "finished", status_payload
                _log_test_progress(f"custom simulation job {job_id} finished successfully")
            finally:
                browser.close()

    return job_id


def _run_python_custom_reference(input_paths, output_dir):
    params_dir = output_dir.parent / "params"
    python_dir = output_dir.parent / "python"
    params_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_label = output_dir.parent.name
    _log_test_progress(f"preparing direct Python custom reference for {trial_label}")

    shutil.copy2(input_paths["network_params"], params_dir / "network_params.py")
    shutil.copy2(input_paths["simulation_params"], params_dir / "simulation_params.py")
    shutil.copy2(input_paths["network_py"], python_dir / "network.py")
    shutil.copy2(input_paths["simulation_py"], python_dir / "simulation.py")

    import ncpi

    simulation = ncpi.Simulation(
        param_folder=str(params_dir),
        python_folder=str(python_dir),
        output_folder=str(output_dir),
    )
    _log_test_progress(f"running direct Python network build for {trial_label}")
    simulation.network("network.py", "network_params.py")
    _log_test_progress(f"running direct Python simulation for {trial_label}")
    simulation.simulate("simulation.py", "simulation_params.py")
    _log_test_progress(f"completed direct Python custom reference for {trial_label}")

    return output_dir


def _assert_custom_webui_matches_direct_python_reference(
    live_webui_server,
    job_id,
    input_paths,
    tmp_path,
    bin_width_ms=100.0,
):
    _log_test_progress(f"downloading WebUI results for job {job_id}")
    ui_zip_path = _shared._download_simulation_zip(
        live_webui_server,
        job_id,
        tmp_path / "webui_simulation_results.zip",
    )
    ui_output_dir = _shared._extract_zip(ui_zip_path, tmp_path / "webui_results")
    webui_firing_rate_dfs = _shared._mean_firing_rate_by_bin_dataframes(ui_output_dir, bin_width_ms=bin_width_ms)

    python_output_dir = _run_python_custom_reference(input_paths, tmp_path / "python_reference" / "output")
    python_firing_rate_dfs = _shared._mean_firing_rate_by_bin_dataframes(
        python_output_dir,
        bin_width_ms=bin_width_ms,
    )

    assert len(webui_firing_rate_dfs) == 1
    assert len(python_firing_rate_dfs) == 1
    pd.testing.assert_frame_equal(webui_firing_rate_dfs[0], python_firing_rate_dfs[0])


def _run_custom_simulation_expect_flash(
    live_webui_server,
    pytestconfig,
    configure_page,
    expected_flash_text,
):
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
                _navigate_to_custom_simulation_form(page, live_webui_server)
                configure_page(page)
                with page.expect_navigation(
                    url="**/simulation/new_sim/custom",
                    wait_until="domcontentloaded",
                ):
                    page.locator("#custom-simulation-form").evaluate("(form) => form.submit()")
                flash = page.locator(".flash-container").first
                flash.wait_for(state="visible")
                expect_text = flash.inner_text()
                assert expected_flash_text in expect_text
            finally:
                browser.close()


def _run_custom_simulation_expect_failed_job(
    live_webui_server,
    pytestconfig,
    input_paths,
):
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
                _navigate_to_custom_simulation_form(page, live_webui_server)
                _set_custom_local_uploads(page, input_paths)
                job_id = _submit_custom_simulation(page)
                status_payload = _shared._wait_for_job_completion(live_webui_server, job_id)
            finally:
                browser.close()

    assert status_payload["status"] == "failed", status_payload
    assert status_payload["error"]
    return status_payload


@pytest.mark.slow
def test_custom_simulation_local_upload_hagen_example_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting custom simulation local-upload validation against direct Python")
    webui_app_module._clear_simulation_output_folder_all_files()
    input_paths = _prepare_deterministic_hagen_custom_inputs(tmp_path / "local_upload_inputs")
    job_id = _run_custom_webui_job_with_local_uploads(
        live_webui_server,
        pytestconfig,
        input_paths,
    )
    _assert_custom_webui_matches_direct_python_reference(
        live_webui_server,
        job_id,
        input_paths,
        tmp_path,
    )


@pytest.mark.slow
def test_custom_simulation_server_path_hagen_example_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting custom simulation server-path validation against direct Python")
    webui_app_module._clear_simulation_output_folder_all_files()
    input_paths = _prepare_deterministic_hagen_custom_inputs(tmp_path / "server_path_inputs")
    job_id = _run_custom_webui_job_with_server_paths(
        live_webui_server,
        pytestconfig,
        input_paths,
    )
    _assert_custom_webui_matches_direct_python_reference(
        live_webui_server,
        job_id,
        input_paths,
        tmp_path,
    )


def test_custom_simulation_server_file_browser_can_select_python_file(
    live_webui_server, pytestconfig
):
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))
    file_path = _hagen_example_input_paths()["network_params"]

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
                _navigate_to_custom_simulation_form(page, live_webui_server)
                _select_custom_server_file_via_modal(
                    page,
                    "network_params_file",
                    file_path,
                )
                summary = page.locator('.custom-file-card[data-field="network_params_file"] [data-role="server-path"]').inner_text()
                assert str(file_path) in summary
            finally:
                browser.close()


def test_custom_simulation_form_requires_all_files_client_side(
    live_webui_server, pytestconfig
):
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
                _navigate_to_custom_simulation_form(page, live_webui_server)
                start_url = page.url
                page.locator('button[type="submit"]').click()
                error_box = page.locator("#custom-form-error")
                error_box.wait_for(state="visible")
                assert "Missing required custom files." in error_box.inner_text()
                assert page.url == start_url
            finally:
                browser.close()


def test_custom_simulation_rejects_non_python_upload(
    live_webui_server, tmp_path, pytestconfig
):
    bad_file = tmp_path / "not_python.txt"
    bad_file.write_text("not python", encoding="utf-8")
    input_paths = _hagen_example_input_paths()

    def configure_page(page):
        _set_custom_local_uploads(
            page,
            {
                "network_params": input_paths["network_params"],
                "network_py": input_paths["network_py"],
                "simulation_params": input_paths["simulation_params"],
                "simulation_py": bad_file,
            },
        )

    _run_custom_simulation_expect_flash(
        live_webui_server,
        pytestconfig,
        configure_page,
        "Custom simulation uploads must be Python files (.py).",
    )


@pytest.mark.slow
def test_custom_simulation_surfaces_uploaded_script_failure(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    _log_test_progress("starting custom simulation failing-script scenario")
    webui_app_module._clear_simulation_output_folder_all_files()
    valid_inputs = _hagen_example_input_paths()
    bad_root = tmp_path / "bad_custom_inputs"
    bad_root.mkdir(parents=True, exist_ok=True)
    input_paths = {}
    for key, source_path in valid_inputs.items():
        target = bad_root / source_path.name
        shutil.copy2(source_path, target)
        input_paths[key] = target

    input_paths["simulation_py"].write_text(
        "raise RuntimeError('intentional custom simulation failure')\n",
        encoding="utf-8",
    )

    status_payload = _run_custom_simulation_expect_failed_job(
        live_webui_server,
        pytestconfig,
        input_paths,
    )
    assert "RuntimeError" in str(status_payload["error"])
