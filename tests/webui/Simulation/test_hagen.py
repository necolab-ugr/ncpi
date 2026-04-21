import os
import importlib
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import threading
import time
import uuid
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pytest
from werkzeug.serving import make_server

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only when Playwright is absent.
    PlaywrightError = None
    sync_playwright = None


REPO_ROOT = Path(__file__).resolve().parents[3]
WEBUI_DIR = REPO_ROOT / "webui"
HAGEN_EXAMPLE_DIR = REPO_ROOT / "examples" / "simulation" / "Hagen_model" / "simulation"


def _ensure_import_paths():
    for entry in (str(REPO_ROOT), str(WEBUI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _reload_webui_app():
    _ensure_import_paths()
    for module_name in ("webui.app", "compute_utils", "tmp_paths"):
        if module_name in sys.modules:
            del sys.modules[module_name]
    module = importlib.import_module("webui.app")
    module.refresh_tmp_paths()
    return module


@contextmanager
def _playwright_temp_environment():
    # Keep Playwright browser profile paths short to avoid Chromium socket/path limits.
    temp_root = Path("/tmp/pw")
    temp_root.mkdir(parents=True, exist_ok=True)

    original_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    original_tempdir = tempfile.tempdir
    try:
        short_tmp = str(temp_root)
        os.environ["TMPDIR"] = short_tmp
        os.environ["TMP"] = short_tmp
        os.environ["TEMP"] = short_tmp
        tempfile.tempdir = short_tmp
        yield short_tmp
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = original_tempdir


def _form_data_from_page(page):
    return page.evaluate(
        """
        () => {
            const form = document.querySelector('#run-simulation-form');
            const data = {};
            for (const [key, value] of new FormData(form).entries()) {
                data[key] = value;
            }
            return data;
        }
        """
    )


def _wait_for_job_completion(base_url, job_id, timeout_seconds=1800):
    status_url = f"{base_url}/status/{job_id}"
    deadline = time.time() + timeout_seconds
    last_payload = None

    while time.time() < deadline:
        with urllib.request.urlopen(status_url) as response:
            last_payload = json.load(response)
        if last_payload["status"] in {"finished", "failed", "cancelled"}:
            return last_payload
        time.sleep(1.5)

    raise TimeoutError(f"Timed out waiting for simulation job {job_id}. Last payload: {last_payload}")


def _download_simulation_zip(base_url, job_id, destination):
    download_url = f"{base_url}/download_results/{job_id}?computation_type=simulation"
    with urllib.request.urlopen(download_url) as response:
        destination.write_bytes(response.read())
    return destination


def _extract_zip(zip_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return destination


def _load_output_pickle(output_dir, filename):
    with open(output_dir / filename, "rb") as handle:
        return pickle.load(handle)


def _mean_firing_rate_by_bin_dataframe(output_dir, bin_width_ms=100.0):
    times = _load_output_pickle(output_dir, "times.pkl")
    network = _load_output_pickle(output_dir, "network.pkl")
    tstop = float(_load_output_pickle(output_dir, "tstop.pkl"))

    populations = list(network["X"])
    population_sizes = {
        str(population): int(size)
        for population, size in zip(populations, network["N_X"])
    }
    bin_edges = np.arange(0.0, tstop + bin_width_ms, bin_width_ms, dtype=float)
    if bin_edges.size < 2 or bin_edges[-1] < tstop:
        bin_edges = np.append(bin_edges, tstop)
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    bin_width_seconds = (bin_ends - bin_starts) / 1000.0

    rows = []
    for population in populations:
        population_key = str(population)
        spike_times = np.asarray(times[population], dtype=float)
        spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
        mean_firing_rate_hz = spike_counts / population_sizes[population_key] / bin_width_seconds
        rows.extend(
            {
                "population": population_key,
                "bin_start_ms": float(start_ms),
                "bin_end_ms": float(end_ms),
                "mean_firing_rate_hz": float(rate_hz),
            }
            for start_ms, end_ms, rate_hz in zip(bin_starts, bin_ends, mean_firing_rate_hz)
        )

    return pd.DataFrame(rows).sort_values(["population", "bin_start_ms"], kind="mergesort").reset_index(drop=True)


def _run_python_hagen_reference(app_module, form_data, output_dir):
    params_dir = output_dir.parent / "params"
    python_dir = output_dir.parent / "python"
    params_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    (params_dir / "network_params.py").write_text(
        app_module._build_hagen_network_params(form_data),
        encoding="utf-8",
    )
    (params_dir / "simulation_params.py").write_text(
        app_module._build_simulation_params(form_data, app_module.HAGEN_DEFAULTS),
        encoding="utf-8",
    )

    shutil.copy2(HAGEN_EXAMPLE_DIR / "python" / "network.py", python_dir / "network.py")
    shutil.copy2(HAGEN_EXAMPLE_DIR / "python" / "simulation.py", python_dir / "simulation.py")
    app_module._enforce_simulation_chunk_seconds(str(python_dir / "simulation.py"), chunk_ms=1000.0)

    import ncpi

    simulation = ncpi.Simulation(
        param_folder=str(params_dir),
        python_folder=str(python_dir),
        output_folder=str(output_dir),
    )
    simulation.network("network.py", "network_params.py")
    simulation.simulate("simulation.py", "simulation_params.py")

    return output_dir


@pytest.fixture(scope="module")
def webui_app_module():
    if importlib.util.find_spec("nest") is None:
        pytest.skip("NEST is required for the Hagen web UI simulation test.")
    if sync_playwright is None:
        pytest.skip("Playwright is required for the web UI simulation tests.")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("NCPI_WEBUI_SESSION_ID", f"pytest_webui_simulation_{uuid.uuid4().hex}")
    monkeypatch.delenv("NCPI_WEBUI_SESSION_ROOT", raising=False)
    module = _reload_webui_app()
    module._clear_simulation_output_folder_all_files()
    yield module
    module._clear_simulation_output_folder_all_files()
    monkeypatch.undo()


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


@pytest.mark.slow
def test_hagen_default_trial_mean_firing_rates_match_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig
):
    webui_app_module._clear_simulation_output_folder_all_files()
    show_browser = bool(pytestconfig.getoption("webui_show_browser"))

    with _playwright_temp_environment():
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
                page.goto(live_webui_server, wait_until="domcontentloaded")

                page.locator("a[href='/simulation']").first.click()
                page.wait_for_url(f"{live_webui_server}/simulation", wait_until="domcontentloaded")

                page.locator("a[href='/simulation/new_sim']").first.click()
                page.wait_for_url(f"{live_webui_server}/simulation/new_sim", wait_until="domcontentloaded")

                page.locator("a[href='/simulation/new_sim/hagen']").first.click()
                page.wait_for_url(f"{live_webui_server}/simulation/new_sim/hagen", wait_until="domcontentloaded")

                # Keep the Hagen model defaults untouched and only enable the optional seed
                # so the web run and direct Python run are reproducible.
                page.locator("#sim-use-numpy-seed").check()
                form_data = _form_data_from_page(page)

                page.get_by_role("button", name="Run trial simulation").click()
                page.wait_for_url("**/job_status/*", wait_until="domcontentloaded")
                job_id = Path(urlparse(page.url).path).name
                status_payload = _wait_for_job_completion(live_webui_server, job_id)
                assert status_payload["status"] == "finished", status_payload
            finally:
                browser.close()

    ui_zip_path = _download_simulation_zip(
        live_webui_server,
        job_id,
        tmp_path / "webui_simulation_results.zip",
    )
    ui_output_dir = _extract_zip(ui_zip_path, tmp_path / "webui_results")
    python_output_dir = _run_python_hagen_reference(
        webui_app_module,
        form_data,
        tmp_path / "python_reference" / "output",
    )

    webui_firing_rates_df = _mean_firing_rate_by_bin_dataframe(ui_output_dir, bin_width_ms=100.0)
    python_firing_rates_df = _mean_firing_rate_by_bin_dataframe(python_output_dir, bin_width_ms=100.0)
    pd.testing.assert_frame_equal(webui_firing_rates_df, python_firing_rates_df)
