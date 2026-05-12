import importlib
import importlib.util
import threading
from pathlib import Path

import pytest
from werkzeug.serving import make_server


REPO_ROOT = Path(__file__).resolve().parents[3]
SIMULATION_TESTS_DIR = Path(__file__).resolve().parents[1] / "Simulation"


def _load_simulation_test_module(filename, module_name):
    module_path = SIMULATION_TESTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_hagen = _load_simulation_test_module("test_hagen.py", "webui_sim_test_hagen")
_cavallari = _load_simulation_test_module("test_cavallari.py", "webui_sim_test_cavallari")
_four_area = _load_simulation_test_module("test_four_area.py", "webui_sim_test_four_area")


@pytest.fixture(scope="module")
def field_potential_webui_app_module(_webui_test_session):
    if importlib.util.find_spec("nest") is None:
        pytest.skip("NEST is required for webui simulation-backed field-potential tests.")
    if _hagen.sync_playwright is None:
        pytest.skip("Playwright is required for webui simulation-backed field-potential tests.")

    module = _hagen._reload_webui_app()
    _hagen._set_fast_simulation_defaults(module)
    module._clear_simulation_output_folder_all_files()
    module._clear_field_potential_data_files()
    yield module
    module._clear_simulation_output_folder_all_files()
    module._clear_field_potential_data_files()


@pytest.fixture(autouse=True)
def _attach_field_potential_terminal_reporter(pytestconfig):
    previous_reporter = _hagen._TERMINAL_REPORTER
    previous_capture_manager = _hagen._CAPTURE_MANAGER
    _hagen._TERMINAL_REPORTER = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    _hagen._CAPTURE_MANAGER = pytestconfig.pluginmanager.get_plugin("capturemanager")
    try:
        yield
    finally:
        _hagen._TERMINAL_REPORTER = previous_reporter
        _hagen._CAPTURE_MANAGER = previous_capture_manager


@pytest.fixture(autouse=True)
def _clear_field_potential_outputs(field_potential_webui_app_module):
    field_potential_webui_app_module._clear_field_potential_data_files()
    yield
    field_potential_webui_app_module._clear_field_potential_data_files()


@pytest.fixture(scope="module")
def compute_utils_module(field_potential_webui_app_module):
    module = importlib.import_module("compute_utils")
    module.refresh_tmp_paths()
    yield module
    module.refresh_tmp_paths()


@pytest.fixture(scope="module")
def field_potential_example_paths():
    return {
        "hagen": {
            "mc_folder": str(REPO_ROOT / "examples" / "simulation" / "Hagen_model" / "simulation"),
            "kernel_params_module": str(
                REPO_ROOT
                / "examples"
                / "simulation"
                / "Hagen_model"
                / "simulation"
                / "params"
                / "analysis_params.py"
            ),
        },
        "cavallari": {
            "mc_folder": str(REPO_ROOT / "examples" / "simulation" / "Cavallari_model" / "MC_simulation"),
            "kernel_params_module": str(
                REPO_ROOT
                / "examples"
                / "simulation"
                / "Cavallari_model"
                / "MC_simulation"
                / "analysis_params.py"
            ),
        },
        "four_area": {
            "mc_folder": str(REPO_ROOT / "examples" / "simulation" / "four_area_cortical_model" / "simulation"),
            "kernel_params_module": str(
                REPO_ROOT
                / "examples"
                / "simulation"
                / "four_area_cortical_model"
                / "simulation"
                / "params"
                / "analysis_params.py"
            ),
        },
    }


@pytest.fixture(scope="module")
def live_webui_server(field_potential_webui_app_module, pytestconfig):
    server_port = int(pytestconfig.getoption("webui_port") or 0)
    server = make_server("127.0.0.1", server_port, field_potential_webui_app_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _run_and_extract_simulation_output(base_url, run_fn, configure_page, pytestconfig, output_root):
    form_data, job_id = run_fn(base_url, pytestconfig, configure_page)
    zip_path = _hagen._download_simulation_zip(base_url, job_id, output_root / f"{job_id}_simulation_results.zip")
    output_dir = _hagen._extract_zip(zip_path, output_root / f"{job_id}_simulation_output")
    return {
        "form_data": form_data,
        "job_id": job_id,
        "output_dir": output_dir,
    }


@pytest.fixture(scope="module")
def hagen_simulation_output(live_webui_server, pytestconfig, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("field_potential_hagen_sim")
    return _run_and_extract_simulation_output(
        live_webui_server,
        _hagen._run_hagen_webui_job,
        lambda page: page.locator("#sim-use-numpy-seed").check(),
        pytestconfig,
        output_root,
    )


@pytest.fixture(scope="module")
def cavallari_simulation_output(live_webui_server, pytestconfig, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("field_potential_cavallari_sim")
    return _run_and_extract_simulation_output(
        live_webui_server,
        _cavallari._run_cavallari_webui_job,
        lambda page: page.locator("#sim-use-numpy-seed").check(),
        pytestconfig,
        output_root,
    )


@pytest.fixture(scope="module")
def four_area_simulation_output(live_webui_server, pytestconfig, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("field_potential_four_area_sim")
    return _run_and_extract_simulation_output(
        live_webui_server,
        _four_area._run_four_area_webui_job,
        lambda page: page.locator("#sim-use-numpy-seed").check(),
        pytestconfig,
        output_root,
    )
