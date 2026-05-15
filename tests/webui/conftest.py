import importlib
import importlib.util
import threading
from pathlib import Path

import pytest
from werkzeug.serving import make_server


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_TESTS_DIR = Path(__file__).resolve().parent / "Simulation"
FIELD_POTENTIAL_HELPERS_PATH = Path(__file__).resolve().parent / "FieldPotential" / "_field_potential_test_helpers.py"


def _is_field_potential_test(request):
    node_path = getattr(request.node, "path", None)
    if node_path is None:
        return False
    return "FieldPotential" in str(node_path)


def _load_simulation_test_module(filename, module_name):
    module_path = SIMULATION_TESTS_DIR / filename
    return _load_test_module(module_path, module_name)


def _load_test_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_hagen = _load_simulation_test_module("test_hagen.py", "webui_sim_test_hagen")
_cavallari = _load_simulation_test_module("test_cavallari.py", "webui_sim_test_cavallari")
_four_area = _load_simulation_test_module("test_four_area.py", "webui_sim_test_four_area")
_field_potential_helpers = _load_test_module(FIELD_POTENTIAL_HELPERS_PATH, "webui_fp_test_helpers")
fake_field_potential_backend = _field_potential_helpers.fake_field_potential_backend

# Field-potential tests intentionally reuse the same Simulation test helpers
# used by the model-specific Simulation suites.
for _module, _required in (
    (_hagen, ("_reload_webui_app", "_set_fast_simulation_defaults", "_run_hagen_webui_job", "_download_simulation_zip", "_extract_zip")),
    (_cavallari, ("_run_cavallari_webui_job",)),
    (_four_area, ("_run_four_area_webui_job",)),
):
    for _name in _required:
        assert hasattr(_module, _name), f"Expected shared Simulation helper '{_name}' in {_module.__name__}."


@pytest.fixture(scope="session")
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
def _attach_field_potential_terminal_reporter(pytestconfig, request):
    if not _is_field_potential_test(request):
        yield
        return
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
def _clear_field_potential_outputs(request):
    if not _is_field_potential_test(request):
        yield
        return
    field_potential_webui_app_module = request.getfixturevalue("field_potential_webui_app_module")
    field_potential_webui_app_module._clear_field_potential_data_files()
    yield
    field_potential_webui_app_module._clear_field_potential_data_files()


@pytest.fixture(scope="session")
def compute_utils_module(field_potential_webui_app_module):
    module = importlib.import_module("compute_utils")
    module.refresh_tmp_paths()
    yield module
    module.refresh_tmp_paths()


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


def _run_and_extract_simulation_output(
    app_module,
    base_url,
    run_fn,
    configure_page,
    pytestconfig,
    output_root,
    model_name,
    scenario_name,
):
    app_module._clear_simulation_output_folder_all_files()
    form_data, job_id = run_fn(base_url, pytestconfig, configure_page)
    zip_path = _hagen._download_simulation_zip(
        base_url,
        job_id,
        output_root / f"{model_name}_{scenario_name}_{job_id}_simulation_results.zip",
    )
    output_dir = _hagen._extract_zip(zip_path, output_root / f"{job_id}_simulation_output")
    return {
        "model": model_name,
        "scenario": scenario_name,
        "form_data": form_data,
        "job_id": job_id,
        "output_dir": output_dir,
    }


def _hagen_scenarios(app_module):
    alternate_values = _hagen._build_hagen_alternate_single_values(app_module.HAGEN_DEFAULTS)
    j_yx_leaf_candidates, _ = _hagen._build_hagen_j_yx_margin_candidates(app_module.HAGEN_DEFAULTS["J_YX"])

    def _default(page):
        page.locator("#sim-use-numpy-seed").check()

    def _alternate(page):
        page.locator("#sim-use-numpy-seed").check()
        _hagen._apply_hagen_single_parameter_overrides(page, alternate_values)

    def _alternate_repetitions(page):
        page.locator("#sim-use-numpy-seed").check()
        _hagen._apply_hagen_single_parameter_overrides(page, alternate_values)
        page.locator("#sim-repetitions").fill("2")

    def _grid(page):
        page.locator("#sim-use-numpy-seed").check()
        page.locator("input[name='sim_run_mode'][value='grid']").check()
        for leaf_index, candidates in enumerate(j_yx_leaf_candidates):
            _hagen._fill_hagen_grid_leaf_candidates(page, "J_YX", leaf_index=leaf_index, candidates=candidates)

    return [
        ("default", _default),
        ("alternate", _alternate),
        ("alternate_repetitions_2", _alternate_repetitions),
        ("grid_j_yx", _grid),
    ]


def _cavallari_scenarios(app_module):
    alternate_values = _cavallari._build_cavallari_alternate_single_values(app_module.CAVALLARI_DEFAULTS)
    recurrent_candidates, _ = _cavallari._build_cavallari_recurrent_margin_candidates(app_module.CAVALLARI_DEFAULTS)

    def _default(page):
        page.locator("#sim-use-numpy-seed").check()

    def _alternate(page):
        page.locator("#sim-use-numpy-seed").check()
        _cavallari._apply_cavallari_single_parameter_overrides(page, alternate_values)

    def _alternate_repetitions(page):
        page.locator("#sim-use-numpy-seed").check()
        _cavallari._apply_cavallari_single_parameter_overrides(page, alternate_values)
        page.locator("#sim-repetitions").fill("2")

    def _grid(page):
        page.locator("#sim-use-numpy-seed").check()
        page.locator("input[name='sim_run_mode'][value='grid']").check()
        for param_name in _cavallari.CAVALLARI_RECURRENT_GRID_KEYS:
            start, end = recurrent_candidates[param_name]
            _cavallari._fill_cavallari_grid_scalar_param(
                page,
                param_name,
                start=float(start),
                step=float(end) - float(start),
                end=float(end),
            )

    return [
        ("default", _default),
        ("alternate", _alternate),
        ("alternate_repetitions_2", _alternate_repetitions),
        ("grid_recurrent_weights", _grid),
    ]


def _four_area_scenarios(app_module):
    alternate_values = _four_area._build_four_area_alternate_single_values(app_module)
    defaults = _four_area._four_area_default_grid_values(app_module)
    j_yx_leaf_candidates, _ = _four_area._build_four_area_local_j_yx_margin_candidates(defaults["J_YX"])

    def _default(page):
        page.locator("#sim-use-numpy-seed").check()

    def _alternate(page):
        page.locator("#sim-use-numpy-seed").check()
        _four_area._shared._apply_hagen_single_parameter_overrides(page, alternate_values)

    def _alternate_repetitions(page):
        page.locator("#sim-use-numpy-seed").check()
        _four_area._shared._apply_hagen_single_parameter_overrides(page, alternate_values)
        page.locator("#sim-repetitions").fill("2")

    def _grid(page):
        page.locator("#sim-use-numpy-seed").check()
        page.locator("input[name='sim_run_mode'][value='grid']").check()
        for leaf_index, candidates in enumerate(j_yx_leaf_candidates):
            _four_area._shared._fill_hagen_grid_leaf_candidates(
                page,
                "area_0.J_YX",
                leaf_index=leaf_index,
                candidates=candidates,
            )

    return [
        ("default", _default),
        ("alternate", _alternate),
        ("alternate_repetitions_2", _alternate_repetitions),
        ("grid_local_j_yx", _grid),
    ]


@pytest.fixture(scope="session")
def field_potential_simulation_scenarios(
    field_potential_webui_app_module,
    live_webui_server,
    pytestconfig,
    tmp_path_factory,
):
    output_root = tmp_path_factory.mktemp("field_potential_simulation_scenarios")
    scenario_builders = {
        "hagen": (_hagen._run_hagen_webui_job, _hagen_scenarios(field_potential_webui_app_module)),
        "cavallari": (_cavallari._run_cavallari_webui_job, _cavallari_scenarios(field_potential_webui_app_module)),
        "four_area": (_four_area._run_four_area_webui_job, _four_area_scenarios(field_potential_webui_app_module)),
    }

    outputs = {}
    for model_name, (run_fn, scenarios) in scenario_builders.items():
        outputs[model_name] = {}
        for scenario_name, configure_page in scenarios:
            outputs[model_name][scenario_name] = _run_and_extract_simulation_output(
                field_potential_webui_app_module,
                live_webui_server,
                run_fn,
                configure_page,
                pytestconfig,
                output_root,
                model_name,
                scenario_name,
            )
    return outputs


@pytest.fixture(scope="session")
def field_potential_simulation_cases(field_potential_simulation_scenarios):
    cases = []
    for model_name in ("hagen", "cavallari", "four_area"):
        for scenario_name in (
            "default",
            "alternate",
            "alternate_repetitions_2",
            "grid_j_yx" if model_name == "hagen" else (
                "grid_recurrent_weights" if model_name == "cavallari" else "grid_local_j_yx"
            ),
        ):
            sim_output = field_potential_simulation_scenarios[model_name][scenario_name]
            cases.append({
                "case_id": f"{model_name}:{scenario_name}",
                "model": model_name,
                "scenario": scenario_name,
                "sim_output": sim_output,
            })
    return cases


@pytest.fixture(scope="session")
def hagen_simulation_output(field_potential_simulation_scenarios):
    return field_potential_simulation_scenarios["hagen"]["default"]


@pytest.fixture(scope="session")
def cavallari_simulation_output(field_potential_simulation_scenarios):
    return field_potential_simulation_scenarios["cavallari"]["default"]


@pytest.fixture(scope="session")
def four_area_simulation_output(field_potential_simulation_scenarios):
    return field_potential_simulation_scenarios["four_area"]["default"]
