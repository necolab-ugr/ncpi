import ast
import importlib.util
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest


_SHARED_PATH = Path(__file__).resolve().with_name("test_hagen.py")
_SHARED_SPEC = importlib.util.spec_from_file_location("test_hagen_shared", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_SHARED_SPEC)
assert _SHARED_SPEC.loader is not None
_SHARED_SPEC.loader.exec_module(_shared)


REPO_ROOT = _shared.REPO_ROOT
FOUR_AREA_EXAMPLE_DIR = REPO_ROOT / "examples" / "simulation" / "four_area_cortical_model" / "simulation"
FOUR_AREA_RELATIVE_MARGIN = 0.10
FOUR_AREA_ALTERNATE_VECTOR_SCALES = {
    "N_X": [1.10, 0.90],
    "C_m_X": [0.90, 1.10],
    "tau_m_X": [1.10, 0.90],
}
FOUR_AREA_ALTERNATE_INTER_AREA_J_YX_SCALES = {
    (0, 0): 1.10,
    (0, 1): 0.90,
}
FOUR_AREA_LOCAL_J_YX_MATRIX_SHAPE = (2, 2)
FOUR_AREA_RUNTIME_CONTROL_KEYS = {"local_num_threads"}

sync_playwright = _shared.sync_playwright
PlaywrightError = _shared.PlaywrightError
webui_app_module = _shared.webui_app_module
live_webui_server = _shared.live_webui_server
_attach_test_four_area_terminal_reporter = _shared._attach_test_hagen_terminal_reporter


def _log_test_progress(message):
    text = f"[test_four_area] {message}"
    if _shared._CAPTURE_MANAGER is not None:
        with _shared._CAPTURE_MANAGER.global_and_fixture_disabled():
            _shared._write_progress_line(text)
        return
    _shared._write_progress_line(text)


def _four_area_default_grid_values(app_module):
    defaults = {
        key: app_module.FOUR_AREA_DEFAULTS[key]
        for key in app_module.FOUR_AREA_GRID_KEYS
        if key in app_module.FOUR_AREA_DEFAULTS
    }
    inter_area_p = app_module.FOUR_AREA_DEFAULTS["inter_area_p"]
    inter_area_scale = app_module.FOUR_AREA_DEFAULTS["inter_area_scale"]
    inter_area_delay = app_module.FOUR_AREA_DEFAULTS["inter_area_delay"]
    defaults["inter_area.C_YX"] = [[inter_area_p, inter_area_p], [0.0, 0.0]]
    defaults["inter_area.J_YX"] = [
        [
            app_module.FOUR_AREA_DEFAULTS["J_EE"] * inter_area_scale,
            app_module.FOUR_AREA_DEFAULTS["J_IE"] * inter_area_scale,
        ],
        [0.0, 0.0],
    ]
    defaults["inter_area.delay_YX"] = [[inter_area_delay, inter_area_delay], [0.0, 0.0]]
    return defaults


def _mean_firing_rate_by_bin_dataframe_from_payload(times, network, tstop, bin_width_ms=100.0):
    areas = list(network["areas"])
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
    for area in areas:
        area_key = str(area)
        for population in populations:
            population_key = str(population)
            spike_times = np.asarray(times[area][population], dtype=float)
            spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
            mean_firing_rate_hz = spike_counts / population_sizes[population_key] / bin_width_seconds
            rows.extend(
                {
                    "area": area_key,
                    "population": population_key,
                    "bin_start_ms": float(start_ms),
                    "bin_end_ms": float(end_ms),
                    "mean_firing_rate_hz": float(rate_hz),
                }
                for start_ms, end_ms, rate_hz in zip(bin_starts, bin_ends, mean_firing_rate_hz)
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["area", "population", "bin_start_ms"], kind="mergesort")
        .reset_index(drop=True)
    )


def _mean_firing_rate_by_bin_dataframes(output_dir, bin_width_ms=100.0):
    times_payloads = _shared._coerce_trial_payloads(_shared._load_output_pickle(output_dir, "times.pkl"))
    network_payloads = _shared._coerce_trial_payloads(_shared._load_output_pickle(output_dir, "network.pkl"))
    tstop_payloads = _shared._coerce_trial_payloads(_shared._load_output_pickle(output_dir, "tstop.pkl"))

    trial_count = len(times_payloads)
    if len(network_payloads) != trial_count or len(tstop_payloads) != trial_count:
        raise AssertionError(
            "Inconsistent repeated-trial payload sizes across times.pkl, network.pkl, and tstop.pkl."
        )

    return [
        _mean_firing_rate_by_bin_dataframe_from_payload(
            times,
            network,
            float(tstop),
            bin_width_ms=bin_width_ms,
        )
        for times, network, tstop in zip(times_payloads, network_payloads, tstop_payloads)
    ]


def _run_python_four_area_reference(app_module, form_data, output_dir):
    params_dir = output_dir.parent / "params"
    python_dir = output_dir.parent / "python"
    params_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_label = output_dir.parent.name
    _log_test_progress(f"preparing python reference for {trial_label} in {output_dir}")

    (params_dir / "network_params.py").write_text(
        app_module._build_four_area_network_params(form_data),
        encoding="utf-8",
    )
    (params_dir / "simulation_params.py").write_text(
        app_module._build_simulation_params(form_data, app_module.FOUR_AREA_DEFAULTS),
        encoding="utf-8",
    )

    _shared.shutil.copy2(FOUR_AREA_EXAMPLE_DIR / "python" / "network.py", python_dir / "network.py")
    _shared.shutil.copy2(FOUR_AREA_EXAMPLE_DIR / "python" / "simulation.py", python_dir / "simulation.py")
    app_module._enforce_simulation_chunk_seconds(str(python_dir / "simulation.py"), chunk_ms=1000.0)

    import ncpi

    simulation = ncpi.Simulation(
        param_folder=str(params_dir),
        python_folder=str(python_dir),
        output_folder=str(output_dir),
    )
    _log_test_progress(f"running python network build for {trial_label}")
    simulation.network("network.py", "network_params.py")
    _log_test_progress(f"running python simulation for {trial_label}")
    simulation.simulate("simulation.py", "simulation_params.py")
    _log_test_progress(f"completed python reference for {trial_label}")

    return output_dir


def _run_python_four_area_reference_dataframes(app_module, form_data, output_root, bin_width_ms=100.0):
    _, expanded_forms = app_module._expand_simulation_forms("four_area", form_data)
    total_trials = len(expanded_forms)

    trial_dataframes = []
    for trial_index, expanded_form in enumerate(expanded_forms):
        combo_index = int(expanded_form.get("sim_combo_index", trial_index))
        repeat_index = int(expanded_form.get("sim_repeat_index", 0))
        _log_test_progress(
            f"python reference trial {trial_index + 1}/{total_trials} "
            f"(configuration={combo_index}, repetition={repeat_index})"
        )
        output_dir = _run_python_four_area_reference(
            app_module,
            expanded_form,
            output_root / f"trial_{trial_index}" / "output",
        )
        dataframes = _mean_firing_rate_by_bin_dataframes(output_dir, bin_width_ms=bin_width_ms)
        assert len(dataframes) == 1
        trial_dataframes.append(dataframes[0])

    return trial_dataframes


def _independent_parse_four_area_value(raw_value, default):
    if raw_value is None:
        return default
    text = str(raw_value).strip()
    if text == "":
        return default
    if isinstance(default, str):
        return text
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    return ast.literal_eval(text)


def _independent_parse_four_area_grid_candidates(raw_value, default, key):
    if raw_value is None:
        return [default]
    text = str(raw_value).strip()
    if text == "":
        return [default]
    if not text.lower().startswith("grid="):
        return [_independent_parse_four_area_value(text, default)]

    spec = text[5:].strip()
    if isinstance(default, (int, float)) and ":" in spec and not any(ch in spec for ch in "[]{}(),"):
        candidates = _shared._independent_grid_range_candidates(spec, key)
    else:
        parsed = ast.literal_eval(spec)
        candidates = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]

    if isinstance(default, int) and not isinstance(default, bool):
        return [int(float(candidate)) for candidate in candidates]
    if isinstance(default, float):
        return [float(candidate) for candidate in candidates]
    return candidates


def _independent_expected_four_area_trials(app_module, form_data):
    defaults = _four_area_default_grid_values(app_module)
    run_mode = str(form_data.get("sim_run_mode", "single")).strip().lower() or "single"
    repetitions = int(float(str(form_data.get("sim_repetitions", "1")).strip() or "1"))

    grid_keys = [key for key in app_module.FOUR_AREA_GRID_KEYS if key in defaults]
    candidate_lists = {
        key: _independent_parse_four_area_grid_candidates(form_data.get(key), defaults[key], key)
        for key in grid_keys
    }
    numpy_seed = None
    if form_data.get("sim_use_numpy_seed") is not None:
        numpy_seed = int(float(str(form_data.get("sim_numpy_seed", "0")).strip() or "0"))

    if run_mode == "single":
        base_values = {key: candidate_lists[key][0] for key in grid_keys}
        combos = [base_values]
    elif run_mode == "grid":
        ordered_keys = list(candidate_lists)
        combos = [
            {key: value for key, value in zip(ordered_keys, combo_values)}
            for combo_values in product(*(candidate_lists[key] for key in ordered_keys))
        ]
    else:
        raise ValueError(f"Unsupported independent four-area run mode: {run_mode}")

    expected_trials = []
    for combo_index, values in enumerate(combos):
        network = {
            "areas": values["areas"],
            "X": values["X"],
            "N_X": values["N_X"],
            "C_m_X": values["C_m_X"],
            "tau_m_X": values["tau_m_X"],
            "E_L_X": values["E_L_X"],
            "C_YX": values["C_YX"],
            "J_YX": values["J_YX"],
            "delay_YX": values["delay_YX"],
            "tau_syn_YX": values["tau_syn_YX"],
            "n_ext": values["n_ext"],
            "nu_ext": values["nu_ext"],
            "J_ext": values["J_ext"],
            "model": values["model"],
            "inter_area": {
                "C_YX": values["inter_area.C_YX"],
                "J_YX": values["inter_area.J_YX"],
                "delay_YX": values["inter_area.delay_YX"],
            },
        }
        simulation = {
            "tstop": values["tstop"],
            "local_num_threads": values["local_num_threads"],
            "dt": values["dt"],
            "numpy_seed": numpy_seed,
        }
        for repeat_index in range(repetitions):
            expected_trials.append({
                "combo_index": combo_index,
                "repeat_index": repeat_index,
                "network": network,
                "simulation": simulation,
            })
    return expected_trials


def _assert_four_area_generated_parameter_files_match_expected(app_module, form_data, captured_param_dirs):
    expected_trials = _independent_expected_four_area_trials(app_module, form_data)
    assert len(captured_param_dirs) == len(expected_trials), (
        f"Captured {len(captured_param_dirs)} WebUI parameter-file snapshots, "
        f"expected {len(expected_trials)}."
    )

    for trial_index, (param_dir, expected_trial) in enumerate(zip(captured_param_dirs, expected_trials), start=1):
        actual_network = _shared._load_generated_python_namespace(param_dir / "network_params.py")["LIF_params"]
        actual_simulation = _shared._load_generated_python_namespace(param_dir / "simulation_params.py")

        assert _shared._independent_values_equal(actual_network, expected_trial["network"]), (
            f"Unexpected WebUI-generated network parameters for trial {trial_index}: "
            f"actual={actual_network!r}, expected={expected_trial['network']!r}"
        )
        assert _shared._independent_values_equal(actual_simulation, expected_trial["simulation"]), (
            f"Unexpected WebUI-generated simulation parameters for trial {trial_index}: "
            f"actual={actual_simulation!r}, expected={expected_trial['simulation']!r}"
        )


def _navigate_to_four_area_form(page, live_webui_server):
    page.goto(live_webui_server, wait_until="domcontentloaded")
    page.locator("a[href='/simulation']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation", wait_until="domcontentloaded")
    page.locator("a[href='/simulation/new_sim']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation/new_sim", wait_until="domcontentloaded")
    page.locator("a[href='/simulation/new_sim/four_area']").first.click()
    page.wait_for_url(f"{live_webui_server}/simulation/new_sim/four_area", wait_until="domcontentloaded")
    page.wait_for_function(
        """
        () => {
            const container = document.querySelector('#parameter-container');
            const nXInput = document.querySelector('.param-input[data-param="N_X"]');
            const interAreaInput = document.querySelector('.param-input[data-param="inter_area.J_YX"]');
            return Boolean(
                container &&
                container.querySelector('.parameter-section') &&
                nXInput &&
                nXInput.parentElement?.querySelector('.single-array-controls') &&
                interAreaInput &&
                interAreaInput.parentElement?.querySelector('.single-array-controls')
            );
        }
        """
    )


def _scale_four_area_parameter(value, factor):
    return _shared._scale_hagen_parameter(value, factor)


def _build_four_area_alternate_single_values(app_module):
    defaults = _four_area_default_grid_values(app_module)
    alternate = {
        param_name: [
            _scale_four_area_parameter(value, factor)
            for value, factor in zip(defaults[param_name], factors)
        ]
        for param_name, factors in FOUR_AREA_ALTERNATE_VECTOR_SCALES.items()
    }

    inter_area_j_yx = np.array(defaults["inter_area.J_YX"], dtype=float, copy=True)
    for (row_index, column_index), factor in FOUR_AREA_ALTERNATE_INTER_AREA_J_YX_SCALES.items():
        inter_area_j_yx[row_index, column_index] = _scale_four_area_parameter(
            inter_area_j_yx[row_index, column_index],
            factor,
        )
    alternate["inter_area.J_YX"] = inter_area_j_yx.tolist()
    return alternate


def _expected_four_area_form_values(app_module, overrides=None):
    expected_values = _four_area_default_grid_values(app_module)
    if overrides:
        expected_values.update(overrides)
    return expected_values


def _assert_four_area_form_values(app_module, form_data, overrides=None, ignored_keys=None):
    ignored_keys = set(ignored_keys or ())
    ignored_keys.update(FOUR_AREA_RUNTIME_CONTROL_KEYS)
    expected_values = _expected_four_area_form_values(app_module, overrides=overrides)
    normalized_form_values = app_module._normalize_simulation_form_values(form_data, expected_values)
    for key, expected_value in expected_values.items():
        if key in ignored_keys or key not in form_data:
            continue
        actual_value = normalized_form_values[key]
        assert app_module._values_equal_for_grid(actual_value, expected_value), (
            f"Unexpected normalized value for {key}: raw={form_data[key]!r}, parsed={actual_value!r}, expected={expected_value!r}"
        )


def _expanded_four_area_trials(app_module, form_data):
    run_mode, expanded_forms = app_module._expand_simulation_forms("four_area", form_data)
    defaults = app_module._simulation_grid_defaults("four_area")
    normalized_trials = [
        app_module._normalize_simulation_form_values(form, defaults)
        for form in expanded_forms
    ]
    return run_mode, expanded_forms, normalized_trials


def _matrix_to_tuple(matrix):
    return _shared._matrix_to_tuple(matrix)


def _build_four_area_local_j_yx_margin_candidates(default_j_yx):
    default_matrix = np.array(default_j_yx, dtype=float, copy=False)
    leaf_candidates = [
        (
            _scale_four_area_parameter(value, 1.0 - FOUR_AREA_RELATIVE_MARGIN),
            _scale_four_area_parameter(value, 1.0 + FOUR_AREA_RELATIVE_MARGIN),
        )
        for value in default_matrix.ravel()
    ]
    matrices = []
    for leaf_values in product(*leaf_candidates):
        matrices.append(np.array(leaf_values, dtype=float).reshape(FOUR_AREA_LOCAL_J_YX_MATRIX_SHAPE).tolist())
    return leaf_candidates, matrices


def _run_four_area_webui_job(live_webui_server, pytestconfig, configure_page):
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
                _log_test_progress(f"opening four-area form at {live_webui_server}")
                _navigate_to_four_area_form(page, live_webui_server)
                _log_test_progress("configuring four-area form")
                configure_page(page)
                form_data = _shared._normalized_form_data_from_page(page)
                _log_test_progress("submitting WebUI simulation")

                page.locator("button[type='submit']").click()
                page.wait_for_url("**/job_status/*", wait_until="domcontentloaded")
                job_id = Path(urlparse(page.url).path).name
                _log_test_progress(f"submitted WebUI simulation as job {job_id}")
                status_payload = _shared._wait_for_job_completion(live_webui_server, job_id)
                assert status_payload["status"] == "finished", status_payload
                _log_test_progress(f"WebUI job {job_id} finished successfully")
            finally:
                browser.close()

    return form_data, job_id


def _assert_four_area_webui_matches_python_reference(
    app_module,
    live_webui_server,
    job_id,
    form_data,
    tmp_path,
    expected_trial_count,
    bin_width_ms=100.0,
):
    _log_test_progress(f"downloading WebUI results for job {job_id}")
    ui_zip_path = _shared._download_simulation_zip(
        live_webui_server,
        job_id,
        tmp_path / "webui_simulation_results.zip",
    )
    ui_output_dir = _shared._extract_zip(ui_zip_path, tmp_path / "webui_results")
    _log_test_progress("computing firing-rate summaries from WebUI output")
    webui_firing_rate_dfs = _mean_firing_rate_by_bin_dataframes(ui_output_dir, bin_width_ms=bin_width_ms)
    _log_test_progress("running direct Python reference simulations")
    python_firing_rate_dfs = _run_python_four_area_reference_dataframes(
        app_module,
        form_data,
        tmp_path / "python_reference",
        bin_width_ms=bin_width_ms,
    )

    assert len(webui_firing_rate_dfs) == expected_trial_count
    assert len(python_firing_rate_dfs) == expected_trial_count

    _log_test_progress(f"comparing {expected_trial_count} WebUI/Python firing-rate summaries")
    for trial_index, (webui_df, python_df) in enumerate(zip(webui_firing_rate_dfs, python_firing_rate_dfs), start=1):
        _log_test_progress(f"comparing trial {trial_index}/{expected_trial_count}")
        pd.testing.assert_frame_equal(webui_df, python_df)
    _log_test_progress("WebUI and Python outputs match")


@pytest.mark.slow
def test_four_area_default_configuration_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    _log_test_progress("starting default four-area WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    captured_param_dirs = _shared._capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )
    form_data, job_id = _run_four_area_webui_job(
        live_webui_server,
        pytestconfig,
        lambda page: page.locator("#sim-use-numpy-seed").check(),
    )
    run_mode, expanded_forms, _ = _expanded_four_area_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 1
    _assert_four_area_form_values(webui_app_module, form_data)
    _assert_four_area_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_four_area_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=1,
    )


@pytest.mark.slow
def test_four_area_alternate_configuration_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    _log_test_progress("starting alternate-parameter four-area WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    alternate_values = _build_four_area_alternate_single_values(webui_app_module)
    captured_param_dirs = _shared._capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        page.locator("#sim-use-numpy-seed").check()
        _shared._apply_hagen_single_parameter_overrides(page, alternate_values)

    form_data, job_id = _run_four_area_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, _ = _expanded_four_area_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 1
    _assert_four_area_form_values(webui_app_module, form_data, overrides=alternate_values)
    _assert_four_area_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_four_area_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=1,
    )


@pytest.mark.slow
def test_four_area_alternate_configuration_with_three_repetitions_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    _log_test_progress("starting alternate-parameter four-area comparison with 3 repetitions")
    webui_app_module._clear_simulation_output_folder_all_files()
    alternate_values = _build_four_area_alternate_single_values(webui_app_module)
    captured_param_dirs = _shared._capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        page.locator("#sim-use-numpy-seed").check()
        _shared._apply_hagen_single_parameter_overrides(page, alternate_values)
        page.locator("#sim-repetitions").fill("3")

    form_data, job_id = _run_four_area_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, normalized_trials = _expanded_four_area_trials(webui_app_module, form_data)
    assert run_mode == "single"
    assert len(expanded_forms) == 3
    _assert_four_area_form_values(webui_app_module, form_data, overrides=alternate_values)
    assert all(trial == normalized_trials[0] for trial in normalized_trials[1:])
    _assert_four_area_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_four_area_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=3,
    )


@pytest.mark.slow
def test_four_area_local_synaptic_weight_grid_sweep_matches_direct_python_run(
    webui_app_module, live_webui_server, tmp_path, pytestconfig, monkeypatch
):
    _log_test_progress("starting four-area local J_YX grid sweep WebUI vs Python comparison")
    webui_app_module._clear_simulation_output_folder_all_files()
    defaults = _four_area_default_grid_values(webui_app_module)
    j_yx_leaf_candidates, expected_j_yx_matrices = _build_four_area_local_j_yx_margin_candidates(
        defaults["J_YX"]
    )
    captured_param_dirs = _shared._capture_webui_generated_param_files(
        webui_app_module,
        monkeypatch,
        tmp_path / "captured_webui_params",
    )

    def configure_page(page):
        page.locator("#sim-use-numpy-seed").check()
        page.locator("input[name='sim_run_mode'][value='grid']").check()
        for leaf_index, candidates in enumerate(j_yx_leaf_candidates):
            _shared._fill_hagen_grid_leaf_candidates(page, "J_YX", leaf_index=leaf_index, candidates=candidates)

    form_data, job_id = _run_four_area_webui_job(
        live_webui_server,
        pytestconfig,
        configure_page,
    )
    run_mode, expanded_forms, normalized_trials = _expanded_four_area_trials(webui_app_module, form_data)
    assert run_mode == "grid"
    assert len(expanded_forms) == 2 ** 4
    _assert_four_area_form_values(webui_app_module, form_data, ignored_keys={"J_YX"})

    grid_metadata = webui_app_module._build_simulation_grid_metadata("four_area", run_mode, expanded_forms)
    assert grid_metadata is not None
    assert grid_metadata["changed_keys"] == ["J_YX"]
    assert grid_metadata["trial_count"] == 2 ** 4
    assert grid_metadata["configuration_count"] == 2 ** 4
    assert grid_metadata["repetitions_per_configuration"] == 1
    assert {
        _matrix_to_tuple(trial["J_YX"])
        for trial in normalized_trials
    } == {
        _matrix_to_tuple(matrix)
        for matrix in expected_j_yx_matrices
    }
    _assert_four_area_generated_parameter_files_match_expected(webui_app_module, form_data, captured_param_dirs)

    _assert_four_area_webui_matches_python_reference(
        webui_app_module,
        live_webui_server,
        job_id,
        form_data,
        tmp_path,
        expected_trial_count=2 ** 4,
    )
