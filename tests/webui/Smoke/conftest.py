"""Fixtures for WebUI smoke/pipeline tests driven by Flask's test client (no browser, no NEST)."""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WEBUI_DIR = REPO_ROOT / "webui"


def _load_webui_app():
    for entry in (str(REPO_ROOT), str(WEBUI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    for module_name in ("webui.app", "compute_utils", "tmp_paths"):
        sys.modules.pop(module_name, None)
    module = importlib.import_module("webui.app")
    module.refresh_tmp_paths()
    return module


def _clear_all(module):
    module._clear_simulation_output_folder_all_files()
    module._clear_field_potential_data_files()


@pytest.fixture(scope="module")
def webui_module(_webui_test_session):
    module = _load_webui_app()
    module.app.config["TESTING"] = True
    _clear_all(module)
    yield module
    _clear_all(module)


@pytest.fixture()
def client(webui_module):
    with webui_module.app.test_client() as test_client:
        yield test_client
