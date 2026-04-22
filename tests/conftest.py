import importlib
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBUI_DIR = REPO_ROOT / "webui"
WEBUI_TESTS_DIR = (Path(__file__).resolve().parent / "webui").resolve()


def _ensure_webui_import_paths():
    for entry in (str(REPO_ROOT), str(WEBUI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _tmp_paths_module():
    _ensure_webui_import_paths()
    return importlib.import_module("tmp_paths")


def _sanitize_token(value, fallback):
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_")
    return token or fallback


def _is_webui_test(request):
    node_path = getattr(request.node, "path", None)
    if node_path is None:
        return False
    try:
        return Path(node_path).resolve().is_relative_to(WEBUI_TESTS_DIR)
    except AttributeError:
        resolved = Path(node_path).resolve()
        return WEBUI_TESTS_DIR == resolved or WEBUI_TESTS_DIR in resolved.parents


def pytest_addoption(parser):
    parser.addoption(
        "--webui-show-browser",
        action="store_true",
        default=False,
        help="Show the Playwright browser window for local web UI tests.",
    )
    parser.addoption(
        "--webui-port",
        action="store",
        type=int,
        default=0,
        help="Port used by the local Flask server in web UI tests. Default: auto-select.",
    )


@pytest.fixture(scope="session")
def _webui_test_session():
    monkeypatch = pytest.MonkeyPatch()
    session_id = f"pytest_webui_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("NCPI_WEBUI_SESSION_ID", session_id)
    monkeypatch.delenv("NCPI_WEBUI_SESSION_ROOT", raising=False)

    tmp_paths = _tmp_paths_module()
    session_root = Path(tmp_paths.activate_session_root(session_id=session_id, create=True))
    try:
        yield session_root
    finally:
        monkeypatch.undo()


@pytest.fixture()
def tmp_path(request, tmp_path_factory):
    if not _is_webui_test(request):
        return tmp_path_factory.mktemp(request.node.name)

    tmp_paths = _tmp_paths_module()
    tests_root = Path(tmp_paths.tmp_subdir("tests", create=True))
    node_name = _sanitize_token(request.node.name, "tmp")
    return Path(tempfile.mkdtemp(prefix=f"{node_name}_", dir=str(tests_root)))
