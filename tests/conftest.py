from pathlib import Path


# Keep pytest temporary artifacts in a stable location for local server runs.
Path("/tmp/pytest").mkdir(parents=True, exist_ok=True)


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
