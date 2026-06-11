#!/usr/bin/env python3
"""Start the NCPI WebUI locally or remotely over SSH."""

import argparse
import http.client
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import PurePosixPath

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
SSH_ENVIRONMENT_VARIABLES = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")


LOCAL_COMMAND = "python webui/launcher.py local"
REMOTE_COMMAND = (
    "python webui/launcher.py remote <user@server> "
    "--remote-dir <absolute_ncpi_path> "
    "--python <conda_python_path>"
)


class GuidedArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, guidance=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance = guidance

    def error(self, message):
        self.print_usage(sys.stderr)
        details = f"Error: {message}\n"
        if self.guidance:
            details += f"\n{self.guidance(message)}\n"
        self.exit(2, details)


def _mode_guidance(_message):
    return (
        "You must specify how to run the application:\n"
        f"  Local:  {LOCAL_COMMAND}\n"
        f"  Remote: {REMOTE_COMMAND}"
    )


def _remote_guidance(message):
    missing = []
    if "destination" in message:
        missing.append("  <user@server>  SSH destination, for example user@example.org")
    if "--remote-dir" in message:
        missing.append("  --remote-dir   Absolute path to the NCPI repository on the server")

    lines = ["Remote configuration:"]
    if missing:
        lines.append("The following required fields are missing:")
        lines.extend(missing)
    elif "invalid int value" in message:
        lines.append("The specified port must be an integer.")
    else:
        lines.append("Check the field identified in the error.")
    lines.extend(("", "Template:", f"  {REMOTE_COMMAND}"))
    return "\n".join(lines)



def _remote_runtime_guidance(error, args):
    message = str(error)
    if "status 255" in message:
        fields = (
            f"  destination: {args.destination}\n"
            f"  --ssh-port: {args.ssh_port}\n"
            "Check that these match the values used by your SSH command."
        )
    elif "status 127" in message:
        fields = (
            f"  --remote-dir: {args.remote_dir}\n"
            f"  --python: {args.python}\n"
            "Check that both paths exist on the server."
        )
    else:
        fields = (
            f"  --local-port: {args.local_port}\n"
            f"  --remote-port: {args.remote_port}\n"
            f"  --python: {args.python}\n"
            "Check that Flask can start with these values."
        )
    return f"Fields to check:\n{fields}\n\nTemplate:\n  {REMOTE_COMMAND}"

def is_ssh_session(environ=None):
    environ = os.environ if environ is None else environ
    return any(environ.get(name) for name in SSH_ENVIRONMENT_VARIABLES)


def browser_url(host, port):
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}"


def should_open_browser(host, no_browser=False, environ=None):
    return not no_browser and host in LOOPBACK_HOSTS and not is_ssh_session(environ)


def _open_browser(url):
    if sys.platform == "win32":
        try:
            os.startfile(url)
            return True
        except (AttributeError, OSError):
            pass

    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        pass

    print(f"Could not open a browser automatically. Open {url}", flush=True)
    return False


def _schedule_browser_open(url, debug, environ=None):
    environ = os.environ if environ is None else environ
    # Schedule from Werkzeug's reloader parent. On Windows, launching from the
    # reloader child during process startup can silently fail or open twice.
    if debug and environ.get("WERKZEUG_RUN_MAIN") == "true":
        return
    timer = threading.Timer(1.0, lambda: _open_browser(url))
    timer.daemon = True
    timer.start()


def _build_local_parser():
    parser = argparse.ArgumentParser(description="Run the NCPI WebUI.")
    parser.add_argument(
        "--host",
        default=os.environ.get("NCPI_WEBUI_HOST", "127.0.0.1"),
        help="Flask bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NCPI_WEBUI_PORT", "5000")),
        help="Flask port (default: 5000).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser when running locally.",
    )
    return parser


def run_webui(app, argv=None):
    _run_webui_with_args(app, _build_local_parser().parse_args(argv))


def _run_webui_with_args(app, args):
    url = browser_url(args.host, args.port)
    debug = True
    use_reloader = not is_ssh_session()

    if should_open_browser(args.host, args.no_browser):
        _schedule_browser_open(url, debug)
    elif (
        is_ssh_session()
        and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    ):
        print(
            "\nSSH session detected. The remote process cannot open a browser "
            "on the SSH client.\n"
            f"Open {url} on your local machine through an SSH tunnel, or use "
            "webui/launcher.py remote from the local checkout.\n",
            flush=True,
        )

    app.run(debug=debug, host=args.host, port=args.port, use_reloader=use_reloader)


def build_remote_command(remote_dir, python_executable, remote_app, remote_port):
    app_command = (
        f"{shlex.quote(python_executable)} {shlex.quote(remote_app)} "
        f"--host 127.0.0.1 --port {remote_port} --no-browser"
    )
    return (
        f"cd -- {shlex.quote(remote_dir)} && "
        "trap 'kill \"$ncpi_pid\" 2>/dev/null; wait \"$ncpi_pid\" 2>/dev/null' "
        "HUP INT TERM EXIT; "
        f"{app_command} & ncpi_pid=$!; wait \"$ncpi_pid\""
    )


def build_ssh_command(args):
    forwarding = f"127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}"
    remote_command = build_remote_command(
        args.remote_dir,
        args.python,
        args.remote_app,
        args.remote_port,
    )
    return [
        args.ssh_executable,
        "-tt",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        forwarding,
        "-p",
        str(args.ssh_port),
        args.destination,
        remote_command,
    ]


def ensure_local_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def wait_for_webui(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"SSH exited with status {return_code} before the WebUI started."
            )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read(1)
            return
        except (OSError, http.client.HTTPException):
            time.sleep(0.2)
        finally:
            connection.close()
    raise TimeoutError(
        f"The WebUI did not become available on local port {port} "
        f"within {timeout:g} seconds."
    )


def _build_remote_parser():
    parser = GuidedArgumentParser(
        guidance=_remote_guidance,
        description=(
            "Start NCPI on a remote machine through an SSH tunnel and open "
            "the WebUI in this machine's browser."
        )
    )
    parser.add_argument("destination", help="SSH destination, for example user@server.")
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Absolute path to the NCPI repository on the remote machine.",
    )
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22).")
    parser.add_argument(
        "--local-port",
        type=int,
        default=5000,
        help="Local browser port (default: 5000).",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=5000,
        help="Remote Flask port (default: 5000).",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable on the remote machine (default: python).",
    )
    parser.add_argument(
        "--remote-app",
        default="webui/app.py",
        help="App path relative to --remote-dir (default: webui/app.py).",
    )
    parser.add_argument(
        "--ssh-executable",
        default="ssh",
        help="Local SSH executable (default: ssh).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the WebUI (default: 60).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Create the tunnel without opening the local browser.",
    )
    return parser


def run_remote(argv=None):
    args = _build_remote_parser().parse_args(argv)
    if not PurePosixPath(args.remote_dir).is_absolute():
        raise SystemExit(
            "Error: --remote-dir must be an absolute path on the server.\n\n"
            "Field to correct:\n"
            f"  --remote-dir: {args.remote_dir}\n\n"
            f"Template:\n  {REMOTE_COMMAND}"
        )

    try:
        ensure_local_port_available(args.local_port)
    except OSError as exc:
        raise SystemExit(
            f"Error: local port {args.local_port} is not available: {exc}\n\n"
            "Field to correct:\n"
            f"  --local-port: {args.local_port}\n"
            "Try another local port, for example --local-port 5002.\n\n"
            f"Template:\n  {REMOTE_COMMAND}"
        ) from exc

    command = build_ssh_command(args)
    url = f"http://127.0.0.1:{args.local_port}"
    print(f"Starting remote NCPI WebUI at {url}", flush=True)
    try:
        process = subprocess.Popen(command)
    except OSError as exc:
        raise SystemExit(
            f"Error: SSH could not be executed: {exc}\n\n"
            "Field to check:\n"
            f"  --ssh-executable: {args.ssh_executable}\n\n"
            f"Template:\n  {REMOTE_COMMAND}"
        ) from exc

    try:
        wait_for_webui(process, args.local_port, args.startup_timeout)
        if not args.no_browser:
            _open_browser(url)
        return process.wait()
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        print(_remote_runtime_guidance(exc, args), file=sys.stderr, flush=True)
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _build_mode_parser():
    parser = GuidedArgumentParser(
        guidance=_mode_guidance,
        description="Start the NCPI WebUI locally or remotely over SSH."
    )
    parser.add_argument("mode", choices=("local", "remote"))
    return parser


def _load_webui_app():
    try:
        from .app import app
    except ImportError:
        from app import app
    return app


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _build_mode_parser().parse_args(argv)

    mode = argv.pop(0)
    if mode == "local":
        args = _build_local_parser().parse_args(argv)
        _run_webui_with_args(_load_webui_app(), args)
        return 0
    if mode == "remote":
        return run_remote(argv)
    _build_mode_parser().error(f"invalid mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
