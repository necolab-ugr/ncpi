from argparse import Namespace

import pytest

from webui import launcher


def test_browser_only_opens_for_local_non_ssh_session():
    assert launcher.should_open_browser("127.0.0.1", environ={})
    assert not launcher.should_open_browser(
        "127.0.0.1",
        environ={"SSH_CONNECTION": "client 123 server 22"},
    )
    assert not launcher.should_open_browser("0.0.0.0", environ={})
    assert not launcher.should_open_browser(
        "127.0.0.1",
        no_browser=True,
        environ={},
    )


def test_ssh_session_disables_flask_reloader(monkeypatch):
    class FakeApp:
        run_arguments = None

        def run(self, **kwargs):
            self.run_arguments = kwargs

    app = FakeApp()
    monkeypatch.setenv("SSH_CONNECTION", "client 123 server 22")

    launcher.run_webui(app, ["--no-browser"])

    assert app.run_arguments["debug"] is True
    assert app.run_arguments["use_reloader"] is False


def test_local_session_keeps_flask_reloader(monkeypatch):
    class FakeApp:
        run_arguments = None

        def run(self, **kwargs):
            self.run_arguments = kwargs

    app = FakeApp()
    for name in launcher.SSH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    launcher.run_webui(app, ["--no-browser"])

    assert app.run_arguments["use_reloader"] is True

def test_browser_url_uses_loopback_for_wildcard_bind_address():
    assert launcher.browser_url("0.0.0.0", 5000) == "http://127.0.0.1:5000"


def test_build_ssh_command_for_remote_webui():
    args = Namespace(
        destination="alice@example.org",
        remote_dir="/srv/ncpi project",
        ssh_port=2222,
        local_port=5050,
        remote_port=5000,
        python="/opt/ncpi env/bin/python",
        remote_app="webui/app.py",
        ssh_executable="ssh",
    )

    assert launcher.build_ssh_command(args) == [
        "ssh",
        "-tt",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        "127.0.0.1:5050:127.0.0.1:5000",
        "-p",
        "2222",
        "alice@example.org",
        "cd -- '/srv/ncpi project' && "
        "trap 'kill \"$ncpi_pid\" 2>/dev/null; wait \"$ncpi_pid\" 2>/dev/null' "
        "HUP INT TERM EXIT; '/opt/ncpi env/bin/python' webui/app.py "
        "--host 127.0.0.1 --port 5000 --no-browser & ncpi_pid=$!; "
        "wait \"$ncpi_pid\"",
    ]


def test_local_port_check_enables_address_reuse(monkeypatch):
    calls = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def setsockopt(self, level, option, value):
            calls.append(("setsockopt", level, option, value))

        def bind(self, address):
            calls.append(("bind", address))

    monkeypatch.setattr(launcher.socket, "socket", lambda *args: FakeSocket())

    launcher.ensure_local_port_available(5001)

    assert calls == [
        (
            "setsockopt",
            launcher.socket.SOL_SOCKET,
            launcher.socket.SO_REUSEADDR,
            1,
        ),
        ("bind", ("127.0.0.1", 5001)),
    ]


def test_wait_for_webui_fails_when_ssh_exits():
    class ExitedProcess:
        @staticmethod
        def poll():
            return 255

    with pytest.raises(RuntimeError, match="SSH exited with status 255"):
        launcher.wait_for_webui(ExitedProcess(), 5000, timeout=1)


def test_main_dispatches_local_mode(monkeypatch):
    calls = []
    fake_app = object()

    monkeypatch.setattr(launcher, "_load_webui_app", lambda: fake_app)
    monkeypatch.setattr(
        launcher,
        "_run_webui_with_args",
        lambda app, args: calls.append((app, args.port)),
    )

    assert launcher.main(["local", "--port", "5050", "--no-browser"]) == 0
    assert calls == [(fake_app, 5050)]

def test_main_requires_explicit_mode():
    with pytest.raises(SystemExit) as exc_info:
        launcher.main([])

    assert exc_info.value.code == 2


def test_main_dispatches_remote_mode(monkeypatch):
    monkeypatch.setattr(launcher, "run_remote", lambda argv: ("remote", argv))

    assert launcher.main(["remote", "user@server"]) == (
        "remote",
        ["user@server"],
    )


def test_missing_mode_prints_local_and_remote_commands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        launcher.main([])

    error = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "You must specify how to run the application" in error
    assert launcher.LOCAL_COMMAND in error
    assert launcher.REMOTE_COMMAND in error


def test_remote_missing_fields_prints_required_configuration(capsys):
    with pytest.raises(SystemExit) as exc_info:
        launcher.run_remote([])

    error = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "The following required fields are missing" in error
    assert "<user@server>" in error
    assert "--remote-dir" in error
    assert launcher.REMOTE_COMMAND in error


def test_relative_remote_dir_prints_field_to_correct():
    with pytest.raises(SystemExit) as exc_info:
        launcher.run_remote(["user@server", "--remote-dir", "ncpi"])

    message = str(exc_info.value)
    assert "--remote-dir must be an absolute path" in message
    assert "--remote-dir: ncpi" in message
    assert launcher.REMOTE_COMMAND in message


def test_remote_runtime_guidance_targets_ssh_and_python_fields():
    args = Namespace(
        destination="user@server",
        remote_dir="/srv/ncpi",
        ssh_port=22,
        local_port=5001,
        remote_port=5001,
        python="/env/bin/python",
    )

    ssh_guidance = launcher._remote_runtime_guidance(
        RuntimeError("SSH exited with status 255"), args
    )
    python_guidance = launcher._remote_runtime_guidance(
        RuntimeError("SSH exited with status 127"), args
    )

    assert "destination: user@server" in ssh_guidance
    assert "--ssh-port: 22" in ssh_guidance
    assert "--remote-dir: /srv/ncpi" in python_guidance
    assert "--python: /env/bin/python" in python_guidance
