import os
import tempfile


def _is_writable_dir(path):
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def resolve_tmp_root():
    """Resolve a shared writable temp root for webui modules."""
    primary = os.path.realpath("/tmp")
    if _is_writable_dir(primary):
        return primary

    home_tmp = os.path.realpath(os.path.join(os.path.expanduser("~"), "tmp"))
    try:
        os.makedirs(home_tmp, exist_ok=True)
    except OSError:
        home_tmp = None
    if _is_writable_dir(home_tmp):
        return home_tmp

    # Last-resort fallback to keep runtime functional if both preferred roots fail.
    fallback = os.path.realpath(tempfile.gettempdir())
    if _is_writable_dir(fallback):
        return fallback
    return os.path.realpath(os.path.expanduser("~"))


TMP_ROOT = resolve_tmp_root()


def configure_temp_environment():
    tempfile.tempdir = TMP_ROOT
    os.environ["TMPDIR"] = TMP_ROOT
    return TMP_ROOT


def tmp_subdir(name, create=False):
    path = os.path.realpath(os.path.join(TMP_ROOT, name))
    if create:
        os.makedirs(path, exist_ok=True)
    return path
