import os
import tempfile
import uuid
import json

SESSION_PREFIX = "ncpi_webui_session_"
SESSION_METADATA_FILENAME = "session.json"


def _is_writable_dir(path):
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def resolve_tmp_base_root():
    """Resolve the writable parent temp directory used by webui."""
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


def _sanitize_session_id(value):
    raw = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or "").strip())
    return raw or uuid.uuid4().hex


def session_folder_name(session_id):
    return f"{SESSION_PREFIX}{_sanitize_session_id(session_id)}"


def session_id_from_root(path):
    basename = os.path.basename(os.path.realpath(str(path or "")))
    if basename.startswith(SESSION_PREFIX):
        return _sanitize_session_id(basename[len(SESSION_PREFIX):])
    return _sanitize_session_id(basename)


def _session_root_from_id(base_root, session_id):
    return os.path.realpath(os.path.join(base_root, session_folder_name(session_id)))


def _validate_session_root(path):
    base_root = os.path.realpath(TMP_BASE_ROOT)
    session_root = os.path.realpath(str(path or ""))
    if not session_root:
        raise ValueError("Missing session folder path.")
    if os.path.dirname(session_root) != base_root:
        raise ValueError(f"Session folder must be directly under {base_root}.")
    if not os.path.basename(session_root).startswith(SESSION_PREFIX):
        raise ValueError(f"Session folder must start with {SESSION_PREFIX}.")
    return session_root


def get_session_metadata(session_root):
    """Read the file 'session.json' inside session_root and returns a dictionary"""
    metadata_path = os.path.join(session_root, SESSION_METADATA_FILENAME)
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def get_session_display_name(session_root):
    """Return the display_name if it exists; if it doesn't, return the base name of the folder."""
    meta = get_session_metadata(session_root)
    return meta.get("display_name") or os.path.basename(session_root)


def save_session_metadata(session_root, metadata):
    """Save the metadata dictionary in session.json inside session_root."""
    metadata_path = os.path.join(session_root, SESSION_METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def ensure_session_metadata(session_root):
    """If session.json doesn't exist, create it with display_name by default (foldername)."""
    if not os.path.isfile(os.path.join(session_root, SESSION_METADATA_FILENAME)):
        default_name = os.path.basename(session_root)
        save_session_metadata(session_root, {"display_name": default_name})

        

def resolve_tmp_root():
    """Resolve the run-scoped temp root for the current webui process."""
    base_root = resolve_tmp_base_root()
    session_id = _sanitize_session_id(os.environ.get("NCPI_WEBUI_SESSION_ID") or uuid.uuid4().hex)
    session_root = os.environ.get("NCPI_WEBUI_SESSION_ROOT")

    if session_root:
        session_root = _validate_session_root(session_root)
        session_id = session_id_from_root(session_root)
    else:
        session_root = _session_root_from_id(base_root, session_id)

    os.makedirs(session_root, exist_ok=True)
    ensure_session_metadata(session_root)
    os.environ["NCPI_WEBUI_SESSION_ID"] = session_id
    os.environ["NCPI_WEBUI_SESSION_ROOT"] = session_root
    return session_root


TMP_BASE_ROOT = resolve_tmp_base_root()
TMP_ROOT = resolve_tmp_root()
SESSION_ID = os.environ["NCPI_WEBUI_SESSION_ID"]


def configure_temp_environment():
    tempfile.tempdir = TMP_ROOT
    os.environ["TMPDIR"] = TMP_ROOT
    os.environ["TMP"] = TMP_ROOT
    os.environ["TEMP"] = TMP_ROOT
    return TMP_ROOT


def tmp_subdir(name, create=False):
    path = os.path.realpath(os.path.join(TMP_ROOT, name))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def activate_session_root(session_root=None, session_id=None, create=False):
    global TMP_ROOT, SESSION_ID

    if session_root:
        target_root = _validate_session_root(session_root)
        session_id = session_id_from_root(target_root)
        if not os.path.isdir(target_root):
            if not create:
                raise FileNotFoundError(f"Session folder not found: {target_root}")
            os.makedirs(target_root, exist_ok=True)
            ensure_session_metadata(target_root)
    else:
        session_id = _sanitize_session_id(session_id or uuid.uuid4().hex)
        target_root = _session_root_from_id(TMP_BASE_ROOT, session_id)
        if create or not os.path.isdir(target_root):
            os.makedirs(target_root, exist_ok=True)

    TMP_ROOT = target_root
    SESSION_ID = session_id
    os.environ["NCPI_WEBUI_SESSION_ID"] = SESSION_ID
    os.environ["NCPI_WEBUI_SESSION_ROOT"] = TMP_ROOT
    configure_temp_environment()
    return TMP_ROOT


def list_session_roots():
    base_root = os.path.realpath(TMP_BASE_ROOT)
    if not os.path.isdir(base_root):
        return []

    session_roots = []
    try:
        entries = os.listdir(base_root)
    except OSError:
        return []

    for name in entries:
        if not name.startswith(SESSION_PREFIX):
            continue
        candidate = os.path.realpath(os.path.join(base_root, name))
        if os.path.isdir(candidate):
            session_roots.append(candidate)

    def _sort_key(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    return sorted(session_roots, key=_sort_key, reverse=True)
