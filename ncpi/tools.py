from __future__ import annotations
from typing import Type
import importlib.util
from functools import wraps
import time
import sys
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from typing import Any, Dict, Optional

try:
    # Python 3.8+
    import importlib.metadata as ilmd
except Exception:  # pragma: no cover
    ilmd = None


def _run_pip(args, *, quiet=False):
    """ Run pip with the given arguments."""
    cmd = [sys.executable, "-m", "pip", *args]
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.STDOUT
    subprocess.check_call(cmd, **kwargs)


def _distribution_version(dist_name: str):
    """Return installed distribution version (pip package name), or None if not installed."""
    if ilmd is None:
        return None
    try:
        return ilmd.version(dist_name)
    except ilmd.PackageNotFoundError:
        return None


def _version_satisfies(installed: str, requirement_spec: str) -> bool:
    """
    Best-effort version check.
    Supports:
      - exact pin: '==1.2.3'
      - simple comparisons: '>=1.0', '<2.0', etc. (comma-separated allowed)
    If we can't parse, we return True (don't force reinstall).
    """
    if not installed or not requirement_spec:
        return True

    # Exact pin fast-path
    spec = requirement_spec.strip()
    if spec.startswith("=="):
        return installed == spec[2:].strip()

    # Lightweight comparator without external deps.
    # For full PEP 440, prefer packaging.specifiers (optional dependency).
    try:
        from packaging.specifiers import SpecifierSet
        return installed in SpecifierSet(spec)
    except Exception:
        # No packaging installed or parsing failed -> don't force reinstall
        return True


def ensure_module(
    module_name: str,
    package: str | None = None,
    *,
    version_spec: str | None = None,
    upgrade: bool = False,
    user: bool = False,
    quiet: bool = False,
    extra_pip_args: list[str] | None = None,
    reload_module: bool = False,
    raise_on_error: bool = False,
) -> bool:
    """
    Ensure a Python importable module exists, and optionally ensure its pip distribution
    satisfies a version constraint.

    Args:
        module_name: Import name, e.g. "rpy2.robjects" or "numpy".
        package: Pip distribution name, e.g. "rpy2" (defaults to module_name).
                 Use this when pip name differs from import name.
        version_spec: Version constraint for the *distribution*, e.g. "==3.5.11", ">=1.2,<2".
        upgrade: If True, pip install will use --upgrade.
        user: If True, pip install will use --user.
        quiet: If True, suppress pip output.
        extra_pip_args: Additional pip args like ["--no-cache-dir"].
        reload_module: If True, reload the imported module after installation.
        raise_on_error: If True, raise exceptions instead of returning False.

    Returns:
        True if module is importable (and version constraint is satisfied when applicable).
    """
    pkg = package or module_name.split(".")[0]
    extra_pip_args = extra_pip_args or []

    def fail(msg, exc=None):
        if raise_on_error:
            if exc:
                raise exc
            raise RuntimeError(msg)
        print(msg)
        return False

    try:
        # 1) Check importability
        spec = importlib.util.find_spec(module_name)
        need_install = spec is None

        # 2) Check version constraint (distribution-level)
        if not need_install and version_spec:
            installed = _distribution_version(pkg)
            # If we can't determine version, don't force reinstall.
            if installed and not _version_satisfies(installed, version_spec):
                need_install = True

        if need_install:
            req = pkg + (version_spec or "")
            pip_args = ["install"]
            if upgrade:
                pip_args.append("--upgrade")
            if user:
                pip_args.append("--user")
            pip_args.extend(extra_pip_args)
            pip_args.append(req)

            print(f"Installing requirement: {req} (import: {module_name})")
            _run_pip(pip_args, quiet=quiet)

            # Refresh import machinery
            importlib.invalidate_caches()

            if importlib.util.find_spec(module_name) is None:
                return fail(
                    f"Installed '{req}', but '{module_name}' is still not importable. "
                    f"Check that the pip package name ('{pkg}') matches the import name."
                )

        # Optional reload
        if reload_module:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)

        return True

    except subprocess.CalledProcessError as e:
        return fail(f"pip failed while installing '{package or module_name}': {e}", e)
    except Exception as e:
        return fail(f"Unexpected error ensuring '{module_name}': {e}", e)


def dynamic_import(
    module_path: str,
    attribute_name: Optional[str] = None,
    package: Optional[str] = None,
    *,
    default: Any = None,
    raise_on_error: bool = True,
    reload_module: bool = False,
    ensure_type: Optional[Type[Any]] = None,
    ensure_callable: bool = False,
) -> Any:
    """
    Dynamically import a module or attribute.

    Supports:
      - module_path="pkg.mod", attribute_name="Thing"
      - module_path="pkg.mod:Thing" (overrides attribute_name)
      - module_path="pkg.mod.Thing" when attribute_name is None (best-effort)
      - nested attributes via dot path: attribute_name="A.B.C"

    Args:
        module_path: Module path (absolute or relative if `package` given).
        attribute_name: Attribute to load from the module (optional).
        package: Package name used for relative imports (importlib behavior).
        default: Value to return on failure when raise_on_error=False.
        raise_on_error: If False, return `default` instead of raising.
        reload_module: If True, reload the imported module before resolving attributes.
        ensure_type: If provided, require the resolved object to be an instance of this type.
        ensure_callable: If True, require the resolved object to be callable.

    Returns:
        Imported module or resolved attribute.

    Raises:
        ImportError / AttributeError / TypeError (depending on failure mode) unless
        raise_on_error=False.
    """

    def fail(exc: Exception, cause: Exception | None = None) -> Any:
        if raise_on_error:
            if cause is not None:
                raise exc from cause
            raise exc
        return default

    # Allow "pkg.mod:attr" shorthand
    if ":" in module_path:
        mod_part, attr_part = module_path.split(":", 1)
        module_path = mod_part.strip()
        attribute_name = attr_part.strip() or None

    # If no attribute_name provided, allow best-effort "pkg.mod.Attr"
    if attribute_name is None and "." in module_path:
        try:
            mod = importlib.import_module(module_path, package=package)
        except ModuleNotFoundError:
            mod_part, attr_part = module_path.rsplit(".", 1)
            module_path, attribute_name = mod_part, attr_part
        else:
            if reload_module:
                mod = importlib.reload(mod)
            return mod

    # Import module (now module_path should be a module)
    try:
        mod = importlib.import_module(module_path, package=package)
        if reload_module:
            mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        # Distinguish "target module missing" vs "dependency missing inside module"
        target_top = module_path.lstrip(".").split(".", 1)[0]
        missing_top = (e.name or "").split(".", 1)[0]

        if missing_top == target_top:
            return fail(ImportError(f"Module not found: '{module_path}'"), e)

        return fail(
            ImportError(
                f"Importing '{module_path}' failed because dependency '{e.name}' was not found."
            ),
            e,
        )
    except Exception as e:
        return fail(ImportError(f"Error importing module '{module_path}': {e}"), e)

    if attribute_name is None:
        return mod

    # Resolve attribute(s), supporting nested "A.B.C"
    try:
        obj: Any = mod
        for part in attribute_name.split("."):
            obj = getattr(obj, part)
    except AttributeError as e:
        return fail(
            AttributeError(f"Module '{module_path}' has no attribute path '{attribute_name}'."),
            e,
        )

    # Optional validation
    if ensure_callable and not callable(obj):
        return fail(TypeError(f"Imported '{module_path}:{attribute_name}' is not callable."))

    if ensure_type is not None and not isinstance(obj, ensure_type):
        return fail(
            TypeError(
                f"Imported '{module_path}:{attribute_name}' is not an instance of {ensure_type!r}."
            )
        )

    return obj


def record_id_from_api_url(api_url: str) -> str:
    """ Extract record ID from Zenodo API URL. """
    m = re.search(r"/api/records/(\d+)", api_url)
    if not m:
        raise ValueError(f"Could not parse record id from api_url: {api_url}")
    return m.group(1)


def ensure_fresh_dir(download_dir: str) -> None:
    """ Ensure the given directory does not exist; create it if not. """
    if os.path.exists(download_dir):
        print(f"Directory {download_dir} already exists. Skipping download.")
        print("If you want to re-download, please delete the directory.")
        raise FileExistsError(download_dir)
    os.makedirs(download_dir, exist_ok=True)


def _is_within_directory(base_dir: str, target_path: str) -> bool:
    """ Check if target_path is within base_dir to prevent path traversal. """
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, target_path])


def safe_extract_zip(zip_path: str, outdir: str) -> bool:
    """ Safely extract a ZIP file to outdir, preventing path traversal. """
    if not zipfile.is_zipfile(zip_path):
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            dest = os.path.join(outdir, member.filename)
            if not _is_within_directory(outdir, dest):
                raise RuntimeError(f"Unsafe ZIP path traversal: {member.filename}")
        zf.extractall(outdir)
    return True


def safe_extract_tar(tar_path: str, outdir: str) -> bool:
    """ Safely extract a TAR file to outdir, preventing path traversal. """
    if not tarfile.is_tarfile(tar_path):
        return False

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            dest = os.path.join(outdir, member.name)
            if not _is_within_directory(outdir, dest):
                raise RuntimeError(f"Unsafe TAR path traversal: {member.name}")
        tf.extractall(path=outdir)
    return True


def extract_and_delete_all_archives(outdir: str, *, verbose: bool = True) -> None:
    """ Recursively extract and delete all ZIP and TAR archives in outdir.
    """
    while True:
        extracted_any = False

        for name in list(os.listdir(outdir)):
            path = os.path.join(outdir, name)
            if not os.path.isfile(path):
                continue

            extracted = False
            if zipfile.is_zipfile(path):
                if verbose:
                    print(f"Extracting ZIP {name}...")
                extracted = safe_extract_zip(path, outdir)

            elif tarfile.is_tarfile(path):
                if verbose:
                    print(f"Extracting TAR {name}...")
                extracted = safe_extract_tar(path, outdir)

            if extracted:
                os.remove(path)
                if verbose:
                    print(f"Deleted {name}")
                extracted_any = True

        if not extracted_any:
            break


def get_requests_and_tqdm():
    """ Get requests and tqdm modules, using dynamic import if available. """
    if "ensure_module" in globals() and "dynamic_import" in globals():
        if not ensure_module("requests"):
            raise ImportError("requests is required for downloading data.")
        requests = dynamic_import("requests")

        if not ensure_module("tqdm"):
            raise ImportError("tqdm is required for progress bars.")
        tqdm = dynamic_import("tqdm", "tqdm")
        return requests, tqdm

    import requests  # type: ignore
    from tqdm import tqdm  # type: ignore
    return requests, tqdm


def fetch_record_json(requests: Any, api_url: str, headers: dict, timeout: int = 60) -> Dict[str, Any]:
    """ Fetch Zenodo record JSON from API URL. """
    r = requests.get(api_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_file_streaming(
    requests: Any,
    tqdm: Any,
    file_url: str,
    dest_path: str,
    *,
    headers: dict,
    timeout: int = 120,
    chunk_size: int = 8192,
    desc: Optional[str] = None,
) -> None:
    """ Download a file from file_url to dest_path with streaming and progress bar. """

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with requests.get(file_url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        label = desc or os.path.basename(dest_path)

        with open(dest_path, "wb") as f, tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            desc=label,
            initial=0,
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))


def is_http_403(err: Exception) -> bool:
    """ Check if the given exception is an HTTP 403 error from requests. """
    status = getattr(getattr(err, "response", None), "status_code", None)
    return status == 403


def download_files_archive_with_wget(record_id: str, outdir: str) -> None:
    """Download Zenodo files-archive using wget as a fallback, then extract it."""
    if shutil.which("wget") is None:
        raise RuntimeError(
            "Fallback requires 'wget' but it was not found in PATH. "
            "Install wget or modify fallback to use curl/requests."
        )

    archive_url = f"https://zenodo.org/api/records/{record_id}/files-archive"
    archive_path = os.path.join(outdir, f"zenodo_{record_id}_files-archive")

    print(f"Falling back to files-archive:\n  {archive_url}")
    print(f"Downloading archive to:\n  {archive_path}")

    res = subprocess.run(
        [
            "wget",
            "--progress=bar:force",   # <- progress bar
            "-O",
            archive_path,
            archive_url,
        ],
        text=True,
    )

    if res.returncode != 0:
        raise RuntimeError("wget failed downloading Zenodo files-archive.")

    extract_and_delete_all_archives(outdir)

    if os.path.exists(archive_path):
        try:
            with open(archive_path, "rb") as f:
                start = f.read(200)
        except Exception:
            start = b""
        os.remove(archive_path)
        raise RuntimeError(
            f"Downloaded files-archive but it doesn't look like ZIP or TAR: {archive_path}\n"
            f"First 200 bytes: {start!r}"
        )


def download_zenodo_record(
    api_url: str,
    download_dir: str = "zenodo_files",
    fallback_files_archive: bool = True,
) -> None:
    """
    Download files from a Zenodo record. Downloads each file individually from the record's
    JSON metadata. If a 403 Forbidden error is encountered, optionally falls back to downloading
    the files-archive using wget.
    Args:
        api_url (str): The Zenodo API URL for the record (e.g., "https://zenodo.org/api/records/123456").
        download_dir (str): Directory to save downloaded files. Defaults to "zenodo_files".
        fallback_files_archive (bool): If True, falls back to downloading the files-archive
                                      using wget on 403 errors. Defaults to True.
    Raises:
        Exception: Propagates exceptions from network errors or file operations.
    """
    # Skip if already downloaded
    try:
        ensure_fresh_dir(download_dir)
    except FileExistsError:
        return

    requests, tqdm = get_requests_and_tqdm()
    headers = {"User-Agent": "Mozilla/5.0"}

    # --- Primary: fetch record JSON (with 403 fallback) ---
    try:
        record = fetch_record_json(requests, api_url, headers=headers, timeout=60)
    except Exception as e:
        if fallback_files_archive and is_http_403(e):
            rid = record_id_from_api_url(api_url)
            download_files_archive_with_wget(rid, download_dir)
            return
        raise

    # Primary: download each file
    files = record.get("files", []) or []
    for file_info in files:
        file_url = file_info["links"]["self"]
        file_name = file_info["key"]
        file_path = os.path.join(download_dir, file_name)

        print(f"Downloading {file_name}...")

        try:
            download_file_streaming(
                requests,
                tqdm,
                file_url,
                file_path,
                headers=headers,
                timeout=120,
                chunk_size=8192,
                desc=file_name,
            )
        except Exception as e:
            # Per-file 403 fallback
            if fallback_files_archive and is_http_403(e):
                rid = record_id_from_api_url(api_url)
                download_files_archive_with_wget(rid, download_dir)
                return
            raise

        print(f"Saved to {file_path}")

        # After each download: extract+delete archives
        extract_and_delete_all_archives(download_dir)

    # Final pass: extract+delete any archives revealed by prior extraction
    extract_and_delete_all_archives(download_dir)


def timer(description=None):
    """
    Decorator that measures and prints execution time of a function.
    
    Args:
        description (str): A custom message describing what the function does.
                         This will be printed before the function executes.
    
    Returns:
        function: A decorator that can be applied to any function to add timing.
    
    Example:
        @timer("Downloading simulation data and ML models.")
        def download_data():
            # function content
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper function that adds timing before and after function execution.

            Args:
                *args: Variable length argument list passed to the original function.
                **kwargs: Arbitrary keyword arguments passed to the original function.
            """
            # Print the custom description message before function execution
            print(f'\n--- {description}')

            start_time = time.time()
            
            # Execute the original function with all its arguments
            result = func(*args, **kwargs)
            
            end_time = time.time()

            # Calculate elapsed time
            elapsed = end_time - start_time

            if elapsed >= 60:
                minutes, seconds = divmod(int(elapsed), 60)
                print(
                    f"Done in {elapsed:.2f} seconds, "
                    f"which is equivalent to {minutes}:{seconds:02d} min"
                )
            else:
                print(f"Done in {elapsed:.2f} seconds")

            return result
        return wrapper
    return decorator
