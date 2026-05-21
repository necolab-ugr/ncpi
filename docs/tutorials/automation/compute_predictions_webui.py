#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for computing predictions in the simulation pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright._impl._errors import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import Locator, Page, sync_playwright


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
VIDEO_DIR = SCRIPT_DIR / "videos"
OUTPUT_VIDEO = VIDEO_DIR / "compute-predictions.webm"
OUTPUT_POSTER = VIDEO_DIR / "compute-predictions-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_SESSION_LABEL = "webui docs"
DEFAULT_SESSION_ROOT = "/tmp/ncpi_webui_session_e6152202b3294223b22bb4fc0bc1682b"
DEFAULT_ASSETS_PATH = "/home/pablomc/Downloads/ML_models/MLP/catch22/model"
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: load a previous session, open Inference -> Compute predictions, "
            "select model and scaler server files, enable scaler usage, and compute predictions."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="WebUI base URL.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser headless (default: true).",
    )
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start webui/app.py automatically (default: true).",
    )
    parser.add_argument(
        "--job-timeout-sec",
        type=int,
        default=7200,
        help="Timeout waiting for predictions computation completion.",
    )
    parser.add_argument(
        "--ui-timeout-sec",
        type=int,
        default=180,
        help="Default UI action timeout.",
    )
    parser.add_argument(
        "--session-label",
        default=DEFAULT_SESSION_LABEL,
        help="Human-readable saved session label (used in logs only).",
    )
    parser.add_argument(
        "--session-root",
        default=DEFAULT_SESSION_ROOT,
        help="Exact saved session root folder to load from /sessions.",
    )
    parser.add_argument(
        "--assets-dir",
        default=DEFAULT_ASSETS_PATH,
        help="Server path to model file, or folder containing model/scaler files.",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=DEFAULT_VIEWPORT_WIDTH,
        help=f"Video viewport width (default: {DEFAULT_VIEWPORT_WIDTH}).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=DEFAULT_VIEWPORT_HEIGHT,
        help=f"Video viewport height (default: {DEFAULT_VIEWPORT_HEIGHT}).",
    )
    parser.add_argument(
        "--slow-mo-ms",
        type=int,
        default=DEFAULT_SLOW_MO_MS,
        help=f"Playwright slow motion delay in ms (default: {DEFAULT_SLOW_MO_MS}).",
    )
    return parser.parse_args()


def wait_for_server(url: str, timeout_sec: int = 60) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.4)
    raise TimeoutError(f"Server did not become reachable at {host}:{port} within {timeout_sec}s.")


def start_webui_server(base_url: str) -> subprocess.Popen:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000
    webui_dir = REPO_ROOT / "webui"
    cmd = [
        sys.executable,
        "-c",
        (
            "from app import app; "
            f"app.run(host={host!r}, port={port}, debug=False, use_reloader=False)"
        ),
    ]
    env = os.environ.copy()
    env.setdefault("NCPI_WEBUI_RUNTIME_MODE", "server")
    env.setdefault("NCPI_ENABLE_NATIVE_PATH_PICKER", "0")
    proc = subprocess.Popen(cmd, cwd=webui_dir, env=env)
    wait_for_server(base_url, timeout_sec=90)
    return proc


def stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def ensure_demo_cursor(page: Page, start_x: float = 24.0, start_y: float = 24.0) -> None:
    try:
        page.evaluate(
            """
            ({ startX, startY }) => {
              if (!document.getElementById('pw-demo-cursor-style')) {
                const style = document.createElement('style');
                style.id = 'pw-demo-cursor-style';
                style.textContent = `
                  #pw-demo-cursor {
                    position: fixed;
                    left: 24px;
                    top: 24px;
                    width: 24px;
                    height: 24px;
                    border-radius: 999px;
                    border: 3px solid #ffffff;
                    background: rgba(220, 38, 38, 0.95);
                    box-shadow: 0 0 0 3px rgba(2, 6, 23, 0.58), 0 0 0 8px rgba(220, 38, 38, 0.22), 0 5px 12px rgba(2, 6, 23, 0.45);
                    transform: translate(-50%, -50%);
                    transition: transform 90ms ease;
                    z-index: 2147483647;
                    pointer-events: none;
                  }
                  #pw-demo-cursor.clicking {
                    transform: translate(-50%, -50%) scale(0.78);
                  }
                `;
                document.head.appendChild(style);
              }
              if (!document.getElementById('pw-demo-cursor')) {
                const node = document.createElement('div');
                node.id = 'pw-demo-cursor';
                node.style.left = `${startX}px`;
                node.style.top = `${startY}px`;
                document.body.appendChild(node);
              }
              if (!window.__pwDemoCursorMouseBound) {
                window.__pwDemoCursorMouseBound = true;
                document.addEventListener('mousemove', (evt) => {
                  const node = document.getElementById('pw-demo-cursor');
                  if (!node) return;
                  node.style.left = `${evt.clientX}px`;
                  node.style.top = `${evt.clientY}px`;
                }, { passive: true });
              }
            }
            """,
            {"startX": start_x, "startY": start_y},
        )
    except PlaywrightError as exc:
        if "Execution context was destroyed" in str(exc):
            return
        raise


def move_demo_cursor(page: Page, click: bool = False) -> None:
    if not click:
        return
    try:
        page.evaluate(
            """
            () => {
              const node = document.getElementById('pw-demo-cursor');
              if (!node) return;
              node.classList.remove('clicking');
              void node.offsetWidth;
              node.classList.add('clicking');
              setTimeout(() => node.classList.remove('clicking'), 120);
            }
            """
        )
    except PlaywrightError as exc:
        if "Execution context was destroyed" in str(exc):
            return
        raise


def move_to_locator(page: Page, locator: Locator, click: bool = False, pause_ms: int = 180) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    locator.wait_for(state="visible")
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    if box is None:
        move_demo_cursor(page, click=click)
        if pause_ms > 0:
            page.wait_for_timeout(pause_ms)
        return
    x = box["x"] + (box["width"] / 2.0)
    y = box["y"] + (box["height"] / 2.0)
    dx = x - _DEMO_CURSOR_POS["x"]
    dy = y - _DEMO_CURSOR_POS["y"]
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    steps = max(28, min(110, int(distance / 12.0)))
    travel_ms = max(260, min(920, int(distance * 0.75)))
    page.mouse.move(x, y, steps=steps)
    move_demo_cursor(page, click=click)
    _DEMO_CURSOR_POS = {"x": x, "y": y}
    if pause_ms > 0:
        page.wait_for_timeout(max(pause_ms, int(travel_ms * 0.72)))


def smooth_click(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    page.wait_for_timeout(after_ms)


def smooth_check(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    if not locator.is_checked():
        locator.check()
    page.wait_for_timeout(after_ms)


def _extract_job_id_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "job_status":
        return parts[1]
    raise RuntimeError(f"Could not parse job id from URL: {url}")


def wait_for_job_finished(page: Page, base_url: str, job_id: str, timeout_sec: int) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    status_url = f"{base_url.rstrip('/')}/status/{job_id}"
    last_status = None
    while time.time() < deadline:
        response = page.request.get(status_url)
        if not response.ok:
            time.sleep(1.0)
            continue
        payload = response.json()
        status = str((payload or {}).get("status") or "").strip().lower()
        if status:
            last_status = status
        if status == "finished":
            return
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Job {job_id} ended with status '{status}'.")
        time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for job {job_id} to finish. Last status: {last_status or 'unknown'}"
    )


def wait_for_enabled(page: Page, locator: Locator, timeout_sec: int = 30) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        try:
            if locator.count() > 0 and locator.first.is_visible() and locator.first.is_enabled():
                return
        except PlaywrightError:
            pass
        page.wait_for_timeout(200)
    raise RuntimeError("Timed out waiting for enabled UI control.")


def _split_abs_path(path_value: str) -> list[str]:
    clean = str(path_value or "").strip().replace("\\", "/")
    if not clean:
        return []
    return [segment for segment in clean.split("/") if segment]


def _normalize_abs_path(path_value: str) -> str:
    segments = _split_abs_path(path_value)
    return "/" + "/".join(segments) if segments else "/"


def _wait_file_modal_idle(page: Page) -> None:
    loading = page.locator("#inferenceServerFileBrowserLoading")
    if loading.count() == 0:
        return
    try:
        loading.wait_for(state="hidden", timeout=20_000)
    except PlaywrightTimeoutError:
        pass


def _read_file_modal_current_path(page: Page) -> str:
    current = page.locator("#inferenceServerFileBrowserPath").inner_text(timeout=5_000).strip()
    return _normalize_abs_path(current or "/")


def _click_file_modal_dir(page: Page, segment: str, timeout_ms: int = 15_000) -> None:
    entries = page.locator("#inferenceServerFileBrowserEntries button")
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        _wait_file_modal_idle(page)
        count = entries.count()
        for idx in range(count):
            btn = entries.nth(idx)
            name = btn.inner_text().strip().rstrip("/")
            if name == segment:
                before = _read_file_modal_current_path(page)
                smooth_click(page, btn, after_ms=420)
                end_deadline = time.time() + 12.0
                while time.time() < end_deadline:
                    _wait_file_modal_idle(page)
                    after = _read_file_modal_current_path(page)
                    if after != before:
                        return
                    page.wait_for_timeout(140)
                return
        page.wait_for_timeout(180)
    raise RuntimeError(f"Could not find directory entry '{segment}' in server file browser.")


def _choose_file_from_modal_by_prefix(page: Page, prefix: str) -> None:
    labels = page.locator("#inferenceServerFileBrowserEntries label")
    count = labels.count()
    prefix_lc = prefix.lower().strip()
    fallback = None
    for idx in range(count):
        label = labels.nth(idx)
        name = label.locator("span.font-mono").first.inner_text().strip()
        if fallback is None:
            fallback = label
        if name.lower().startswith(prefix_lc):
            smooth_click(page, label, after_ms=280)
            return
    if fallback is not None:
        smooth_click(page, fallback, after_ms=280)
        return
    raise RuntimeError("No file entries found in server file browser.")


def _pick_local_file_by_prefix(directory: str, prefix: str) -> str:
    path = Path(directory)
    if not path.is_dir():
        raise RuntimeError(f"Assets directory does not exist or is not a folder: {directory}")
    prefix_lc = prefix.lower().strip()
    files = sorted(path.iterdir(), key=lambda p: p.name.lower())
    for candidate in files:
        if candidate.is_file() and candidate.name.lower().startswith(prefix_lc):
            return str(candidate)
    raise RuntimeError(f"Could not find a file starting with '{prefix}' in: {directory}")


def _resolve_model_and_scaler_paths(path_value: str) -> tuple[str, str]:
    path = Path(path_value)
    if path.is_file():
        parent = path.parent
        model_path = str(path)
        scaler_candidate = parent / "scaler"
        if scaler_candidate.is_file():
            return model_path, str(scaler_candidate)
        return model_path, _pick_local_file_by_prefix(str(parent), "scaler")

    if path.is_dir():
        model_path = _pick_local_file_by_prefix(str(path), "model")
        scaler_path = _pick_local_file_by_prefix(str(path), "scaler")
        return model_path, scaler_path

    raise RuntimeError(f"Assets path does not exist: {path_value}")


def _set_server_file_input(page: Page, input_id: str, file_path: str) -> None:
    page.evaluate(
        """
        ({ inputId, selectedPath }) => {
          const input = document.getElementById(inputId);
          if (!input) throw new Error(`Missing input: ${inputId}`);
          const source = document.getElementById('model-assets-source');
          if (source) {
            source.value = 'server-path';
            source.dispatchEvent(new Event('input', { bubbles: true }));
            source.dispatchEvent(new Event('change', { bubbles: true }));
          }
          input.value = selectedPath;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
        """,
        {"inputId": input_id, "selectedPath": file_path},
    )


def scroll_to_prediction_assets(page: Page) -> None:
    section = page.locator("h2:has-text('Load Prediction Assets')").first
    move_to_locator(page, section, click=False, pause_ms=120)
    page.evaluate(
        """
        () => {
          const heading = Array.from(document.querySelectorAll('h2'))
            .find((el) => (el.textContent || '').trim().includes('Load Prediction Assets'));
          if (!heading) return;
          heading.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
          window.scrollBy({ top: 180, left: 0, behavior: 'smooth' });
        }
        """
    )
    page.wait_for_timeout(900)


def select_server_file_via_modal(
    page: Page,
    open_btn: Locator,
    target_file_path: str,
    target_input_id: str,
) -> None:
    target_file = Path(target_file_path)
    if not target_file.is_file():
        raise RuntimeError(f"Target artifact file does not exist: {target_file_path}")
    target_dir = str(target_file.parent)
    target_name = target_file.name

    smooth_click(page, open_btn, after_ms=500)
    modal = page.locator("#inferenceServerFileBrowserModal")
    modal.wait_for(state="visible", timeout=15_000)
    _wait_file_modal_idle(page)

    current_path = _read_file_modal_current_path(page)
    target_segments = _split_abs_path(target_dir)
    current_segments = _split_abs_path(current_path)

    common = 0
    max_common = min(len(current_segments), len(target_segments))
    while common < max_common and current_segments[common] == target_segments[common]:
        common += 1

    up_btn = page.locator("#inferenceServerFileBrowserUp")
    for _ in range(len(current_segments) - common):
        smooth_click(page, up_btn, after_ms=320)
        _wait_file_modal_idle(page)

    for segment in target_segments[common:]:
        _click_file_modal_dir(page, segment, timeout_ms=18_000)

    labels = page.locator("#inferenceServerFileBrowserEntries label")
    selected = None
    for idx in range(labels.count()):
        label = labels.nth(idx)
        name = label.locator("span.font-mono").first.inner_text().strip()
        if name == target_name:
            selected = label
            break
    if selected is None:
        for idx in range(labels.count()):
            label = labels.nth(idx)
            name = label.locator("span.font-mono").first.inner_text().strip().lower()
            if name.startswith(target_name.lower()):
                selected = label
                break
    if selected is None:
        raise RuntimeError(f"Could not find file '{target_name}' in server browser at: {target_dir}")

    smooth_click(page, selected, after_ms=280)
    smooth_click(page, page.locator("#inferenceServerFileBrowserSelect"), after_ms=620)
    modal.wait_for(state="hidden", timeout=15_000)

    hidden_input = page.locator(f"#{target_input_id}")
    hidden_input.wait_for(state="attached", timeout=5_000)
    assigned = hidden_input.input_value().strip()
    if not assigned:
        _set_server_file_input(page, target_input_id, str(target_file))
    page.wait_for_timeout(420)


def _has_any_file(dir_path: Path, suffixes: Optional[set[str]] = None) -> bool:
    if not dir_path.is_dir():
        return False
    for entry in dir_path.rglob("*"):
        if not entry.is_file():
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue
        return True
    return False


def _session_has_simulation_artifacts(session_root: str) -> bool:
    base = Path(session_root)
    return _has_any_file(base / "simulation" / "data", suffixes={".pkl"})


def _session_has_field_potential_artifacts(session_root: str) -> bool:
    base = Path(session_root)
    return _has_any_file(base / "field_potential", suffixes={".pkl"})


def _session_has_features_artifacts(session_root: str) -> bool:
    base = Path(session_root)
    return _has_any_file(base / "features" / "data")


def load_requested_session(page: Page, session_root: str, session_label: str) -> None:
    smooth_click(page, page.get_by_role("link", name="Previous Sessions"))
    page.wait_for_url("**/sessions**")
    page.wait_for_timeout(500)
    entries = page.evaluate(
        """
        () => {
          const parseStamp = (text) => {
            const raw = String(text || '').trim();
            if (!raw) return null;
            let ts = Date.parse(raw);
            if (!Number.isNaN(ts)) return ts;
            const m = raw.match(/(\\d{4})-(\\d{2})-(\\d{2})[ T](\\d{2}):(\\d{2})(?::(\\d{2}))?/);
            if (m) {
              const y = Number(m[1]);
              const mo = Number(m[2]) - 1;
              const d = Number(m[3]);
              const h = Number(m[4]);
              const mi = Number(m[5]);
              const s = Number(m[6] || '0');
              return new Date(y, mo, d, h, mi, s).getTime();
            }
            return null;
          };

          const cards = Array.from(document.querySelectorAll('article'));
          const entries = cards.map((card, idx) => {
            const input = card.querySelector("input[name='session_root']");
            const dtNodes = Array.from(card.querySelectorAll('dt'));
            const ddNodes = Array.from(card.querySelectorAll('dd'));
            let updated = '';
            for (const dt of dtNodes) {
              if ((dt.textContent || '').trim().toLowerCase() === 'last modified') {
                const dd = dt.parentElement ? dt.parentElement.querySelector('dd') : null;
                updated = (dd ? dd.textContent : '').trim();
                break;
              }
            }
            if (!updated && ddNodes.length >= 3) {
              updated = (ddNodes[2].textContent || '').trim();
            }
            let modulesText = '';
            for (const dt of dtNodes) {
              if ((dt.textContent || '').trim().toLowerCase() === 'detected modules') {
                const dd = dt.parentElement ? dt.parentElement.querySelector('dd') : null;
                modulesText = (dd ? dd.textContent : '').trim();
                break;
              }
            }
            const modules = modulesText
              .split(',')
              .map((token) => token.trim().toLowerCase())
              .filter((token) => token && token !== 'no module folders detected yet');
            const isActive = Boolean(card.querySelector('span') && Array.from(card.querySelectorAll('span')).some((n) => (n.textContent || '').trim().toLowerCase() === 'active'));
            return {
              index: idx,
              path: input ? String(input.value || '').trim() : '',
              updated,
              stamp: parseStamp(updated),
              isActive,
              modules,
            };
          }).filter((entry) => entry.path);

          if (!entries.length) return [];
          entries.sort((a, b) => {
            if (a.stamp != null && b.stamp != null) return b.stamp - a.stamp;
            if (a.stamp != null) return -1;
            if (b.stamp != null) return 1;
            if (a.updated && b.updated) return b.updated.localeCompare(a.updated);
            return a.index - b.index;
          });
          return entries;
        }
        """
    )
    if not entries:
        raise RuntimeError("No saved sessions found in /sessions page.")

    required_modules = {"simulation", "field potential", "features"}
    eligible = []
    for entry in entries:
        if bool(entry.get("isActive")):
            continue
        modules = {str(item).strip().lower() for item in (entry.get("modules") or [])}
        if not required_modules.issubset(modules):
            continue
        path_value = str(entry.get("path") or "")
        if not _session_has_simulation_artifacts(path_value):
            continue
        if not _session_has_field_potential_artifacts(path_value):
            continue
        if not _session_has_features_artifacts(path_value):
            continue
        eligible.append(entry)
    if not eligible:
        raise RuntimeError(
            "No previous session contains the artifacts required for prediction "
            "(expected simulation/data/*.pkl, field_potential/**/*.pkl, and files under features/data)."
        )
    target = eligible[0]
    target_path = str(target["path"])
    print(
        f"[automation] loading most recent session with simulation+field potential+features artifacts "
        f"(label={session_label!r}, requested={session_root!r}) -> "
        f"{target_path} (updated={target.get('updated', '')}, modules={target.get('modules', [])})",
        flush=True,
    )
    target_form = page.locator(f"form:has(input[name='session_root'][value='{target_path}'])").first
    if target_form.count() == 0:
        raise RuntimeError(f"Target saved session form not found in UI: {target_path}")
    target_card = page.locator(f"article:has(input[name='session_root'][value='{target_path}'])").first
    if target_card.count() > 0:
        target_card.evaluate(
            "el => el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' })"
        )
        page.wait_for_timeout(900)
        move_to_locator(page, target_card, click=False, pause_ms=220)
    smooth_click(page, target_form.get_by_role("button"))
    page.wait_for_url("**/sessions**")
    page.wait_for_timeout(1200)


def run_tutorial_recording(
    base_url: str,
    headless: bool,
    job_timeout_sec: int,
    ui_timeout_sec: int,
    session_label: str,
    session_root: str,
    assets_dir: str,
    viewport_width: int,
    viewport_height: int,
    slow_mo_ms: int,
) -> Path:
    global _DEMO_CURSOR_POS
    _DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tmp_video_dir = VIDEO_DIR / ".tmp"
    if tmp_video_dir.exists():
        shutil.rmtree(tmp_video_dir)
    tmp_video_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            slow_mo=max(0, int(slow_mo_ms)),
            args=[f"--window-size={int(viewport_width)},{int(viewport_height)}"],
        )
        context = browser.new_context(
            viewport={"width": int(viewport_width), "height": int(viewport_height)},
            screen={"width": int(viewport_width), "height": int(viewport_height)},
            record_video_dir=str(tmp_video_dir),
            record_video_size={"width": int(viewport_width), "height": int(viewport_height)},
            device_scale_factor=1,
        )
        page = context.new_page()
        video = page.video
        page.set_default_timeout(ui_timeout_sec * 1000)

        try:
            print("[automation] opening dashboard...", flush=True)
            page.goto(base_url, wait_until="domcontentloaded")
            model_path, scaler_path = _resolve_model_and_scaler_paths(assets_dir)

            load_requested_session(page, session_root=session_root, session_label=session_label)

            smooth_click(page, page.get_by_role("link", name="Dashboard"))
            page.wait_for_url("**/")
            smooth_click(page, page.locator("a[href='/inference']").first)
            page.wait_for_url("**/inference**")
            smooth_click(page, page.get_by_role("link", name="Compute predictions").first)
            page.wait_for_url("**/inference/compute_predictions**")
            scroll_to_prediction_assets(page)

            print("[automation] selecting model artifact from server path...", flush=True)
            select_server_file_via_modal(
                page,
                open_btn=page.locator("#assets-model-source-server-btn"),
                target_file_path=model_path,
                target_input_id="inference-model-server-file-path",
            )

            print("[automation] selecting scaler artifact from server path...", flush=True)
            select_server_file_via_modal(
                page,
                open_btn=page.locator("#assets-scaler-source-server-btn"),
                target_file_path=scaler_path,
                target_input_id="inference-scaler-server-file-path",
            )

            use_scaler = page.locator("#inference-use-scaler")
            wait_for_enabled(page, use_scaler, timeout_sec=60)
            smooth_check(page, use_scaler, after_ms=600)

            compute_btn = page.get_by_role("button", name="Compute Predictions").first
            wait_for_enabled(page, compute_btn, timeout_sec=60)
            print("[automation] submitting predictions computation...", flush=True)
            smooth_click(page, compute_btn, after_ms=700)

            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for predictions job {job_id}...", flush=True)
            wait_for_job_finished(page, base_url, job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] predictions job {job_id} finished.", flush=True)
            page.wait_for_timeout(2200)

            print("[automation] capturing poster...", flush=True)
            page.screenshot(path=str(OUTPUT_POSTER), full_page=False, timeout=10_000)

        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Playwright timeout: {exc}") from exc
        finally:
            print("[automation] finalizing browser context...", flush=True)
            context.close()
            if video is None:
                browser.close()
                raise RuntimeError("Playwright did not attach a video stream.")
            if OUTPUT_VIDEO.exists():
                OUTPUT_VIDEO.unlink()
            print("[automation] saving video...", flush=True)
            video.save_as(str(OUTPUT_VIDEO))
            browser.close()

    if not OUTPUT_VIDEO.exists():
        raise RuntimeError(f"Expected output video was not created: {OUTPUT_VIDEO}")

    shutil.rmtree(tmp_video_dir, ignore_errors=True)
    return OUTPUT_VIDEO


def main() -> int:
    args = parse_args()
    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.start_server:
            print("Starting WebUI server...")
            server_proc = start_webui_server(args.base_url)
        else:
            wait_for_server(args.base_url, timeout_sec=30)

        print("Running Playwright flow...")
        output_video = run_tutorial_recording(
            base_url=args.base_url,
            headless=args.headless,
            job_timeout_sec=args.job_timeout_sec,
            ui_timeout_sec=args.ui_timeout_sec,
            session_label=args.session_label,
            session_root=args.session_root,
            assets_dir=args.assets_dir,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            slow_mo_ms=args.slow_mo_ms,
        )
        print(f"Saved tutorial video to: {output_video}")
        print(f"Saved tutorial poster to: {OUTPUT_POSTER}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(server_proc)


if __name__ == "__main__":
    sys.exit(main())
