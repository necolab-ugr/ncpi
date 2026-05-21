#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for computing field potentials from a saved session."""

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
OUTPUT_VIDEO = VIDEO_DIR / "compute-field-potentials.webm"
OUTPUT_POSTER = VIDEO_DIR / "compute-field-potentials-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_SESSION_LABEL = "webui docs"
DEFAULT_SESSION_ROOT = "/tmp/ncpi_webui_session_e6152202b3294223b22bb4fc0bc1682b"
DEFAULT_MC_FOLDER = "/home/pablomc/Downloads/multicompartment_neuron_network"
DEFAULT_OUTPUT_SIM_FOLDER = (
    "/home/pablomc/Downloads/multicompartment_neuron_network/output/"
    "adb947bfb931a5a8d09ad078a6d256b0/"
)
DEFAULT_ANALYSIS_PARAMS = (
    REPO_ROOT / "examples/simulation/Hagen_model/simulation/params/analysis_params.py"
)
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: load a saved session, compute CDM/LFP with kernel "
            "configuration, then plot CDM and LFP in Analysis."
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
        help="Timeout waiting for field-potential computation completion.",
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
        "--mc-folder",
        default=DEFAULT_MC_FOLDER,
        help="Multicompartment neuron model folder path.",
    )
    parser.add_argument(
        "--output-sim-folder",
        default=DEFAULT_OUTPUT_SIM_FOLDER,
        help="Multicompartment simulation output folder path.",
    )
    parser.add_argument(
        "--analysis-params-path",
        default=str(DEFAULT_ANALYSIS_PARAMS),
        help=(
            "Fallback server path for KernelParams module (.py). "
            "Used if the loaded session does not already provide one."
        ),
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


def smooth_fill(page: Page, locator: Locator, value: str, type_delay_ms: int = 120, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    locator.press("Control+A")
    locator.press("Backspace")
    locator.type(value, delay=max(0, int(type_delay_ms)))
    page.wait_for_timeout(after_ms)


def smooth_check(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    if not locator.is_checked():
        locator.check()
    page.wait_for_timeout(after_ms)


def smooth_select_option(page: Page, locator: Locator, value: str, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    page.wait_for_timeout(220)
    state = locator.evaluate(
        """
        (el, targetValue) => {
          const options = Array.from(el.options || []);
          const values = options.map((opt) => String(opt.value || ""));
          const labels = options.map((opt) => String((opt.textContent || "").trim()));
          let targetIndex = values.findIndex((item) => item === targetValue);
          if (targetIndex < 0) {
            targetIndex = labels.findIndex((item) => item === targetValue);
          }
          const currentIndex = Number.isInteger(el.selectedIndex) ? el.selectedIndex : -1;
          return { targetIndex, currentIndex, values, labels };
        }
        """,
        value,
    )
    target_index = int(state.get("targetIndex", -1))
    current_index = int(state.get("currentIndex", -1))
    if target_index < 0:
        raise RuntimeError(
            f"Dropdown option {value!r} not found. "
            f"Available values: {state.get('values', [])}, labels: {state.get('labels', [])}"
        )
    if current_index < 0:
        locator.press("Home")
        page.wait_for_timeout(120)
        current_index = 0
    if target_index != current_index:
        key = "ArrowDown" if target_index > current_index else "ArrowUp"
        for _ in range(abs(target_index - current_index)):
            locator.press(key)
            page.wait_for_timeout(120)
    locator.press("Enter")
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


def wait_for_analysis_load_all(page: Page, timeout_sec: int = 180) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    load_all = page.get_by_role("button", name="Load all")
    while time.time() < deadline:
        if load_all.count() > 0 and load_all.first.is_visible():
            return
        page.wait_for_timeout(1500)
        page.reload(wait_until="domcontentloaded")
    raise RuntimeError(
        "Analysis did not expose 'Load all' simulation files in time. "
        "Simulation/field-potential outputs may still be unavailable."
    )


def wait_for_plot_rendered(page: Page, timeout_sec: int) -> None:
    timeout_ms = max(1, int(timeout_sec)) * 1000
    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    plot_image = page.locator("#plot-image")
    plot_image.wait_for(state="attached", timeout=timeout_ms)
    try:
        plot_image.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        loaded = page.evaluate(
            """
            () => {
              const img = document.getElementById('plot-image');
              if (!img) return false;
              return !!(img.complete && img.naturalWidth > 0);
            }
            """
        )
        if not loaded:
            raise


def scroll_plot_into_view(page: Page) -> None:
    page.evaluate(
        """
        () => {
          const img = document.getElementById('plot-image');
          if (!img) return;
          img.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        }
        """
    )
    page.wait_for_timeout(1200)


def _split_abs_path(path_value: str) -> list[str]:
    clean = str(path_value or "").strip().replace("\\", "/")
    if not clean:
        return []
    return [segment for segment in clean.split("/") if segment]


def _list_server_folder_entries(page: Page) -> list[str]:
    return page.locator("#kernel-server-folder-list button span.font-mono").all_text_contents()


def _normalize_abs_path(path_value: str) -> str:
    segments = _split_abs_path(path_value)
    return "/" + "/".join(segments) if segments else "/"


def _read_server_folder_current_path(page: Page) -> str:
    current = page.locator("#kernel-server-folder-current").inner_text(timeout=5_000).strip()
    return _normalize_abs_path(current or "/")


def _wait_server_folder_modal_idle(page: Page) -> None:
    loading = page.locator("#kernel-server-folder-loading")
    if loading.count() == 0:
        return
    try:
        loading.wait_for(state="hidden", timeout=20_000)
    except PlaywrightTimeoutError:
        pass


def _click_server_folder_entry(page: Page, segment: str, timeout_ms: int = 15_000) -> None:
    buttons = page.locator("#kernel-server-folder-list button")
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        _wait_server_folder_modal_idle(page)
        count = buttons.count()
        for idx in range(count):
            btn = buttons.nth(idx)
            name = btn.locator("span.font-mono").first.inner_text().strip()
            if name == segment:
                before = _read_server_folder_current_path(page)
                smooth_click(page, btn, after_ms=420)
                end_deadline = time.time() + 12.0
                while time.time() < end_deadline:
                    _wait_server_folder_modal_idle(page)
                    after = _read_server_folder_current_path(page)
                    if after != before:
                        return
                    page.wait_for_timeout(140)
                return
        page.wait_for_timeout(180)
    entries = ", ".join(_list_server_folder_entries(page))
    raise RuntimeError(f"Could not find folder entry '{segment}'. Visible entries: [{entries}]")


def select_server_folder_in_kernel_modal(page: Page, target_path: str) -> None:
    segments = _split_abs_path(target_path)
    if not segments:
        raise RuntimeError("Target server folder path is empty.")

    modal = page.locator("#kernel-server-folder-browser")
    modal.wait_for(state="visible", timeout=15_000)
    page.wait_for_timeout(450)

    normalized_target = _normalize_abs_path(target_path)
    current_path = _read_server_folder_current_path(page)
    print(f"[automation] server-folder modal current={current_path} target={normalized_target}", flush=True)
    current_segments = _split_abs_path(current_path)

    common = 0
    max_common = min(len(current_segments), len(segments))
    while common < max_common and current_segments[common] == segments[common]:
        common += 1

    up_btn = page.locator("#kernel-server-folder-up")
    for _ in range(len(current_segments) - common):
        smooth_click(page, up_btn, after_ms=320)
        _wait_server_folder_modal_idle(page)

    for segment in segments[common:]:
        _click_server_folder_entry(page, segment, timeout_ms=18_000)

    smooth_click(page, page.locator("#kernel-server-folder-select"), after_ms=620)
    modal.wait_for(state="hidden")


def set_kernel_params_fallback(
    page: Page,
    fallback_analysis_params: str,
) -> None:
    page.evaluate(
        """
        ({ fallbackParamsPath }) => {
          const sourceMode = document.getElementById('kernelParamsModuleSourceMode');
          const paramsServerPath = document.getElementById('kernelParamsModuleServerPath');
          if (sourceMode && paramsServerPath) {
            const currentPath = String(paramsServerPath.value || '').trim();
            if (!currentPath) {
              sourceMode.value = 'server-path';
              paramsServerPath.value = fallbackParamsPath;
              sourceMode.dispatchEvent(new Event('change', { bubbles: true }));
              paramsServerPath.dispatchEvent(new Event('input', { bubbles: true }));
              paramsServerPath.dispatchEvent(new Event('change', { bubbles: true }));
            }
          }

          const submitBtn = document.querySelector('button[type="submit"][form="kernel-compute-form"]');
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.removeAttribute('aria-disabled');
            submitBtn.title = '';
          }
        }
        """,
        {
            "fallbackParamsPath": fallback_analysis_params,
        },
    )


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
    data_dir = base / "simulation" / "data"
    return _has_any_file(data_dir, suffixes={".pkl"})


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

    required_modules = {"simulation"}
    eligible = []
    for entry in entries:
        if bool(entry.get("isActive")):
            continue
        modules = {str(item).strip().lower() for item in (entry.get("modules") or [])}
        if not required_modules.issubset(modules):
            continue
        if not _session_has_simulation_artifacts(str(entry.get("path") or "")):
            continue
        eligible.append(entry)
    if not eligible:
        raise RuntimeError(
            "No previous session contains simulation artifacts required for field potential "
            "tutorial (expected .pkl files under simulation/data)."
        )
    target = eligible[0]
    target_path = str(target["path"])
    print(
        f"[automation] loading most recent session with simulation artifacts "
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
    mc_folder: str,
    output_sim_folder: str,
    analysis_params_path: str,
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

            load_requested_session(page, session_root=session_root, session_label=session_label)

            smooth_click(page, page.get_by_role("link", name="Dashboard"))
            page.wait_for_url("**/")
            smooth_click(page, page.locator("a[href='/field_potential']").first)
            page.wait_for_url("**/field_potential")
            smooth_click(page, page.get_by_role("link", name="Kernel"))
            page.wait_for_url("**/field_potential/kernel**")

            print("[automation] selecting multicompartment model folder in kernel browser...", flush=True)
            mc_card = page.locator("#kernelMcFolderUploadCard")
            print("[automation] opening MC server-folder modal...", flush=True)
            smooth_click(page, mc_card)
            select_server_folder_in_kernel_modal(page, mc_folder)
            page.wait_for_timeout(900)

            print("[automation] selecting multicompartment output folder in kernel browser...", flush=True)
            output_card = page.locator("#kernelOutputSimUploadCard")
            print("[automation] opening output server-folder modal...", flush=True)
            smooth_click(page, output_card)
            select_server_folder_in_kernel_modal(page, output_sim_folder)
            page.wait_for_timeout(900)

            print("[automation] applying kernel params fallback (if needed)...", flush=True)
            set_kernel_params_fallback(
                page,
                fallback_analysis_params=analysis_params_path,
            )
            page.wait_for_timeout(700)

            smooth_click(page, page.get_by_role("button", name="Current dipole moment/LFP"))
            smooth_check(page, page.locator("input[name='probe_gauss_cylinder']"))
            smooth_fill(page, page.locator("#cdm-decimation-factor"), "10")

            print("[automation] submitting CDM/LFP computation...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Compute CDM/LFP"))
            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for field-potential job {job_id}...", flush=True)
            wait_for_job_finished(page, base_url, job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] field-potential job {job_id} finished.", flush=True)

            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_url("**/")
            smooth_click(page, page.locator("a[href='/analysis']").first)
            page.wait_for_url("**/analysis**")

            wait_for_analysis_load_all(page, timeout_sec=180)
            load_all = page.get_by_role("button", name="Load all")
            print("[automation] loading detected simulation/field-potential files...", flush=True)
            smooth_click(page, load_all)

            smooth_click(page, page.get_by_role("button", name="Simulation outputs"))
            smooth_select_option(page, page.locator("#sim-plot-type"), "cdm")
            print("[automation] plotting CDM...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Plot simulation outputs"))
            wait_for_plot_rendered(page, timeout_sec=ui_timeout_sec)
            scroll_plot_into_view(page)
            page.wait_for_timeout(2200)

            smooth_click(page, page.get_by_role("link", name="Back to analysis"))
            page.wait_for_url("**/analysis**")
            smooth_click(page, page.get_by_role("button", name="Simulation outputs"))
            smooth_select_option(page, page.locator("#sim-plot-type"), "lfp")
            print("[automation] plotting LFP...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Plot simulation outputs"))
            wait_for_plot_rendered(page, timeout_sec=ui_timeout_sec)
            scroll_plot_into_view(page)
            page.wait_for_timeout(2600)

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
            mc_folder=args.mc_folder,
            output_sim_folder=args.output_sim_folder,
            analysis_params_path=args.analysis_params_path,
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
