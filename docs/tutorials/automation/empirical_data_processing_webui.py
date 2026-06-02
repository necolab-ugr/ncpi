#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow: empirical data -> features -> predictions."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from playwright._impl._errors import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import Locator, Page, sync_playwright


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
VIDEO_DIR = SCRIPT_DIR / "videos"
OUTPUT_VIDEO = VIDEO_DIR / "empirical-data-processing.webm"
OUTPUT_POSTER = VIDEO_DIR / "empirical-data-processing-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
CURSOR_SPEED_PX_PER_MS = 0.72
DEFAULT_DATA_FOLDER = (
    "/home/pablomc/Downloads/empirical_data/LFP"
)
DEFAULT_MODEL_PATH = "/home/pablomc/Downloads/ML_models/MLP/catch22/model"
DEFAULT_SCALER_PATH = "/home/pablomc/Downloads/ML_models/MLP/catch22/scaler"
DEFAULT_FEATURES_N_JOBS = 32
DEFAULT_INFERENCE_N_JOBS = 32
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: Features -> Compute new features -> New data -> "
            "server folder selection -> parser setup -> catch22 -> compute features -> "
            "Continue to Inference -> Compute predictions -> server model/scaler -> compute predictions."
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
        help="Timeout waiting for feature/prediction computations.",
    )
    parser.add_argument(
        "--ui-timeout-sec",
        type=int,
        default=180,
        help="Default UI action timeout.",
    )
    parser.add_argument(
        "--data-folder",
        default=DEFAULT_DATA_FOLDER,
        help="Server folder for feature extraction.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Server file path for model artifact.",
    )
    parser.add_argument(
        "--scaler-path",
        default=DEFAULT_SCALER_PATH,
        help="Server file path for scaler artifact.",
    )
    parser.add_argument(
        "--features-n-jobs",
        type=int,
        default=DEFAULT_FEATURES_N_JOBS,
        help=f"Parallel workers for feature extraction (default: {DEFAULT_FEATURES_N_JOBS}).",
    )
    parser.add_argument(
        "--inference-n-jobs",
        type=int,
        default=DEFAULT_INFERENCE_N_JOBS,
        help=f"Parallel workers for prediction (default: {DEFAULT_INFERENCE_N_JOBS}).",
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


def move_cursor_to_point(
    page: Page,
    x: float,
    y: float,
    *,
    click: bool = False,
    extra_pause_ms: int = 0,
) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    dx = float(x) - _DEMO_CURSOR_POS["x"]
    dy = float(y) - _DEMO_CURSOR_POS["y"]
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    steps = max(24, min(120, int(distance / 10.0)))
    page.mouse.move(float(x), float(y), steps=steps)
    move_demo_cursor(page, click=click)
    _DEMO_CURSOR_POS = {"x": float(x), "y": float(y)}
    travel_ms = max(260, min(980, int(distance / CURSOR_SPEED_PX_PER_MS)))
    page.wait_for_timeout(travel_ms + max(0, int(extra_pause_ms)))


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
    move_cursor_to_point(page, x, y, click=click, extra_pause_ms=max(0, int(pause_ms)))


def smooth_click(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    page.wait_for_timeout(after_ms)


def smooth_mouse_click(page: Page, locator: Locator, after_ms: int = 450) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    handle = locator.element_handle(timeout=5_000)
    if handle is None:
        raise RuntimeError("Could not resolve element handle for mouse click.")
    handle.evaluate(
        "el => el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' })"
    )
    page.wait_for_timeout(40)
    box = handle.bounding_box()
    if box is None:
        raise RuntimeError("Could not resolve element box for mouse click.")
    x = box["x"] + (box["width"] / 2.0)
    y = box["y"] + (box["height"] / 2.0)
    move_cursor_to_point(page, x, y, click=True, extra_pause_ms=40)
    page.mouse.click(x, y, delay=80)
    page.wait_for_timeout(after_ms)


def smooth_check(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    if not locator.is_checked():
        locator.check()
    page.wait_for_timeout(after_ms)


def smooth_fill(page: Page, locator: Locator, value: str, type_delay_ms: int = 120, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    locator.press("Control+A")
    locator.press("Backspace")
    locator.type(value, delay=max(0, int(type_delay_ms)))
    page.wait_for_timeout(after_ms)


def smooth_select_option(page: Page, locator: Locator, value: str, after_ms: int = 450) -> None:
    smooth_select_option_match(
        page=page,
        locator=locator,
        matcher=lambda option_value, option_label: option_value == value or option_label == value,
        description=value,
        after_ms=after_ms,
    )


def smooth_select_option_match(
    page: Page,
    locator: Locator,
    matcher: Callable[[str, str], bool],
    description: str,
    after_ms: int = 450,
) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.click(delay=80)
    page.wait_for_timeout(220)
    state = locator.evaluate(
        """
        (el) => {
          const options = Array.from(el.options || []);
          const values = options.map((opt) => String(opt.value || ""));
          const labels = options.map((opt) => String((opt.textContent || "").trim()));
          const currentIndex = Number.isInteger(el.selectedIndex) ? el.selectedIndex : -1;
          return { values, labels, currentIndex };
        }
        """
    )
    values = [str(item) for item in (state.get("values") or [])]
    labels = [str(item) for item in (state.get("labels") or [])]
    current_index = int(state.get("currentIndex", -1))

    target_index = -1
    for idx, (opt_value, opt_label) in enumerate(zip(values, labels)):
        if matcher(opt_value, opt_label):
            target_index = idx
            break
    if target_index < 0:
        raise RuntimeError(
            f"Dropdown option match not found ({description}). "
            f"Available values: {values}, labels: {labels}"
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


def find_visible_labeled_select(page: Page, label_text: str, select_name: str, timeout_sec: int = 60) -> Locator:
    selector = (
        f"label:has(span:has-text('{label_text}')) "
        f"select[name='{select_name}']:visible"
    )
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        locator = page.locator(selector).first
        try:
            if locator.count() > 0 and locator.is_visible():
                return locator
        except PlaywrightError:
            pass
        page.wait_for_timeout(150)
    raise RuntimeError(f"Timed out waiting for visible labeled select: {select_name} ({label_text})")


def set_labeled_select_exact_resilient(
    page: Page,
    *,
    label_text: str,
    select_name: str,
    wanted_value: str,
    timeout_sec: int = 120,
    after_ms: int = 120,
) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    last_values: list[str] = []
    last_labels: list[str] = []
    wanted = str(wanted_value)
    while time.time() < deadline:
        select = find_visible_labeled_select(page, label_text, select_name, timeout_sec=5)
        wait_for_enabled(page, select, timeout_sec=10)
        move_to_locator(page, select, click=False, pause_ms=120)
        move_to_locator(page, select, click=True, pause_ms=50)
        state = select.evaluate(
            """
            (el, wanted) => {
              const options = Array.from(el?.options || []);
              const values = options.map((opt) => String(opt.value || ''));
              const labels = options.map((opt) => String((opt.textContent || '').trim()));
              let idx = values.findIndex((value) => value === wanted);
              if (idx < 0) idx = labels.findIndex((label) => label === wanted);
              if (idx < 0) {
                return { ok: false, values, labels, current: String(el && el.value != null ? el.value : '') };
              }
              const target = options[idx];
              target.selected = true;
              el.value = String(target.value || '');
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
              return { ok: true, values, labels, current: String(el && el.value != null ? el.value : '') };
            }
            """,
            wanted,
        )
        last_values = [str(item) for item in (state.get("values") or [])]
        last_labels = [str(item) for item in (state.get("labels") or [])]
        if bool(state.get("ok")):
            page.wait_for_timeout(after_ms)
            return
        page.wait_for_timeout(180)
    raise RuntimeError(
        f"Failed to select option {wanted!r} for {select_name}. "
        f"Last available values: {last_values}, labels: {last_labels}"
    )


def _split_abs_path(path_value: str) -> list[str]:
    clean = str(path_value or "").strip().replace("\\", "/")
    if not clean:
        return []
    return [segment for segment in clean.split("/") if segment]


def _normalize_abs_path(path_value: str) -> str:
    segments = _split_abs_path(path_value)
    return "/" + "/".join(segments) if segments else "/"


def _features_folder_modal(page: Page) -> Locator:
    return page.locator("div.fixed.inset-0.z-50:has(h3:has-text('Browse server folders'))").first


def _wait_features_folder_modal_idle(page: Page, modal: Locator) -> None:
    loading = modal.locator("div:has-text('Loading directories...')").first
    if loading.count() == 0:
        return
    try:
        loading.wait_for(state="hidden", timeout=4_000)
    except PlaywrightTimeoutError:
        pass


def _read_features_folder_current_path(page: Page, modal: Locator) -> str:
    current = modal.locator("p:has-text('Current folder:') span.font-mono").first.inner_text(
        timeout=5_000
    )
    return _normalize_abs_path(current or "/")


def _list_features_folder_entries(modal: Locator) -> list[str]:
    return [entry.strip() for entry in modal.locator("div.max-h-80 button span.font-mono").all_text_contents()]


def _click_features_toolbar_button(
    page: Page,
    modal: Locator,
    label: str,
    *,
    after_ms: int = 320,
    timeout_ms: int = 8_000,
    wait_idle: bool = True,
) -> None:
    buttons = modal.locator("div.p-4.space-y-3 > div.flex.flex-wrap.gap-2 > button")
    deadline = time.time() + (timeout_ms / 1000.0)
    wanted = str(label or "").strip().lower()
    while time.time() < deadline:
        if wait_idle:
            _wait_features_folder_modal_idle(page, modal)
        count = buttons.count()
        for idx in range(count):
            btn = buttons.nth(idx)
            name = btn.inner_text().strip().lower()
            if name == wanted:
                _fast_features_modal_click(page, btn, after_ms=after_ms)
                return
        page.wait_for_timeout(80)
    raise RuntimeError(f"Could not find toolbar button '{label}' in features folder browser.")


def _click_features_folder_entry(
    page: Page,
    modal: Locator,
    segment: str,
    timeout_ms: int = 15_000,
    *,
    wait_for_path_change: bool = True,
) -> None:
    entries = modal.locator("div.max-h-80 button")
    labels = modal.locator("div.max-h-80 button span.font-mono")
    wanted = str(segment or "").strip().rstrip("/").lower()
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        count = entries.count()
        names = labels.all_text_contents()
        for idx, raw_name in enumerate(names):
            if idx >= count:
                break
            name = str(raw_name or "").strip().rstrip("/").lower()
            if name != wanted:
                continue
            btn = entries.nth(idx)
            before = _read_features_folder_current_path(page, modal) if wait_for_path_change else ""
            _fast_features_modal_click(page, btn, after_ms=0)
            if not wait_for_path_change:
                print(
                    f"[automation] features-folder final step '{segment}' selected.",
                    flush=True,
                )
                return
            end_deadline = time.time() + 10.0
            while time.time() < end_deadline:
                after = _read_features_folder_current_path(page, modal)
                if after != before:
                    print(
                        f"[automation] features-folder step '{segment}' -> {after}",
                        flush=True,
                    )
                    return
                page.wait_for_timeout(80)
            break
        page.wait_for_timeout(80)
    entries = ", ".join(_list_features_folder_entries(modal))
    raise RuntimeError(
        f"Could not find folder entry '{segment}' in features folder browser. Visible entries: [{entries}]"
    )


def select_server_folder_in_features_modal(page: Page, target_path: str) -> None:
    segments = _split_abs_path(target_path)
    if not segments:
        raise RuntimeError("Target data folder path is empty.")
    modal = _features_folder_modal(page)
    modal.wait_for(state="visible", timeout=15_000)

    normalized_target = _normalize_abs_path(target_path)
    current_path = _read_features_folder_current_path(page, modal)
    print(
        f"[automation] features-folder modal current={current_path} target={normalized_target}",
        flush=True,
    )
    current_segments = _split_abs_path(current_path)

    common = 0
    max_common = min(len(current_segments), len(segments))
    while common < max_common and current_segments[common] == segments[common]:
        common += 1

    for _ in range(len(current_segments) - common):
        _click_features_toolbar_button(page, modal, "Up", after_ms=280)
        _wait_features_folder_modal_idle(page, modal)

    nav_segments = segments[common:]
    for idx, segment in enumerate(nav_segments):
        print(f"[automation] navigating to folder segment: {segment}", flush=True)
        is_final_segment = idx == (len(nav_segments) - 1)
        _click_features_folder_entry(
            page,
            modal,
            segment,
            timeout_ms=18_000,
            wait_for_path_change=not is_final_segment,
        )

    _click_features_toolbar_button(page, modal, "Add this folder", after_ms=0, wait_idle=False)
    modal.wait_for(state="hidden", timeout=15_000)


def wait_for_parser_inspection_ready(page: Page, timeout_sec: int = 240) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        if page.locator("div:has-text('Full-folder inspection completed.')").first.count() > 0:
            if page.locator("div:has-text('Full-folder inspection completed.')").first.is_visible():
                return
        if page.locator("select[name='parser_axis_channels']:visible").count() > 0:
            return
        if page.locator("div:has-text('Failed to inspect simulation outputs folder.')").first.count() > 0:
            raise RuntimeError("Parser inspection failed while loading server folder.")
        page.wait_for_timeout(350)
    raise RuntimeError("Timed out waiting for parser inspection to complete.")


def wait_for_parser_loading_idle(page: Page, timeout_sec: int = 180) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    indicator = page.locator("span:has-text('Inspecting selected data...')").first
    while time.time() < deadline:
        busy = False
        try:
            busy = indicator.count() > 0 and indicator.is_visible()
        except PlaywrightError:
            busy = False
        if not busy:
            # Confirm parser loading indicator stays hidden briefly.
            page.wait_for_timeout(240)
            try:
                if not (indicator.count() > 0 and indicator.is_visible()):
                    return
            except PlaywrightError:
                return
        page.wait_for_timeout(180)
    raise RuntimeError("Timed out waiting for parser loading to become idle.")


def scroll_to_parser_metadata_sources(page: Page) -> None:
    subject_label = page.locator("span:has-text('Subject ID source')").first
    move_to_locator(page, subject_label, click=False, pause_ms=120)
    page.evaluate(
        """
        () => {
          const label = Array.from(document.querySelectorAll('span'))
            .find((el) => (el.textContent || '').trim() === 'Subject ID source');
          if (!label) return;
          label.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        }
        """
    )
    page.wait_for_timeout(420)


def open_dropdown_menu(page: Page, select: Locator) -> None:
    move_to_locator(page, select, click=False, pause_ms=0)
    page.wait_for_timeout(1000)
    move_demo_cursor(page, click=True)
    select.click(delay=0)
    page.wait_for_timeout(260)


def set_dropdown_option_direct(
    page: Page,
    select: Locator,
    *,
    value: str,
    label_fallback_contains: Optional[str] = None,
    after_ms: int = 140,
) -> None:
    wanted = str(value)
    try:
        selected = select.select_option(value=wanted)
        if not selected and label_fallback_contains:
            token = str(label_fallback_contains).strip().lower()
            state = select.evaluate(
                """
                (el) => {
                  const options = Array.from(el?.options || []);
                  return options.map((opt) => ({
                    value: String(opt.value || ''),
                    label: String((opt.textContent || '').trim()),
                  }));
                }
                """
            )
            for item in (state or []):
                label = str(item.get("label") or "")
                if token and token in label.lower():
                    select.select_option(value=str(item.get("value") or ""))
                    break
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to select dropdown option {wanted!r}: {exc}") from exc
    page.wait_for_timeout(after_ms)


def choose_select_option_with_menu_cursor(
    page: Page,
    select: Locator,
    *,
    value: str,
    label_fallback_contains: Optional[str] = None,
    after_ms: int = 180,
) -> None:
    global _DEMO_CURSOR_POS

    wanted = str(value)
    fallback_token = str(label_fallback_contains or "").strip().lower()
    smooth_select_option_match(
        page=page,
        locator=select,
        matcher=lambda option_value, option_label: (
            option_value == wanted
            or (bool(fallback_token) and fallback_token in option_label.lower())
        ),
        description=wanted,
        after_ms=after_ms,
    )
    return

    move_to_locator(page, select, click=False, pause_ms=120)
    move_to_locator(page, select, click=True, pause_ms=60)

    state = select.evaluate(
        """
        (el) => {
          const options = Array.from(el?.options || []);
          return {
            values: options.map((opt) => String(opt.value || '')),
            labels: options.map((opt) => String((opt.textContent || '').trim())),
            currentIndex: Number.isInteger(el?.selectedIndex) ? el.selectedIndex : -1,
          };
        }
        """
    )
    values = [str(item) for item in (state.get("values") or [])]
    labels = [str(item) for item in (state.get("labels") or [])]
    current_index = int(state.get("currentIndex", -1))

    target_index = -1
    for idx, opt_value in enumerate(values):
        if opt_value == wanted:
            target_index = idx
            break
    if target_index < 0 and label_fallback_contains:
        token = str(label_fallback_contains).strip().lower()
        for idx, opt_label in enumerate(labels):
            if token and token in opt_label.lower():
                target_index = idx
                wanted = values[idx] if idx < len(values) else wanted
                break
    if target_index < 0:
        raise RuntimeError(
            f"Could not find dropdown option {value!r}. "
            f"Available values: {values}, labels: {labels}"
        )
    select.evaluate("el => el && el.focus()")

    overlay_target = page.evaluate(
        """
        ({ labels, targetIndex }) => {
          const selectEl = document.activeElement && document.activeElement.tagName === 'SELECT'
            ? document.activeElement
            : null;
          const fallbackSelect = document.querySelector('select:focus');
          const anchor = selectEl || fallbackSelect;
          if (!anchor) return null;

          const old = document.getElementById('pw-demo-select-menu');
          if (old) old.remove();

          const rect = anchor.getBoundingClientRect();
          const optionHeight = Math.max(24, Math.min(32, rect.height || 28));
          const count = Array.isArray(labels) ? labels.length : 0;
          const visibleRows = Math.max(4, Math.min(10, count || 6));
          const menuHeight = visibleRows * optionHeight;
          const spaceBelow = window.innerHeight - (rect.bottom + 8);
          const spaceAbove = rect.top - 8;
          const opensUpward = spaceBelow < menuHeight && spaceAbove > spaceBelow;

          const menu = document.createElement('div');
          menu.id = 'pw-demo-select-menu';
          menu.style.position = 'fixed';
          menu.style.left = `${Math.max(10, Math.min(window.innerWidth - rect.width - 10, rect.left))}px`;
          menu.style.width = `${Math.max(180, rect.width)}px`;
          menu.style.maxHeight = `${menuHeight}px`;
          menu.style.overflow = 'hidden';
          menu.style.border = '1px solid rgba(100,116,139,0.65)';
          menu.style.borderRadius = '10px';
          menu.style.background = '#ffffff';
          menu.style.boxShadow = '0 14px 28px rgba(15,23,42,0.30)';
          menu.style.zIndex = '2147483646';
          menu.style.pointerEvents = 'none';

          const start = Math.max(0, Math.min(
            targetIndex - Math.floor(visibleRows / 2),
            Math.max(0, count - visibleRows),
          ));
          const end = Math.min(count, start + visibleRows);
          const rows = [];
          for (let i = start; i < end; i += 1) {
            const row = document.createElement('div');
            row.textContent = String(labels[i] || '');
            row.style.height = `${optionHeight}px`;
            row.style.lineHeight = `${optionHeight}px`;
            row.style.padding = '0 10px';
            row.style.fontSize = '13px';
            row.style.color = '#111827';
            row.style.background = '#ffffff';
            row.style.borderTop = i > start ? '1px solid rgba(226,232,240,0.9)' : 'none';
            menu.appendChild(row);
            rows.push(row);
          }

          const menuTop = opensUpward
            ? Math.max(10, rect.top - menuHeight - 6)
            : Math.min(window.innerHeight - menuHeight - 10, rect.bottom + 6);
          menu.style.top = `${menuTop}px`;

          document.body.appendChild(menu);

          const hoverColor = 'rgba(59,130,246,0.18)';
          const clearRows = () => {
            for (const row of rows) {
              row.style.background = '#ffffff';
            }
          };
          const onMouseMove = (evt) => {
            const x = Number(evt?.clientX ?? -1);
            const y = Number(evt?.clientY ?? -1);
            const left = parseFloat(menu.style.left) || 0;
            const width = parseFloat(menu.style.width) || 0;
            const top = menuTop;
            const height = (end - start) * optionHeight;
            const inside = x >= left && x <= (left + width) && y >= top && y <= (top + height);
            if (!inside) {
              clearRows();
              return;
            }
            const idx = Math.floor((y - top) / optionHeight);
            clearRows();
            if (idx >= 0 && idx < rows.length) {
              rows[idx].style.background = hoverColor;
            }
          };
          document.addEventListener('mousemove', onMouseMove, { passive: true });
          window.__pwDemoSelectMenuCleanup = () => {
            try {
              document.removeEventListener('mousemove', onMouseMove);
            } catch (_err) {}
          };

          const activeLocal = Math.max(0, targetIndex - start);
          const targetY = menuTop + ((activeLocal + 0.5) * optionHeight);
          const targetX = parseFloat(menu.style.left) + (parseFloat(menu.style.width) / 2.0);
          return { x: targetX, y: targetY };
        }
        """,
        {"labels": labels, "targetIndex": target_index},
    )

    if overlay_target:
        menu_x = float(overlay_target.get("x"))
        menu_y = float(overlay_target.get("y"))
        move_cursor_to_point(page, menu_x, menu_y)
        page.wait_for_timeout(1000)
        move_demo_cursor(page, click=True)
        page.mouse.click(menu_x, menu_y, delay=0)
        page.evaluate(
            """
            () => {
              if (typeof window.__pwDemoSelectMenuCleanup === 'function') {
                try { window.__pwDemoSelectMenuCleanup(); } catch (_err) {}
              }
              window.__pwDemoSelectMenuCleanup = null;
              const node = document.getElementById('pw-demo-select-menu');
              if (node) node.remove();
            }
            """
        )

    # Deterministic fallback to guarantee final selected value.
    set_dropdown_option_direct(
        page,
        select,
        value=wanted,
        label_fallback_contains=label_fallback_contains,
        after_ms=after_ms,
    )


def select_mat_data_source(page: Page) -> None:
    blocks = page.locator("div:has(h4:has-text('Data source selection'))")
    count = blocks.count()
    if count == 0:
        raise RuntimeError("Could not find 'Data source selection' section.")
    for idx in range(count):
        block_select = blocks.nth(idx).locator("select").first
        if block_select.count() == 0 or not block_select.is_visible():
            continue
        # Avoid re-selecting if current option is already .mat
        current = block_select.evaluate(
            """
            (el) => {
              const selected = el && el.options && el.selectedIndex >= 0 ? el.options[el.selectedIndex] : null;
              return {
                value: String(el && el.value != null ? el.value : ''),
                label: String(selected && selected.textContent ? selected.textContent : ''),
              };
            }
            """
        )
        current_value = str((current or {}).get("value") or "").strip().lower()
        current_label = str((current or {}).get("label") or "").strip().lower()
        if ".mat" in current_value or ".mat" in current_label:
            return
        try:
            smooth_select_option_match(
                page=page,
                locator=block_select,
                matcher=lambda value, label: ".mat" in value.lower() or ".mat" in label.lower(),
                description="option containing .mat",
                after_ms=520,
            )
            return
        except RuntimeError:
            continue
    raise RuntimeError("No 'Data source selection' dropdown with a .mat option was found.")


def set_aggregate_over_sensor(page: Page) -> None:
    select = page.locator("select[multiple][x-model='parserAggregateOverSelected']").first
    if select.count() == 0:
        select = page.locator("select[multiple]:visible").first
    if select.count() == 0:
        raise RuntimeError("Aggregate-over multiselect was not found.")
    move_to_locator(page, select, click=False, pause_ms=150)
    option = select.locator("option", has_text="sensor").first
    option.wait_for(state="visible", timeout=10_000)
    move_to_locator(page, option, click=False, pause_ms=0)
    page.wait_for_timeout(1000)
    move_demo_cursor(page, click=True)
    option.click()
    select.evaluate(
        """
        (el) => {
          const opts = Array.from(el.options || []);
          opts.forEach((opt) => {
            const text = String((opt.textContent || '').trim()).toLowerCase();
            const value = String(opt.value || '').trim().toLowerCase();
            opt.selected = text === 'sensor' || value === 'sensor';
          });
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        """
    )
    page.wait_for_timeout(420)


def _wait_file_modal_idle(page: Page) -> None:
    loading = page.locator("#inferenceServerFileBrowserLoading")
    if loading.count() == 0:
        return
    try:
        loading.wait_for(state="hidden", timeout=1_200)
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


def _resolve_artifact_file(path_value: str, kind: str) -> Path:
    target = Path(path_value).expanduser()
    key = "model" if kind == "model" else "scaler"
    alt_key = "posterior" if key == "model" else None

    if target.is_file():
        return target

    candidates: list[Path] = []
    if target.is_dir():
        for entry in sorted(target.iterdir()):
            if not entry.is_file():
                continue
            low = entry.name.lower()
            if low == key or low.startswith(f"{key}."):
                candidates.append(entry)
            elif alt_key and (low == alt_key or low.startswith(f"{alt_key}.")):
                candidates.append(entry)
    elif target.parent.is_dir():
        base = target.name.lower()
        for entry in sorted(target.parent.iterdir()):
            if not entry.is_file():
                continue
            low = entry.name.lower()
            if low == base or low.startswith(f"{base}."):
                candidates.append(entry)
            elif alt_key and (
                low == alt_key or low.startswith(f"{alt_key}.")
            ):
                candidates.append(entry)

    if candidates:
        return candidates[0]

    parent = target if target.is_dir() else target.parent
    parent_hint = str(parent)
    nearby = []
    if parent.is_dir():
        for entry in sorted(parent.iterdir()):
            if entry.is_file():
                nearby.append(entry.name)
            if len(nearby) >= 12:
                break
    nearby_hint = ", ".join(nearby) if nearby else "none"
    raise RuntimeError(
        f"Target artifact file does not exist: {path_value}. "
        f"Checked in: {parent_hint}. Nearby files: [{nearby_hint}]"
    )


def select_server_file_via_modal(
    page: Page,
    open_btn: Locator,
    target_file_path: str,
    target_input_id: str,
    kind: str,
) -> None:
    target_file = _resolve_artifact_file(target_file_path, kind=kind)
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


from tutorial_cursor import (  # noqa: E402
    ensure_demo_cursor,
    move_cursor_to_point,
    move_demo_cursor,
    move_to_locator,
    reset_demo_cursor_position,
    show_cursor_transition,
    smooth_check,
    smooth_click,
    smooth_fill,
    smooth_mouse_click,
    smooth_scroll_locator_into_view,
    smooth_select_option,
    smooth_select_option_match,
)


def _fast_features_modal_click(page: Page, locator: Locator, after_ms: int = 0) -> None:
    """Click inside the server-folder modal without the global tutorial pause."""
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    locator.wait_for(state="visible", timeout=5_000)
    handle = locator.element_handle(timeout=5_000)
    if handle is None:
        raise RuntimeError("Could not resolve element handle for folder-browser click.")
    handle.evaluate(
        "el => el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' })"
    )
    box = handle.bounding_box()
    if box is None:
        raise RuntimeError("Could not resolve element box for folder-browser click.")
    x = box["x"] + (box["width"] / 2.0)
    y = box["y"] + (box["height"] / 2.0)
    move_cursor_to_point(page, x, y)
    move_demo_cursor(page, click=True)
    page.mouse.click(x, y, delay=0)
    if after_ms > 0:
        page.wait_for_timeout(after_ms)


def run_tutorial_recording(
    base_url: str,
    headless: bool,
    job_timeout_sec: int,
    ui_timeout_sec: int,
    data_folder: str,
    model_path: str,
    scaler_path: str,
    features_n_jobs: int,
    inference_n_jobs: int,
    viewport_width: int,
    viewport_height: int,
    slow_mo_ms: int,
) -> Path:
    global _DEMO_CURSOR_POS
    _DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}
    reset_demo_cursor_position()
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

            print("[automation] opening Features...", flush=True)
            smooth_click(page, page.locator("a[href='/features']").first)
            page.wait_for_url("**/features**")

            compute_new_features = page.get_by_role("button", name="Compute new features")
            if compute_new_features.count() == 0:
                compute_new_features = page.get_by_role("link", name="Compute new features")
            if compute_new_features.count() == 0:
                compute_new_features = page.locator("a:has-text('Compute new features')").first
            if compute_new_features.count() == 0:
                raise RuntimeError("Could not find 'Compute new features' entry in Features page.")
            smooth_click(page, compute_new_features.first)
            page.wait_for_url("**/features/methods**")

            print("[automation] choosing New data flow...", flush=True)
            smooth_click(page, page.get_by_role("button", name="New data").first)
            page.locator("h2:has-text('New data configuration')").first.wait_for(
                state="visible", timeout=30_000
            )

            data_folder_card = page.locator(
                "div:has(h3:has-text('Data folder for feature extraction'))"
            ).first
            smooth_mouse_click(
                page,
                data_folder_card.get_by_role("button", name="Add server folder").first,
                after_ms=0,
            )

            print(f"[automation] selecting feature-extraction folder: {data_folder}", flush=True)
            select_server_folder_in_features_modal(page, data_folder)
            wait_for_parser_inspection_ready(page, timeout_sec=360)

            print("[automation] selecting .mat data source...", flush=True)
            select_mat_data_source(page)
            wait_for_parser_loading_idle(page, timeout_sec=180)

            print("[automation] configuring parser fields...", flush=True)
            channel_names_source_select = find_visible_labeled_select(
                page, "Channel names source", "parser_ch_names_source", timeout_sec=150
            )
            choose_select_option_with_menu_cursor(
                page,
                channel_names_source_select,
                value="__autocomplete__",
                label_fallback_contains="autocomplete",
                after_ms=80,
            )
            print("[automation] channel source set to autocomplete.", flush=True)
            wait_for_parser_loading_idle(page, timeout_sec=120)
            scroll_to_parser_metadata_sources(page)

            subject_id_source_select = find_visible_labeled_select(
                page, "Subject ID source", "parser_metadata_subject_id_source", timeout_sec=120
            )
            choose_select_option_with_menu_cursor(
                page,
                subject_id_source_select,
                value="__file_id__",
                label_fallback_contains="file_id",
                after_ms=140,
            )
            print("[automation] subject ID source set to file_ID.", flush=True)

            group_source_select = find_visible_labeled_select(
                page, "Group source", "parser_metadata_group_source", timeout_sec=120
            )
            choose_select_option_with_menu_cursor(
                page,
                group_source_select,
                value="LFP.age",
                label_fallback_contains="lfp.age",
                after_ms=160,
            )
            print("[automation] group source set to LFP.age.", flush=True)

            epoching_checkbox = page.locator(
                "label:has-text('Enable epoching') input[type='checkbox']:visible"
            ).first
            aggregation_checkbox = page.locator(
                "label:has-text('Enable aggregation') input[type='checkbox']:visible"
            ).first
            smooth_check(page, epoching_checkbox)
            smooth_check(page, aggregation_checkbox)
            set_aggregate_over_sensor(page)

            smooth_click(page, page.get_by_role("button", name="Next: Select method").first)
            page.locator("h2:has-text('Select Method')").first.wait_for(state="visible")

            print("[automation] selecting catch22...", flush=True)
            catch22_radio = page.locator(
                "input[type='radio'][name='select-method'][value='catch22']:visible"
            ).first
            if catch22_radio.count() == 0:
                raise RuntimeError("Could not find catch22 method selector.")
            smooth_check(page, catch22_radio)
            smooth_click(page, page.get_by_role("button", name="Next step").first)
            page.locator("h1:has-text('Method Configuration')").first.wait_for(state="visible")

            smooth_fill(
                page,
                page.locator("input[name='features_n_jobs']").first,
                str(max(1, int(features_n_jobs))),
                type_delay_ms=80,
                after_ms=420,
            )
            smooth_fill(
                page,
                page.locator("input[name='features_subsample_percent']:visible").first,
                "10",
                type_delay_ms=80,
                after_ms=420,
            )

            compute_features_btn = page.get_by_role("button", name="Compute features").first
            wait_for_enabled(page, compute_features_btn, timeout_sec=600)
            print("[automation] submitting feature extraction...", flush=True)
            smooth_click(page, compute_features_btn, after_ms=700)

            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            features_job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for features job {features_job_id}...", flush=True)
            wait_for_job_finished(page, base_url, features_job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] features job {features_job_id} finished.", flush=True)
            continue_to_inference = page.get_by_role("link", name="Continue to Inference").first
            continue_to_inference.wait_for(state="visible", timeout=60_000)
            smooth_click(page, continue_to_inference, after_ms=820)

            page.wait_for_url("**/inference**")
            print("[automation] opening Compute predictions...", flush=True)
            smooth_click(page, page.get_by_role("link", name="Compute predictions").first)
            page.wait_for_url("**/inference/compute_predictions**")
            scroll_to_prediction_assets(page)

            print(f"[automation] selecting model file: {model_path}", flush=True)
            select_server_file_via_modal(
                page,
                open_btn=page.locator("#assets-model-source-server-btn"),
                target_file_path=model_path,
                target_input_id="inference-model-server-file-path",
                kind="model",
            )

            print(f"[automation] selecting scaler file: {scaler_path}", flush=True)
            select_server_file_via_modal(
                page,
                open_btn=page.locator("#assets-scaler-source-server-btn"),
                target_file_path=scaler_path,
                target_input_id="inference-scaler-server-file-path",
                kind="scaler",
            )

            use_scaler = page.locator("#inference-use-scaler")
            wait_for_enabled(page, use_scaler, timeout_sec=60)
            smooth_check(page, use_scaler, after_ms=520)

            smooth_fill(
                page,
                page.locator("input[name='inference_n_jobs']").first,
                str(max(1, int(inference_n_jobs))),
                type_delay_ms=80,
                after_ms=420,
            )
            smooth_fill(
                page,
                page.locator("input[name='inference_subsample_percent']:visible").first,
                "10",
                type_delay_ms=80,
                after_ms=420,
            )

            compute_predictions_btn = page.get_by_role("button", name="Compute Predictions").first
            wait_for_enabled(page, compute_predictions_btn, timeout_sec=120)
            print("[automation] submitting predictions computation...", flush=True)
            smooth_click(page, compute_predictions_btn, after_ms=700)

            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            predictions_job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for predictions job {predictions_job_id}...", flush=True)
            wait_for_job_finished(page, base_url, predictions_job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] predictions job {predictions_job_id} finished.", flush=True)
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
            data_folder=args.data_folder,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            features_n_jobs=args.features_n_jobs,
            inference_n_jobs=args.inference_n_jobs,
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
