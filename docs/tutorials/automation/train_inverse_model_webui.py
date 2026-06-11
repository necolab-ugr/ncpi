#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for inverse model training."""

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
OUTPUT_VIDEO = VIDEO_DIR / "train-inverse-model.webm"
OUTPUT_POSTER = VIDEO_DIR / "train-inverse-model-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_FEATURES_PATH = "/home/pablomc/Downloads/training/sim_x"
DEFAULT_PARAMETERS_PATH = "/home/pablomc/Downloads/training/sm_theta"
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: load a previous session, open Inference -> New training, "
            "select features and parameters files from server paths, choose Ridge model, and start training."
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
        help="Timeout waiting for training completion.",
    )
    parser.add_argument(
        "--ui-timeout-sec",
        type=int,
        default=180,
        help="Default UI action timeout.",
    )
    parser.add_argument(
        "--features-path",
        default=DEFAULT_FEATURES_PATH,
        help="Server path for features training file.",
    )
    parser.add_argument(
        "--parameters-path",
        default=DEFAULT_PARAMETERS_PATH,
        help="Server path for parameters training file.",
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


def _split_abs_path(path_value: str) -> list[str]:
    clean = str(path_value or "").strip().replace("\\", "/")
    if not clean:
        return []
    return [segment for segment in clean.split("/") if segment]


def _normalize_abs_path(path_value: str) -> str:
    segments = _split_abs_path(path_value)
    return "/" + "/".join(segments) if segments else "/"


def _resolve_path_fuzzy(path_value: str) -> str:
    target = Path(path_value)
    if target.is_file():
        return str(target)

    parent = target.parent
    name = target.name
    if parent.is_dir():
        children = [p for p in parent.iterdir() if p.is_file()]
        by_exact_ci = next((p for p in children if p.name.lower() == name.lower()), None)
        if by_exact_ci:
            return str(by_exact_ci)
        if name.lower().startswith("sm_"):
            alt = "sim_" + name[3:]
            alt_match = next((p for p in children if p.name.lower() == alt.lower()), None)
            if alt_match:
                return str(alt_match)
        if "theta" in name.lower():
            theta_match = next((p for p in children if "theta" in p.name.lower()), None)
            if theta_match:
                return str(theta_match)
        if "sim" in name.lower() and "x" in name.lower():
            x_match = next((p for p in children if p.name.lower().startswith("sim_") and "x" in p.name.lower()), None)
            if x_match:
                return str(x_match)
    raise RuntimeError(f"Could not resolve file path: {path_value}")


def _wait_training_modal_idle(page: Page) -> None:
    loading = page.locator("#inferenceTrainingServerFileBrowserLoading")
    if loading.count() == 0:
        return
    try:
        loading.wait_for(state="hidden", timeout=20_000)
    except PlaywrightTimeoutError:
        pass


def _read_training_modal_current_path(page: Page) -> str:
    current = page.locator("#inferenceTrainingServerFileBrowserPath").inner_text(timeout=5_000).strip()
    return _normalize_abs_path(current or "/")


def _click_training_modal_dir(page: Page, segment: str, timeout_ms: int = 15_000) -> None:
    entries = page.locator("#inferenceTrainingServerFileBrowserEntries button")
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        _wait_training_modal_idle(page)
        count = entries.count()
        for idx in range(count):
            btn = entries.nth(idx)
            name = btn.inner_text().strip().rstrip("/")
            if name == segment:
                before = _read_training_modal_current_path(page)
                smooth_click(page, btn, after_ms=420)
                end_deadline = time.time() + 12.0
                while time.time() < end_deadline:
                    _wait_training_modal_idle(page)
                    after = _read_training_modal_current_path(page)
                    if after != before:
                        return
                    page.wait_for_timeout(140)
                return
        page.wait_for_timeout(180)
    raise RuntimeError(f"Could not find directory entry '{segment}' in training server file browser.")


def _select_training_modal_file(page: Page, target_name: str) -> None:
    """Select a file in the training modal. Try several robust Playwright queries and scrolling
    to handle cases where the modal is scrolled or elements are off-screen.
    """
    entries_container = page.locator("#inferenceTrainingServerFileBrowserEntries")
    # Reset scroll and give the UI a brief moment to render
    try:
        entries_container.evaluate("el => { el.scrollTop = 0; }")
    except PlaywrightError:
        pass
    page.wait_for_timeout(120)

    selected = None

    # 1) Prefer a direct locator that matches a label containing the target name in the font-mono span
    try:
        candidate = entries_container.locator(f"label:has(span.font-mono:has-text(\"{target_name}\"))").first
        if candidate.count() > 0:
            selected = candidate
    except PlaywrightError:
        # fall through to other strategies
        pass

    labels = page.locator("#inferenceTrainingServerFileBrowserEntries label")

    # 2) Iterate labels and compare exact text (scrolling each into view first)
    if selected is None:
        for idx in range(labels.count()):
            label = labels.nth(idx)
            try:
                label.scroll_into_view_if_needed()
            except PlaywrightError:
                pass
            try:
                name = label.locator("span.font-mono").first.inner_text().strip()
            except PlaywrightError:
                name = label.inner_text().strip()
            if name == target_name or name.lower() == target_name.lower():
                selected = label
                break

    # 3) Try a loose text search inside the entries container and get the ancestor label
    if selected is None:
        try:
            span = entries_container.get_by_text(target_name, exact=False)
            if span.count() > 0:
                # get nearest ancestor label
                ancestor = span.first.locator("xpath=ancestor::label")
                if ancestor.count() > 0:
                    selected = ancestor.first
        except PlaywrightError:
            pass

    # 4) Try scrolling the container from top to bottom and re-check labels
    if selected is None:
        try:
            entries_container.evaluate("el => { el.scrollTop = 0; }")
            page.wait_for_timeout(150)
            entries_container.evaluate("el => { el.scrollTop = el.scrollHeight; }")
            page.wait_for_timeout(150)
        except PlaywrightError:
            pass

        for idx in range(labels.count()):
            label = labels.nth(idx)
            try:
                label.scroll_into_view_if_needed()
            except PlaywrightError:
                pass
            try:
                name = label.locator("span.font-mono").first.inner_text().strip()
            except PlaywrightError:
                name = label.inner_text().strip()
            if name == target_name or name.lower() == target_name.lower() or target_name.lower() in name.lower():
                selected = label
                break

    if selected is None:
        raise RuntimeError(f"Could not find file '{target_name}' in training server browser.")

    smooth_click(page, selected, after_ms=300)
    smooth_click(page, page.locator("#inferenceTrainingServerFileBrowserSelect"), after_ms=620)


def select_training_file_via_modal(page: Page, field: str, file_path: str) -> None:
    card = page.locator(f".custom-file-card[data-field='{field}']").first
    page.evaluate(
        """
        (fieldName) => {
          const modeInput = document.getElementById(`${fieldName}_source_mode`);
          if (!modeInput) return;
          modeInput.value = 'server-path';
          modeInput.dispatchEvent(new Event('input', { bubbles: true }));
          modeInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
        """,
        field,
    )
    page.wait_for_timeout(220)
    smooth_click(page, card, after_ms=500)

    modal = page.locator("#inferenceTrainingServerFileBrowserModal")
    modal.wait_for(state="visible", timeout=15_000)
    _wait_training_modal_idle(page)

    resolved = Path(file_path)
    target_dir = _normalize_abs_path(str(resolved.parent))
    target_name = resolved.name

    current_path = _read_training_modal_current_path(page)
    current_segments = _split_abs_path(current_path)
    target_segments = _split_abs_path(target_dir)

    common = 0
    max_common = min(len(current_segments), len(target_segments))
    while common < max_common and current_segments[common] == target_segments[common]:
        common += 1

    up_btn = page.locator("#inferenceTrainingServerFileBrowserUp")
    for _ in range(len(current_segments) - common):
        smooth_click(page, up_btn, after_ms=320)
        _wait_training_modal_idle(page)

    for segment in target_segments[common:]:
        _click_training_modal_dir(page, segment, timeout_ms=18_000)

    _select_training_modal_file(page, target_name=target_name)
    modal.wait_for(state="hidden", timeout=15_000)
    page.wait_for_timeout(500)


def scroll_to_dataset_upload(page: Page) -> None:
    heading = page.locator("h2:has-text('Dataset Upload')").first
    move_to_locator(page, heading, click=False, pause_ms=120)
    page.evaluate(
        """
        () => {
          const h = Array.from(document.querySelectorAll('h2'))
            .find((el) => (el.textContent || '').trim().includes('Dataset Upload'));
          if (!h) return;
          h.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
          window.scrollBy({ top: 140, left: 0, behavior: 'smooth' });
        }
        """
    )
    page.wait_for_timeout(850)


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


def run_tutorial_recording(
    base_url: str,
    headless: bool,
    job_timeout_sec: int,
    ui_timeout_sec: int,
    features_path: str,
    parameters_path: str,
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
            resolved_features = _resolve_path_fuzzy(features_path)
            resolved_parameters = _resolve_path_fuzzy(parameters_path)

            smooth_click(page, page.locator("a[href='/inference']").first)
            page.wait_for_url("**/inference**")
            smooth_click(page, page.get_by_role("link", name="New training").first)
            page.wait_for_url("**/inference/new_training**")
            scroll_to_dataset_upload(page)

            print("[automation] selecting features file from server browser...", flush=True)
            select_training_file_via_modal(page, field="training_features_file", file_path=resolved_features)
            print("[automation] selecting parameters file from server browser...", flush=True)
            select_training_file_via_modal(page, field="training_parameters_file", file_path=resolved_parameters)

            smooth_select_option(page, page.locator("#training-model-name"), "Ridge", after_ms=600)

            print("[automation] submitting training...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Start Training"), after_ms=700)

            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for training job {job_id}...", flush=True)
            wait_for_job_finished(page, base_url, job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] training job {job_id} finished.", flush=True)
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
            features_path=args.features_path,
            parameters_path=args.parameters_path,
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
