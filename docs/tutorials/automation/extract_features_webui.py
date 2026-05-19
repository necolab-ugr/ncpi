#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for feature extraction in the simulation pipeline."""

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
OUTPUT_VIDEO = VIDEO_DIR / "extract-features.webm"
OUTPUT_POSTER = VIDEO_DIR / "extract-features-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_SESSION_LABEL = "webui docs"
DEFAULT_SESSION_ROOT = "/tmp/ncpi_webui_session_2c3af880cfe741ceb12bc3a66e9750bf"
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: load a previous session, continue simulation pipeline in Features, "
            "select cdm.pkl, configure parser (CDM + simulated + epoching), choose catch22, and compute features."
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
        help="Timeout waiting for features computation completion.",
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


def smooth_check(page: Page, locator: Locator, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    if not locator.is_checked():
        locator.check()
    page.wait_for_timeout(after_ms)


def smooth_select_option(page: Page, locator: Locator, value: str, after_ms: int = 450) -> None:
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)
    locator.select_option(value)
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


def load_requested_session(page: Page, session_root: str, session_label: str) -> None:
    smooth_click(page, page.get_by_role("link", name="Previous Sessions"))
    page.wait_for_url("**/sessions**")
    print(f"[automation] loading saved session '{session_label}' -> {session_root}", flush=True)
    target_form = page.locator(f"form:has(input[name='session_root'][value='{session_root}'])").first
    if target_form.count() == 0:
        raise RuntimeError(f"Saved session not found in UI: {session_root}")
    smooth_click(page, target_form.get_by_role("button"))
    page.wait_for_url("**/sessions**")
    page.wait_for_timeout(1200)


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


def run_tutorial_recording(
    base_url: str,
    headless: bool,
    job_timeout_sec: int,
    ui_timeout_sec: int,
    session_label: str,
    session_root: str,
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

            smooth_click(page, page.locator("a[href='/features']").first)
            page.wait_for_url("**/features")

            compute_new_features = page.get_by_role("button", name="Compute new features")
            if compute_new_features.count() == 0:
                compute_new_features = page.get_by_role("link", name="Compute new features")
            if compute_new_features.count() == 0:
                compute_new_features = page.locator("a:has-text('Compute new features')").first
            if compute_new_features.count() == 0:
                raise RuntimeError("Could not find 'Compute new features' entry in Features page.")
            smooth_click(page, compute_new_features.first)
            page.wait_for_url("**/features/methods**")

            smooth_click(page, page.get_by_role("button", name="Continue simulation pipeline"))
            pipeline_section = page.locator("div:has(h3:has-text('Continue simulation pipeline'))").first
            pipeline_section.wait_for(state="visible", timeout=120_000)
            cdm_entry = pipeline_section.locator("button", has_text="cdm.pkl").first
            if cdm_entry.count() == 0:
                cdm_entry = page.get_by_role("button", name="cdm.pkl").first
            if cdm_entry.count() == 0:
                # Fallback: pick any pipeline button whose label contains cdm.pkl (case-insensitive).
                cdm_entry = page.locator("button:has-text('cdm.pkl'), button:has-text('CDM.PKL')").first
            if cdm_entry.count() == 0:
                raise RuntimeError("Could not find 'cdm.pkl' in detected simulation pipeline files.")
            smooth_click(page, cdm_entry)

            parser_source = page.locator("select[name='parser_recording_type_source']:visible").first
            parser_source.wait_for(state="visible", timeout=120_000)
            smooth_select_option(page, parser_source, "__value__")
            smooth_select_option(page, page.locator("select[name='parser_recording_type']:visible").first, "CDM")

            simulated_radio = page.locator("input[name='parser_metadata_mode'][value='simulated']:visible").first
            smooth_check(page, simulated_radio)

            epoching_checkbox = page.locator("label:has-text('Enable epoching') input[type='checkbox']:visible").first
            smooth_check(page, epoching_checkbox)

            smooth_click(page, page.get_by_role("button", name="Next: Select method"))
            page.locator("h2:has-text('Select Method')").first.wait_for(state="visible")

            smooth_click(page, page.locator("label:has-text('Catch22')").first)
            smooth_click(page, page.get_by_role("button", name="Next step"))
            page.locator("h1:has-text('Method Configuration')").first.wait_for(state="visible")

            compute_button = page.get_by_role("button", name="Compute features").first
            wait_for_enabled(page, compute_button, timeout_sec=60)
            smooth_click(page, compute_button, after_ms=600)

            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for features job {job_id}...", flush=True)
            wait_for_job_finished(page, base_url, job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] features job {job_id} finished.", flush=True)
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
