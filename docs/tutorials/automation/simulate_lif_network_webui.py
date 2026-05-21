#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for a Hagen single-trial simulation."""

from __future__ import annotations

import argparse
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
OUTPUT_VIDEO = VIDEO_DIR / "simulate-lif-network-raster.webm"
OUTPUT_POSTER = VIDEO_DIR / "simulate-lif-network-raster-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_CAPTION_DURATION_MS = 5600
DEFAULT_CAPTION_TAIL_PAUSE_MS = 900
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: Hagen single-trial simulation (threads=32), "
            "then Analysis -> Load all -> raster plot."
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
        default=3600,
        help="Timeout waiting for simulation completion.",
    )
    parser.add_argument(
        "--ui-timeout-sec",
        type=int,
        default=120,
        help="Default UI action timeout.",
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
    parser.add_argument(
        "--caption-duration-ms",
        type=int,
        default=DEFAULT_CAPTION_DURATION_MS,
        help=f"Overlay caption visible duration in ms (default: {DEFAULT_CAPTION_DURATION_MS}).",
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
    proc = subprocess.Popen(cmd, cwd=webui_dir)
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


def show_overlay_caption(page: Page, text: str, duration_ms: int = 3000) -> None:
    page.evaluate(
        """
        async ({ message, duration }) => {
          if (!document.getElementById('pw-tutorial-overlay-style')) {
            const style = document.createElement('style');
            style.id = 'pw-tutorial-overlay-style';
            style.textContent = `
              #pw-tutorial-overlay {
                position: fixed;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: min(94vw, 1760px);
                min-height: min(72vh, 860px);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 42px 56px;
                border-radius: 18px;
                background: #ffffff;
                border: 2px solid #111827;
                color: #0f172a;
                font: 700 clamp(36px, 4.6vw, 74px)/1.25 "Space Grotesk", "Manrope", sans-serif;
                letter-spacing: 0;
                text-align: center;
                z-index: 2147483647;
                pointer-events: none;
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
                opacity: 0;
                transition: opacity 420ms ease, transform 420ms ease;
                text-shadow: none;
              }
            `;
            document.head.appendChild(style);
          }

          let node = document.getElementById('pw-tutorial-overlay');
          if (!node) {
            node = document.createElement('div');
            node.id = 'pw-tutorial-overlay';
            document.body.appendChild(node);
          }

          node.textContent = message;
          node.style.opacity = '0';
          node.style.transform = 'translate(-50%, calc(-50% - 12px))';
          await new Promise((resolve) => requestAnimationFrame(resolve));
          await new Promise((resolve) => requestAnimationFrame(resolve));
          node.style.opacity = '1';
          node.style.transform = 'translate(-50%, -50%)';
          await new Promise((resolve) => setTimeout(resolve, Math.max(2200, duration - 700)));
          node.style.opacity = '0';
          node.style.transform = 'translate(-50%, calc(-50% - 12px))';
          await new Promise((resolve) => setTimeout(resolve, 700));
        }
        """,
        {"message": text, "duration": duration_ms},
    )


def show_step_caption(
    page: Page,
    text: str,
    duration_ms: int,
    tail_pause_ms: int = DEFAULT_CAPTION_TAIL_PAUSE_MS,
) -> None:
    show_overlay_caption(page, text, duration_ms=duration_ms)
    if tail_pause_ms > 0:
        page.wait_for_timeout(tail_pause_ms)


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


def move_demo_cursor(page: Page, x: float, y: float, click: bool = False, travel_ms: int = 360) -> None:
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
        move_demo_cursor(page, _DEMO_CURSOR_POS["x"], _DEMO_CURSOR_POS["y"], click=click, travel_ms=220)
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
    move_demo_cursor(page, x, y, click=click, travel_ms=travel_ms)
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


def wait_for_simulation_job_finished(page: Page, base_url: str, job_id: str, timeout_sec: int) -> None:
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
            raise RuntimeError(f"Simulation job {job_id} ended with status '{status}'.")
        time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for simulation job {job_id} to finish. Last status: {last_status or 'unknown'}"
    )


def wait_for_analysis_load_all(page: Page, timeout_sec: int = 120) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    load_all = page.get_by_role("button", name="Load all")
    while time.time() < deadline:
        if load_all.count() > 0 and load_all.first.is_visible():
            return
        page.wait_for_timeout(1500)
        page.reload(wait_until="domcontentloaded")
    raise RuntimeError(
        "Analysis did not expose 'Load all' simulation files in time. "
        "Simulation outputs may still be unavailable."
    )


def wait_for_raster_plot_rendered(page: Page, timeout_sec: int) -> None:
    timeout_ms = max(1, int(timeout_sec)) * 1000
    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    plot_image = page.locator("#plot-image")
    plot_image.wait_for(state="attached", timeout=timeout_ms)
    try:
        plot_image.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        # Fallback for cases where classes do not toggle but image bytes are loaded.
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


def scroll_simulation_outputs_controls_into_view(page: Page) -> None:
    page.evaluate(
        """
        () => {
          const plotType = document.getElementById('sim-plot-type');
          if (!plotType) return;
          plotType.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        }
        """
    )
    page.wait_for_timeout(900)


def run_tutorial_recording(
    base_url: str,
    headless: bool,
    job_timeout_sec: int,
    ui_timeout_sec: int,
    viewport_width: int,
    viewport_height: int,
    slow_mo_ms: int,
    caption_duration_ms: int,
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

            smooth_click(page, page.locator("a[href='/simulation']").first)
            smooth_click(page, page.locator("a[href='/simulation/new_sim']").first)
            page.wait_for_url("**/simulation/new_sim**")
            smooth_click(page, page.locator("a[href='/simulation/new_sim/hagen']").first)
            page.wait_for_url("**/simulation/new_sim/hagen**")

            smooth_fill(page, page.locator("input[data-param='local_num_threads']"), "32")
            print("[automation] submitting simulation...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Run trial simulation"))
            page.wait_for_url("**/job_status/**", timeout=ui_timeout_sec * 1000)
            job_id = _extract_job_id_from_url(page.url)
            print(f"[automation] waiting for simulation job {job_id}...", flush=True)
            wait_for_simulation_job_finished(page, base_url, job_id, timeout_sec=job_timeout_sec)
            print(f"[automation] simulation job {job_id} finished.", flush=True)

            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_url("**/")
            smooth_click(page, page.locator("a[href='/analysis']").first)
            page.wait_for_url("**/analysis**")

            wait_for_analysis_load_all(page, timeout_sec=180)
            load_all = page.get_by_role("button", name="Load all")
            load_all.wait_for(state="visible")
            print("[automation] loading detected simulation files...", flush=True)
            smooth_click(page, load_all)

            smooth_click(page, page.get_by_role("button", name="Simulation outputs"))
            scroll_simulation_outputs_controls_into_view(page)
            smooth_select_option(page, page.locator("#sim-plot-type"), "raster")
            print("[automation] plotting raster...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Plot simulation outputs"))

            wait_for_raster_plot_rendered(page, timeout_sec=ui_timeout_sec)
            scroll_plot_into_view(page)
            page.wait_for_timeout(1500)

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
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            slow_mo_ms=args.slow_mo_ms,
            caption_duration_ms=args.caption_duration_ms,
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
