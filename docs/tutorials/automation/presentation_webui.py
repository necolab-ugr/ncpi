#!/usr/bin/env python3
"""Record the ncpi WebUI overview used in the repository README."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

import simulate_lif_network_webui as simulation_flow


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = SCRIPT_DIR / "videos"
OUTPUT_VIDEO = VIDEO_DIR / "ncpi-webui-overview.webm"
OUTPUT_POSTER = VIDEO_DIR / "ncpi-webui-overview-poster.png"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_UI_TIMEOUT_MS = 30_000
DEFAULT_JOB_TIMEOUT_SEC = 600.0


def wait_for_job_finished(
    page: Page,
    base_url: str,
    job_id: str,
    job_name: str,
    timeout_sec: float,
) -> None:
    """Poll a WebUI background job until it finishes or fails."""
    deadline = time.monotonic() + timeout_sec
    status_url = f"{base_url.rstrip('/')}/status/{job_id}"
    last_status = "unknown"

    while time.monotonic() < deadline:
        response = page.request.get(status_url)
        if not response.ok:
            page.wait_for_timeout(1_000)
            continue
        payload = response.json() or {}
        status = str(payload.get("status") or "").strip().lower()
        if status:
            last_status = status

        if status == "finished":
            return
        if status in {"error", "failed", "cancelled"}:
            error = payload.get("error") or status
            raise RuntimeError(f"{job_name} failed: {error}")

        page.wait_for_timeout(1_000)

    raise TimeoutError(
        f"{job_name} did not finish within {timeout_sec:.0f} seconds; "
        f"last status: {last_status}"
    )


def return_to_dashboard(page: Page, base_url: str) -> None:
    """Return to the dashboard while preserving the active WebUI session."""
    dashboard_link = page.locator("a[href='/']").first
    try:
        simulation_flow.smooth_click(page, dashboard_link)
        page.wait_for_url(f"{base_url.rstrip('/')}/", wait_until="domcontentloaded")
    except PlaywrightTimeoutError:
        page.goto(base_url, wait_until="domcontentloaded")


def slow_scroll_down_to_locator(
    page: Page,
    locator: Locator,
    step_px: int = 180,
    pause_ms: int = 260,
) -> None:
    """Scroll down incrementally until the target control is visible."""
    locator.wait_for(state="attached")
    while True:
        position = locator.evaluate(
            """
            (el) => {
              const rect = el.getBoundingClientRect();
              return { top: rect.top, bottom: rect.bottom, viewportHeight: window.innerHeight };
            }
            """
        )
        if position["top"] >= 80 and position["bottom"] <= position["viewportHeight"] - 80:
            break
        previous_y = int(page.evaluate("() => window.scrollY") or 0)
        page.mouse.wheel(0, max(80, int(step_px)))
        page.wait_for_timeout(max(120, int(pause_ms)))
        current_y = int(page.evaluate("() => window.scrollY") or 0)
        if current_y <= previous_y:
            break
    simulation_flow.smooth_scroll_to_locator(page, locator, wait_ms=900)


def record_presentation(
    base_url: str,
    *,
    headless: bool,
    start_server: bool,
    job_timeout_sec: float,
    ui_timeout_ms: int,
    viewport_width: int,
    viewport_height: int,
    slow_mo_ms: int,
) -> None:
    """Record simulation, proxy computation, and analysis in one WebUI video."""
    server_process: Optional[subprocess.Popen] = None

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    temporary_video_dir = VIDEO_DIR / ".tmp-presentation"
    if temporary_video_dir.exists():
        shutil.rmtree(temporary_video_dir)
    temporary_video_dir.mkdir(parents=True)

    try:
        if start_server:
            server_process = simulation_flow.start_webui_server(base_url)
        simulation_flow.wait_for_server(base_url)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
            context = browser.new_context(
                viewport={"width": viewport_width, "height": viewport_height},
                screen={"width": viewport_width, "height": viewport_height},
                record_video_dir=str(temporary_video_dir),
                record_video_size={"width": viewport_width, "height": viewport_height},
                device_scale_factor=1,
            )
            page = context.new_page()
            video = page.video
            page.set_default_timeout(ui_timeout_ms)
            simulation_flow.reset_demo_cursor_position()

            # Keep the opening sequence identical to simulate_lif_network_webui.py.
            page.goto(base_url, wait_until="domcontentloaded")
            simulation_flow.smooth_click(page, page.locator("a[href='/simulation']").first)
            simulation_flow.smooth_click(page, page.locator("a[href='/simulation/new_sim']").first)
            page.wait_for_url("**/simulation/new_sim**")
            simulation_flow.smooth_click(page, page.locator("a[href='/simulation/new_sim/hagen']").first)
            page.wait_for_url("**/simulation/new_sim/hagen**")
            run_trial_button = page.get_by_role("button", name="Run trial simulation")
            slow_scroll_down_to_locator(page, run_trial_button)
            simulation_flow.smooth_click(
                page,
                run_trial_button,
            )
            page.wait_for_url("**/job_status/**")
            simulation_job_id = simulation_flow._extract_job_id_from_url(page.url)
            wait_for_job_finished(
                page,
                base_url,
                simulation_job_id,
                "Simulation",
                job_timeout_sec,
            )

            return_to_dashboard(page, base_url)
            simulation_flow.smooth_click(page, page.locator("a[href='/field_potential']").first)
            page.wait_for_url("**/field_potential")
            simulation_flow.smooth_click(
                page,
                page.locator("a[href='/field_potential/proxy']").first,
            )
            page.wait_for_url("**/field_potential/proxy")
            compute_proxy = page.get_by_role("button", name="Compute proxy")
            simulation_flow.smooth_scroll_to_locator(page, compute_proxy)
            simulation_flow.smooth_click(page, compute_proxy)
            page.wait_for_url("**/job_status/**")
            proxy_job_id = simulation_flow._extract_job_id_from_url(page.url)
            wait_for_job_finished(page, base_url, proxy_job_id, "Proxy computation", job_timeout_sec)

            return_to_dashboard(page, base_url)
            simulation_flow.smooth_click(page, page.locator("a[href='/analysis']").first)
            page.wait_for_url("**/analysis")
            simulation_flow.wait_for_analysis_load_all(page)
            simulation_flow.smooth_click(page, page.get_by_role("button", name="Load all"))
            simulation_flow.smooth_click(
                page,
                page.get_by_role("button", name="Simulation outputs"),
            )
            simulation_flow.scroll_simulation_outputs_controls_into_view(page)

            plot_type = page.locator("#sim-plot-type")
            simulation_flow.smooth_select_option(page, plot_type, "raster")
            simulation_flow.smooth_click(
                page,
                page.get_by_role("button", name="Plot simulation outputs"),
            )
            simulation_flow.wait_for_raster_plot_rendered(
                page,
                timeout_sec=max(1, ui_timeout_ms // 1_000),
            )
            simulation_flow.scroll_plot_into_view(page)
            page.wait_for_timeout(2_500)

            simulation_flow.smooth_click(page, page.get_by_role("link", name="Back to analysis"))
            page.wait_for_url("**/analysis")
            simulation_flow.smooth_click(
                page,
                page.get_by_role("button", name="Simulation outputs"),
            )
            simulation_flow.scroll_simulation_outputs_controls_into_view(page)
            simulation_flow.smooth_select_option(page, plot_type, "proxy")
            simulation_flow.smooth_click(
                page,
                page.get_by_role("button", name="Plot simulation outputs"),
            )
            simulation_flow.wait_for_raster_plot_rendered(
                page,
                timeout_sec=max(1, ui_timeout_ms // 1_000),
            )
            simulation_flow.scroll_plot_into_view(page)
            page.wait_for_timeout(3_000)
            page.screenshot(path=str(OUTPUT_POSTER), full_page=False)

            context.close()
            if video is None:
                browser.close()
                raise RuntimeError("Playwright did not attach a video stream")
            if OUTPUT_VIDEO.exists():
                OUTPUT_VIDEO.unlink()
            video.save_as(str(OUTPUT_VIDEO))
            browser.close()

        if not OUTPUT_VIDEO.exists():
            raise RuntimeError("Playwright did not produce a video")
    finally:
        simulation_flow.stop_process(server_process)
        shutil.rmtree(temporary_video_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Use an already running WebUI server",
    )
    parser.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT_SEC)
    parser.add_argument("--ui-timeout", type=int, default=DEFAULT_UI_TIMEOUT_MS)
    parser.add_argument("--viewport-width", type=int, default=1600)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--slow-mo", type=int, default=75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record_presentation(
            args.base_url,
            headless=not args.headed,
            start_server=not args.no_start_server,
            job_timeout_sec=args.job_timeout,
            ui_timeout_ms=args.ui_timeout,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            slow_mo_ms=args.slow_mo,
        )
    except Exception as exc:
        print(f"Presentation recording failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved video to {OUTPUT_VIDEO}")
    print(f"Saved poster to {OUTPUT_POSTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
