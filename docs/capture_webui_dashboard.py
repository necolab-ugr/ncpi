#!/usr/bin/env python3
"""Capture the WebUI dashboard for the documentation image gallery."""

from __future__ import annotations

import argparse
import base64
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
OUTPUT_SVG = DOCS_DIR / "img" / "webui.svg"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the ncpi WebUI dashboard, hide the shared header/footer, "
            "and save a documentation-ready SVG screenshot to docs/img/webui.svg."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="WebUI base URL.")
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start webui/app.py automatically when no server is reachable.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Chromium headless.",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=DEFAULT_VIEWPORT_WIDTH,
        help=f"Capture viewport width (default: {DEFAULT_VIEWPORT_WIDTH}).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=DEFAULT_VIEWPORT_HEIGHT,
        help=f"Capture viewport height (default: {DEFAULT_VIEWPORT_HEIGHT}).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=90,
        help="Timeout while waiting for the WebUI server.",
    )
    return parser.parse_args()


def is_server_reachable(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def wait_for_server(url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if is_server_reachable(url):
            return
        time.sleep(0.4)
    raise TimeoutError(f"Server did not become reachable at {url} within {timeout_sec}s.")


def start_webui_server(base_url: str, timeout_sec: int) -> Optional[subprocess.Popen]:
    if is_server_reachable(base_url):
        return None

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
    wait_for_server(base_url, timeout_sec=timeout_sec)
    return proc


def stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def png_bytes_to_svg(png_bytes: bytes, width: int, height: int) -> str:
    image_data = base64.b64encode(png_bytes).decode("ascii")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="ncpi WebUI dashboard">\n'
        f'  <image href="data:image/png;base64,{image_data}" width="{width}" '
        f'height="{height}" />\n'
        "</svg>\n"
    )


def capture_dashboard_svg(
    base_url: str,
    *,
    headless: bool,
    viewport_width: int,
    viewport_height: int,
) -> None:
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            screen={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(base_url, wait_until="networkidle")
        page.add_style_tag(
            content="""
                header,
                footer {
                    display: none !important;
                }
                main {
                    padding-top: 2.5rem !important;
                    padding-bottom: 2.5rem !important;
                }
                main p.max-w-2xl {
                    max-width: 72rem !important;
                    font-size: 2.25rem !important;
                    line-height: 1.3 !important;
                }
                main a[href] div:last-child h2 {
                    font-size: 1.65rem !important;
                    line-height: 1.25 !important;
                }
                main a[href] div:last-child p {
                    font-size: 1.18rem !important;
                    line-height: 1.45 !important;
                }
                .pipeline-node + div span[class*="text-\\[13px\\]"],
                .pipeline-node + div span[class*="text-\\[10px\\]"],
                .pipeline-node + div div[class*="text-\\[10px\\]"],
                .pipeline-node + div button[class*="text-\\[10px\\]"] {
                    font-size: 1.25rem !important;
                    line-height: 1.3 !important;
                }
            """
        )
        page.locator("main").wait_for(state="visible", timeout=30_000)
        page.locator("p", has_text="Select a module to begin").first.evaluate(
            """(el) => {
                el.style.setProperty("max-width", "72rem", "important");
                el.style.setProperty("font-size", "2.25rem", "important");
                el.style.setProperty("line-height", "1.3", "important");
            }"""
        )
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(500)

        main = page.locator("main").first
        box = main.bounding_box()
        if not box:
            raise RuntimeError("Could not determine dashboard content bounds.")

        png_bytes = main.screenshot(type="png", timeout=30_000)
        width = max(1, round(box["width"]))
        height = max(1, round(box["height"]))
        OUTPUT_SVG.write_text(png_bytes_to_svg(png_bytes, width, height), encoding="utf-8")

        context.close()
        browser.close()


def main() -> None:
    args = parse_args()
    server_process: Optional[subprocess.Popen] = None
    try:
        if args.start_server:
            server_process = start_webui_server(args.base_url, timeout_sec=args.timeout_sec)
        else:
            wait_for_server(args.base_url, timeout_sec=args.timeout_sec)

        capture_dashboard_svg(
            args.base_url,
            headless=args.headless,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
        )
        print(f"Saved {OUTPUT_SVG}")
    finally:
        stop_process(server_process)


if __name__ == "__main__":
    main()
