#!/usr/bin/env python3
"""Record the ncpi WebUI tutorial flow for plotting empirical predictions in Analysis."""

from __future__ import annotations

import argparse
import os
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
OUTPUT_VIDEO = VIDEO_DIR / "plot-empirical-results.webm"
OUTPUT_POSTER = VIDEO_DIR / "plot-empirical-results-poster.png"
DEFAULT_VIEWPORT_WIDTH = 1600
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_SLOW_MO_MS = 300
DEFAULT_SESSION_LABEL = "webui docs"
DEFAULT_SESSION_ROOT = "/tmp/ncpi_webui_session_e6152202b3294223b22bb4fc0bc1682b"
DEFAULT_GROUP_BY = "group"
DEFAULT_Y_VARIABLE = "Predictions"
DEFAULT_CONTROL_GROUP = "2.0"
_DEMO_CURSOR_POS = {"x": 24.0, "y": 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate and record: load latest eligible session, go to Analysis, load "
            "predictions.pkl, configure Boxplot (group vs Predictions), enable effect size "
            "vs control (2.0), plot, and scroll to the bottom."
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
        "--boxplot-group-by",
        default=DEFAULT_GROUP_BY,
        help="Boxplot X-axis grouping factor.",
    )
    parser.add_argument(
        "--boxplot-y-variable",
        default=DEFAULT_Y_VARIABLE,
        help="Boxplot Y-axis variable.",
    )
    parser.add_argument(
        "--control-group",
        default=DEFAULT_CONTROL_GROUP,
        help="Control group value used when effect size vs control is enabled.",
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


def _session_has_features_artifacts(session_root: str) -> bool:
    base = Path(session_root)
    return _has_any_file(base / "features" / "data")


def _session_has_inference_artifacts(session_root: str) -> bool:
    base = Path(session_root)
    return _has_any_file(base / "inference" / "predictions", suffixes={".pkl", ".pickle"}) or _has_any_file(
        base / "inference" / "data",
        suffixes={".pkl", ".pickle"},
    )


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
          const values = cards.map((card, idx) => {
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

          if (!values.length) return [];
          values.sort((a, b) => {
            if (a.stamp != null && b.stamp != null) return b.stamp - a.stamp;
            if (a.stamp != null) return -1;
            if (b.stamp != null) return 1;
            if (a.updated && b.updated) return b.updated.localeCompare(a.updated);
            return a.index - b.index;
          });
          return values;
        }
        """
    )
    if not entries:
        raise RuntimeError("No saved sessions found in /sessions page.")

    eligible = []
    for entry in entries:
        if bool(entry.get("isActive")):
            continue
        modules = {str(item).strip().lower() for item in (entry.get("modules") or [])}
        has_features_module = "features" in modules
        has_inference_module = ("inference" in modules) or ("predictions" in modules)
        if not (has_features_module and has_inference_module):
            continue
        path_value = str(entry.get("path") or "")
        if not _session_has_features_artifacts(path_value):
            continue
        if not _session_has_inference_artifacts(path_value):
            continue
        eligible.append(entry)
    if not eligible:
        raise RuntimeError(
            "No previous session contains both features and inference artifacts "
            "(expected files under features/data and inference/predictions or inference/data)."
        )
    target = eligible[0]
    target_path = str(target["path"])
    print(
        f"[automation] loading most recent session with features+inference artifacts "
        f"(label={session_label!r}, requested={session_root!r}) -> "
        f"{target_path} (updated={target.get('updated', '')}, modules={target.get('modules', [])})",
        flush=True,
    )
    target_form = page.locator(f"form:has(input[name='session_root'][value='{target_path}'])").first
    if target_form.count() == 0:
        raise RuntimeError(f"Target saved session form not found in UI: {target_path}")
    target_card = page.locator(f"article:has(input[name='session_root'][value='{target_path}'])").first
    if target_card.count() > 0:
        target_card.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
        page.wait_for_timeout(900)
        move_to_locator(page, target_card, click=False, pause_ms=220)
    smooth_click(page, target_form.get_by_role("button"))
    # Wait for page to reload after session load redirect
    page.wait_for_load_state("domcontentloaded", timeout=90_000)
    page.wait_for_timeout(1_000)


def smooth_select_option(page: Page, locator: Locator, value: str, after_ms: int = 450) -> None:
    global _DEMO_CURSOR_POS
    wanted = str(value)
    move_to_locator(page, locator, click=False, pause_ms=180)
    move_to_locator(page, locator, click=True, pause_ms=80)

    state = locator.evaluate(
        """
        (el) => {
          const options = Array.from(el?.options || []);
          return {
            values: options.map((opt) => String(opt.value || '')),
            labels: options.map((opt) => String((opt.textContent || '').trim())),
          };
        }
        """
    )
    values = [str(item) for item in (state.get("values") or [])]
    labels = [str(item) for item in (state.get("labels") or [])]

    target_index = -1
    for idx, opt_value in enumerate(values):
        if opt_value == wanted:
            target_index = idx
            break
    if target_index < 0:
        # Fallback to label match
        token = wanted.strip().lower()
        for idx, opt_label in enumerate(labels):
            if token and token in opt_label.lower():
                target_index = idx
                wanted = values[idx]
                break

    if target_index < 0:
        raise RuntimeError(
            f"Dropdown option {value!r} not found. "
            f"Available values: {values}, labels: {labels}"
        )

    # Use overlay to simulate dropdown menu and allow demo cursor movement
    overlay_target = locator.evaluate(
        """
        (anchor, { labels, targetIndex }) => {
          if (!anchor) return null;

          const old = document.getElementById('pw-demo-select-menu');
          if (old) old.remove();

          const rect = anchor.getBoundingClientRect();
          const optionHeight = 28;
          const count = labels.length;
          const visibleRows = Math.max(4, Math.min(10, count));
          const menuHeight = visibleRows * optionHeight;
          const spaceBelow = window.innerHeight - (rect.bottom + 8);
          const spaceAbove = rect.top - 8;
          const opensUpward = spaceBelow < menuHeight && spaceAbove > spaceBelow;

          const menu = document.createElement('div');
          menu.id = 'pw-demo-select-menu';
          menu.style.position = 'fixed';
          menu.style.left = `${rect.left}px`;
          menu.style.width = `${Math.max(180, rect.width)}px`;
          menu.style.maxHeight = `${menuHeight}px`;
          menu.style.overflow = 'hidden';
          menu.style.border = '1px solid rgba(100,116,139,0.65)';
          menu.style.borderRadius = '8px';
          menu.style.background = '#ffffff';
          menu.style.boxShadow = '0 10px 25px rgba(0,0,0,0.2)';
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
            row.textContent = labels[i];
            row.style.height = `${optionHeight}px`;
            row.style.lineHeight = `${optionHeight}px`;
            row.style.padding = '0 10px';
            row.style.fontSize = '13px';
            row.style.color = '#111827';
            row.style.background = '#ffffff';
            menu.appendChild(row);
            rows.push(row);
          }

          const menuTop = opensUpward
            ? rect.top - menuHeight - 4
            : rect.bottom + 4;
          menu.style.top = `${menuTop}px`;
          document.body.appendChild(menu);

          const onMouseMove = (evt) => {
            const y = evt.clientY;
            const x = evt.clientX;
            const menuRect = menu.getBoundingClientRect();
            if (x >= menuRect.left && x <= menuRect.right && y >= menuRect.top && y <= menuRect.bottom) {
              const idx = Math.floor((y - menuRect.top) / optionHeight);
              for (let i = 0; i < rows.length; i++) {
                if (i === idx) {
                  rows[i].style.background = '#3b82f6';
                  rows[i].style.color = '#ffffff';
                } else {
                  rows[i].style.background = '#ffffff';
                  rows[i].style.color = '#111827';
                }
              }
            }
          };
          document.addEventListener('mousemove', onMouseMove, { passive: true });
          window.__pwDemoSelectMenuCleanup = () => {
            document.removeEventListener('mousemove', onMouseMove);
          };

          const activeLocal = targetIndex - start;
          const targetY = menuTop + ((activeLocal + 0.5) * optionHeight);
          const targetX = rect.left + (Math.max(180, rect.width) / 2.0);
          return { x: targetX, y: targetY };
        }
        """,
        {"labels": labels, "targetIndex": target_index},
    )

    if overlay_target:
        tx = float(overlay_target["x"])
        ty = float(overlay_target["y"])

        # Move cursor to the option in the overlay
        dx = tx - _DEMO_CURSOR_POS["x"]
        dy = ty - _DEMO_CURSOR_POS["y"]
        dist = (dx * dx + dy * dy) ** 0.5
        steps = max(24, min(80, int(dist / 10)))
        page.mouse.move(tx, ty, steps=steps)
        _DEMO_CURSOR_POS = {"x": tx, "y": ty}

        page.wait_for_timeout(150)
        move_demo_cursor(page, click=True)
        page.wait_for_timeout(100)

        page.evaluate(
            """
            () => {
              if (typeof window.__pwDemoSelectMenuCleanup === 'function') {
                window.__pwDemoSelectMenuCleanup();
                window.__pwDemoSelectMenuCleanup = null;
              }
              const node = document.getElementById('pw-demo-select-menu');
              if (node) node.remove();
            }
            """
        )

    locator.select_option(value=wanted)
    page.wait_for_timeout(after_ms)


def _wait_for_control_group_ready(page: Page, desired_value: str, timeout_sec: int = 60) -> None:
    select = page.locator("#boxplot-control-group").first
    deadline = time.time() + max(1, int(timeout_sec))
    token = str(desired_value).strip().lower()
    while time.time() < deadline:
        if select.count() > 0 and select.is_visible() and select.is_enabled():
            state = select.evaluate(
                """
                (el) => Array.from(el?.options || []).map((opt) => ({
                  value: String(opt.value || ''),
                  label: String((opt.textContent || '').trim()),
                }))
                """
            )
            for item in (state or []):
                value = str(item.get("value") or "").strip().lower()
                label = str(item.get("label") or "").strip().lower()
                if token and (token == value or token == label or token in label):
                    return
        page.wait_for_timeout(220)
    raise RuntimeError(f"Timed out waiting for control-group option: {desired_value}")


def load_predictions_from_detected_entries(page: Page) -> None:
    print("[automation] loading predictions.pkl in Analysis...", flush=True)
    container = page.locator("div:has(p:has-text('Detected data files from previous modules'))").first
    container.wait_for(state="visible", timeout=30_000)

    deadline = time.time() + 60
    best = None
    fallback = None
    while time.time() < deadline:
        buttons = container.locator("button")
        count = buttons.count()
        for idx in range(count):
            btn = buttons.nth(idx)
            if not btn.is_visible():
                continue
            text = btn.inner_text().strip()
            low = text.lower()
            if "predictions" in low and "predictions.pkl" in low:
                best = btn
                break
            if "predictions" in low and fallback is None:
                fallback = btn
        if best is not None:
            break
        if fallback is not None:
            break
        page.wait_for_timeout(240)

    target = best or fallback
    if target is None:
        raise RuntimeError("Could not find a detected Predictions dataframe entry in Analysis.")
    smooth_click(page, target, after_ms=800)

    # Wait until dataframe mode is active and columns are available.
    columns_panel = page.locator("h3:has-text('Detected columns')").first
    columns_panel.wait_for(state="visible", timeout=60_000)
    page.wait_for_timeout(700)


def slow_scroll_to_bottom(page: Page, step_px: int = 220, pause_ms: int = 260) -> None:
    page.wait_for_timeout(500)
    # Use a small loop to re-evaluate height in case it's lazy-loading
    for _ in range(2):
        total_height = int(
            page.evaluate(
                """
                () => Math.max(
                  document.body ? document.body.scrollHeight : 0,
                  document.documentElement ? document.documentElement.scrollHeight : 0
                )
                """
            )
            or 0
        )
        viewport_h = int(page.evaluate("() => window.innerHeight") or 900)
        max_scroll = max(0, total_height - viewport_h)
        pos = int(page.evaluate("() => window.scrollY") or 0)

        if max_scroll <= pos:
            break

        while pos < max_scroll:
            pos = min(max_scroll, pos + max(80, int(step_px)))
            page.evaluate("(y) => window.scrollTo(0, y)", pos)
            page.wait_for_timeout(max(120, int(pause_ms)))

            # Re-check height in case it grew
            new_total = int(page.evaluate("() => document.documentElement.scrollHeight") or 0)
            if new_total > total_height:
                total_height = new_total
                max_scroll = max(0, total_height - viewport_h)

    page.wait_for_timeout(700)


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
    ui_timeout_sec: int,
    session_label: str,
    session_root: str,
    boxplot_group_by: str,
    boxplot_y_variable: str,
    control_group: str,
    viewport_width: int,
    viewport_height: int,
    slow_mo_ms: int,
) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, slow_mo=max(0, int(slow_mo_ms)))
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": viewport_width, "height": viewport_height},
        )
        context.set_default_timeout(max(1, int(ui_timeout_sec)) * 1000)
        page = context.new_page()
        video = page.video
        try:
            print("[automation] opening dashboard...", flush=True)
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_timeout(900)
            reset_demo_cursor_position()
            ensure_demo_cursor(page)

            load_requested_session(page, session_root=session_root, session_label=session_label)

            print("[automation] returning to dashboard...", flush=True)
            smooth_click(page, page.locator("a[href='/']").first, after_ms=700)
            page.wait_for_load_state("domcontentloaded", timeout=90_000)
            page.wait_for_timeout(900)

            print("[automation] opening Analysis...", flush=True)
            smooth_click(page, page.locator("a[href='/analysis']").first, after_ms=700)
            page.wait_for_url("**/analysis**")
            page.wait_for_timeout(900)

            load_predictions_from_detected_entries(page)

            print("[automation] opening Boxplot tab...", flush=True)
            smooth_click(page, page.get_by_role("button", name="Boxplot").first, after_ms=620)

            print(
                f"[automation] selecting boxplot axes: x={boxplot_group_by!r}, y={boxplot_y_variable!r}...",
                flush=True,
            )
            group_select = page.locator("#boxplot-group-by").first
            value_select = page.locator("#boxplot-value-col").first
            group_select.wait_for(state="visible", timeout=60_000)
            value_select.wait_for(state="visible", timeout=60_000)
            smooth_select_option(
                page,
                group_select,
                value=boxplot_group_by,
                after_ms=280,
            )
            smooth_select_option(
                page,
                value_select,
                value=boxplot_y_variable,
                after_ms=280,
            )

            print("[automation] enabling effect-size vs control...", flush=True)
            smooth_check(page, page.locator("#boxplot-show-cohend").first, after_ms=350)
            _wait_for_control_group_ready(page, control_group, timeout_sec=90)
            smooth_select_option(
                page,
                page.locator("#boxplot-control-group").first,
                value=control_group,
                after_ms=300,
            )

            print("[automation] plotting boxplot...", flush=True)
            plot_btn = page.locator("button[data-plot-action='boxplot']").first
            # Scroll down a little before clicking
            page.evaluate("window.scrollBy({ top: 300, behavior: 'smooth' })")
            page.wait_for_timeout(800)
            smooth_click(page, plot_btn, after_ms=650)
            
            # Wait for either the URL to change or the plot to become visible
            # Some implementations might not change the URL if it's an AJAX plot.
            try:
                page.wait_for_url("**/analysis/plot/boxplot**", timeout=5000)
            except PlaywrightTimeoutError:
                print("[automation] URL did not change to boxplot, checking for plot image...", flush=True)

            plot_img = page.locator("#plot-image").first
            plot_img.wait_for(state="visible", timeout=180_000)
            plot_img.scroll_into_view_if_needed()
            page.wait_for_timeout(450)
            page.evaluate("window.scrollBy({ top: 180, behavior: 'smooth' })")
            page.wait_for_timeout(900)

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


def main() -> None:
    args = parse_args()
    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.start_server:
            print("[automation] starting webui/app.py...", flush=True)
            server_proc = start_webui_server(args.base_url)
            print("[automation] server started.", flush=True)
        else:
            wait_for_server(args.base_url, timeout_sec=30)

        print("[automation] running Playwright flow...", flush=True)
        run_tutorial_recording(
            base_url=args.base_url,
            headless=args.headless,
            ui_timeout_sec=args.ui_timeout_sec,
            session_label=args.session_label,
            session_root=args.session_root,
            boxplot_group_by=args.boxplot_group_by,
            boxplot_y_variable=args.boxplot_y_variable,
            control_group=args.control_group,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            slow_mo_ms=args.slow_mo_ms,
        )
        print(f"[automation] done. video={OUTPUT_VIDEO} poster={OUTPUT_POSTER}", flush=True)
    finally:
        stop_process(server_proc)


if __name__ == "__main__":
    main()
