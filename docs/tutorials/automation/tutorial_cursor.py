"""Shared cursor, scroll, and click timing for WebUI tutorial recordings."""

from __future__ import annotations

from typing import Callable

from playwright._impl._errors import Error as PlaywrightError
from playwright.sync_api import Locator, Page


CURSOR_START_X = 24.0
CURSOR_START_Y = 24.0
CURSOR_SPEED_PX_PER_MS = 1.15
CURSOR_MIN_TRAVEL_MS = 320
CURSOR_MAX_TRAVEL_MS = 1050
PRE_CLICK_WAIT_MS = 1000
POST_ACTION_WAIT_MS = 220
SCROLL_WAIT_MS = 620
CLICK_FLASH_MS = 140
_DEMO_CURSOR_POS = {"x": CURSOR_START_X, "y": CURSOR_START_Y}


def reset_demo_cursor_position(x: float = CURSOR_START_X, y: float = CURSOR_START_Y) -> None:
    global _DEMO_CURSOR_POS
    _DEMO_CURSOR_POS = {"x": float(x), "y": float(y)}


def ensure_demo_cursor(page: Page, start_x: float = CURSOR_START_X, start_y: float = CURSOR_START_Y) -> None:
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
            (flashMs) => {
              const node = document.getElementById('pw-demo-cursor');
              if (!node) return;
              node.classList.remove('clicking');
              void node.offsetWidth;
              node.classList.add('clicking');
              setTimeout(() => node.classList.remove('clicking'), flashMs);
            }
            """,
            CLICK_FLASH_MS,
        )
    except PlaywrightError as exc:
        if "Execution context was destroyed" in str(exc):
            return
        raise


def _cursor_travel_ms(distance: float) -> int:
    return max(CURSOR_MIN_TRAVEL_MS, min(CURSOR_MAX_TRAVEL_MS, int(distance / CURSOR_SPEED_PX_PER_MS)))


def smooth_scroll_locator_into_view(page: Page, locator: Locator, wait_ms: int = SCROLL_WAIT_MS) -> None:
    locator.wait_for(state="visible")
    handle = locator.element_handle(timeout=5_000)
    if handle is None:
        return
    handle.evaluate(
        """
        (el) => {
          const rect = el.getBoundingClientRect();
          const margin = 96;
          const visible = (
            rect.top >= margin &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight - margin) &&
            rect.right <= window.innerWidth
          );
          if (!visible) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
          }
        }
        """
    )
    page.wait_for_timeout(max(0, int(wait_ms)))


def move_cursor_to_point(
    page: Page,
    x: float,
    y: float,
    *,
    click: bool = False,
    extra_pause_ms: int = 0,
    min_travel_ms: int = CURSOR_MIN_TRAVEL_MS,
    max_travel_ms: int = CURSOR_MAX_TRAVEL_MS,
) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    x = float(x)
    y = float(y)
    dx = x - _DEMO_CURSOR_POS["x"]
    dy = y - _DEMO_CURSOR_POS["y"]
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    travel_ms = max(int(min_travel_ms), min(int(max_travel_ms), int(distance / CURSOR_SPEED_PX_PER_MS)))
    steps = max(28, min(120, int(travel_ms / 12)))
    page.mouse.move(x, y, steps=steps)
    move_demo_cursor(page, click=click)
    _DEMO_CURSOR_POS = {"x": x, "y": y}
    page.wait_for_timeout(travel_ms + max(0, int(extra_pause_ms)))


def animate_demo_cursor_to_locator(page: Page, locator: Locator, duration_ms: int = 900, scroll: bool = True) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    if scroll:
        smooth_scroll_locator_into_view(page, locator)
    box = locator.bounding_box()
    if box is None:
        return
    x = float(box["x"] + (box["width"] / 2.0))
    y = float(box["y"] + (box["height"] / 2.0))
    page.evaluate(
        """
        async ({ fromX, fromY, toX, toY, duration }) => {
          const node = document.getElementById('pw-demo-cursor');
          if (!node) return;
          const total = Math.max(120, Number(duration) || 900);
          node.style.left = `${fromX}px`;
          node.style.top = `${fromY}px`;
          await new Promise((resolve) => {
            const start = performance.now();
            const step = (now) => {
              const t = Math.min(1, (now - start) / total);
              const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
              node.style.left = `${fromX + ((toX - fromX) * eased)}px`;
              node.style.top = `${fromY + ((toY - fromY) * eased)}px`;
              if (t < 1) {
                requestAnimationFrame(step);
              } else {
                resolve();
              }
            };
            requestAnimationFrame(step);
          });
        }
        """,
        {
            "fromX": float(_DEMO_CURSOR_POS["x"]),
            "fromY": float(_DEMO_CURSOR_POS["y"]),
            "toX": x,
            "toY": y,
            "duration": int(duration_ms),
        },
    )
    _DEMO_CURSOR_POS = {"x": x, "y": y}


def animate_demo_cursor_with_scroll(page: Page, delta_y: int = 320, duration_ms: int = 900) -> None:
    global _DEMO_CURSOR_POS
    ensure_demo_cursor(page, start_x=_DEMO_CURSOR_POS["x"], start_y=_DEMO_CURSOR_POS["y"])
    target_y = max(80.0, min(820.0, float(_DEMO_CURSOR_POS["y"]) + 120.0))
    page.evaluate(
        """
        async ({ fromX, fromY, toY, scrollDelta, duration }) => {
          const node = document.getElementById('pw-demo-cursor');
          if (!node) return;
          const total = Math.max(120, Number(duration) || 900);
          const startScroll = window.scrollY || window.pageYOffset || 0;
          node.style.left = `${fromX}px`;
          node.style.top = `${fromY}px`;
          await new Promise((resolve) => {
            const start = performance.now();
            const step = (now) => {
              const t = Math.min(1, (now - start) / total);
              const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
              node.style.left = `${fromX}px`;
              node.style.top = `${fromY + ((toY - fromY) * eased)}px`;
              window.scrollTo({ top: startScroll + (Number(scrollDelta) * eased), left: 0 });
              if (t < 1) {
                requestAnimationFrame(step);
              } else {
                resolve();
              }
            };
            requestAnimationFrame(step);
          });
        }
        """,
        {
            "fromX": float(_DEMO_CURSOR_POS["x"]),
            "fromY": float(_DEMO_CURSOR_POS["y"]),
            "toY": target_y,
            "scrollDelta": int(delta_y),
            "duration": int(duration_ms),
        },
    )
    _DEMO_CURSOR_POS = {"x": float(_DEMO_CURSOR_POS["x"]), "y": target_y}


def move_to_locator(page: Page, locator: Locator, click: bool = False, pause_ms: int = 0) -> None:
    smooth_scroll_locator_into_view(page, locator)
    box = locator.bounding_box()
    if box is None:
        move_demo_cursor(page, click=click)
        if pause_ms > 0:
            page.wait_for_timeout(pause_ms)
        return
    x = box["x"] + (box["width"] / 2.0)
    y = box["y"] + (box["height"] / 2.0)
    move_cursor_to_point(page, x, y, click=click, extra_pause_ms=max(0, int(pause_ms)))


def _wait_then_click(page: Page, locator: Locator) -> None:
    move_to_locator(page, locator, click=False, pause_ms=0)
    page.wait_for_timeout(PRE_CLICK_WAIT_MS)
    move_demo_cursor(page, click=True)
    locator.click(delay=0)


def smooth_click(page: Page, locator: Locator, after_ms: int = POST_ACTION_WAIT_MS) -> None:
    _wait_then_click(page, locator)
    page.wait_for_timeout(after_ms)


def smooth_mouse_click(page: Page, locator: Locator, after_ms: int = POST_ACTION_WAIT_MS) -> None:
    smooth_scroll_locator_into_view(page, locator)
    handle = locator.element_handle(timeout=5_000)
    if handle is None:
        raise RuntimeError("Could not resolve element handle for mouse click.")
    box = handle.bounding_box()
    if box is None:
        raise RuntimeError("Could not resolve element box for mouse click.")
    x = box["x"] + (box["width"] / 2.0)
    y = box["y"] + (box["height"] / 2.0)
    move_cursor_to_point(page, x, y)
    page.wait_for_timeout(PRE_CLICK_WAIT_MS)
    move_demo_cursor(page, click=True)
    page.mouse.click(x, y, delay=0)
    page.wait_for_timeout(after_ms)


def smooth_check(page: Page, locator: Locator, after_ms: int = POST_ACTION_WAIT_MS) -> None:
    move_to_locator(page, locator)
    page.wait_for_timeout(PRE_CLICK_WAIT_MS)
    move_demo_cursor(page, click=True)
    if not locator.is_checked():
        locator.check()
    page.wait_for_timeout(after_ms)


def smooth_fill(page: Page, locator: Locator, value: str, type_delay_ms: int = 120, after_ms: int = POST_ACTION_WAIT_MS) -> None:
    _wait_then_click(page, locator)
    locator.press("Control+A")
    locator.press("Backspace")
    locator.type(value, delay=max(0, int(type_delay_ms)))
    page.wait_for_timeout(after_ms)


def smooth_select_option(page: Page, locator: Locator, value: str, after_ms: int = POST_ACTION_WAIT_MS) -> None:
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
    after_ms: int = POST_ACTION_WAIT_MS,
) -> None:
    move_to_locator(page, locator, click=False, pause_ms=0)
    page.wait_for_timeout(PRE_CLICK_WAIT_MS)
    move_demo_cursor(page, click=True)
    locator.evaluate("el => { if (el && typeof el.focus === 'function') el.focus(); }")
    state = locator.evaluate(
        """
        (el) => {
          const options = Array.from(el.options || []);
          const values = options.map((opt) => String(opt.value || ""));
          const labels = options.map((opt) => String((opt.textContent || "").trim()));
          const disabled = options.map((opt) => Boolean(opt.disabled));
          const currentIndex = Number.isInteger(el.selectedIndex) ? el.selectedIndex : -1;
          return { values, labels, disabled, currentIndex };
        }
        """
    )
    values = [str(item) for item in (state.get("values") or [])]
    labels = [str(item) for item in (state.get("labels") or [])]
    disabled = [bool(item) for item in (state.get("disabled") or [])]
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
    if target_index < len(disabled) and disabled[target_index]:
        raise RuntimeError(f"Dropdown option match is disabled ({description}).")

    overlay_target = locator.evaluate(
        """
        (el, payload) => {
          const old = document.getElementById('pw-demo-select-menu');
          if (old) old.remove();
          if (typeof window.__pwDemoSelectMenuCleanup === 'function') {
            try { window.__pwDemoSelectMenuCleanup(); } catch (_err) {}
          }
          window.__pwDemoSelectMenuCleanup = null;

          const rect = el.getBoundingClientRect();
          const labels = payload.labels || [];
          const targetIndex = Number(payload.targetIndex || 0);
          const currentIndex = Number(payload.currentIndex || 0);
          const optionHeight = 30;
          const visibleRows = Math.min(9, Math.max(1, labels.length));
          const start = Math.max(0, Math.min(
            Math.max(0, labels.length - visibleRows),
            targetIndex - Math.floor(visibleRows / 2)
          ));
          const end = Math.min(labels.length, start + visibleRows);
          const menu = document.createElement('div');
          menu.id = 'pw-demo-select-menu';
          menu.style.position = 'fixed';
          menu.style.left = `${Math.max(8, rect.left)}px`;
          const below = rect.bottom + 8;
          const above = rect.top - ((end - start) * optionHeight) - 8;
          const fitsBelow = below + ((end - start) * optionHeight) < window.innerHeight - 8;
          menu.style.top = `${fitsBelow ? below : Math.max(8, above)}px`;
          menu.style.width = `${Math.max(220, rect.width)}px`;
          menu.style.background = '#ffffff';
          menu.style.border = '1px solid #334155';
          menu.style.borderRadius = '8px';
          menu.style.boxShadow = '0 18px 40px rgba(15, 23, 42, 0.28)';
          menu.style.overflow = 'hidden';
          menu.style.zIndex = '2147483646';
          menu.style.font = '500 14px/1.2 "Manrope", "Space Grotesk", sans-serif';
          menu.style.color = '#0f172a';
          menu.style.pointerEvents = 'auto';

          const rows = [];
          for (let idx = start; idx < end; idx += 1) {
            const row = document.createElement('div');
            row.textContent = labels[idx] || '(empty)';
            row.dataset.index = String(idx);
            row.style.height = `${optionHeight}px`;
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.padding = '0 12px';
            row.style.whiteSpace = 'nowrap';
            row.style.overflow = 'hidden';
            row.style.textOverflow = 'ellipsis';
            row.style.background = idx === currentIndex ? '#e0f2fe' : '#ffffff';
            row.style.color = idx === targetIndex ? '#0f172a' : '#334155';
            if (idx === targetIndex) {
              row.style.fontWeight = '800';
            }
            menu.appendChild(row);
            rows.push(row);
          }
          document.body.appendChild(menu);

          const clearRows = () => {
            for (const row of rows) {
              const idx = Number(row.dataset.index || -1);
              row.style.background = idx === currentIndex ? '#e0f2fe' : '#ffffff';
              row.style.color = idx === targetIndex ? '#0f172a' : '#334155';
            }
          };
          const onMouseMove = (evt) => {
            const x = Number(evt?.clientX ?? -1);
            const y = Number(evt?.clientY ?? -1);
            const left = parseFloat(menu.style.left) || 0;
            const top = parseFloat(menu.style.top) || 0;
            const width = parseFloat(menu.style.width) || 0;
            const height = (end - start) * optionHeight;
            const inside = x >= left && x <= (left + width) && y >= top && y <= (top + height);
            clearRows();
            if (!inside) return;
            const local = Math.floor((y - top) / optionHeight);
            if (local >= 0 && local < rows.length) {
              rows[local].style.background = '#2563eb';
              rows[local].style.color = '#ffffff';
            }
          };
          document.addEventListener('mousemove', onMouseMove, { passive: true });
          window.__pwDemoSelectMenuCleanup = () => {
            try { document.removeEventListener('mousemove', onMouseMove); } catch (_err) {}
          };

          let pathStart = start;
          if (currentIndex >= start && currentIndex < end) {
            pathStart = currentIndex;
          } else if (currentIndex >= 0 && currentIndex > targetIndex) {
            pathStart = end - 1;
          }
          const path = [];
          const direction = targetIndex >= pathStart ? 1 : -1;
          for (let idx = pathStart; direction > 0 ? idx <= targetIndex : idx >= targetIndex; idx += direction) {
            if (idx < start || idx >= end) continue;
            const local = idx - start;
            path.push({
              x: (parseFloat(menu.style.left) || 0) + ((parseFloat(menu.style.width) || rect.width) / 2.0),
              y: (parseFloat(menu.style.top) || 0) + ((local + 0.5) * optionHeight),
            });
          }
          const activeLocal = Math.max(0, targetIndex - start);
          if (path.length === 0) {
            path.push({
              x: (parseFloat(menu.style.left) || 0) + ((parseFloat(menu.style.width) || rect.width) / 2.0),
              y: (parseFloat(menu.style.top) || 0) + ((activeLocal + 0.5) * optionHeight),
            });
          }
          return {
            x: (parseFloat(menu.style.left) || 0) + ((parseFloat(menu.style.width) || rect.width) / 2.0),
            y: (parseFloat(menu.style.top) || 0) + ((activeLocal + 0.5) * optionHeight),
            path,
          };
        }
        """,
        {"labels": labels, "targetIndex": target_index, "currentIndex": current_index},
    )
    if overlay_target:
        move_cursor_to_point(
            page,
            float(overlay_target["x"]),
            float(overlay_target["y"]),
            min_travel_ms=80,
            max_travel_ms=180,
        )
        page.wait_for_timeout(PRE_CLICK_WAIT_MS)
        move_demo_cursor(page, click=True)
        page.mouse.click(float(overlay_target["x"]), float(overlay_target["y"]), delay=0)
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

    target_value = values[target_index] if target_index < len(values) else ""
    if target_value != "":
        locator.select_option(value=target_value)
    else:
        locator.select_option(label=labels[target_index])
    page.wait_for_timeout(after_ms)


def show_cursor_transition(page: Page, locator: Locator, pause_ms: int = 280) -> None:
    move_to_locator(page, locator, click=False, pause_ms=max(0, int(pause_ms)))
