import re
import os
import time
import pickle
from pathlib import Path
import numpy as np
import pytest
from playwright.sync_api import Playwright, sync_playwright, expect, Page, Locator


# Sampling percentage
sampling_percentage = "10"

# Tolerance error range
tolerance = 1e-1

# Parallel workers (n_jobs)
parallel_workers = "8"


def navigate_and_select(page, target_path, folder=True):
    """
    Go to any target path dynamically, deciding the next button to press to achieve the target path
    """
    # Normalize the path to avoid errors with final / (e.g. /DATOS/ vs /DATOS) 
    target_path = os.path.normpath(target_path)

    # If it's a file, we need to reach the parent directory first
    if folder:
        destination_folder = target_path
        item_name = None
    else:
        destination_folder = os.path.dirname(target_path)
        item_name = os.path.basename(target_path)

    while True:
        # 1. Get and clean the current path
        # This picks the active path display in the modal
        current_folder_locator = page.get_by_text("Current folder:").filter(visible=True).last
        
        # Get the full text (e.g., "Current folder: /home/user")
        full_text = current_folder_locator.inner_text()

        # Clean the string to get just the path
        current_path = os.path.normpath(full_text.replace("Current folder:", "").strip())

        # CASE A: We are already in the target path
        if current_path == destination_folder:
            if folder:
                break
            else:
                # If it's a file, click it and we are done
                page.get_by_role("button", name=item_name, exact=True).click()
                return

        # CASE B: The current path is "father" or part of the path to the target path
        # E.g.: We're in "/DATOS" and the target is "/DATOS/empirical"
        if destination_folder.startswith(current_path + os.sep) or current_path == "/":
            # Decide the next folder to click
            path_suffix = destination_folder[len(current_path):].lstrip(os.sep)
            next_step = path_suffix.split(os.sep)[0]
            
            page.get_by_role("button", name=next_step, exact=True).click()
        
        # CASE C: We're in a totally different path (e.g: /home/user)
        # or we're deeper than we should (e.g: /DATOS/empirical/local/data/sub-001)
        else:
            page.get_by_role("button", name="Up", exact=True).click()

        # Small pause to wait until the DOM is updated after the click
        page.wait_for_load_state("networkidle")

    if folder:
        # Confirm the folder
        page.get_by_role("button", name="Add this folder").click()


def wait_and_get_feature_average(
    page: Page,
    method_name: str = "catch22",
    timeout_terminal: int = 240,
    timeout_file: int = 60,
    timeout_completion: int = 1_500_000  # 25 min
) -> float:

    # --- 1. Search output path in the terminal text ---
    log_selector = "#output-terminal"
    found_line = None
    start = time.time()
    while not found_line and (time.time() - start) < timeout_terminal:
        # Get the whole text of the terminal element
        log_text = page.text_content(log_selector) if page.locator(log_selector).count() else ""
        # Search the line with the text "Persisted dashboard features file"
        for line in log_text.splitlines():
            if "Persisted dashboard features file:" in line:
                found_line = line
                break
        if not found_line:
            time.sleep(2)

    if not found_line:
        raise Exception("The output filepath could not be found in the UI or the timeout was too low for this data")

    # Extract UUID with regex
    match = re.search(r"/tmp/ncpi_webui_session_([a-f0-9]+)/", found_line)
    if not match:
        raise Exception("Session UUID could not be found in the terminal element")
    session_uuid = match.group(1)
    print(f"Session UUID extracted from the terminal: {session_uuid}")

    # --- 2. Expect the end of the computation in the UI ---
    try:
        expect(page.get_by_role("main")).to_contain_text(
            "Features Computed", timeout=timeout_completion
        )
        print("Computation finished (UI indicates 'Features Computed')")
    except Exception:
        print("No UI indicator of 'completed' could be detected. Trying to read the file anyway...")

    expect(page.get_by_role("link", name="Continue to Inference")).to_be_visible(
        timeout=timeout_completion
    )

    # --- 3. Wait and load the pickle file ---
    output_file_path = Path(f"/tmp/ncpi_webui_session_{session_uuid}/features/data/{method_name}_features.pkl")

    start = time.time()
    while not output_file_path.exists() and (time.time() - start) < timeout_file:
        time.sleep(1)
        print("Output file doesnt exist yet, reloading... (60 seconds maximum timeout)")
    if not output_file_path.exists():
        raise FileNotFoundError(
            f"Pickle file {output_file_path} couldnt be found in the expected path after a timeout of {timeout_file}s"
        )
    print(f"File found in: {output_file_path}")

    with open(output_file_path, "rb") as f:
        emp_data = pickle.load(f)

    # --- 4. Calculate the average of all data (ignoring NaN) ---
    features = emp_data["Features"].tolist()
    avg = np.nanmean([np.nanmean(np.asarray(x)) for x in features])
    return float(avg)


# Variable global a nivel de módulo para recordar la última posición del cursor
# Ayuda a que el movimiento sea más suave
_last_cursor_pos = {"x": 24.0, "y": 24.0}


def ensure_demo_cursor(page: Page) -> None:
    """Asegura que el cursor rojo y su event listener existen en la página actual."""
    # Inyectamos el CSS y el div si no están presentes
    page.evaluate(
        """
        () => {
            // Crear estilos si no existen
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
                        border-radius: 50%;
                        background: rgba(220, 38, 38, 0.95);
                        border: 3px solid white;
                        box-shadow: 0 0 0 3px rgba(0,0,0,0.5), 0 0 0 8px rgba(220,38,38,0.2);
                        transform: translate(-50%, -50%);
                        transition: transform 0.09s ease;
                        pointer-events: none;
                        z-index: 2147483647;
                    }
                    #pw-demo-cursor.clicking {
                        transform: translate(-50%, -50%) scale(0.78);
                    }
                `;
                document.head.appendChild(style);
            }

            // Crear el div del cursor si no existe
            if (!document.getElementById('pw-demo-cursor')) {
                const cursor = document.createElement('div');
                cursor.id = 'pw-demo-cursor';
                cursor.style.left = '24px';
                cursor.style.top = '24px';
                document.body.appendChild(cursor);
            }

            // Asegurar el event listener de mousemove (solo una vez por página)
            if (!window.__pwDemoCursorMouseBound) {
                window.__pwDemoCursorMouseBound = true;
                document.addEventListener('mousemove', (e) => {
                    const c = document.getElementById('pw-demo-cursor');
                    if (c) {
                        c.style.left = e.clientX + 'px';
                        c.style.top = e.clientY + 'px';
                    }
                });
            }
        }
        """
    )


def move_demo_cursor(page: Page, target_x: float, target_y: float, move_duration_ms: int = 400) -> None:
    """
    Mueve el ratón real de Playwright en línea recta hacia (target_x, target_y)
    con un número fijo de steps, y espera el tiempo indicado para que la animación sea visible.
    Además, actualiza la posición global almacenada.
    """
    global _last_cursor_pos
    # Asegurar que el cursor existe antes de mover
    ensure_demo_cursor(page)

    # Usamos steps=40 por simplicidad (funciona bien en ventanas de hasta 1600x900)
    page.mouse.move(target_x, target_y, steps=40)
    # Pausa para que el vídeo capture el movimiento
    page.wait_for_timeout(move_duration_ms)
    # Actualizar posición recordada
    _last_cursor_pos = {"x": target_x, "y": target_y}


def click_with_demo_cursor(page: Page, locator: Locator, move_duration_ms: int = 400, click_delay: int = 80) -> None:
    """Mueve el cursor al elemento, muestra efecto de click, y hace clic real."""
    # Obtener coordenadas del centro del elemento
    box = locator.bounding_box()
    if not box:
        raise ValueError("No se pudo obtener bounding box del elemento")
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2

    move_demo_cursor(page, target_x, target_y, move_duration_ms)

    # Efecto visual de click (encogimiento)
    page.evaluate("""
        () => {
            const c = document.getElementById('pw-demo-cursor');
            if (c) {
                c.classList.add('clicking');
                setTimeout(() => c.classList.remove('clicking'), 120);
            }
        }
    """)

    # Clic real de Playwright
    locator.click(delay=click_delay)
    page.wait_for_timeout(200)  # Pequeña pausa para apreciar el click


def select_option_with_cursor(page: Page, locator: Locator, value: str, move_duration_ms: int = 400) -> None:
    """Mueve el cursor al select, luego selecciona la opción."""
    box = locator.bounding_box()
    if not box:
        raise ValueError("No se pudo obtener bounding box del select")
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2

    move_demo_cursor(page, target_x, target_y, move_duration_ms)
    locator.select_option(value)
    page.wait_for_timeout(150)


def fill_with_cursor(page: Page, locator: Locator, text: str, move_duration_ms: int = 400) -> None:
    """Mueve el cursor al campo de texto y luego escribe el texto."""
    box = locator.bounding_box()
    if not box:
        raise ValueError("No se pudo obtener bounding box del campo")
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2

    move_demo_cursor(page, target_x, target_y, move_duration_ms)
    locator.fill(text)
    page.wait_for_timeout(150)


def check_with_cursor(page: Page, locator: Locator, move_duration_ms: int = 400) -> None:
    """Mueve el cursor a un radio/checkbox y luego lo marca."""
    box = locator.bounding_box()
    if not box:
        raise ValueError("No se pudo obtener bounding box del elemento")
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2

    move_demo_cursor(page, target_x, target_y, move_duration_ms)
    # Opcional: añadir efecto de click si quieres
    # page.evaluate("() => { const c = document.getElementById('pw-demo-cursor'); if(c) c.classList.add('clicking'); setTimeout(() => c.classList.remove('clicking'), 120); }")
    locator.check()
    page.wait_for_timeout(150)