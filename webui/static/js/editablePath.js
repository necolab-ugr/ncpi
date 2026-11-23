// Makes the "Current folder" path of the file/folder browsers editable.
// Mark the read-only path element with `data-editable-path`; an input is placed next to it,
// kept in sync with its text, and Enter/"Go" dispatches a bubbling `ncpi-path-submit`
// event ({ detail: { path } }) from that element so each browser can load the typed path.
(function () {
    const ATTR = "data-editable-path";
    const INPUT_CLASS = "flex-1 min-w-0 h-8 px-2 rounded-lg border border-gray-300 dark:border-[#323b67] "
        + "bg-white dark:bg-[#111422] text-xs font-mono text-gray-700 dark:text-gray-200 "
        + "focus:outline-none focus:ring-1 focus:ring-primary";
    const BUTTON_CLASS = "h-8 px-3 rounded-lg border border-primary text-xs font-semibold text-primary "
        + "hover:bg-primary/10";

    function readPath(el) {
        const text = String(el.textContent || "").trim();
        return text === "/" ? "" : text;
    }

    function attach(el) {
        if (!el || el.dataset.editablePathAttached === "1") return;
        el.dataset.editablePathAttached = "1";

        const wrapper = document.createElement("span");
        wrapper.className = "mt-1 flex w-full items-center gap-2";

        const input = document.createElement("input");
        input.type = "text";
        input.className = INPUT_CLASS;
        input.spellcheck = false;
        input.autocomplete = "off";
        input.placeholder = "Type or paste a path and press Enter";
        input.setAttribute("aria-label", "Folder or file path");
        input.value = readPath(el);

        const button = document.createElement("button");
        button.type = "button";
        button.className = BUTTON_CLASS;
        button.textContent = "Go";

        const submit = () => {
            const path = String(input.value || "").trim();
            if (!path) return;
            el.dispatchEvent(new CustomEvent("ncpi-path-submit", { bubbles: true, detail: { path } }));
        };
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                // Avoid submitting the surrounding form
                event.preventDefault();
                event.stopPropagation();
                submit();
            }
        });
        button.addEventListener("click", submit);

        wrapper.appendChild(input);
        wrapper.appendChild(button);
        el.style.display = "none";
        el.insertAdjacentElement("afterend", wrapper);

        // Keep the input in sync when the browser navigates (textContent / x-text updates)
        new MutationObserver(() => {
            input.value = readPath(el);
        }).observe(el, { childList: true, characterData: true, subtree: true });
    }

    function attachAll(root) {
        (root || document).querySelectorAll(`[${ATTR}]`).forEach(attach);
    }

    window.ncpiEditablePath = { attach, attachAll };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => attachAll(), { once: true });
    } else {
        attachAll();
    }
})();
