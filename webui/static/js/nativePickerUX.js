(function () {
    let overlay = null;
    let pendingCount = 0;
    let originalFetch = null;

    function isSelectFolderRequest(input, init) {
        let url = "";
        let method = "";

        if (typeof input === "string") {
            url = input;
        } else if (input && typeof input.url === "string") {
            url = input.url;
        }

        if (init && typeof init.method === "string") {
            method = init.method;
        } else if (input && typeof input.method === "string") {
            method = input.method;
        }

        const normalizedMethod = String(method || "GET").trim().toUpperCase();
        if (normalizedMethod !== "POST") {
            return false;
        }
        return /\/features\/select_folder(?:\?|$)/.test(String(url || ""));
    }

    function ensureOverlay() {
        if (overlay || !document.body) {
            return overlay;
        }

        const root = document.createElement("div");
        root.id = "native-picker-wait-overlay";
        root.setAttribute(
            "style",
            [
                "position:fixed",
                "inset:0",
                "z-index:9999",
                "display:none",
                "align-items:center",
                "justify-content:center",
                "background:rgba(15,23,42,0.32)",
                "backdrop-filter:blur(1px)",
                "pointer-events:none",
            ].join(";")
        );

        const panel = document.createElement("div");
        panel.setAttribute(
            "style",
            [
                "max-width:min(520px,92vw)",
                "border-radius:12px",
                "padding:14px 16px",
                "background:#0f172a",
                "color:#e2e8f0",
                "box-shadow:0 12px 32px rgba(2,6,23,0.45)",
                "font:500 13px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif",
            ].join(";")
        );

        const title = document.createElement("div");
        title.textContent = "Opening server file browser...";
        title.setAttribute("style", "font-weight:700;margin-bottom:4px;");

        const subtitle = document.createElement("div");
        subtitle.id = "native-picker-wait-message";
        subtitle.textContent = "Waiting for the native picker window. If it is behind, check the taskbar/sidebar.";
        subtitle.setAttribute("style", "opacity:0.94;");

        panel.appendChild(title);
        panel.appendChild(subtitle);
        root.appendChild(panel);
        document.body.appendChild(root);
        overlay = root;
        return overlay;
    }

    function setOverlayVisible(visible, message) {
        const el = ensureOverlay();
        if (!el) {
            return;
        }
        const msgEl = document.getElementById("native-picker-wait-message");
        if (msgEl && message) {
            msgEl.textContent = message;
        }
        el.style.display = visible ? "flex" : "none";
    }

    async function withPending(action, options) {
        if (typeof action !== "function") {
            return undefined;
        }

        const message = String(
            (options && options.message) ||
            "Waiting for the native picker window. If it is behind, check the taskbar/sidebar."
        );

        pendingCount += 1;
        if (pendingCount === 1) {
            setOverlayVisible(true, message);
        } else {
            setOverlayVisible(true, message);
        }

        try {
            return await action();
        } finally {
            pendingCount = Math.max(0, pendingCount - 1);
            if (pendingCount === 0) {
                setOverlayVisible(false);
            }
        }
    }

    function patchFetch() {
        if (typeof window.fetch !== "function") {
            return;
        }
        if (originalFetch) {
            return;
        }
        originalFetch = window.fetch.bind(window);
        window.fetch = function patchedFetch(input, init) {
            if (!isSelectFolderRequest(input, init)) {
                return originalFetch(input, init);
            }
            return withPending(() => originalFetch(input, init));
        };
    }

    window.ncpiNativePickerUX = {
        withPending,
        show(message) {
            pendingCount += 1;
            setOverlayVisible(true, String(message || ""));
        },
        hide() {
            pendingCount = Math.max(0, pendingCount - 1);
            if (pendingCount === 0) {
                setOverlayVisible(false);
            }
        },
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", patchFetch, { once: true });
    } else {
        patchFetch();
    }
})();
