// Global helper to show backend/Flask errors for async requests.
(function () {
    const banner = document.getElementById('server-error-banner');
    const messageNode = document.getElementById('server-error-banner-message');
    const closeButton = document.getElementById('server-error-banner-close');

    const normalizeMessage = (value) => String(value || '').replace(/\s+/g, ' ').trim();

    const hide = () => {
        if (!banner) return;
        banner.classList.add('hidden');
        banner.setAttribute('aria-hidden', 'true');
    };

    const show = (message) => {
        if (!banner || !messageNode) return;
        const normalized = normalizeMessage(message);
        if (!normalized) return;
        messageNode.textContent = normalized;
        banner.classList.remove('hidden');
        banner.removeAttribute('aria-hidden');
    };

    const extractFromObject = (payload) => {
        if (!payload || typeof payload !== 'object') return '';
        const candidates = [
            payload.error,
            payload.message,
            payload.detail,
            payload.details,
        ];
        for (const item of candidates) {
            const normalized = normalizeMessage(item);
            if (normalized) return normalized;
        }
        return '';
    };

    const extractFromHtml = (rawText) => {
        const raw = String(rawText || '');
        if (!raw) return '';
        try {
            const doc = new DOMParser().parseFromString(raw, 'text/html');
            const messages = Array.from(doc.querySelectorAll('.flash-container'))
                .map((node) => normalizeMessage(node.textContent))
                .map((text) => text.replace(/^Error:\s*/i, '').trim())
                .filter(Boolean);
            if (messages.length > 0) {
                return messages.join(' ');
            }

            const titleText = normalizeMessage(doc.querySelector('title')?.textContent || '');
            if (titleText) {
                return titleText;
            }
        } catch (_err) {
            return '';
        }
        return '';
    };

    const extractMessage = (rawText, fallbackMessage = '') => {
        const raw = String(rawText || '').trim();
        if (!raw) {
            return normalizeMessage(fallbackMessage);
        }

        try {
            const parsed = JSON.parse(raw);
            const objectMessage = extractFromObject(parsed);
            if (objectMessage) {
                return objectMessage;
            }
        } catch (_err) {
            // Not JSON, continue with HTML/raw parsing.
        }

        const htmlMessage = extractFromHtml(raw);
        if (htmlMessage) {
            return htmlMessage;
        }

        const normalized = normalizeMessage(raw);
        if (!normalized) {
            return normalizeMessage(fallbackMessage);
        }
        return normalized.length > 300 ? `${normalized.slice(0, 300).trim()}...` : normalized;
    };

    const showFromResponseText = (rawText, fallbackMessage = '') => {
        const message = extractMessage(rawText, fallbackMessage);
        if (message) {
            show(message);
        }
        return message;
    };

    const showFromError = (error, fallbackMessage = '') => {
        const errorMessage = normalizeMessage(error && error.message ? error.message : '');
        const message = errorMessage || normalizeMessage(fallbackMessage);
        if (message) {
            show(message);
        }
        return message;
    };

    if (closeButton) {
        closeButton.addEventListener('click', hide);
    }

    window.ncpiServerErrorBanner = {
        show,
        hide,
        extractMessage,
        showFromResponseText,
        showFromError,
    };
})();
