(function () {
    function toArray(value) {
        if (!value) {
            return [];
        }
        if (Array.isArray(value)) {
            return value;
        }
        return Array.from(value);
    }

    function getElementByIdSafe(id) {
        if (!id) {
            return null;
        }
        return document.getElementById(id);
    }

    function normalizePaths(paths) {
        const unique = [];
        const seen = new Set();
        toArray(paths).forEach((rawPath) => {
            const cleanPath = String(rawPath || "").trim();
            if (!cleanPath || seen.has(cleanPath)) {
                return;
            }
            seen.add(cleanPath);
            unique.push(cleanPath);
        });
        return unique;
    }

    function parentDirPath(pathValue) {
        const value = String(pathValue || "").trim();
        if (!value) {
            return "";
        }
        const normalized = value.replace(/\\/g, "/");
        const index = normalized.lastIndexOf("/");
        if (index <= 0) {
            return "";
        }
        return normalized.slice(0, index);
    }

    function handleFileUpload(event) {
        if (!event || !event.target) {
            return;
        }

        const input = event.target;
        const file = input.files && input.files[0];
        const inputId = input.id;
        if (!file || !inputId) {
            return;
        }

        const alpine = window.Alpine;
        if (!alpine || typeof alpine.$data !== "function") {
            return;
        }

        const owner = input.closest("[x-data]");
        if (!owner) {
            return;
        }

        const component = alpine.$data(owner);
        if (!component || typeof component !== "object") {
            return;
        }

        if (!component.uploads) {
            component.uploads = {};
        }

        component.uploads[inputId] = {
            name: file.name,
            uploaded: false,
        };

        if (typeof component.onFileSelected === "function") {
            component.onFileSelected(file, inputId);
        }

        if (typeof component.$nextTick === "function") {
            component.$nextTick();
        }
    }

    function initUnifiedUploadFlow(config) {
        const cfg = config || {};
        const runtime = cfg.runtime || window.__webuiRuntime || {};
        const allowServerSource = Object.prototype.hasOwnProperty.call(cfg, "allowServerSource")
            ? Boolean(cfg.allowServerSource)
            : runtime.is_server_runtime !== false;

        const elements = {
            form: getElementByIdSafe(cfg.formId),
            fileInput: getElementByIdSafe(cfg.fileInputId),
            dropZone: getElementByIdSafe(cfg.dropZoneId),
            sourceModeInput: getElementByIdSafe(cfg.sourceModeInputId),
            serverPathInput: getElementByIdSafe(cfg.serverPathInputId),
            modeAutoBtn: getElementByIdSafe(cfg.modeAutoBtnId),
            modeUploadBtn: getElementByIdSafe(cfg.modeUploadBtnId),
            modeServerBtn: getElementByIdSafe(cfg.modeServerBtnId),
            dropZoneHint: getElementByIdSafe(cfg.dropZoneHintId),
            selectedLocalContainer: getElementByIdSafe(cfg.selectedLocalContainerId),
            selectedLocalList: getElementByIdSafe(cfg.selectedLocalListId),
            selectedServerContainer: getElementByIdSafe(cfg.selectedServerContainerId),
            selectedServerList: getElementByIdSafe(cfg.selectedServerListId),
            selectedDetectedContainer: getElementByIdSafe(cfg.selectedDetectedContainerId),
            modalRoot: getElementByIdSafe(cfg.modalRootId),
            modalPath: getElementByIdSafe(cfg.modalPathId),
            modalEntries: getElementByIdSafe(cfg.modalEntriesId),
            modalError: getElementByIdSafe(cfg.modalErrorId),
            modalLoading: getElementByIdSafe(cfg.modalLoadingId),
            modalUp: getElementByIdSafe(cfg.modalUpId),
            modalRefresh: getElementByIdSafe(cfg.modalRefreshId),
            modalSelect: getElementByIdSafe(cfg.modalSelectId),
            modalClose: getElementByIdSafe(cfg.modalCloseId),
        };

        const required = [
            elements.form,
            elements.fileInput,
            elements.dropZone,
            elements.sourceModeInput,
            elements.serverPathInput,
            elements.modeUploadBtn,
            elements.modeServerBtn,
            elements.dropZoneHint,
            elements.selectedLocalContainer,
            elements.selectedLocalList,
            elements.selectedServerContainer,
            elements.selectedServerList,
            elements.modalRoot,
            elements.modalPath,
            elements.modalEntries,
            elements.modalError,
            elements.modalLoading,
            elements.modalUp,
            elements.modalRefresh,
            elements.modalSelect,
            elements.modalClose,
        ];
        if (required.some((item) => !item)) {
            return null;
        }

        const browseDirsUrl = String(cfg.browseDirsUrl || "").trim();
        const selectPathUrl = String(cfg.selectPathUrl || "").trim();
        if (!browseDirsUrl || !selectPathUrl) {
            return null;
        }

        const activeButtonClasses = toArray(cfg.activeButtonClasses).filter(Boolean);
        const inactiveButtonClasses = toArray(cfg.inactiveButtonClasses).filter(Boolean);
        const dragActiveClasses = toArray(cfg.dragActiveClasses).filter(Boolean);
        const extensions = String(cfg.extensions || ".pkl,.pickle");
        const localPickerId = String(cfg.localPickerId || cfg.historyKey || cfg.formId || "ncpi_local_picker");
        const fileInputResetOnClick = cfg.fileInputResetOnClick !== false;
        const allowMultipleServerSelection = cfg.allowMultipleServerSelection !== false;
        const allowAutoDetectedSource = Boolean(cfg.allowAutoDetectedSource);
        const detectedAvailable = Boolean(cfg.detectedAvailable);
        const noMatchesText = String(cfg.noMatchesText || "No matching files or subdirectories found.");
        const noServerSelectionText = String(cfg.noServerSelectionText || "Select at least one file.");
        const serverListErrorText = String(cfg.serverListErrorText || "Failed to list server files.");
        const directoryEntryClass = String(
            cfg.directoryEntryClass ||
            "w-full px-3 py-2 text-left text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
        );
        const fileEntryClass = String(
            cfg.fileEntryClass ||
            "w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer"
        );
        const emptyEntryClass = String(
            cfg.emptyEntryClass || "px-3 py-2 text-xs text-slate-500 dark:text-slate-400"
        );
        const nativeUnavailableNeedle = String(
            cfg.nativeUnavailableNeedle || "native file/folder picker is unavailable"
        ).toLowerCase();
        const loadUploadHintText = String(cfg.uploadHintText || "Drag and drop pkl files or click to browse");
        const loadServerHintText = String(cfg.serverHintText || loadUploadHintText);
        const loadAutoHintText = String(cfg.autoHintText || "Using detected files from previous steps");

        const onLocalError = typeof cfg.onLocalError === "function" ? cfg.onLocalError : null;
        const onLocalErrorClear = typeof cfg.onLocalErrorClear === "function" ? cfg.onLocalErrorClear : null;
        const processLocalFilesCustom = typeof cfg.processLocalFiles === "function" ? cfg.processLocalFiles : null;

        let sourceMode = "upload";
        let serverBrowserCurrentPath = "";
        let serverBrowserParentPath = "";
        let serverBrowserSelectedPaths = [];

        if (!allowServerSource) {
            elements.modeServerBtn.classList.add("hidden");
            elements.selectedServerContainer.classList.add("hidden");
        }

        function clearLocalError() {
            if (onLocalErrorClear) {
                onLocalErrorClear();
            }
        }

        function showLocalError(message) {
            if (onLocalError) {
                onLocalError(message);
            }
        }

        function submitIfReady() {
            if (sourceMode === "auto-detected") {
                if (detectedAvailable) {
                    elements.form.submit();
                }
                return;
            }
            if (sourceMode === "upload" && elements.fileInput.files && elements.fileInput.files.length > 0) {
                elements.form.submit();
                return;
            }
            if (sourceMode === "server-path" && String(elements.serverPathInput.value || "").trim()) {
                elements.form.submit();
            }
        }

        function renderSelectedLocalFiles(files) {
            const fileList = toArray(files || []);
            elements.selectedLocalList.innerHTML = "";
            fileList.forEach((file) => {
                const line = document.createElement("span");
                line.className = "block";
                line.textContent = file.name;
                elements.selectedLocalList.appendChild(line);
            });
            elements.selectedLocalContainer.classList.toggle("hidden", fileList.length === 0);
        }

        function hideSelectedLocalFiles() {
            elements.selectedLocalContainer.classList.add("hidden");
            elements.selectedLocalList.innerHTML = "";
        }

        function parseSelectedServerFiles() {
            return normalizePaths(
                String(elements.serverPathInput.value || "")
                    .split(/\r?\n/)
                    .map((item) => item.trim())
                    .filter(Boolean)
            );
        }

        function setSelectedServerFiles(filePaths) {
            const unique = normalizePaths(filePaths);
            elements.serverPathInput.value = unique.join("\n");
            elements.selectedServerList.innerHTML = "";
            unique.forEach((pathValue) => {
                const line = document.createElement("div");
                line.className = "font-mono break-all";
                line.textContent = pathValue;
                elements.selectedServerList.appendChild(line);
            });
            elements.selectedServerContainer.classList.toggle("hidden", unique.length === 0);
        }

        function setInputFiles(files) {
            const list = toArray(files || []);
            const dataTransfer = new DataTransfer();
            list.forEach((file) => dataTransfer.items.add(file));
            elements.fileInput.files = dataTransfer.files;
        }

        function parseAllowedExtensions(rawExtensions) {
            return normalizePaths(
                String(rawExtensions || "")
                    .split(",")
                    .map((item) => item.trim().toLowerCase())
                    .filter((item) => item.startsWith("."))
            );
        }

        async function tryPickLocalFilesWithBrowserMemory() {
            if (typeof window.showOpenFilePicker !== "function" || !window.isSecureContext) {
                return false;
            }

            try {
                const pickerOptions = {
                    multiple: Boolean(elements.fileInput && elements.fileInput.multiple),
                    id: localPickerId,
                };

                const allowedExts = parseAllowedExtensions(extensions);
                if (allowedExts.length > 0) {
                    pickerOptions.types = [{
                        description: "Supported files",
                        accept: {
                            "application/octet-stream": allowedExts,
                        },
                    }];
                }

                const handles = await window.showOpenFilePicker(pickerOptions);
                const files = [];
                for (const handle of toArray(handles)) {
                    if (!handle || typeof handle.getFile !== "function") {
                        continue;
                    }
                    const file = await handle.getFile();
                    if (file) {
                        files.push(file);
                    }
                }
                if (!files.length) {
                    return true;
                }
                processLocalFiles(files);
                return true;
            } catch (error) {
                const errName = String(error && error.name ? error.name : "");
                if (errName === "AbortError") {
                    return true;
                }
                return false;
            }
        }

        function setSourceMode(mode) {
            if (allowAutoDetectedSource && detectedAvailable && mode === "auto-detected") {
                sourceMode = "auto-detected";
            } else if (allowServerSource && mode === "server-path") {
                sourceMode = "server-path";
            } else {
                sourceMode = "upload";
            }
            elements.sourceModeInput.value = sourceMode;
            clearLocalError();

            if (elements.modeAutoBtn) {
                elements.modeAutoBtn.classList.remove(...activeButtonClasses);
                elements.modeAutoBtn.classList.add(...inactiveButtonClasses);
            }
            elements.modeUploadBtn.classList.remove(...activeButtonClasses);
            elements.modeUploadBtn.classList.add(...inactiveButtonClasses);
            elements.modeServerBtn.classList.remove(...activeButtonClasses);
            elements.modeServerBtn.classList.add(...inactiveButtonClasses);

            if (sourceMode === "auto-detected") {
                if (elements.modeAutoBtn) {
                    elements.modeAutoBtn.classList.add(...activeButtonClasses);
                    elements.modeAutoBtn.classList.remove(...inactiveButtonClasses);
                }
                elements.dropZoneHint.textContent = loadAutoHintText;
                elements.selectedLocalContainer.classList.add("hidden");
                elements.selectedServerContainer.classList.add("hidden");
                if (elements.selectedDetectedContainer) {
                    elements.selectedDetectedContainer.classList.remove("hidden");
                }
            } else if (sourceMode === "upload") {
                elements.modeUploadBtn.classList.add(...activeButtonClasses);
                elements.modeUploadBtn.classList.remove(...inactiveButtonClasses);
                elements.dropZoneHint.textContent = loadUploadHintText;
                elements.selectedServerContainer.classList.add("hidden");
                renderSelectedLocalFiles(elements.fileInput.files || []);
                if (elements.selectedDetectedContainer) {
                    elements.selectedDetectedContainer.classList.add("hidden");
                }
            } else {
                elements.modeServerBtn.classList.add(...activeButtonClasses);
                elements.modeServerBtn.classList.remove(...inactiveButtonClasses);
                elements.dropZoneHint.textContent = loadServerHintText;
                elements.selectedLocalContainer.classList.add("hidden");
                const selectedServerPaths = parseSelectedServerFiles();
                serverBrowserSelectedPaths = selectedServerPaths.slice();
                setSelectedServerFiles(selectedServerPaths);
                if (elements.selectedDetectedContainer) {
                    elements.selectedDetectedContainer.classList.add("hidden");
                }
            }
        }

        function showServerBrowserError(message) {
            const clean = String(message || "").trim();
            elements.modalError.textContent = clean;
            elements.modalError.classList.toggle("hidden", !clean);
        }

        function setServerBrowserLoading(loading) {
            elements.modalLoading.classList.toggle("hidden", !loading);
        }

        function renderServerBrowserEntries(dirs, files) {
            elements.modalEntries.innerHTML = "";
            const directoryItems = Array.isArray(dirs) ? dirs : [];
            const fileItems = Array.isArray(files) ? files : [];
            const inputType = allowMultipleServerSelection ? "checkbox" : "radio";

            directoryItems.forEach((entry) => {
                const button = document.createElement("button");
                button.type = "button";
                button.className = directoryEntryClass;
                button.textContent = `${entry.name}/`;
                button.addEventListener("click", () => {
                    void loadServerFiles(entry.path);
                });
                elements.modalEntries.appendChild(button);
            });

            fileItems.forEach((entry) => {
                const label = document.createElement("label");
                label.className = fileEntryClass;

                const input = document.createElement("input");
                input.type = inputType;
                input.name = "unified-server-file-browser-selection";
                input.value = entry.path;
                input.className = "text-primary focus:ring-primary";
                if (serverBrowserSelectedPaths.includes(entry.path)) {
                    input.checked = true;
                }

                input.addEventListener("change", (event) => {
                    if (!allowMultipleServerSelection) {
                        serverBrowserSelectedPaths = event.target.checked ? [entry.path] : [];
                        return;
                    }
                    if (event.target.checked) {
                        if (!serverBrowserSelectedPaths.includes(entry.path)) {
                            serverBrowserSelectedPaths.push(entry.path);
                        }
                    } else {
                        serverBrowserSelectedPaths = serverBrowserSelectedPaths.filter(
                            (selectedPath) => selectedPath !== entry.path
                        );
                    }
                });

                const span = document.createElement("span");
                span.className = "font-mono";
                span.textContent = entry.name;
                label.appendChild(input);
                label.appendChild(span);
                elements.modalEntries.appendChild(label);
            });

            if (directoryItems.length === 0 && fileItems.length === 0) {
                const empty = document.createElement("p");
                empty.className = emptyEntryClass;
                empty.textContent = noMatchesText;
                elements.modalEntries.appendChild(empty);
            }
        }

        async function loadServerFiles(path) {
            setServerBrowserLoading(true);
            showServerBrowserError("");
            try {
                const params = new URLSearchParams();
                const targetPath = String(path || "").trim();
                if (targetPath) {
                    params.set("path", targetPath);
                }
                if (cfg.historyKey) {
                    params.set("history_key", cfg.historyKey);
                }
                params.set("include_files", "1");
                params.set("extensions", extensions);
                const response = await fetch(`${browseDirsUrl}?${params.toString()}`, { method: "GET" });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.error || serverListErrorText);
                }
                serverBrowserCurrentPath = String(payload.path || "");
                serverBrowserParentPath = String(payload.parent || "");
                elements.modalPath.textContent = serverBrowserCurrentPath || "/";
                renderServerBrowserEntries(payload.dirs || [], payload.files || []);
                elements.modalUp.disabled = !serverBrowserParentPath;
            } catch (error) {
                showServerBrowserError((error && error.message) || serverListErrorText);
            } finally {
                setServerBrowserLoading(false);
            }
        }

        async function pickNativeServerFiles() {
            const formData = new FormData();
            formData.append("mode", String(cfg.nativePickerMode || "file"));
            formData.append("extensions", extensions);
            formData.append("allow_multiple", allowMultipleServerSelection ? "1" : "0");
            if (cfg.historyKey) {
                formData.append("history_key", cfg.historyKey);
            }
            const response = await fetch(selectPathUrl, {
                method: "POST",
                body: formData,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || "Path selection cancelled.");
            }
            const paths = Array.isArray(payload.paths) ? payload.paths : (payload.path ? [payload.path] : []);
            const cleaned = normalizePaths(paths);
            if (!cleaned.length) {
                throw new Error("No files selected.");
            }
            return cleaned;
        }

        async function openServerFileModal() {
            elements.modalRoot.classList.remove("hidden");
            elements.modalRoot.classList.add("flex");
            serverBrowserSelectedPaths = parseSelectedServerFiles();
            const firstPath = serverBrowserSelectedPaths.length > 0 ? serverBrowserSelectedPaths[0] : "";
            await loadServerFiles(parentDirPath(firstPath));
        }

        async function openServerFilePicker() {
            try {
                const selected = await pickNativeServerFiles();
                setSelectedServerFiles(selected);
                submitIfReady();
                return;
            } catch (error) {
                const errorMessage = String((error && error.message) || "").toLowerCase();
                const shouldFallbackToBrowser = errorMessage.includes(nativeUnavailableNeedle);
                if (!shouldFallbackToBrowser) {
                    return;
                }
            }
            await openServerFileModal();
        }

        function closeServerFileModal() {
            elements.modalRoot.classList.add("hidden");
            elements.modalRoot.classList.remove("flex");
            showServerBrowserError("");
        }

        const api = {
            submitIfReady,
            setSourceMode,
            setInputFiles,
            renderLocalFiles: renderSelectedLocalFiles,
            hideLocalFiles: hideSelectedLocalFiles,
            setServerPaths: setSelectedServerFiles,
            parseServerPaths: parseSelectedServerFiles,
            clearLocalError,
            showLocalError,
        };

        function processLocalFiles(files) {
            if (processLocalFilesCustom) {
                processLocalFilesCustom(files, api);
                return;
            }
            const selected = toArray(files || []);
            clearLocalError();
            if (!selected.length) {
                setInputFiles([]);
                hideSelectedLocalFiles();
                return;
            }
            setInputFiles(selected);
            renderSelectedLocalFiles(selected);
            submitIfReady();
        }

        elements.fileInput.addEventListener("change", () => {
            processLocalFiles(elements.fileInput.files);
        });

        elements.dropZone.addEventListener("click", () => {
            if (sourceMode === "auto-detected") {
                return;
            }
            if (sourceMode === "upload") {
                const supportsFsaPicker = typeof window.showOpenFilePicker === "function" && window.isSecureContext;
                if (!supportsFsaPicker) {
                    // Keep the click synchronous so browsers preserve user activation.
                    if (fileInputResetOnClick) {
                        elements.fileInput.value = "";
                    }
                    elements.fileInput.click();
                    return;
                }

                void (async () => {
                    const usedFsaPicker = await tryPickLocalFilesWithBrowserMemory();
                    if (usedFsaPicker) {
                        return;
                    }
                    // If the native picker failed even though the API exists, fallback to
                    // the standard file input as a best effort.
                    if (fileInputResetOnClick) {
                        elements.fileInput.value = "";
                    }
                    elements.fileInput.click();
                })();
                return;
            }
            void openServerFilePicker();
        });

        elements.modeUploadBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            setSourceMode("upload");
        });

        elements.modeServerBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            setSourceMode("server-path");
        });

        if (elements.modeAutoBtn) {
            if (!(allowAutoDetectedSource && detectedAvailable)) {
                elements.modeAutoBtn.disabled = true;
                elements.modeAutoBtn.setAttribute("aria-disabled", "true");
            }
            elements.modeAutoBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                if (!(allowAutoDetectedSource && detectedAvailable)) {
                    return;
                }
                setSourceMode("auto-detected");
                submitIfReady();
            });
        }

        ["dragenter", "dragover"].forEach((eventName) => {
            elements.dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                if (sourceMode !== "upload") {
                    return;
                }
                elements.dropZone.classList.add(...dragActiveClasses);
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            elements.dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                if (sourceMode !== "upload") {
                    return;
                }
                elements.dropZone.classList.remove(...dragActiveClasses);
            });
        });

        elements.dropZone.addEventListener("drop", (event) => {
            if (sourceMode !== "upload") {
                return;
            }
            const files = event.dataTransfer && event.dataTransfer.files;
            if (!files || files.length === 0) {
                return;
            }
            processLocalFiles(files);
        });

        elements.modalUp.addEventListener("click", () => {
            if (serverBrowserParentPath) {
                void loadServerFiles(serverBrowserParentPath);
            }
        });

        elements.modalRefresh.addEventListener("click", () => {
            void loadServerFiles(serverBrowserCurrentPath);
        });

        elements.modalSelect.addEventListener("click", () => {
            if (!serverBrowserSelectedPaths.length) {
                showServerBrowserError(noServerSelectionText);
                return;
            }
            setSelectedServerFiles(serverBrowserSelectedPaths);
            closeServerFileModal();
            submitIfReady();
        });

        elements.modalClose.addEventListener("click", closeServerFileModal);
        elements.modalRoot.addEventListener("click", (event) => {
            if (event.target === elements.modalRoot) {
                closeServerFileModal();
            }
        });

        const defaultModeKey = String(cfg.defaultModeKey || "default_simulation_source_mode");
        const runtimeDefault = String(cfg.defaultMode || runtime[defaultModeKey] || "").toLowerCase();
        if (allowAutoDetectedSource && detectedAvailable && runtimeDefault === "auto-detected") {
            setSourceMode("auto-detected");
        } else {
            setSourceMode(runtimeDefault === "server-path" ? "server-path" : "upload");
        }

        return api;
    }

    window.handleFileUpload = handleFileUpload;
    window.initUnifiedUploadFlow = initUnifiedUploadFlow;
})();
