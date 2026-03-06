function updateFormVisibility() {
    const method = document.getElementById('proxyMethod').value;

    // Simulation Step: LRWS, ERWS1, ERWS2
    const simulationStep = document.getElementById('simulationStepContainer');
    simulationStep.classList.toggle('hidden', !['LRWS', 'ERWS1', 'ERWS2'].includes(method));

    // FR bin size: FR only
    const frBinSize = document.getElementById('frBinSizeContainer');
    frBinSize.classList.toggle('hidden', method !== 'FR');

    // Spike times/gids: Method = FR
    const timesUpload = document.getElementById('timesUploadContainer');
    const gidsUpload = document.getElementById('gidsUploadContainer');
    const showFR = method === 'FR';
    timesUpload.classList.toggle('hidden', !showFR);
    gidsUpload.classList.toggle('hidden', !showFR);

    // Vm Upload: Method = Vm
    const vmUpload = document.getElementById('vmUploadContainer');
    vmUpload.classList.toggle('hidden', method !== 'Vm');

    // AMPA Upload: AMPA, I, I_abs, LRWS, ERWS1, ERWS2
    const ampaMethods = ['AMPA', 'I', 'I_abs', 'LRWS', 'ERWS1', 'ERWS2'];
    const ampaUpload = document.getElementById('ampaUploadContainer');
    ampaUpload.classList.toggle('hidden', !ampaMethods.includes(method));

    // GABA Upload: GABA, I, I_abs, LRWS, ERWS1, ERWS2
    const gabaMethods = ['GABA', 'I', 'I_abs', 'LRWS', 'ERWS1', 'ERWS2'];
    const gabaUpload = document.getElementById('gabaUploadContainer');
    gabaUpload.classList.toggle('hidden', !gabaMethods.includes(method));

    // External Background Rate: ERWS2 only
    const externalRate = document.getElementById('externalRateContainer');
    externalRate.classList.toggle('hidden', method !== 'ERWS2');
}

function basenameFromPath(pathValue) {
    const value = String(pathValue || '').trim();
    if (!value) return '';
    const normalized = value.replace(/\\/g, '/');
    const parts = normalized.split('/').filter(Boolean);
    return parts.length ? parts[parts.length - 1] : value;
}

function parentDirPath(pathValue) {
    const value = String(pathValue || '').trim();
    if (!value) return '';
    const normalized = value.replace(/\\/g, '/');
    const idx = normalized.lastIndexOf('/');
    if (idx <= 0) return '';
    return normalized.slice(0, idx);
}

function createProxyServerFileBrowser() {
    const browseUrl = String(window.__proxyBrowseDirsUrl || '').trim();
    const modal = document.getElementById('proxyServerFileBrowserModal');
    const pathEl = document.getElementById('proxyServerFileBrowserPath');
    const entriesEl = document.getElementById('proxyServerFileBrowserEntries');
    const errorEl = document.getElementById('proxyServerFileBrowserError');
    const loadingEl = document.getElementById('proxyServerFileBrowserLoading');
    const upBtn = document.getElementById('proxyServerFileBrowserUp');
    const refreshBtn = document.getElementById('proxyServerFileBrowserRefresh');
    const selectBtn = document.getElementById('proxyServerFileBrowserSelect');
    const closeBtn = document.getElementById('proxyServerFileBrowserClose');

    if (
        !browseUrl || !modal || !pathEl || !entriesEl || !errorEl || !loadingEl ||
        !upBtn || !refreshBtn || !selectBtn || !closeBtn
    ) {
        return null;
    }

    const runtime = window.__webuiRuntime || {};
    const BROWSE_HISTORY_KEY = 'field_potential_proxy:file_browser';
    let currentPath = '';
    let parentPath = '';
    let selectedFilePath = '';
    let currentExtensions = '.pkl,.pickle';
    let onSelect = null;

    const setError = (message) => {
        const text = String(message || '').trim();
        errorEl.textContent = text;
        errorEl.classList.toggle('hidden', !text);
    };

    const setLoading = (loading) => {
        loadingEl.classList.toggle('hidden', !loading);
    };

    const close = () => {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
        setError('');
    };

    const renderEntries = (dirs, files) => {
        entriesEl.innerHTML = '';
        const dirItems = Array.isArray(dirs) ? dirs : [];
        const fileItems = Array.isArray(files) ? files : [];

        dirItems.forEach((entry) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'w-full px-3 py-2 text-left text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800';
            button.textContent = `${entry.name}/`;
            button.addEventListener('click', () => {
                void load(entry.path);
            });
            entriesEl.appendChild(button);
        });

        fileItems.forEach((entry) => {
            const label = document.createElement('label');
            label.className = 'w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'proxy-server-file-selected';
            input.value = entry.path;
            input.className = 'text-primary focus:ring-primary';
            input.checked = entry.path === selectedFilePath;
            input.addEventListener('change', () => {
                selectedFilePath = entry.path;
            });

            const name = document.createElement('span');
            name.className = 'font-mono';
            name.textContent = entry.name;

            label.appendChild(input);
            label.appendChild(name);
            entriesEl.appendChild(label);
        });

        if (dirItems.length === 0 && fileItems.length === 0) {
            const empty = document.createElement('p');
            empty.className = 'px-3 py-2 text-xs text-slate-500 dark:text-slate-400';
            empty.textContent = 'No matching files or subdirectories found.';
            entriesEl.appendChild(empty);
        }
    };

    const load = async (path) => {
        setLoading(true);
        setError('');
        try {
            const params = new URLSearchParams();
            const targetPath = String(path || '').trim();
            if (targetPath) params.set('path', targetPath);
            params.set('history_key', BROWSE_HISTORY_KEY);
            params.set('include_files', '1');
            params.set('extensions', currentExtensions || '.pkl,.pickle');
            const response = await fetch(`${browseUrl}?${params.toString()}`, { method: 'GET' });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || 'Failed to list server files.');
            }
            currentPath = String(payload.path || '');
            parentPath = String(payload.parent || '');
            pathEl.textContent = currentPath || '/';
            renderEntries(payload.dirs || [], payload.files || []);
            upBtn.disabled = !parentPath;
        } catch (error) {
            setError(error && error.message ? error.message : 'Failed to list server files.');
        } finally {
            setLoading(false);
        }
    };

    upBtn.addEventListener('click', () => {
        if (parentPath) {
            void load(parentPath);
        }
    });
    refreshBtn.addEventListener('click', () => {
        void load(currentPath);
    });
    selectBtn.addEventListener('click', () => {
        if (!selectedFilePath) {
            setError('Select a file.');
            return;
        }
        if (typeof onSelect === 'function') {
            onSelect(selectedFilePath);
        }
        close();
    });
    closeBtn.addEventListener('click', close);
    modal.addEventListener('click', (event) => {
        if (event.target === modal) close();
    });

    return {
        open({ startPath = '', selectedPath = '', extensions = '.pkl,.pickle', onFileSelect = null } = {}) {
            currentExtensions = String(extensions || '.pkl,.pickle');
            selectedFilePath = String(selectedPath || '').trim();
            onSelect = typeof onFileSelect === 'function' ? onFileSelect : null;
            modal.classList.remove('hidden');
            modal.classList.add('flex');
            const resolvedStart = String(startPath || '').trim() || '';
            void load(resolvedStart);
        },
    };
}

function setupUploadZones() {
    const runtime = window.__webuiRuntime || {};
    const defaultMode = String(runtime.default_simulation_source_mode || '').trim().toLowerCase() === 'server-path'
        ? 'server-path'
        : 'upload';
    const serverBrowser = createProxyServerFileBrowser();
    const zones = document.querySelectorAll('[data-upload]');
    zones.forEach((zone) => {
        const inputId = zone.getAttribute('data-upload');
        const input = document.getElementById(inputId);
        const sourceModeInputName = zone.getAttribute('data-source-mode-input');
        const serverPathInputName = zone.getAttribute('data-server-path-input');
        const sourceModeInput = sourceModeInputName ? document.querySelector(`input[name="${sourceModeInputName}"]`) : null;
        const serverPathInput = serverPathInputName ? document.querySelector(`input[name="${serverPathInputName}"]`) : null;
        const filename = zone.querySelector('[data-filename]');
        const uploadedBox = zone.querySelector('[data-uploaded]');
        const uploadedName = zone.querySelector('[data-uploaded-name]');
        const uploadedLabel = zone.querySelector('[data-uploaded-label]');
        const clearSelectionBtn = zone.querySelector('[data-clear-upload]');
        const defaultName = uploadedBox ? String(uploadedBox.dataset.defaultName || '') : '';

        if (!input) {
            return;
        }

        const syncUploadedBoxToDefault = () => {
            if (!uploadedBox || !uploadedName) return;
            if (defaultName) {
                uploadedName.textContent = defaultName;
                if (uploadedLabel) uploadedLabel.textContent = 'Loaded';
                uploadedBox.classList.remove('hidden');
                if (clearSelectionBtn) clearSelectionBtn.classList.add('hidden');
            } else {
                uploadedName.textContent = '';
                uploadedBox.classList.add('hidden');
                if (clearSelectionBtn) clearSelectionBtn.classList.add('hidden');
            }
        };

        const isServerPathMode = () => {
            if (!sourceModeInput) return false;
            const value = String(sourceModeInput.value || '').trim().toLowerCase();
            return value === 'server-path';
        };

        if (!sourceModeInput || !serverPathInput) {
            // Keep legacy local-upload behavior if mode inputs are unavailable.
            zone.addEventListener('click', () => input.click());
            input.addEventListener('change', () => {
                if (!filename) return;
                filename.textContent = (input.files && input.files[0]) ? input.files[0].name : 'Not selected';
            });
            return;
        }

        const controls = document.createElement('div');
        controls.className = 'w-full flex flex-wrap items-center justify-center gap-2 mb-1';
        controls.innerHTML = `
            <button type="button" data-upload-action="true" class="proxy-upload-local px-2.5 py-1 text-[10px] rounded border">Local upload</button>
            <button type="button" data-upload-action="true" class="proxy-upload-server px-2.5 py-1 text-[10px] rounded border">Server file</button>
        `;
        zone.insertBefore(controls, zone.firstChild);

        const localBtn = controls.querySelector('.proxy-upload-local');
        const serverBtn = controls.querySelector('.proxy-upload-server');

        const applyModeVisuals = (mode) => {
            const activeClasses = ['border-primary', 'bg-primary/10', 'text-primary', 'font-semibold'];
            const inactiveClasses = ['border-slate-300', 'dark:border-slate-700', 'text-slate-600', 'dark:text-slate-300'];
            [localBtn, serverBtn].forEach((btn) => {
                btn.classList.remove(...activeClasses);
                btn.classList.add(...inactiveClasses);
            });
            if (mode === 'server-path') {
                serverBtn.classList.add(...activeClasses);
                serverBtn.classList.remove(...inactiveClasses);
            } else {
                localBtn.classList.add(...activeClasses);
                localBtn.classList.remove(...inactiveClasses);
            }
        };

        const syncServerSelectionUi = () => {
            const selected = String(serverPathInput.value || '').trim();
            if (filename) {
                filename.textContent = selected ? basenameFromPath(selected) : 'Server file not selected';
            }
            if (uploadedBox && uploadedName) {
                if (selected) {
                    uploadedName.textContent = basenameFromPath(selected);
                    if (uploadedLabel) uploadedLabel.textContent = 'Selected';
                    uploadedBox.classList.remove('hidden');
                    if (clearSelectionBtn) clearSelectionBtn.classList.remove('hidden');
                } else {
                    syncUploadedBoxToDefault();
                }
            }
        };

        const syncLocalSelectionUi = () => {
            const selectedFile = input.files && input.files[0] ? input.files[0] : null;
            if (filename) {
                filename.textContent = selectedFile ? selectedFile.name : 'Local file not selected';
            }
            if (uploadedBox && uploadedName) {
                if (selectedFile) {
                    uploadedName.textContent = selectedFile.name;
                    if (uploadedLabel) uploadedLabel.textContent = 'Uploaded';
                    uploadedBox.classList.remove('hidden');
                    if (clearSelectionBtn) clearSelectionBtn.classList.remove('hidden');
                } else {
                    syncUploadedBoxToDefault();
                }
            }
        };

        const setMode = (mode) => {
            const nextMode = mode === 'server-path' ? 'server-path' : 'upload';
            sourceModeInput.value = nextMode;
            applyModeVisuals(nextMode);
            zone.classList.remove('border-primary', 'bg-primary/5');
            if (nextMode === 'upload') {
                serverPathInput.value = '';
                syncLocalSelectionUi();
            } else {
                input.value = '';
                syncServerSelectionUi();
            }
        };

        const openServerPicker = () => {
            if (!serverBrowser) {
                return;
            }
            const selected = String(serverPathInput.value || '').trim();
            const startPath = parentDirPath(selected) || '';
            serverBrowser.open({
                startPath,
                selectedPath: selected,
                extensions: '.pkl,.pickle',
                onFileSelect: (pickedPath) => {
                    serverPathInput.value = String(pickedPath || '').trim();
                    syncServerSelectionUi();
                },
            });
        };

        if (localBtn) {
            localBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                setMode('upload');
            });
        }
        if (serverBtn) {
            serverBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                setMode('server-path');
                openServerPicker();
            });
        }
        if (clearSelectionBtn) {
            clearSelectionBtn.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                input.value = '';
                serverPathInput.value = '';
                if (isServerPathMode()) {
                    syncServerSelectionUi();
                } else {
                    syncLocalSelectionUi();
                }
            });
        }

        zone.addEventListener('click', (event) => {
            if (event.target && event.target.closest('[data-upload-action]')) {
                return;
            }
            if (isServerPathMode()) {
                openServerPicker();
                return;
            }
            input.click();
        });
        zone.addEventListener('dragover', (event) => {
            event.preventDefault();
            if (isServerPathMode()) {
                return;
            }
            zone.classList.add('border-primary', 'bg-primary/5');
        });
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('border-primary', 'bg-primary/5');
        });
        zone.addEventListener('drop', (event) => {
            event.preventDefault();
            if (isServerPathMode()) {
                return;
            }
            zone.classList.remove('border-primary', 'bg-primary/5');
            if (event.dataTransfer.files && event.dataTransfer.files[0]) {
                input.files = event.dataTransfer.files;
                serverPathInput.value = '';
                syncLocalSelectionUi();
            }
        });

        input.addEventListener('change', () => {
            if (isServerPathMode()) {
                return;
            }
            serverPathInput.value = '';
            syncLocalSelectionUi();
        });

        const initialMode = String(sourceModeInput.value || '').trim().toLowerCase();
        setMode(initialMode === 'server-path' ? 'server-path' : defaultMode);
    });

    const extInput = document.getElementById('extBackRateFile');
    const extButton = document.getElementById('extBackRateButton');
    const extFilename = document.getElementById('extBackRateFilename');
    const extUploadedBox = document.getElementById('extBackRateUploaded');
    const extUploadedName = document.getElementById('extBackRateUploadedName');
    const extUploadedLabel = document.getElementById('extBackRateUploadedLabel');
    const extClearBtn = document.getElementById('extBackRateClear');
    const extDefaultName = extUploadedBox ? extUploadedBox.dataset.defaultName : '';
    if (extInput && extButton) {
        extButton.addEventListener('click', () => extInput.click());
        extInput.addEventListener('change', () => {
            if (extFilename) {
                if (extInput.files && extInput.files[0]) {
                    extFilename.textContent = extInput.files[0].name;
                } else {
                    extFilename.textContent = 'Local file not selected';
                }
            }
            if (extUploadedBox && extUploadedName) {
                if (extInput.files && extInput.files[0]) {
                    extUploadedName.textContent = extInput.files[0].name;
                    if (extUploadedLabel) {
                        extUploadedLabel.textContent = 'Uploaded';
                    }
                    extUploadedBox.classList.remove('hidden');
                    if (extClearBtn) {
                        extClearBtn.classList.remove('hidden');
                    }
                } else {
                    if (extDefaultName) {
                        extUploadedName.textContent = extDefaultName;
                        if (extUploadedLabel) {
                            extUploadedLabel.textContent = 'Loaded';
                        }
                        extUploadedBox.classList.remove('hidden');
                        if (extClearBtn) {
                            extClearBtn.classList.add('hidden');
                        }
                    } else {
                        extUploadedName.textContent = '';
                        extUploadedBox.classList.add('hidden');
                        if (extClearBtn) {
                            extClearBtn.classList.add('hidden');
                        }
                    }
                }
            }
        });
        if (extClearBtn) {
            extClearBtn.addEventListener('click', () => {
                extInput.value = '';
                extInput.dispatchEvent(new Event('change'));
            });
            if (extInput.files && extInput.files[0]) {
                extClearBtn.classList.remove('hidden');
            } else {
                extClearBtn.classList.add('hidden');
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const hasProxyMethod = document.getElementById('proxyMethod');
    if (hasProxyMethod) {
        updateFormVisibility();
    }
    setupUploadZones();
});
