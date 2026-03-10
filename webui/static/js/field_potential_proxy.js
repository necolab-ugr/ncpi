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

function createNuExtManager() {
    const inferUrl = String(window.__proxyInferTrialsUrl || '').trim();
    const modeSelect = document.getElementById('nuExtMode');
    const sharedContainer = document.getElementById('nuExtSharedContainer');
    const sharedInput = document.getElementById('nuExtSharedValue');
    const perTrialContainer = document.getElementById('nuExtPerTrialContainer');
    const perTrialInputs = document.getElementById('nuExtPerTrialInputs');
    const detectedTrials = document.getElementById('nuExtDetectedTrials');
    const detectionStatus = document.getElementById('nuExtDetectionStatus');
    const valuesJsonInput = document.getElementById('nuExtValuesJson');

    if (
        !modeSelect || !sharedContainer || !sharedInput || !perTrialContainer ||
        !perTrialInputs || !detectedTrials || !detectionStatus || !valuesJsonInput
    ) {
        return null;
    }

    let trialCount = 1;
    let debounceTimer = null;
    let requestCounter = 0;

    const setStatus = (message, isError = false) => {
        detectionStatus.textContent = String(message || '').trim();
        detectionStatus.classList.toggle('text-red-600', !!isError);
        detectionStatus.classList.toggle('dark:text-red-400', !!isError);
        detectionStatus.classList.toggle('text-slate-500', !isError);
        detectionStatus.classList.toggle('dark:text-slate-400', !isError);
    };

    const getMode = () => String(modeSelect.value || 'shared').trim().toLowerCase() === 'per-trial'
        ? 'per-trial'
        : 'shared';

    const getPerTrialValues = () => {
        const values = [];
        const inputs = perTrialInputs.querySelectorAll('[data-nu-ext-trial-input]');
        inputs.forEach((input) => {
            values.push(String(input.value || '').trim());
        });
        return values;
    };

    const syncHiddenValues = () => {
        if (getMode() !== 'per-trial') {
            valuesJsonInput.value = '';
            return;
        }
        valuesJsonInput.value = JSON.stringify(getPerTrialValues());
    };

    const buildPerTrialInputs = (seedValues = []) => {
        perTrialInputs.innerHTML = '';
        for (let idx = 0; idx < trialCount; idx += 1) {
            const wrapper = document.createElement('div');
            wrapper.className = 'flex flex-col gap-1';

            const label = document.createElement('label');
            label.className = 'text-xs font-semibold text-slate-700 dark:text-slate-200';
            label.textContent = `Trial ${idx + 1}`;

            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.1';
            input.className = 'form-input h-10 rounded-lg border-slate-300 dark:border-slate-700 dark:bg-slate-900 focus:ring-primary focus:border-primary';
            input.setAttribute('data-nu-ext-trial-input', '1');
            if (idx < seedValues.length) {
                input.value = String(seedValues[idx] || '').trim();
            }
            input.addEventListener('input', () => {
                syncHiddenValues();
                setStatus(`Detected ${trialCount} trial(s). Set one nu_ext value per trial.`);
            });

            wrapper.appendChild(label);
            wrapper.appendChild(input);
            perTrialInputs.appendChild(wrapper);
        }
        syncHiddenValues();
    };

    const applyModeUi = () => {
        const mode = getMode();
        const perTrial = mode === 'per-trial';
        sharedContainer.classList.toggle('hidden', perTrial);
        perTrialContainer.classList.toggle('hidden', !perTrial);
        if (perTrial) {
            const existing = getPerTrialValues();
            const sharedValue = String(sharedInput.value || '').trim();
            if (existing.length !== trialCount) {
                const seed = [];
                for (let idx = 0; idx < trialCount; idx += 1) {
                    if (idx < existing.length && String(existing[idx] || '').trim()) {
                        seed.push(existing[idx]);
                    } else {
                        seed.push(sharedValue);
                    }
                }
                buildPerTrialInputs(seed);
            } else {
                syncHiddenValues();
            }
            setStatus(`Detected ${trialCount} trial(s). Set one nu_ext value per trial.`);
        } else {
            valuesJsonInput.value = '';
            setStatus(`Detected ${trialCount} trial(s). One shared nu_ext value will be applied to all trials.`);
        }
    };

    const parseTrialCount = (payload) => {
        const raw = Number(payload && payload.trial_count);
        if (!Number.isFinite(raw) || raw < 1) return 1;
        return Math.max(1, Math.floor(raw));
    };

    const setTrialCount = (count, sourceLabel = '') => {
        trialCount = Math.max(1, Number(count) || 1);
        detectedTrials.textContent = String(trialCount);
        if (sourceLabel) {
            setStatus(`Detected ${trialCount} trial(s) from ${sourceLabel}.`);
        } else {
            setStatus(`Detected ${trialCount} trial(s).`);
        }
        applyModeUi();
    };

    const chooseCandidate = (candidates) => {
        const items = Array.isArray(candidates) ? candidates : [];
        const local = items.find((item) => item && item.kind === 'local' && item.file);
        if (local) return local;
        const server = items.find((item) => item && item.kind === 'server' && item.path);
        if (server) return server;
        const fallback = items.find((item) => item && item.kind === 'default' && item.useDefault);
        return fallback || null;
    };

    const inferTrialCount = async (candidate) => {
        if (!candidate || !candidate.fileKey || !inferUrl) {
            setTrialCount(1);
            return;
        }

        const requestId = ++requestCounter;
        setStatus('Detecting trial count from selected simulation data...');
        try {
            const formData = new FormData();
            formData.append('file_key', candidate.fileKey);
            if (candidate.kind === 'local' && candidate.file) {
                formData.append('local_file', candidate.file);
            } else if (candidate.kind === 'server' && candidate.path) {
                formData.append('server_path', candidate.path);
            } else if (candidate.kind === 'default') {
                formData.append('use_default', '1');
            } else {
                setTrialCount(1);
                return;
            }

            const response = await fetch(inferUrl, { method: 'POST', body: formData });
            const payload = await response.json().catch(() => ({}));
            if (requestId !== requestCounter) {
                return;
            }
            if (!response.ok) {
                throw new Error(payload.error || 'Failed to detect trial count.');
            }
            const sourceKind = String(payload.source_kind || candidate.kind || 'selected data');
            setTrialCount(parseTrialCount(payload), sourceKind);
        } catch (error) {
            if (requestId !== requestCounter) {
                return;
            }
            setTrialCount(1);
            setStatus(error && error.message ? error.message : 'Failed to detect trial count.', true);
        }
    };

    const scheduleInferFromCandidates = (candidates) => {
        if (debounceTimer) {
            clearTimeout(debounceTimer);
        }
        debounceTimer = setTimeout(() => {
            const candidate = chooseCandidate(candidates);
            void inferTrialCount(candidate);
        }, 120);
    };

    const validateForSubmit = () => {
        if (document.getElementById('proxyMethod')?.value !== 'ERWS2') {
            return true;
        }
        if (getMode() === 'per-trial') {
            const values = getPerTrialValues();
            if (values.length !== trialCount) {
                setStatus(`Per-trial nu_ext requires ${trialCount} values.`, true);
                return false;
            }
            for (let idx = 0; idx < values.length; idx += 1) {
                if (values[idx] === '' || Number.isNaN(Number(values[idx]))) {
                    setStatus(`Invalid nu_ext value for trial ${idx + 1}.`, true);
                    return false;
                }
            }
            valuesJsonInput.value = JSON.stringify(values);
            return true;
        }
        const shared = String(sharedInput.value || '').trim();
        if (shared === '' || Number.isNaN(Number(shared))) {
            setStatus('Provide a numeric shared nu_ext value.', true);
            return false;
        }
        valuesJsonInput.value = '';
        return true;
    };

    modeSelect.addEventListener('change', applyModeUi);
    sharedInput.addEventListener('input', () => {
        if (getMode() === 'shared') {
            setStatus(`Detected ${trialCount} trial(s). One shared nu_ext value will be applied to all trials.`);
        }
    });
    applyModeUi();

    return {
        scheduleInferFromCandidates,
        validateForSubmit,
    };
}

function setupUploadZones() {
    const runtime = window.__webuiRuntime || {};
    const defaultMode = String(runtime.default_simulation_source_mode || '').trim().toLowerCase() === 'server-path'
        ? 'server-path'
        : 'upload';
    const nuExtManager = createNuExtManager();
    const serverBrowser = createProxyServerFileBrowser();
    const zoneCandidateGetters = [];
    const notifyNuExtTrialDetection = () => {
        if (!nuExtManager) return;
        const candidates = zoneCandidateGetters
            .map((getter) => {
                try {
                    return getter();
                } catch (error) {
                    return null;
                }
            })
            .filter(Boolean);
        nuExtManager.scheduleInferFromCandidates(candidates);
    };
    const zones = document.querySelectorAll('[data-upload]');
    zones.forEach((zone) => {
        const inputId = zone.getAttribute('data-upload');
        const input = document.getElementById(inputId);
        const sourceModeInputName = zone.getAttribute('data-source-mode-input');
        const serverPathInputName = zone.getAttribute('data-server-path-input');
        const ignoreDefaultInputName = zone.getAttribute('data-ignore-default-input');
        const sourceModeInput = sourceModeInputName ? document.querySelector(`input[name="${sourceModeInputName}"]`) : null;
        const serverPathInput = serverPathInputName ? document.querySelector(`input[name="${serverPathInputName}"]`) : null;
        const ignoreDefaultInput = ignoreDefaultInputName ? document.querySelector(`input[name="${ignoreDefaultInputName}"]`) : null;
        const filename = zone.querySelector('[data-filename]');
        const uploadedBox = zone.querySelector('[data-uploaded]');
        const uploadedName = zone.querySelector('[data-uploaded-name]');
        const uploadedLabel = zone.querySelector('[data-uploaded-label]');
        const clearSelectionBtn = zone.querySelector('[data-clear-upload]');
        const defaultName = uploadedBox ? String(uploadedBox.dataset.defaultName || '') : '';
        const fileKey = sourceModeInputName
            ? String(sourceModeInputName).replace(/_source_mode$/, '')
            : '';

        const isDefaultIgnored = () => {
            if (!ignoreDefaultInput) return false;
            const value = String(ignoreDefaultInput.value || '').trim().toLowerCase();
            return value === '1' || value === 'true' || value === 'yes';
        };

        const setDefaultIgnored = (ignored) => {
            if (!ignoreDefaultInput) return;
            ignoreDefaultInput.value = ignored ? '1' : '0';
        };

        if (!input) {
            return;
        }

        const syncUploadedBoxToDefault = () => {
            if (!uploadedBox || !uploadedName) return;
            if (defaultName && !isDefaultIgnored()) {
                uploadedName.textContent = defaultName;
                if (uploadedLabel) uploadedLabel.textContent = 'Loaded';
                uploadedBox.classList.remove('hidden');
                if (clearSelectionBtn) clearSelectionBtn.classList.remove('hidden');
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
                    // A new explicit selection replaces the preloaded default.
                    setDefaultIgnored(true);
                    uploadedName.textContent = basenameFromPath(selected);
                    if (uploadedLabel) uploadedLabel.textContent = 'Selected server file';
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
                    // A new explicit selection replaces the preloaded default.
                    setDefaultIgnored(true);
                    uploadedName.textContent = selectedFile.name;
                    if (uploadedLabel) uploadedLabel.textContent = 'Selected local file';
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
                    notifyNuExtTrialDetection();
                },
            });
        };

        if (localBtn) {
            localBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                setMode('upload');
                notifyNuExtTrialDetection();
            });
        }
        if (serverBtn) {
            serverBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                setMode('server-path');
                notifyNuExtTrialDetection();
            });
        }
        if (clearSelectionBtn) {
            clearSelectionBtn.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                input.value = '';
                serverPathInput.value = '';
                // Clearing is explicit: keep defaults disabled for this field.
                if (defaultName) setDefaultIgnored(true);
                if (isServerPathMode()) {
                    syncServerSelectionUi();
                } else {
                    syncLocalSelectionUi();
                }
                notifyNuExtTrialDetection();
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
                notifyNuExtTrialDetection();
            }
        });

        input.addEventListener('change', () => {
            if (isServerPathMode()) {
                return;
            }
            serverPathInput.value = '';
            syncLocalSelectionUi();
            notifyNuExtTrialDetection();
        });

        zoneCandidateGetters.push(() => {
            const selectedFile = input.files && input.files[0] ? input.files[0] : null;
            if (selectedFile) {
                return { fileKey, kind: 'local', file: selectedFile };
            }
            const selectedServerPath = String(serverPathInput.value || '').trim();
            if (selectedServerPath) {
                return { fileKey, kind: 'server', path: selectedServerPath };
            }
            if (defaultName && !isDefaultIgnored()) {
                return { fileKey, kind: 'default', useDefault: true };
            }
            return null;
        });

        const initialMode = String(sourceModeInput.value || '').trim().toLowerCase();
        setMode(initialMode === 'server-path' ? 'server-path' : defaultMode);
    });

    notifyNuExtTrialDetection();

    const form = document.querySelector('form[action*="field_potential_proxy"]');
    if (form && nuExtManager) {
        form.addEventListener('submit', (event) => {
            if (!nuExtManager.validateForSubmit()) {
                event.preventDefault();
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const hasProxyMethod = document.getElementById('proxyMethod');
    if (hasProxyMethod) {
        updateFormVisibility();
    }
    setupUploadZones();
});
