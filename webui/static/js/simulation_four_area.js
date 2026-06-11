// static/js/simulation_four_area.js
document.addEventListener('DOMContentLoaded', function() {
    const simulationOnlyParams = new Set(['tstop', 'dt', 'local_num_threads']);
    const fixedGridParams = new Set(['X', 'areas', 'model']);
    const populationIndexedVectorParams = new Set(['X', 'N_X', 'C_m_X', 'tau_m_X', 'E_L_X', 'n_ext']);
    const areaIndexedVectorParams = new Set(['areas']);
    const sourceTargetMatrixParams = new Set([
        'C_YX', 'J_YX', 'delay_YX',
        'inter_area.C_YX', 'inter_area.J_YX', 'inter_area.delay_YX',
    ]);
    const areaSpecificLocalParams = [
        'N_X',
        'C_m_X',
        'tau_m_X',
        'E_L_X',
        'C_YX',
        'J_YX',
        'delay_YX',
        'tau_syn_YX',
        'n_ext',
        'nu_ext',
        'J_ext',
    ];
    const areaSpecificLocalParamSet = new Set(areaSpecificLocalParams);
    const interAreaParamNames = ['inter_area.C_YX', 'inter_area.J_YX', 'inter_area.delay_YX'];
    const interAreaSourceAreaColors = ['#dc2626', '#2563eb', '#16a34a', '#a855f7'];
    const jointParamUnits = {
        C_m_X: 'pF',
        tau_m_X: 'ms',
        E_L_X: 'mV',
        J_YX: 'nA',
        delay_YX: 'ms',
        tau_syn_YX: 'ms',
        nu_ext: 'Hz',
        J_ext: 'nA',
        'inter_area.J_YX': 'nA',
        'inter_area.delay_YX': 'ms',
    };
    // Preset values for four-area cortical model configuration
    const ncpiPresets = {
        tstop: 12000.0,
        dt: 0.0625,
        local_num_threads: 64,
        areas: "['frontal', 'parietal', 'temporal', 'occipital']",
        X: "['E', 'I']",
        N_X: "[8192, 1024]",
        C_m_X: "[289.1, 110.7]",
        tau_m_X: "[10.0, 10.0]",
        E_L_X: "[-65.0, -65.0]",
        model: "iaf_psc_exp",
        C_YX: "[[0.2, 0.2], [0.2, 0.2]]",
        J_YX: "[[1.589, 2.020], [-23.84, -8.441]]",
        delay_YX: "[[2.520, 1.714], [1.585, 1.149]]",
        tau_syn_YX: "[[0.5, 0.5], [0.5, 0.5]]",
        n_ext: "[465, 160]",
        nu_ext: 40.0,
        J_ext: 29.89,
        "inter_area.C_YX": "[[0.02, 0.02], [0.0, 0.0]]",
        "inter_area.J_YX": "[[0.23835, 0.303], [0.0, 0.0]]",
        "inter_area.delay_YX": "[[10.0, 10.0], [0.0, 0.0]]"
    };

    // Get DOM Elements
    const elements = {
        // Parameters Section of the form
        paramSection: document.getElementById('parameter-section'),
        // Add a container where to append the cloned section
        container: document.getElementById('parameter-container'),
        form: document.getElementById('run-simulation-form'),
        runModeInputs: document.querySelectorAll('input[name="sim_run_mode"]'),
        buttonLabel: document.getElementById('run-simulation-button-label'),
        gridHelp: document.getElementById('grid-mode-help'),
        jointGroupsContainer: document.getElementById('joint-groups-container'),
        addJointGroupButton: document.getElementById('add-joint-group-button'),
        jointGroupsInput: document.getElementById('sim-joint-groups'),
        repetitionsInput: document.getElementById('sim-repetitions'),
        useNumpySeedInput: document.getElementById('sim-use-numpy-seed'),
        numpySeedInput: document.getElementById('sim-numpy-seed'),
        areaSelector: document.getElementById('four-area-selector'),
        interAreaModeStatus: document.getElementById('inter-area-mode-status'),
    };
    const restoreForm = readSimulationRestoreForm();

    function readSimulationRestoreForm() {
        const node = document.getElementById('simulation-restore-form');
        if (!node) {
            return {};
        }
        try {
            const parsed = JSON.parse(node.textContent || '{}');
            return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
        } catch (_err) {
            return {};
        }
    }

    function hasRestoredValue(name) {
        return Object.prototype.hasOwnProperty.call(restoreForm, name);
    }

    function restoredValueOrPreset(name, preset) {
        return hasRestoredValue(name) ? restoreForm[name] : preset;
    }

    function parseSimpleGridRangeSpec(value) {
        const text = String(value ?? '').trim();
        if (!text.toLowerCase().startsWith('grid=')) {
            return null;
        }
        const spec = text.slice(5).trim();
        if (!spec || /[\[\]{}(),]/.test(spec)) {
            return null;
        }
        const parts = spec.split(':').map(part => part.trim());
        if (parts.length !== 3) {
            return null;
        }
        if (!parts.every(part => Number.isFinite(Number(part)))) {
            return null;
        }
        return { start: parts[0], stop: parts[1], step: parts[2] };
    }
    let areaLocalState = [];
    let activeAreaIndex = 0;

    function ensureInputCanCarryValue(input, value) {
        if (!input) {
            return;
        }
        if (!input.dataset.originalType) {
            input.dataset.originalType = input.type || 'text';
        }
        const originalType = input.dataset.originalType;
        const textValue = String(value ?? '');
        const wantsGridSpec = textValue.toLowerCase().startsWith('grid=');
        const wantsInvalidNumericText = originalType === 'number' && textValue !== '' && !Number.isFinite(Number(textValue));
        if ((wantsGridSpec || wantsInvalidNumericText) && originalType === 'number') {
            input.type = 'text';
            input.inputMode = 'decimal';
            return;
        }
        if (input.type !== originalType) {
            input.type = originalType;
        }
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function toPythonLiteral(value) {
        if (Array.isArray(value)) {
            return `[${value.map(toPythonLiteral).join(', ')}]`;
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
            return String(value);
        }
        if (typeof value === 'boolean') {
            return value ? 'True' : 'False';
        }
        if (value === null || value === undefined) {
            return 'None';
        }
        return `'${String(value).replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
    }

    function parseStructured(raw) {
        const trimmed = String(raw ?? '').trim();
        if (trimmed === '') {
            return '';
        }
        const numeric = Number(trimmed);
        if (Number.isFinite(numeric)) {
            return numeric;
        }
        try {
            return JSON.parse(trimmed);
        } catch (_e) {
            // ignore
        }
        try {
            return JSON.parse(trimmed.replace(/'/g, '"'));
        } catch (_e) {
            // ignore
        }
        return trimmed;
    }

    function _firstCandidateLiteralFromGridSpec(raw) {
        const text = String(raw ?? '').trim();
        if (!text.toLowerCase().startsWith('grid=')) {
            return text;
        }
        const spec = text.slice(5).trim();
        if (!spec) {
            return text;
        }
        if (spec.includes(':') && !/[,\[\]\{\}\(\)]/.test(spec)) {
            const parts = spec.split(':').map(part => String(part).trim());
            if (parts.length === 3) {
                const start = Number(parts[0]);
                if (Number.isFinite(start)) {
                    return String(start);
                }
            }
            return text;
        }
        const parsed = parseStructured(spec);
        if (Array.isArray(parsed) && parsed.length > 0) {
            return toPythonLiteral(parsed[0]);
        }
        return text;
    }

    function parseArrayInput(paramName, fallbackValue) {
        const input = document.querySelector(`.param-input[data-param="${paramName}"]`);
        const parsed = parseStructured(input ? input.value : fallbackValue);
        return Array.isArray(parsed) ? parsed : [];
    }

    function getLiveSingleArrayValues(paramName) {
        const input = document.querySelector(`.param-input[data-param="${paramName}"]`);
        if (!input || !input.parentElement) {
            return [];
        }
        const controls = input.parentElement.querySelector('.single-array-controls');
        if (!controls) {
            return [];
        }
        const rows = Array.from(controls.querySelectorAll('[data-single-array-row="1"]'))
            .filter(row => (row.dataset.paramName || '') === paramName);
        if (rows.length === 0) {
            return [];
        }
        return rows.map(row => String((row.querySelector('input')?.value ?? '').trim()));
    }

    function getPopulationNames() {
        const liveValues = getLiveSingleArrayValues('X');
        if (liveValues.length > 0) {
            return liveValues.map(value => String(value));
        }
        return parseArrayInput('X', ncpiPresets.X).map(value => String(value));
    }

    function getAreaNames() {
        const liveValues = getLiveSingleArrayValues('areas');
        if (liveValues.length > 0) {
            return liveValues.map(value => String(value));
        }
        return parseArrayInput('areas', ncpiPresets.areas).map(value => String(value));
    }

    function getParamInput(paramName) {
        return document.querySelector(`.param-input[data-param="${paramName}"]`);
    }

    function localParamState(paramName) {
        const input = getParamInput(paramName);
        if (!input || !input.parentElement) {
            return null;
        }
        const singleControls = input.parentElement.querySelector('.single-array-controls');
        const gridControls = input.parentElement.querySelector('.grid-parameter-controls');
        const singleRows = singleControls
            ? Array.from(singleControls.querySelectorAll('[data-single-array-row="1"] input')).map(rowInput => String(rowInput.value ?? ''))
            : [];
        const gridRows = gridControls
            ? Array.from(gridControls.querySelectorAll('[data-grid-row="1"]')).map(row => ({
                start: String(row.querySelector('[data-grid-role="start"]')?.value ?? ''),
                step: String(row.querySelector('[data-grid-role="step"]')?.value ?? ''),
                end: String(row.querySelector('[data-grid-role="end"]')?.value ?? ''),
            }))
            : [];
        return {
            inputValue: String(input.value ?? ''),
            singleRows,
            gridRows,
        };
    }

    function applyLocalParamState(paramName, state) {
        const input = getParamInput(paramName);
        if (!input || !state || !input.parentElement) {
            return;
        }
        if (typeof state.inputValue === 'string') {
            ensureInputCanCarryValue(input, state.inputValue);
            input.value = state.inputValue;
        }
        const singleControls = input.parentElement.querySelector('.single-array-controls');
        if (singleControls && Array.isArray(state.singleRows)) {
            const inputs = Array.from(singleControls.querySelectorAll('[data-single-array-row="1"] input'));
            inputs.forEach((singleInput, idx) => {
                if (idx < state.singleRows.length) {
                    singleInput.value = state.singleRows[idx];
                }
            });
        }
        const gridControls = input.parentElement.querySelector('.grid-parameter-controls');
        if (gridControls && Array.isArray(state.gridRows)) {
            const rows = Array.from(gridControls.querySelectorAll('[data-grid-row="1"]'));
            rows.forEach((row, idx) => {
                if (idx >= state.gridRows.length) {
                    return;
                }
                const values = state.gridRows[idx] || {};
                const startInput = row.querySelector('[data-grid-role="start"]');
                const stepInput = row.querySelector('[data-grid-role="step"]');
                const endInput = row.querySelector('[data-grid-role="end"]');
                if (startInput && typeof values.start === 'string') {
                    startInput.value = values.start;
                }
                if (stepInput && typeof values.step === 'string') {
                    stepInput.value = values.step;
                }
                if (endInput && typeof values.end === 'string') {
                    endInput.value = values.end;
                }
            });
        }
    }

    function cloneLocalAreaState(state) {
        if (!state || typeof state !== 'object') {
            return {};
        }
        return JSON.parse(JSON.stringify(state));
    }

    function captureAreaLocalState(areaIndex) {
        if (areaIndex < 0 || areaIndex >= areaLocalState.length) {
            return;
        }
        const snapshot = {};
        areaSpecificLocalParams.forEach(paramName => {
            snapshot[paramName] = localParamState(paramName);
        });
        areaLocalState[areaIndex] = snapshot;
    }

    function applyAreaLocalState(areaIndex) {
        if (areaIndex < 0 || areaIndex >= areaLocalState.length) {
            return;
        }
        const snapshot = areaLocalState[areaIndex] || {};
        areaSpecificLocalParams.forEach(paramName => {
            applyLocalParamState(paramName, snapshot[paramName]);
        });
    }

    function upsertHiddenFormValue(name, value) {
        if (!elements.form) {
            return;
        }
        let input = elements.form.querySelector(`input[type="hidden"][name="${name}"]`);
        if (!input) {
            input = document.createElement('input');
            input.type = 'hidden';
            input.name = name;
            elements.form.appendChild(input);
        }
        input.value = String(value ?? '');
    }

    function refreshAreaSelectorOptions() {
        if (!elements.areaSelector) {
            return;
        }
        const areaNames = getAreaNames();
        if (areaNames.length === 0) {
            return;
        }
        const options = Array.from(elements.areaSelector.options);
        options.forEach((option, index) => {
            option.textContent = areaNames[index] || `area_${index}`;
        });
        const selectedAreaName = areaNames[Math.max(0, Math.min(activeAreaIndex, areaNames.length - 1))]
            || `area_${activeAreaIndex}`;
        [
            document.getElementById('network-selected-area'),
            document.getElementById('recurrent-selected-area'),
        ].forEach(indicator => {
            if (indicator) {
                indicator.textContent = `Selected area: ${selectedAreaName}`;
            }
        });
    }

    function initializeAreaLocalState() {
        const areaNames = getAreaNames();
        const totalAreas = Math.max(1, areaNames.length || 4);
        const baseSnapshot = {};
        areaSpecificLocalParams.forEach(paramName => {
            baseSnapshot[paramName] = localParamState(paramName);
        });
        areaLocalState = Array.from({ length: totalAreas }, () => cloneLocalAreaState(baseSnapshot));
        activeAreaIndex = 0;
    }

    function localStateFromRawParamValue(paramName, rawValue) {
        const state = cloneLocalAreaState(localParamState(paramName) || {});
        state.inputValue = String(rawValue ?? '');

        const parsed = parseStructured(rawValue);
        if (Array.isArray(parsed)) {
            state.singleRows = collectLeaves(parsed).map(leaf => String(leaf.value ?? ''));
        }

        const simpleGridRange = parseSimpleGridRangeSpec(rawValue);
        if (simpleGridRange && Array.isArray(state.gridRows) && state.gridRows.length > 0) {
            state.gridRows = state.gridRows.map((row, index) => {
                if (index !== 0) {
                    return row;
                }
                return {
                    start: simpleGridRange.start,
                    step: simpleGridRange.step,
                    end: simpleGridRange.stop,
                };
            });
        }

        return state;
    }

    function restoreAreaLocalStateFromForm() {
        if (!Array.isArray(areaLocalState) || areaLocalState.length === 0) {
            return;
        }
        let restoredAny = false;
        for (let areaIndex = 0; areaIndex < areaLocalState.length; areaIndex += 1) {
            const snapshot = cloneLocalAreaState(areaLocalState[areaIndex] || {});
            areaSpecificLocalParams.forEach(paramName => {
                const formKey = `area_${areaIndex}.${paramName}`;
                if (!hasRestoredValue(formKey)) {
                    return;
                }
                snapshot[paramName] = localStateFromRawParamValue(paramName, restoreForm[formKey]);
                restoredAny = true;
            });
            areaLocalState[areaIndex] = snapshot;
        }
        if (restoredAny) {
            applyAreaLocalState(activeAreaIndex);
        }
    }

    function indexedName(names, index, fallbackPrefix) {
        if (index >= 0 && index < names.length) {
            return names[index];
        }
        return `${fallbackPrefix}[${index}]`;
    }

    function populationName(index) {
        return indexedName(getPopulationNames(), index, 'pop');
    }

    function areaName(index) {
        return indexedName(getAreaNames(), index, 'area');
    }

    function areaPopulationName(flatIndex) {
        const populations = getPopulationNames();
        const popCount = populations.length;
        if (popCount <= 0) {
            return `node[${flatIndex}]`;
        }
        const areaIdx = Math.floor(flatIndex / popCount);
        const popIdx = ((flatIndex % popCount) + popCount) % popCount;
        return `${areaName(areaIdx)}.${populationName(popIdx)}`;
    }

    function matrixShape(value) {
        if (!Array.isArray(value) || value.length === 0 || !Array.isArray(value[0])) {
            return null;
        }
        const cols = value[0].length;
        for (let row = 0; row < value.length; row += 1) {
            if (!Array.isArray(value[row]) || value[row].length !== cols) {
                return null;
            }
        }
        return [value.length, cols];
    }

    function formatPath(path) {
        return path.map(idx => `[${idx}]`).join('');
    }

    function _hexToRgba(hex, alpha) {
        const raw = String(hex || '').replace('#', '').trim();
        if (raw.length !== 6) {
            return `rgba(100, 116, 139, ${alpha})`;
        }
        const r = Number.parseInt(raw.slice(0, 2), 16);
        const g = Number.parseInt(raw.slice(2, 4), 16);
        const b = Number.parseInt(raw.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function _interAreaSourceAreaIndex(paramName, path) {
        if (!String(paramName || '').startsWith('inter_area.')) {
            return null;
        }
        if (!Array.isArray(path) || path.length !== 2) {
            return null;
        }
        const populations = getPopulationNames();
        const popCount = populations.length;
        const areaCount = getAreaNames().length;
        const totalNodes = popCount * areaCount;
        if (popCount <= 0 || totalNodes <= 0) {
            return null;
        }
        if (path[0] < 0 || path[0] >= totalNodes || path[1] < 0 || path[1] >= totalNodes) {
            return null;
        }
        return Math.floor(path[0] / popCount);
    }

    function _interAreaSourceAreaTitle(areaIndex) {
        if (!Number.isInteger(areaIndex) || areaIndex < 0) {
            return 'unknown';
        }
        const names = getAreaNames();
        return names[areaIndex] || `area_${areaIndex}`;
    }

    function _interAreaGroupTitleText(areaIndex, rowCount) {
        const count = Number(rowCount || 0);
        const label = count === 1 ? 'parameter' : 'parameters';
        return `Source area: ${_interAreaSourceAreaTitle(areaIndex)} (${count} ${label})`;
    }

    function _refreshInterAreaGroupTitles() {
        document.querySelectorAll('[data-inter-area-group-title="1"]').forEach(title => {
            const areaIndex = Number.parseInt(String(title.dataset.sourceAreaIndex ?? '-1'), 10);
            const rowCount = Number.parseInt(String(title.dataset.rowCount ?? '0'), 10);
            title.textContent = _interAreaGroupTitleText(areaIndex, rowCount);
        });
    }

    function _decorateInterAreaControlsBySourceArea(controls, paramName, rowSelector) {
        if (!controls || !String(paramName || '').startsWith('inter_area.')) {
            return;
        }
        const rows = Array.from(controls.querySelectorAll(rowSelector));
        if (rows.length === 0) {
            return;
        }
        const groups = new Map();
        const order = [];
        rows.forEach(row => {
            const path = JSON.parse(row.dataset.path || '[]');
            const sourceAreaIdx = _interAreaSourceAreaIndex(paramName, path);
            const key = Number.isInteger(sourceAreaIdx) ? sourceAreaIdx : -1;
            if (!groups.has(key)) {
                groups.set(key, []);
                order.push(key);
            }
            groups.get(key).push(row);
        });

        controls.replaceChildren();
        order.forEach(sourceAreaIdx => {
            const groupRows = groups.get(sourceAreaIdx) || [];
            if (groupRows.length === 0) {
                return;
            }
            const group = document.createElement('div');
            group.className = 'rounded-lg border border-slate-200 bg-white/80 p-2 dark:border-slate-700 dark:bg-[#111729]';

            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'flex w-full items-center justify-between gap-2 rounded-md px-2 py-1 text-left text-xs font-semibold text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800';
            toggle.setAttribute('aria-expanded', 'false');

            const title = document.createElement('span');
            title.dataset.interAreaGroupTitle = '1';
            title.dataset.sourceAreaIndex = String(sourceAreaIdx);
            title.dataset.rowCount = String(groupRows.length);
            title.textContent = _interAreaGroupTitleText(sourceAreaIdx, groupRows.length);

            const caret = document.createElement('span');
            caret.textContent = '+';
            caret.className = 'text-slate-500 dark:text-slate-400';

            toggle.appendChild(title);
            toggle.appendChild(caret);

            const body = document.createElement('div');
            body.className = 'hidden mt-2 flex flex-col gap-2';
            body.dataset.interAreaGroupBody = '1';
            groupRows.forEach(row => {
                body.appendChild(row);
            });

            toggle.addEventListener('click', () => {
                const willOpen = body.classList.contains('hidden');
                body.classList.toggle('hidden', !willOpen);
                toggle.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
                caret.textContent = willOpen ? '-' : '+';
            });

            group.appendChild(toggle);
            group.appendChild(body);
            controls.appendChild(group);
        });
    }

    function _isIgnoredSameAreaInterAreaPath(paramName, path) {
        if (!String(paramName || '').startsWith('inter_area.')) {
            return false;
        }
        if (!Array.isArray(path) || path.length !== 2) {
            return false;
        }
        const populations = getPopulationNames();
        const popCount = populations.length;
        const areaCount = getAreaNames().length;
        const totalNodes = popCount * areaCount;
        if (popCount <= 0 || totalNodes <= 0) {
            return false;
        }
        if (path[0] < 0 || path[0] >= totalNodes || path[1] < 0 || path[1] >= totalNodes) {
            return false;
        }
        const srcArea = Math.floor(path[0] / popCount);
        const tgtArea = Math.floor(path[1] / popCount);
        return srcArea === tgtArea;
    }

    function _styleLeafLabelRow(row, paramName, path) {
        if (!row) {
            return;
        }
        const labelNode = row.querySelector('[data-param-leaf-label="1"]');
        row.style.borderLeft = '';
        row.style.paddingLeft = '';
        row.style.backgroundColor = '';
        if (labelNode) {
            labelNode.style.color = '';
            labelNode.style.fontWeight = '';
        }
        const srcAreaIdx = _interAreaSourceAreaIndex(paramName, path);
        if (srcAreaIdx === null) {
            return;
        }
        const color = interAreaSourceAreaColors[srcAreaIdx % interAreaSourceAreaColors.length];
        row.style.borderLeft = `3px solid ${color}`;
        row.style.paddingLeft = '0.5rem';
        row.style.backgroundColor = _hexToRgba(color, 0.07);
        if (labelNode) {
            labelNode.style.color = color;
            labelNode.style.fontWeight = '600';
        }
    }

    function describeLeafLabel(paramName, path, index, isArrayKind) {
        if (!Array.isArray(path) || path.length === 0) {
            return isArrayKind ? `Component ${index + 1}` : 'Value';
        }
        if (path.length === 1 && populationIndexedVectorParams.has(paramName)) {
            return `population: ${populationName(path[0])}`;
        }
        if (path.length === 1 && areaIndexedVectorParams.has(paramName)) {
            return `area: ${areaName(path[0])}`;
        }
        if (path.length === 2 && sourceTargetMatrixParams.has(paramName)) {
            if (paramName.startsWith('inter_area.')) {
                const populations = getPopulationNames();
                const popCount = populations.length;
                const areaCount = getAreaNames().length;
                const totalNodes = popCount * areaCount;
                if (popCount > 0 && totalNodes > 0 && path[0] < totalNodes && path[1] < totalNodes) {
                    return `source: ${areaPopulationName(path[0])} -> target: ${areaPopulationName(path[1])}`;
                }
                if (popCount > 0) {
                    return `source: ${populationName(path[0])} -> target: ${populationName(path[1])}`;
                }
            }
            return `source: ${populationName(path[0])} -> target: ${populationName(path[1])}`;
        }
        if (path.length === 2 && paramName === 'tau_syn_YX') {
            const synType = path[1] === 0
                ? 'excitatory input'
                : (path[1] === 1 ? 'inhibitory input' : `input[${path[1]}]`);
            return `target: ${populationName(path[0])}, ${synType}`;
        }
        return formatPath(path);
    }

    function refreshLeafLabels() {
        document.querySelectorAll('[data-single-array-row="1"]').forEach(row => {
            const paramName = row.dataset.paramName || '';
            const path = JSON.parse(row.dataset.path || '[]');
            const labelNode = row.querySelector('[data-param-leaf-label="1"]');
            if (labelNode) {
                labelNode.textContent = describeLeafLabel(paramName, path, 0, true);
            }
            _styleLeafLabelRow(row, paramName, path);
        });
        document.querySelectorAll('[data-grid-row="1"]').forEach(row => {
            const paramName = row.dataset.paramName || '';
            const path = JSON.parse(row.dataset.path || '[]');
            const labelNode = row.querySelector('[data-param-leaf-label="1"]');
            if (labelNode) {
                labelNode.textContent = describeLeafLabel(paramName, path, 0, true);
            }
            _styleLeafLabelRow(row, paramName, path);
        });
        _refreshInterAreaGroupTitles();
    }

    function collectLeaves(value, path = []) {
        if (Array.isArray(value)) {
            let out = [];
            value.forEach((entry, idx) => {
                out = out.concat(collectLeaves(entry, path.concat(idx)));
            });
            return out;
        }
        return [{ path, value }];
    }

    function setAtPath(target, path, value) {
        let ref = target;
        for (let i = 0; i < path.length - 1; i += 1) {
            ref = ref[path[i]];
        }
        ref[path[path.length - 1]] = value;
    }

    function buildNumericRange(start, end, step) {
        if (step === 0) {
            return start === end ? [start] : null;
        }
        if ((end > start && step < 0) || (end < start && step > 0)) {
            return null;
        }
        const eps = Math.abs(step) * 1e-9 + 1e-12;
        const out = [];
        let current = start;
        if (step > 0) {
            while (current <= end + eps) {
                out.push(Number(current.toFixed(12)));
                current += step;
            }
        } else {
            while (current >= end - eps) {
                out.push(Number(current.toFixed(12)));
                current += step;
            }
        }
        return out.length > 0 ? out : null;
    }

    function cartesianProduct(arrays) {
        return arrays.reduce((acc, arr) => {
            const next = [];
            acc.forEach(prefix => {
                arr.forEach(item => {
                    next.push(prefix.concat([item]));
                });
            });
            return next;
        }, [[]]);
    }

    function leafPathKey(path) {
        if (!Array.isArray(path) || path.length === 0) {
            return '';
        }
        return path.map(part => String(part)).join('.');
    }

    function makeJointToken(paramName, path) {
        const key = leafPathKey(path);
        return key ? `${paramName}::${key}` : paramName;
    }

    function parseJointToken(token) {
        const raw = String(token || '').trim();
        if (!raw) {
            return { paramName: '', pathKey: '' };
        }
        const sepIndex = raw.indexOf('::');
        if (sepIndex < 0) {
            return { paramName: raw, pathKey: '' };
        }
        const paramName = raw.slice(0, sepIndex).trim();
        const pathKey = raw.slice(sepIndex + 2).trim();
        return { paramName, pathKey };
    }

    function parseAreaQualifiedLocalParam(paramName) {
        const text = String(paramName || '').trim();
        const match = text.match(/^area_(\d+)\.(.+)$/);
        if (!match) {
            return null;
        }
        const areaIndex = Number(match[1]);
        const localParam = String(match[2] || '').trim();
        if (!Number.isInteger(areaIndex) || areaIndex < 0 || !areaSpecificLocalParamSet.has(localParam)) {
            return null;
        }
        return { areaIndex, localParam };
    }

    function qualifyAreaLocalParamName(paramName, areaIndex) {
        const raw = String(paramName || '').trim();
        if (!raw) {
            return raw;
        }
        if (parseAreaQualifiedLocalParam(raw)) {
            return raw;
        }
        if (!areaSpecificLocalParamSet.has(raw)) {
            return raw;
        }
        const idx = Number.isInteger(areaIndex) && areaIndex >= 0 ? areaIndex : 0;
        return `area_${idx}.${raw}`;
    }

    function qualifyAreaToken(token, areaIndex) {
        const parsed = parseJointToken(token);
        const qualifiedParam = qualifyAreaLocalParamName(parsed.paramName, areaIndex);
        if (!parsed.pathKey) {
            return qualifiedParam;
        }
        return `${qualifiedParam}::${parsed.pathKey}`;
    }

    function sweepableParamNames() {
        if (!elements.form) {
            return [];
        }
        return Array.from(elements.form.querySelectorAll('.param-input'))
            .map(input => input.dataset.param || input.name || '')
            .filter(paramName => paramName && !simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName));
    }

    function sweepableParamOptions() {
        if (!elements.form) {
            return [];
        }
        const seen = new Set();
        const options = [];
        const displayLabel = (paramName) => {
            const unit = jointParamUnits[paramName];
            return unit ? `${paramName} (${unit})` : paramName;
        };
        const inputs = Array.from(elements.form.querySelectorAll('.param-input'))
            .filter(input => {
                const paramName = input.dataset.param || input.name || '';
                return paramName && !simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName);
            });

        inputs.forEach(input => {
            const paramName = input.dataset.param || input.name || '';
            const controls = input.parentElement
                ? input.parentElement.querySelector('.grid-parameter-controls')
                : null;
            const isArrayKind = controls && (controls.dataset.gridKind || 'scalar') === 'array';
            const rows = controls
                ? Array.from(controls.querySelectorAll('[data-grid-row="1"]'))
                : [];

            if (!isArrayKind || rows.length <= 1) {
                if (!seen.has(paramName)) {
                    seen.add(paramName);
                    options.push({ value: paramName, label: displayLabel(paramName) });
                }
                return;
            }

            rows.forEach((row, index) => {
                const path = JSON.parse(row.dataset.path || '[]');
                const token = makeJointToken(paramName, path);
                if (seen.has(token)) {
                    return;
                }
                seen.add(token);
                const leafLabel = describeLeafLabel(paramName, path, index, true);
                options.push({ value: token, label: `${displayLabel(paramName)} - ${leafLabel}` });
            });
        });

        return options;
    }

    function jointGroupCards() {
        if (!elements.jointGroupsContainer) {
            return [];
        }
        return Array.from(elements.jointGroupsContainer.querySelectorAll('[data-joint-group-card="1"]'));
    }

    function readJointGroupSelection(card) {
        const select = card?.querySelector('[data-joint-group-select="1"]');
        if (!select) {
            return [];
        }
        return Array.from(select.selectedOptions).map(option => option.value);
    }

    function syncJointGroupsInput(groups) {
        if (!elements.jointGroupsInput) {
            return;
        }
        elements.jointGroupsInput.value = JSON.stringify(groups);
    }

    function refreshJointGroupsUi(changedSelect = null) {
        const cards = jointGroupCards();
        const selections = cards.map(card => readJointGroupSelection(card));
        const globallySelected = new Set(selections.flat());
        const options = sweepableParamOptions();
        const optionLabelByValue = new Map(options.map(option => [option.value, option.label]));

        cards.forEach((card, index) => {
            const title = card.querySelector('[data-joint-group-title="1"]');
            if (title) {
                const areaNames = getAreaNames();
                const areaIndexRaw = Number(String(card?.dataset?.areaIndex ?? 0));
                const areaIndex = Number.isInteger(areaIndexRaw) && areaIndexRaw >= 0 ? areaIndexRaw : 0;
                const areaLabel = areaNames[areaIndex] || `area_${areaIndex}`;
                title.textContent = `Group ${index + 1} (${areaLabel})`;
            }
            const summary = card.querySelector('[data-joint-group-summary="1"]');
            if (summary) {
                const selected = selections[index] || [];
                if (selected.length === 0) {
                    summary.textContent = 'Select at least two parameters';
                } else {
                    const labels = selected.map(value => optionLabelByValue.get(value) || value);
                    const preview = labels.slice(0, 6);
                    const suffix = labels.length > preview.length ? `, +${labels.length - preview.length} more` : '';
                    summary.textContent = `Selected: ${preview.join(', ')}${suffix}`;
                }
            }

            const select = card.querySelector('[data-joint-group-select="1"]');
            if (!select) {
                return;
            }
            const previousScrollTop = select.scrollTop;
            const shouldPreserveInPlace = select === changedSelect || select === document.activeElement;
            const ownSelection = new Set(selections[index]);
            if (shouldPreserveInPlace) {
                const existingOptionValues = new Set(Array.from(select.options).map(node => node.value));
                Array.from(select.options).forEach(node => {
                    node.disabled = globallySelected.has(node.value) && !ownSelection.has(node.value);
                    node.selected = ownSelection.has(node.value);
                });
                options.forEach(option => {
                    if (existingOptionValues.has(option.value)) {
                        return;
                    }
                    const node = document.createElement('option');
                    node.value = option.value;
                    node.textContent = option.label;
                    node.selected = ownSelection.has(option.value);
                    node.disabled = globallySelected.has(option.value) && !ownSelection.has(option.value);
                    select.appendChild(node);
                });
            } else {
                const fragment = document.createDocumentFragment();
                options.forEach(option => {
                    const node = document.createElement('option');
                    node.value = option.value;
                    node.textContent = option.label;
                    node.selected = ownSelection.has(option.value);
                    node.disabled = globallySelected.has(option.value) && !ownSelection.has(option.value);
                    fragment.appendChild(node);
                });
                select.replaceChildren(fragment);
            }
            select.scrollTop = previousScrollTop;
            window.requestAnimationFrame(() => {
                select.scrollTop = previousScrollTop;
            });
        });

        syncJointGroupsInput(
            selections.filter(group => group.length > 0)
        );
    }

    function addJointGroupCard(selectedValues = []) {
        if (!elements.jointGroupsContainer) {
            return null;
        }
        const card = document.createElement('div');
        card.dataset.jointGroupCard = '1';
        card.dataset.areaIndex = String(activeAreaIndex);
        card.className = 'rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-[#141a2d]';
        card.innerHTML = `
            <div class="flex items-start justify-between gap-3">
                <div class="flex flex-col gap-1">
                    <h4 class="text-sm font-semibold text-slate-900 dark:text-white" data-joint-group-title="1"></h4>
                    <p class="text-xs text-slate-500 dark:text-slate-400" data-joint-group-summary="1"></p>
                </div>
                <button
                    type="button"
                    class="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800"
                    data-remove-joint-group="1"
                >
                    Remove
                </button>
            </div>
            <label class="mt-3 flex flex-col gap-2" data-field-help-skip="1">
                <span class="text-xs font-medium text-slate-700 dark:text-slate-300">Grouped parameters</span>
                <select
                    multiple
                    size="8"
                    class="form-input min-h-[12rem] rounded-lg border-slate-300 bg-background-light text-sm text-slate-900 dark:border-slate-700 dark:bg-[#191e33] dark:text-white"
                    data-joint-group-select="1"
                ></select>
            </label>
            <p class="mt-2 text-xs text-slate-500 dark:text-slate-400">Use Ctrl/Cmd-click to select multiple parameters.</p>
        `;
        elements.jointGroupsContainer.appendChild(card);

        const select = card.querySelector('[data-joint-group-select="1"]');
        if (select) {
            const selectedSet = new Set(selectedValues);
            sweepableParamOptions().forEach(option => {
                const node = document.createElement('option');
                node.value = option.value;
                node.textContent = option.label;
                node.selected = selectedSet.has(option.value);
                select.appendChild(node);
            });
            select.addEventListener('change', () => {
                refreshJointGroupsUi(select);
            });
        }
        const removeButton = card.querySelector('[data-remove-joint-group="1"]');
        if (removeButton) {
            removeButton.addEventListener('click', () => {
                card.remove();
                refreshJointGroupsUi();
            });
        }
        refreshJointGroupsUi();
        return card;
    }

    function normalizeJointGroupsInput() {
        const cards = jointGroupCards();
        const groupEntries = cards
            .map(card => ({ card, group: readJointGroupSelection(card) }))
            .filter(entry => entry.group.length > 0);
        const groups = groupEntries.map(entry => entry.group);

        const seenGroupTokens = new Set();
        const seenBackendParams = new Set();
        const backendGroups = [];
        const countGroups = [];
        const leafGroupsByParam = new Map();

        for (let index = 0; index < groups.length; index += 1) {
            const group = groups[index];
            const card = groupEntries[index]?.card || null;
            const groupAreaIndexRaw = Number(String(card?.dataset?.areaIndex ?? activeAreaIndex));
            const groupAreaIndex = Number.isInteger(groupAreaIndexRaw) && groupAreaIndexRaw >= 0
                ? groupAreaIndexRaw
                : Math.max(0, activeAreaIndex);
            if (group.length < 2) {
                return { ok: false, message: `Joint sweep group ${index + 1} must contain at least two parameters.` };
            }

            const parsedGroup = group.map(parseJointToken);
            const backendGroup = [];
            const backendSeenInGroup = new Set();
            const localLeafPathsByParam = new Map();
            const countGroup = [];

            for (let tokenIndex = 0; tokenIndex < group.length; tokenIndex += 1) {
                const parsed = parsedGroup[tokenIndex];
                const paramName = String(parsed.paramName || '').trim();
                const pathKey = String(parsed.pathKey || '').trim();
                const qualifiedParamName = qualifyAreaLocalParamName(paramName, groupAreaIndex);
                const qualifiedToken = pathKey
                    ? `${qualifiedParamName}::${pathKey}`
                    : qualifiedParamName;

                if (!qualifiedParamName) {
                    return { ok: false, message: `Joint sweep group ${index + 1} contains an invalid parameter target.` };
                }
                if (seenGroupTokens.has(qualifiedToken)) {
                    return { ok: false, message: `Parameter "${qualifiedToken}" cannot belong to more than one joint sweep group.` };
                }
                seenGroupTokens.add(qualifiedToken);
                countGroup.push(qualifiedToken);

                if (!backendSeenInGroup.has(qualifiedParamName)) {
                    backendSeenInGroup.add(qualifiedParamName);
                    backendGroup.push(qualifiedParamName);
                }

                if (!pathKey) {
                    continue;
                }
                if (!localLeafPathsByParam.has(qualifiedParamName)) {
                    localLeafPathsByParam.set(qualifiedParamName, new Set());
                }
                const leafPathSet = localLeafPathsByParam.get(qualifiedParamName);
                if (leafPathSet.has(pathKey)) {
                    return { ok: false, message: `Joint sweep group ${index + 1} contains duplicate leaf targets.` };
                }
                leafPathSet.add(pathKey);
            }

            for (const [paramName, pathSet] of localLeafPathsByParam.entries()) {
                if (!leafGroupsByParam.has(paramName)) {
                    leafGroupsByParam.set(paramName, []);
                }
                leafGroupsByParam.get(paramName).push(Array.from(pathSet));
            }

            if (backendGroup.length >= 2) {
                for (const paramName of backendGroup) {
                    if (seenBackendParams.has(paramName)) {
                        return { ok: false, message: `Parameter "${paramName}" cannot belong to more than one joint sweep group.` };
                    }
                }
                backendGroup.forEach(paramName => seenBackendParams.add(paramName));
                countGroups.push(countGroup);
                backendGroups.push(backendGroup);
            }
        }

        syncJointGroupsInput(backendGroups);
        return { ok: true, groups, backendGroups, leafGroupsByParam, countGroups };
    }

    function addGridControls(input, preset) {
        const initial = (preset !== undefined && preset !== null) ? preset : (input.value || '');
        const simpleGridRange = parseSimpleGridRangeSpec(initial);
        const parsed = simpleGridRange ? Number(simpleGridRange.start) : parseStructured(initial);
        const controls = document.createElement('div');
        controls.className = 'grid-parameter-controls hidden mt-2 flex flex-col gap-2';
        const paramName = input.dataset.param || input.name || '';
        controls.dataset.paramName = paramName;

        const baseClass = 'grid-param-input form-input rounded-lg border-slate-300 dark:border-slate-700 bg-background-light dark:bg-[#191e33] text-slate-900 dark:text-white';
        const leaves = Array.isArray(parsed) ? collectLeaves(parsed) : [{ path: [], value: parsed }];
        controls.dataset.gridKind = Array.isArray(parsed) ? 'array' : 'scalar';
        controls.dataset.baseValue = JSON.stringify(parsed);

        leaves.forEach((leaf, index) => {
            if (_isIgnoredSameAreaInterAreaPath(paramName, leaf.path)) {
                return;
            }
            const label = describeLeafLabel(paramName, leaf.path, index, controls.dataset.gridKind === 'array');
            const startValue = String(leaf.value ?? '');
            const stepValue = simpleGridRange && leaves.length === 1 ? simpleGridRange.step : '0';
            const endValue = simpleGridRange && leaves.length === 1 ? simpleGridRange.stop : startValue;
            controls.insertAdjacentHTML('beforeend', `
                <div class="grid grid-cols-1 md:grid-cols-4 gap-2" data-grid-row="1" data-path="${escapeHtml(JSON.stringify(leaf.path))}" data-param-name="${escapeHtml(paramName)}" data-template-type="${typeof leaf.value}">
                    <div class="text-xs text-slate-600 dark:text-slate-300 md:pt-2" data-param-leaf-label="1">${escapeHtml(label)}</div>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">Start</span>
                        <input type="text" data-grid-role="start" class="${baseClass}" value="${escapeHtml(startValue)}" />
                    </label>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">Step</span>
                        <input type="text" data-grid-role="step" class="${baseClass}" value="${escapeHtml(stepValue)}" />
                    </label>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">End</span>
                        <input type="text" data-grid-role="end" class="${baseClass}" value="${escapeHtml(endValue)}" />
                    </label>
                </div>
            `);
        });

        _decorateInterAreaControlsBySourceArea(controls, paramName, '[data-grid-row="1"]');
        input.insertAdjacentElement('afterend', controls);
    }

    function addSingleArrayControls(input, preset) {
        const initial = (preset !== undefined && preset !== null) ? preset : (input.value || '');
        const parsed = parseStructured(initial);
        if (!Array.isArray(parsed)) {
            return;
        }
        const paramName = input.dataset.param || input.name || '';

        const controls = document.createElement('div');
        controls.className = String(paramName).startsWith('inter_area.')
            ? 'single-array-controls mt-2 grid grid-cols-1 gap-2'
            : 'single-array-controls mt-2 grid grid-cols-1 md:grid-cols-2 gap-2';
        controls.dataset.baseValue = JSON.stringify(parsed);
        controls.dataset.paramName = paramName;

        const leaves = collectLeaves(parsed);
        const baseClass = 'single-array-input form-input rounded-lg border-slate-300 dark:border-slate-700 bg-background-light dark:bg-[#191e33] text-slate-900 dark:text-white';

        leaves.forEach((leaf, index) => {
            if (_isIgnoredSameAreaInterAreaPath(paramName, leaf.path)) {
                return;
            }
            const label = describeLeafLabel(paramName, leaf.path, index, true);
            controls.insertAdjacentHTML('beforeend', `
                <label class="flex flex-col gap-1" data-single-array-row="1" data-path="${escapeHtml(JSON.stringify(leaf.path))}" data-param-name="${escapeHtml(paramName)}">
                    <span class="text-xs text-slate-600 dark:text-slate-300" data-param-leaf-label="1">${escapeHtml(label)}</span>
                    <input type="text" class="${baseClass}" value="${escapeHtml(String(leaf.value ?? ''))}" />
                </label>
            `);
        });

        _decorateInterAreaControlsBySourceArea(controls, paramName, '[data-single-array-row="1"]');
        input.insertAdjacentElement('afterend', controls);
    }

    function clearParamDynamicControls(input) {
        if (!input || !input.parentElement) {
            return;
        }
        const single = input.parentElement.querySelector('.single-array-controls');
        if (single) {
            single.remove();
        }
        const grid = input.parentElement.querySelector('.grid-parameter-controls');
        if (grid) {
            grid.remove();
        }
    }

    function rebuildParamControls(input) {
        if (!input) {
            return;
        }
        const paramName = input.dataset.param || input.name || '';
        clearParamDynamicControls(input);
        addSingleArrayControls(input, input.value);
        if (!simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName)) {
            addGridControls(input, input.value);
        }
    }

    function setStructuredParamValue(paramName, value) {
        const input = getParamInput(paramName);
        if (!input) {
            return false;
        }
        const literal = toPythonLiteral(value);
        ensureInputCanCarryValue(input, literal);
        input.value = literal;
        rebuildParamControls(input);
        return true;
    }

    function buildFullInterAreaMatrixFromPopulationMatrix(popMatrix, popCount, areaCount) {
        const shape = matrixShape(popMatrix);
        if (!shape || shape[0] !== popCount || shape[1] !== popCount) {
            return null;
        }
        const totalNodes = popCount * areaCount;
        const fullMatrix = Array.from({ length: totalNodes }, () => Array(totalNodes).fill(0));
        for (let src = 0; src < totalNodes; src += 1) {
            for (let tgt = 0; tgt < totalNodes; tgt += 1) {
                fullMatrix[src][tgt] = popMatrix[src % popCount][tgt % popCount];
            }
        }
        return fullMatrix;
    }

    function updateInterAreaModeStatus() {
        if (!elements.interAreaModeStatus) {
            return;
        }
        elements.interAreaModeStatus.textContent = 'Inter-area connectivity is always shown as full 8x8 area-population matrices (frontal.E, frontal.I, ..., occipital.I). Same-area entries are ignored by the simulator (these connections are configured in the Recurrent connectivity section).';
    }

    function ensureFullInterAreaMatrices() {
        const popCount = getPopulationNames().length;
        const areaCount = getAreaNames().length;
        const totalNodes = popCount * areaCount;
        if (popCount <= 0 || areaCount <= 0) {
            return false;
        }

        let changed = false;
        interAreaParamNames.forEach(paramName => {
            const input = getParamInput(paramName);
            if (!input) {
                return;
            }
            const parsed = parseStructured(input.value);
            const shape = matrixShape(parsed);
            if (shape && shape[0] === totalNodes && shape[1] === totalNodes) {
                return;
            }
            const nextValue = buildFullInterAreaMatrixFromPopulationMatrix(parsed, popCount, areaCount);
            if (!nextValue) {
                return;
            }
            changed = true;
            setStructuredParamValue(paramName, nextValue);
        });

        if (changed) {
            refreshLeafLabels();
            refreshJointGroupsUi();
            updateModeUi();
        }
        updateInterAreaModeStatus();
        return changed;
    }

    function validateInterAreaMatricesAreFull() {
        const popCount = getPopulationNames().length;
        const areaCount = getAreaNames().length;
        const totalNodes = popCount * areaCount;

        function validateShapeOrGridShape(rawValue) {
            const text = String(rawValue ?? '').trim();
            if (text.toLowerCase().startsWith('grid=')) {
                const spec = text.slice(5).trim();
                if (!spec) {
                    return false;
                }
                const parsed = parseStructured(spec);
                if (!Array.isArray(parsed) || parsed.length === 0) {
                    return false;
                }
                return parsed.every(candidate => {
                    const shape = matrixShape(candidate);
                    return Boolean(shape && shape[0] === totalNodes && shape[1] === totalNodes);
                });
            }
            const shape = matrixShape(parseStructured(text));
            return Boolean(shape && shape[0] === totalNodes && shape[1] === totalNodes);
        }

        for (const paramName of interAreaParamNames) {
            const input = getParamInput(paramName);
            if (!input) {
                continue;
            }
            if (!validateShapeOrGridShape(input.value)) {
                return {
                    ok: false,
                    message: `Parameter "${paramName}" must be a ${totalNodes}x${totalNodes} matrix.`,
                };
            }
        }
        return { ok: true };
    }

    function setupInterAreaMatrixUi() {
        ensureFullInterAreaMatrices();
        interAreaParamNames.forEach(paramName => {
            const input = getParamInput(paramName);
            if (!input) {
                return;
            }
            input.addEventListener('input', () => {
                updateInterAreaModeStatus();
            });
            input.addEventListener('change', () => {
                updateInterAreaModeStatus();
            });
        });
        updateInterAreaModeStatus();
    }

    function normalizeSingleArrayInput(input) {
        const controls = input.parentElement.querySelector('.single-array-controls');
        if (!controls) {
            return { ok: true };
        }

        const baseValue = parseStructured(controls.dataset.baseValue || input.value || '');
        if (!Array.isArray(baseValue)) {
            return { ok: true };
        }

        const out = JSON.parse(JSON.stringify(baseValue));
        const leaves = collectLeaves(baseValue);
        const leafByPath = new Map();
        leaves.forEach(leaf => {
            leafByPath.set(leafPathKey(leaf.path), leaf);
        });
        const rows = Array.from(controls.querySelectorAll('[data-single-array-row="1"]'));

        for (let idx = 0; idx < rows.length; idx += 1) {
            const row = rows[idx];
            const path = JSON.parse(row.dataset.path || '[]');
            const templateLeaf = leafByPath.get(leafPathKey(path));
            const template = templateLeaf ? templateLeaf.value : null;
            const raw = (row.querySelector('input')?.value ?? '').trim();

            if (typeof template === 'number') {
                const numeric = Number(raw);
                if (!Number.isFinite(numeric)) {
                    return { ok: false, message: `Parameter "${input.name}" ${path.map(i => `[${i}]`).join('')}: numeric value required.` };
                }
                setAtPath(out, path, numeric);
                continue;
            }
            setAtPath(out, path, raw);
        }

        input.value = toPythonLiteral(out);
        return { ok: true };
    }

    function normalizeGridInput(input, leafJointGroups = []) {
        const controls = input.parentElement.querySelector('.grid-parameter-controls');
        const paramName = input.dataset.param || input.name || '';
        if (!controls) {
            return { ok: true, candidateCount: 1, candidateCountsByToken: { [paramName]: 1 } };
        }

        const kind = controls.dataset.gridKind || 'scalar';
        const rows = Array.from(controls.querySelectorAll('[data-grid-row="1"]'));

        if (kind === 'scalar') {
            const row = rows[0];
            if (!row) {
                return { ok: true };
            }
            const start = (row.querySelector('[data-grid-role="start"]')?.value ?? '').trim();
            const step = (row.querySelector('[data-grid-role="step"]')?.value ?? '0').trim() || '0';
            const end = (row.querySelector('[data-grid-role="end"]')?.value ?? '').trim() || start;

            const startNum = Number(start);
            const endNum = Number(end);
            const stepNum = Number(step);
            const isNumeric = Number.isFinite(startNum) && Number.isFinite(endNum) && Number.isFinite(stepNum);
            if (isNumeric && stepNum !== 0 && startNum !== endNum) {
                const rangeSpec = `grid=${start}:${end}:${step}`;
                ensureInputCanCarryValue(input, rangeSpec);
                input.value = rangeSpec;
                const range = buildNumericRange(startNum, endNum, stepNum);
                const count = range ? range.length : 0;
                return { ok: true, candidateCount: count, candidateCountsByToken: { [paramName]: count } };
            }
            if (isNumeric && stepNum === 0 && startNum !== endNum) {
                return { ok: false, message: `Parameter "${input.name}": step cannot be 0 when start and end differ.` };
            }
            if (!isNumeric && (step !== '0' || start !== end)) {
                return { ok: false, message: `Parameter "${input.name}": non-numeric values must use start=end and step=0.` };
            }
            input.value = start;
            return { ok: true, candidateCount: 1, candidateCountsByToken: { [paramName]: 1 } };
        }

        const baseValue = parseStructured(controls.dataset.baseValue || input.value || '');
        const leaves = collectLeaves(baseValue);
        const candidatePerLeaf = [];
        const startStructure = JSON.parse(JSON.stringify(baseValue));
        const visibleLeaves = rows.map(row => {
            const path = JSON.parse(row.dataset.path || '[]');
            const pathKey = leafPathKey(path);
            const found = leaves.find(leaf => leafPathKey(leaf.path) === pathKey);
            return found ? found : { path, value: null };
        });
        const leafTokenByIndex = visibleLeaves.map(leaf => makeJointToken(paramName, leaf.path));
        const leafPathToIndex = new Map();
        visibleLeaves.forEach((leaf, idx) => {
            leafPathToIndex.set(leafPathKey(leaf.path), idx);
        });

        for (let idx = 0; idx < rows.length; idx += 1) {
            const row = rows[idx];
            const path = JSON.parse(row.dataset.path || '[]');
            const template = visibleLeaves[idx] ? visibleLeaves[idx].value : null;
            const startRaw = (row.querySelector('[data-grid-role="start"]')?.value ?? '').trim();
            const stepRaw = (row.querySelector('[data-grid-role="step"]')?.value ?? '0').trim() || '0';
            const endRaw = (row.querySelector('[data-grid-role="end"]')?.value ?? '').trim();

            if (typeof template === 'number') {
                const startNum = Number(startRaw);
                const endNum = Number(endRaw === '' ? startRaw : endRaw);
                const stepNum = Number(stepRaw);
                if (!Number.isFinite(startNum) || !Number.isFinite(endNum) || !Number.isFinite(stepNum)) {
                    return { ok: false, message: `Parameter "${input.name}" ${path.map(i => `[${i}]`).join('')}: numeric values required.` };
                }
                const range = buildNumericRange(startNum, endNum, stepNum);
                if (!range) {
                    return { ok: false, message: `Parameter "${input.name}" ${path.map(i => `[${i}]`).join('')}: invalid start/step/end.` };
                }
                setAtPath(startStructure, path, startNum);
                candidatePerLeaf.push(range);
            } else {
                const startVal = startRaw;
                const endVal = endRaw === '' ? startVal : endRaw;
                if (stepRaw !== '0' || startVal !== endVal) {
                    return { ok: false, message: `Parameter "${input.name}" ${path.map(i => `[${i}]`).join('')}: non-numeric components must stay fixed.` };
                }
                setAtPath(startStructure, path, startVal);
                candidatePerLeaf.push([startVal]);
            }
        }

        const groupedLeafIndices = new Set();
        const dimensions = [];
        for (let groupIndex = 0; groupIndex < leafJointGroups.length; groupIndex += 1) {
            const group = leafJointGroups[groupIndex] || [];
            if (!Array.isArray(group) || group.length === 0) {
                continue;
            }
            const indices = [];
            for (const pathKey of group) {
                if (!leafPathToIndex.has(pathKey)) {
                    return { ok: false, message: `Joint sweep group references an unknown leaf in "${paramName}".` };
                }
                const idx = leafPathToIndex.get(pathKey);
                if (groupedLeafIndices.has(idx)) {
                    return { ok: false, message: `Leaf ${pathKey} in "${paramName}" belongs to more than one joint sweep group.` };
                }
                groupedLeafIndices.add(idx);
                indices.push(idx);
            }
            const expected = candidatePerLeaf[indices[0]] ? candidatePerLeaf[indices[0]].length : 1;
            const mismatch = indices.find(idx => (candidatePerLeaf[idx] ? candidatePerLeaf[idx].length : 1) !== expected);
            if (mismatch !== undefined) {
                return { ok: false, message: `Joint sweep group in "${paramName}" must use the same number of candidates across grouped leaves.` };
            }
            dimensions.push({ type: 'group', indices, size: expected });
        }
        for (let idx = 0; idx < candidatePerLeaf.length; idx += 1) {
            if (groupedLeafIndices.has(idx)) {
                continue;
            }
            dimensions.push({ type: 'leaf', index: idx, size: candidatePerLeaf[idx].length });
        }

        const totalCombos = dimensions.reduce((acc, dim) => acc * dim.size, 1);
        if (totalCombos > 256) {
            return { ok: false, message: `Parameter "${input.name}" expands to ${totalCombos} combinations (max 256).` };
        }

        const candidateCountsByToken = { [paramName]: totalCombos };
        leafTokenByIndex.forEach((token, idx) => {
            candidateCountsByToken[token] = candidatePerLeaf[idx].length;
        });

        if (totalCombos === 1) {
            const singleValue = toPythonLiteral(startStructure);
            ensureInputCanCarryValue(input, singleValue);
            input.value = singleValue;
            return { ok: true, candidateCount: 1, candidateCountsByToken };
        }

        const dimensionChoices = dimensions.map(dim => Array.from({ length: dim.size }, (_, idx) => idx));
        const combos = cartesianProduct(dimensionChoices).map(choiceVector => {
            const out = JSON.parse(JSON.stringify(startStructure));
            choiceVector.forEach((choice, dimIndex) => {
                const dim = dimensions[dimIndex];
                if (dim.type === 'group') {
                    dim.indices.forEach(leafIdx => {
                        setAtPath(out, visibleLeaves[leafIdx].path, candidatePerLeaf[leafIdx][choice]);
                    });
                    return;
                }
                setAtPath(out, visibleLeaves[dim.index].path, candidatePerLeaf[dim.index][choice]);
            });
            return out;
        });
        const comboSpec = `grid=${toPythonLiteral(combos)}`;
        ensureInputCanCarryValue(input, comboSpec);
        input.value = comboSpec;
        return { ok: true, candidateCount: totalCombos, candidateCountsByToken };
    }

    function initializeParameterInput(input) {
        const paramName = input.dataset.param;
        const preset = restoredValueOrPreset(paramName, ncpiPresets[paramName]);
        if (!input.name) {
            input.name = paramName;
        }

        if (input.type === 'checkbox') {
            input.checked = Boolean(preset);
            return;
        }

        if (preset !== undefined) {
            if (!input.dataset.originalType) {
                input.dataset.originalType = input.type || 'text';
            }
            ensureInputCanCarryValue(input, preset);
            input.value = preset;
        }

        addSingleArrayControls(input, preset);

        if (!simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName)) {
            addGridControls(input, preset);
        }
    }

    // Function to create parameter section
    function createParameterSection() {
        const clone = elements.paramSection.content.cloneNode(true);
        const section = clone.querySelector('.parameter-section');

        // Configure all parameter inputs
        const inputs = section.querySelectorAll('.param-input');
        inputs.forEach(initializeParameterInput);

        // Append the cloned section to the container
        elements.container.appendChild(section);

        return section;
    }

    function restoreSimulationFormControls() {
        const restoredMode = hasRestoredValue('sim_run_mode') ? String(restoreForm.sim_run_mode || '') : '';
        if (restoredMode) {
            const modeInput = Array.from(elements.runModeInputs).find(input => input.value === restoredMode);
            if (modeInput) {
                modeInput.checked = true;
            }
        }
        if (elements.repetitionsInput && hasRestoredValue('sim_repetitions')) {
            elements.repetitionsInput.value = String(restoreForm.sim_repetitions ?? '');
        }
        if (elements.useNumpySeedInput && Object.keys(restoreForm).length > 0) {
            elements.useNumpySeedInput.checked = hasRestoredValue('sim_use_numpy_seed');
        }
        if (elements.numpySeedInput && hasRestoredValue('sim_numpy_seed')) {
            elements.numpySeedInput.value = String(restoreForm.sim_numpy_seed ?? '');
        }
        if (elements.jointGroupsContainer && hasRestoredValue('sim_joint_groups')) {
            let groups = [];
            try {
                const rawGroups = restoreForm.sim_joint_groups;
                groups = Array.isArray(rawGroups) ? rawGroups : JSON.parse(String(rawGroups || '[]'));
            } catch (_err) {
                groups = [];
            }
            if (Array.isArray(groups)) {
                elements.jointGroupsContainer.replaceChildren();
                groups.forEach(group => {
                    if (Array.isArray(group) && group.length > 0) {
                        addJointGroupCard(group.map(value => String(value)));
                    }
                });
            }
        }
    }

    // Initialize UI
    ['areas', 'X'].forEach(paramName => {
        const sharedInput = document.querySelector(`.param-input[data-param="${paramName}"]`);
        if (sharedInput) {
            initializeParameterInput(sharedInput);
        }
    });
    createParameterSection();
    restoreSimulationFormControls();
    setupInterAreaMatrixUi();
    refreshLeafLabels();
    initializeAreaLocalState();
    restoreAreaLocalStateFromForm();
    refreshAreaSelectorOptions();
    refreshJointGroupsUi();

    if (elements.container) {
        elements.container.addEventListener('keydown', (event) => {
            const target = event.target;
            if (!(target instanceof HTMLInputElement)) {
                return;
            }
            if (target.matches('.param-input[type="number"]') &&
                (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
                event.preventDefault();
            }
        });

        elements.container.addEventListener('wheel', (event) => {
            const target = event.target;
            if (!(target instanceof HTMLInputElement)) {
                return;
            }
            if (target.matches('.param-input[type="number"]') && document.activeElement === target) {
                event.preventDefault();
            }
        }, { passive: false });
    }

    const populationsInput = document.querySelector('.param-input[data-param="X"]');
    if (populationsInput) {
        populationsInput.addEventListener('input', () => {
            ensureFullInterAreaMatrices();
            refreshLeafLabels();
            updateInterAreaModeStatus();
        });
        populationsInput.addEventListener('change', () => {
            ensureFullInterAreaMatrices();
            refreshLeafLabels();
            updateInterAreaModeStatus();
        });
    }
    const areasInput = document.querySelector('.param-input[data-param="areas"]');
    if (areasInput) {
        areasInput.addEventListener('input', () => {
            ensureFullInterAreaMatrices();
            refreshLeafLabels();
            refreshAreaSelectorOptions();
            updateInterAreaModeStatus();
        });
        areasInput.addEventListener('change', () => {
            ensureFullInterAreaMatrices();
            refreshLeafLabels();
            refreshAreaSelectorOptions();
            updateInterAreaModeStatus();
        });
    }
    if (elements.form) {
        const _syncActiveAreaOnLocalEdit = (event) => {
            const target = event.target;
            if (!(target instanceof HTMLInputElement)) {
                return;
            }
            const directParam = target.matches('.param-input')
                ? (target.dataset.param || target.name || '')
                : '';
            const dynamicRow = target.closest('[data-param-name]');
            const dynamicParam = dynamicRow ? (dynamicRow.dataset.paramName || '') : '';
            if (areaSpecificLocalParamSet.has(directParam) || areaSpecificLocalParamSet.has(dynamicParam)) {
                captureAreaLocalState(activeAreaIndex);
            }
        };
        elements.form.addEventListener('input', _syncActiveAreaOnLocalEdit);
        elements.form.addEventListener('change', _syncActiveAreaOnLocalEdit);

        const _refreshOnNameControlEdit = (event) => {
            const row = event.target && event.target.closest
                ? event.target.closest('[data-single-array-row="1"]')
                : null;
            if (!row) {
                return;
            }
            const paramName = row.dataset.paramName || '';
            if (paramName === 'X' || paramName === 'areas') {
                ensureFullInterAreaMatrices();
                refreshLeafLabels();
                if (paramName === 'areas') {
                    refreshAreaSelectorOptions();
                }
                updateInterAreaModeStatus();
            }
        };
        elements.form.addEventListener('input', _refreshOnNameControlEdit);
        elements.form.addEventListener('change', _refreshOnNameControlEdit);
    }
    if (elements.areaSelector) {
        elements.areaSelector.addEventListener('change', (event) => {
            const requestedIndex = Number.parseInt(String(event.target.value ?? '0'), 10);
            const nextAreaIndex = Number.isInteger(requestedIndex) ? requestedIndex : 0;
            captureAreaLocalState(activeAreaIndex);
            activeAreaIndex = Math.max(0, Math.min(nextAreaIndex, areaLocalState.length - 1));
            applyAreaLocalState(activeAreaIndex);
            refreshLeafLabels();
            refreshAreaSelectorOptions();
            refreshJointGroupsUi();
        });
    }

    function updateModeUi() {
        const selected = document.querySelector('input[name="sim_run_mode"]:checked');
        const isGrid = selected && selected.value === 'grid';
        if (elements.buttonLabel) {
            elements.buttonLabel.textContent = isGrid
                ? 'Run parameter grid sweep simulation'
                : 'Run trial simulation';
        }
        if (elements.gridHelp) {
            elements.gridHelp.classList.toggle('hidden', !isGrid);
        }
        document.querySelectorAll('.param-input').forEach(input => {
            const paramName = input.dataset.param || input.name;
            const keepSingle = simulationOnlyParams.has(paramName);
            const singleArrayControls = input.parentElement.querySelector('.single-array-controls');
            const hasGridControls = Boolean(input.parentElement.querySelector('.grid-parameter-controls'));
            const showGridControls = isGrid && hasGridControls && !keepSingle && !fixedGridParams.has(paramName);
            const originalType = input.dataset.originalType || input.type || 'text';
            if (!input.dataset.originalType) {
                input.dataset.originalType = originalType;
            }
            if (isGrid && showGridControls && originalType === 'number') {
                input.type = 'text';
                input.inputMode = 'decimal';
            } else if (input.type !== originalType) {
                input.type = originalType;
            }
            if (singleArrayControls) {
                input.classList.add('hidden');
                singleArrayControls.classList.toggle('hidden', showGridControls);
            } else {
                input.classList.toggle('hidden', showGridControls);
            }
        });
        document.querySelectorAll('.grid-parameter-controls').forEach(ctrl => {
            ctrl.classList.toggle('hidden', !isGrid);
        });
    }

    function updateSeedUi() {
        if (!elements.numpySeedInput) {
            return;
        }
        const useFixedSeed = Boolean(elements.useNumpySeedInput && elements.useNumpySeedInput.checked);
        elements.numpySeedInput.disabled = !useFixedSeed;
    }

    elements.runModeInputs.forEach(input => {
        input.addEventListener('change', updateModeUi);
    });
    if (elements.addJointGroupButton) {
        elements.addJointGroupButton.addEventListener('click', () => {
            addJointGroupCard();
        });
    }
    if (elements.useNumpySeedInput) {
        elements.useNumpySeedInput.addEventListener('change', updateSeedUi);
    }
    updateModeUi();
    updateSeedUi();

    if (elements.form) {
        elements.form.addEventListener('submit', (event) => {
            const selected = document.querySelector('input[name="sim_run_mode"]:checked');
            const isGrid = selected && selected.value === 'grid';
            let jointGroupResult = null;

            ensureFullInterAreaMatrices();
            const interAreaValidation = validateInterAreaMatricesAreFull();
            if (!interAreaValidation.ok) {
                event.preventDefault();
                window.alert(interAreaValidation.message);
                return;
            }

            if (elements.repetitionsInput) {
                const repetitions = Number(String(elements.repetitionsInput.value ?? '').trim());
                if (!Number.isInteger(repetitions) || repetitions < 1) {
                    event.preventDefault();
                    window.alert('Repetitions must be a positive integer.');
                    return;
                }
            }

            if (elements.useNumpySeedInput && elements.useNumpySeedInput.checked) {
                const rawSeed = String(elements.numpySeedInput?.value ?? '').trim();
                const numpySeed = Number(rawSeed);
                if (!Number.isInteger(numpySeed) || numpySeed < 0) {
                    event.preventDefault();
                    window.alert('NumPy seed must be a non-negative integer.');
                    return;
                }
            }

            if (isGrid) {
                jointGroupResult = normalizeJointGroupsInput();
                if (!jointGroupResult.ok) {
                    event.preventDefault();
                    window.alert(jointGroupResult.message);
                    return;
                }
            }

            const gridCandidateCounts = new Map();
            captureAreaLocalState(activeAreaIndex);
            const areaNames = getAreaNames();
            for (let areaIndex = 0; areaIndex < areaLocalState.length; areaIndex += 1) {
                applyAreaLocalState(areaIndex);
                const areaLabel = areaNames[areaIndex] || `area_${areaIndex}`;
                for (const paramName of areaSpecificLocalParams) {
                    const input = getParamInput(paramName);
                    if (!input) {
                        continue;
                    }
                    const singleResult = normalizeSingleArrayInput(input);
                    if (!singleResult.ok) {
                        event.preventDefault();
                        window.alert(`Area "${areaLabel}": ${singleResult.message}`);
                        return;
                    }
                    if (isGrid && !simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName)) {
                        const areaQualifiedParam = `area_${areaIndex}.${paramName}`;
                        const leafGroupsForParam = (jointGroupResult?.leafGroupsByParam.get(areaQualifiedParam))
                            || (jointGroupResult?.leafGroupsByParam.get(paramName))
                            || [];
                        const areaGridResult = normalizeGridInput(input, leafGroupsForParam);
                        if (!areaGridResult.ok) {
                            event.preventDefault();
                            window.alert(`Area "${areaLabel}": ${areaGridResult.message}`);
                            return;
                        }
                        gridCandidateCounts.set(areaQualifiedParam, Number(areaGridResult.candidateCount || 1));
                        Object.entries(areaGridResult.candidateCountsByToken || {}).forEach(([token, count]) => {
                            const qualifiedToken = qualifyAreaToken(token, areaIndex);
                            gridCandidateCounts.set(qualifiedToken, Number(count || 1));
                        });
                    }
                    upsertHiddenFormValue(`area_${areaIndex}.${paramName}`, input.value);
                }
            }
            applyAreaLocalState(activeAreaIndex);
            refreshLeafLabels();
            refreshAreaSelectorOptions();

            // Area-local parameters are submitted through hidden area_{i}.* fields.
            // Collapse the visible shared inputs to a single candidate so they do
            // not add extra global grid dimensions.
            for (const paramName of areaSpecificLocalParams) {
                const input = getParamInput(paramName);
                if (!input) {
                    continue;
                }
                const collapsed = _firstCandidateLiteralFromGridSpec(input.value);
                ensureInputCanCarryValue(input, collapsed);
                input.value = collapsed;
            }

            const params = Array.from(elements.form.querySelectorAll('.param-input'));
            for (const input of params) {
                const singleResult = normalizeSingleArrayInput(input);
                if (!singleResult.ok) {
                    event.preventDefault();
                    window.alert(singleResult.message);
                    return;
                }
            }

            if (!isGrid) {
                syncJointGroupsInput([]);
                return;
            }

            for (const input of params) {
                const paramName = input.dataset.param || input.name;
                if (
                    simulationOnlyParams.has(paramName)
                    || fixedGridParams.has(paramName)
                    || areaSpecificLocalParamSet.has(paramName)
                ) {
                    continue;
                }
                const leafGroupsForParam = jointGroupResult.leafGroupsByParam.get(paramName) || [];
                const result = normalizeGridInput(input, leafGroupsForParam);
                if (!result.ok) {
                    event.preventDefault();
                    window.alert(result.message);
                    return;
                }
                gridCandidateCounts.set(paramName, Number(result.candidateCount || 1));
                Object.entries(result.candidateCountsByToken || {}).forEach(([token, count]) => {
                    gridCandidateCounts.set(token, Number(count || 1));
                });
            }

            const countGroups = Array.isArray(jointGroupResult.countGroups)
                ? jointGroupResult.countGroups
                : jointGroupResult.groups;
            for (let index = 0; index < countGroups.length; index += 1) {
                const group = countGroups[index];
                const expectedCount = gridCandidateCounts.get(group[0]) || 1;
                const mismatch = group.find(token => (gridCandidateCounts.get(token) || 1) !== expectedCount);
                if (mismatch) {
                    event.preventDefault();
                    window.alert(`Joint sweep group ${index + 1} must use the same number of candidates for all parameters.`);
                    return;
                }
            }
        });
    }
});
