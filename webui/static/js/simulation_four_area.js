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
        J_EE: 1.589,
        J_IE: 2.020,
        J_EI: -23.84,
        J_II: -8.441,
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
        numpySeedInput: document.getElementById('sim-numpy-seed')
    };

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
        if (wantsGridSpec && originalType === 'number') {
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

    function formatPath(path) {
        return path.map(idx => `[${idx}]`).join('');
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
                if (popCount > 0) {
                    const srcPop = populations[((path[0] % popCount) + popCount) % popCount];
                    const tgtPop = populations[((path[1] % popCount) + popCount) % popCount];
                    return `source: ${srcPop} -> target: ${tgtPop}`;
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
        });
        document.querySelectorAll('[data-grid-row="1"]').forEach(row => {
            const paramName = row.dataset.paramName || '';
            const path = JSON.parse(row.dataset.path || '[]');
            const labelNode = row.querySelector('[data-param-leaf-label="1"]');
            if (labelNode) {
                labelNode.textContent = describeLeafLabel(paramName, path, 0, true);
            }
        });
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
                    options.push({ value: paramName, label: paramName });
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
                options.push({ value: token, label: `${paramName} - ${leafLabel}` });
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

    function refreshJointGroupsUi() {
        const cards = jointGroupCards();
        const selections = cards.map(card => readJointGroupSelection(card));
        const globallySelected = new Set(selections.flat());
        const options = sweepableParamOptions();

        cards.forEach((card, index) => {
            const title = card.querySelector('[data-joint-group-title="1"]');
            if (title) {
                title.textContent = `Group ${index + 1}`;
            }
            const summary = card.querySelector('[data-joint-group-summary="1"]');
            if (summary) {
                const count = selections[index].length;
                summary.textContent = count > 0 ? `${count} parameter(s) selected` : 'Select at least two parameters';
            }

            const select = card.querySelector('[data-joint-group-select="1"]');
            if (!select) {
                return;
            }
            const ownSelection = new Set(selections[index]);
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
            <label class="mt-3 flex flex-col gap-2">
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
            select.addEventListener('change', refreshJointGroupsUi);
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
        const groups = jointGroupCards()
            .map(card => readJointGroupSelection(card))
            .filter(group => group.length > 0);

        const seenGroupTokens = new Set();
        const seenBackendParams = new Set();
        const seenLeafTokens = new Set();
        const backendGroups = [];
        const leafGroupsByParam = new Map();

        for (let index = 0; index < groups.length; index += 1) {
            const group = groups[index];
            if (group.length < 2) {
                return { ok: false, message: `Joint sweep group ${index + 1} must contain at least two parameters.` };
            }

            const parsedGroup = group.map(parseJointToken);
            const groupHasLeafTokens = parsedGroup.some(item => item.pathKey !== '');
            if (groupHasLeafTokens) {
                const paramNames = new Set(parsedGroup.map(item => item.paramName));
                if (paramNames.size !== 1) {
                    return { ok: false, message: `Joint sweep group ${index + 1} with leaf targets must belong to a single parameter.` };
                }
                if (parsedGroup.some(item => !item.pathKey)) {
                    return { ok: false, message: `Joint sweep group ${index + 1} cannot mix a whole parameter with leaf targets.` };
                }
                const paramName = parsedGroup[0].paramName;
                const leafPathKeys = parsedGroup.map(item => item.pathKey);
                const localLeafSet = new Set();
                for (const token of group) {
                    if (seenGroupTokens.has(token)) {
                        return { ok: false, message: `Parameter "${token}" cannot belong to more than one joint sweep group.` };
                    }
                    seenGroupTokens.add(token);
                    if (seenLeafTokens.has(token)) {
                        return { ok: false, message: `Leaf target "${token}" cannot belong to more than one joint sweep group.` };
                    }
                    seenLeafTokens.add(token);
                }
                for (const key of leafPathKeys) {
                    if (localLeafSet.has(key)) {
                        return { ok: false, message: `Joint sweep group ${index + 1} contains duplicate leaf targets.` };
                    }
                    localLeafSet.add(key);
                }
                if (!leafGroupsByParam.has(paramName)) {
                    leafGroupsByParam.set(paramName, []);
                }
                leafGroupsByParam.get(paramName).push(leafPathKeys);
                continue;
            }

            const backendGroup = parsedGroup.map(item => item.paramName);
            for (const paramName of backendGroup) {
                if (seenGroupTokens.has(paramName)) {
                    return { ok: false, message: `Parameter "${paramName}" cannot belong to more than one joint sweep group.` };
                }
                if (seenBackendParams.has(paramName)) {
                    return { ok: false, message: `Parameter "${paramName}" cannot belong to more than one joint sweep group.` };
                }
                seenGroupTokens.add(paramName);
                seenBackendParams.add(paramName);
            }
            backendGroups.push(backendGroup);
        }

        syncJointGroupsInput(backendGroups);
        return { ok: true, groups, backendGroups, leafGroupsByParam };
    }

    function addGridControls(input, preset) {
        const initial = (preset !== undefined && preset !== null) ? preset : (input.value || '');
        const parsed = parseStructured(initial);
        const controls = document.createElement('div');
        controls.className = 'grid-parameter-controls hidden mt-2 flex flex-col gap-2';
        const paramName = input.dataset.param || input.name || '';

        const baseClass = 'grid-param-input form-input rounded-lg border-slate-300 dark:border-slate-700 bg-background-light dark:bg-[#191e33] text-slate-900 dark:text-white';
        const leaves = Array.isArray(parsed) ? collectLeaves(parsed) : [{ path: [], value: parsed }];
        controls.dataset.gridKind = Array.isArray(parsed) ? 'array' : 'scalar';
        controls.dataset.baseValue = JSON.stringify(parsed);

        leaves.forEach((leaf, index) => {
            const label = describeLeafLabel(paramName, leaf.path, index, controls.dataset.gridKind === 'array');
            const startValue = String(leaf.value ?? '');
            controls.insertAdjacentHTML('beforeend', `
                <div class="grid grid-cols-1 md:grid-cols-4 gap-2" data-grid-row="1" data-path="${escapeHtml(JSON.stringify(leaf.path))}" data-param-name="${escapeHtml(paramName)}" data-template-type="${typeof leaf.value}">
                    <div class="text-xs text-slate-600 dark:text-slate-300 md:pt-2" data-param-leaf-label="1">${escapeHtml(label)}</div>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">Start</span>
                        <input type="text" data-grid-role="start" class="${baseClass}" value="${escapeHtml(startValue)}" />
                    </label>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">Step</span>
                        <input type="text" data-grid-role="step" class="${baseClass}" value="0" />
                    </label>
                    <label class="flex flex-col gap-1">
                        <span class="text-xs text-slate-600 dark:text-slate-300">End</span>
                        <input type="text" data-grid-role="end" class="${baseClass}" value="${escapeHtml(startValue)}" />
                    </label>
                </div>
            `);
        });

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
        controls.className = 'single-array-controls mt-2 grid grid-cols-1 md:grid-cols-2 gap-2';
        controls.dataset.baseValue = JSON.stringify(parsed);

        const leaves = collectLeaves(parsed);
        const baseClass = 'single-array-input form-input rounded-lg border-slate-300 dark:border-slate-700 bg-background-light dark:bg-[#191e33] text-slate-900 dark:text-white';

        leaves.forEach((leaf, index) => {
            const label = describeLeafLabel(paramName, leaf.path, index, true);
            controls.insertAdjacentHTML('beforeend', `
                <label class="flex flex-col gap-1" data-single-array-row="1" data-path="${escapeHtml(JSON.stringify(leaf.path))}" data-param-name="${escapeHtml(paramName)}">
                    <span class="text-xs text-slate-600 dark:text-slate-300" data-param-leaf-label="1">${escapeHtml(label)}</span>
                    <input type="text" class="${baseClass}" value="${escapeHtml(String(leaf.value ?? ''))}" />
                </label>
            `);
        });

        input.insertAdjacentElement('afterend', controls);
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
        const rows = Array.from(controls.querySelectorAll('[data-single-array-row="1"]'));

        for (let idx = 0; idx < rows.length; idx += 1) {
            const row = rows[idx];
            const path = JSON.parse(row.dataset.path || '[]');
            const template = leaves[idx] ? leaves[idx].value : null;
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
        const leafTokenByIndex = leaves.map(leaf => makeJointToken(paramName, leaf.path));
        const leafPathToIndex = new Map();
        leaves.forEach((leaf, idx) => {
            leafPathToIndex.set(leafPathKey(leaf.path), idx);
        });

        for (let idx = 0; idx < rows.length; idx += 1) {
            const row = rows[idx];
            const path = JSON.parse(row.dataset.path || '[]');
            const template = leaves[idx] ? leaves[idx].value : null;
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
                        setAtPath(out, leaves[leafIdx].path, candidatePerLeaf[leafIdx][choice]);
                    });
                    return;
                }
                setAtPath(out, leaves[dim.index].path, candidatePerLeaf[dim.index][choice]);
            });
            return out;
        });
        const comboSpec = `grid=${toPythonLiteral(combos)}`;
        ensureInputCanCarryValue(input, comboSpec);
        input.value = comboSpec;
        return { ok: true, candidateCount: totalCombos, candidateCountsByToken };
    }

    // Function to create parameter section
    function createParameterSection() {
        const clone = elements.paramSection.content.cloneNode(true);
        const section = clone.querySelector('.parameter-section');

        // Configure all parameter inputs
        const inputs = section.querySelectorAll('.param-input');
        inputs.forEach(input => {
            const paramName = input.dataset.param;
            const preset = ncpiPresets[paramName];
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
                input.value = preset;
            }

            addSingleArrayControls(input, preset);

            if (!simulationOnlyParams.has(paramName) && !fixedGridParams.has(paramName)) {
                addGridControls(input, preset);
            }
        });

        // Append the cloned section to the container
        elements.container.appendChild(section);

        return section;
    }

    // Initialize UI
    createParameterSection();
    refreshLeafLabels();
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
        populationsInput.addEventListener('input', refreshLeafLabels);
        populationsInput.addEventListener('change', refreshLeafLabels);
    }
    const areasInput = document.querySelector('.param-input[data-param="areas"]');
    if (areasInput) {
        areasInput.addEventListener('input', refreshLeafLabels);
        areasInput.addEventListener('change', refreshLeafLabels);
    }
    if (elements.container) {
        const _refreshOnNameControlEdit = (event) => {
            const row = event.target && event.target.closest
                ? event.target.closest('[data-single-array-row="1"]')
                : null;
            if (!row) {
                return;
            }
            const paramName = row.dataset.paramName || '';
            if (paramName === 'X' || paramName === 'areas') {
                refreshLeafLabels();
            }
        };
        elements.container.addEventListener('input', _refreshOnNameControlEdit);
        elements.container.addEventListener('change', _refreshOnNameControlEdit);
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

            const jointGroupResult = normalizeJointGroupsInput();
            if (!jointGroupResult.ok) {
                event.preventDefault();
                window.alert(jointGroupResult.message);
                return;
            }

            const gridCandidateCounts = new Map();

            for (const input of params) {
                const paramName = input.dataset.param || input.name;
                if (simulationOnlyParams.has(paramName) || fixedGridParams.has(paramName)) {
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

            for (let index = 0; index < jointGroupResult.groups.length; index += 1) {
                const group = jointGroupResult.groups[index];
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
