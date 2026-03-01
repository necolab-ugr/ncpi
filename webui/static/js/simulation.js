// static/js/simulation.js
document.addEventListener('DOMContentLoaded', function() {
    const simulationOnlyParams = new Set(['tstop', 'dt', 'local_num_threads']);
    // Parameters preset values of ncpi simulation configuration option
    const ncpiPresets = {
        tstop: 12000.0,
        dt: 0.0625,
        local_num_threads: 64,
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
        J_ext: 29.89
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
        gridHelp: document.getElementById('grid-mode-help')
    };

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

    function addGridControls(input, preset) {
        const initial = (preset !== undefined && preset !== null) ? preset : (input.value || '');
        const parsed = parseStructured(initial);
        const controls = document.createElement('div');
        controls.className = 'grid-parameter-controls hidden mt-2 flex flex-col gap-2';

        const baseClass = 'grid-param-input form-input rounded-lg border-slate-300 dark:border-slate-700 bg-background-light dark:bg-[#191e33] text-slate-900 dark:text-white';
        const leaves = Array.isArray(parsed) ? collectLeaves(parsed) : [{ path: [], value: parsed }];
        controls.dataset.gridKind = Array.isArray(parsed) ? 'array' : 'scalar';
        controls.dataset.baseValue = JSON.stringify(parsed);

        leaves.forEach((leaf, index) => {
            const label = leaf.path.length === 0
                ? (controls.dataset.gridKind === 'array' ? `Component ${index + 1}` : 'Value')
                : leaf.path.map(idx => `[${idx}]`).join('');
            const startValue = String(leaf.value ?? '');
            controls.insertAdjacentHTML('beforeend', `
                <div class="grid grid-cols-1 md:grid-cols-4 gap-2" data-grid-row="1" data-path="${escapeHtml(JSON.stringify(leaf.path))}" data-template-type="${typeof leaf.value}">
                    <div class="text-xs text-slate-600 dark:text-slate-300 md:pt-2">${escapeHtml(label)}</div>
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

    function normalizeGridInput(input) {
        const controls = input.parentElement.querySelector('.grid-parameter-controls');
        if (!controls) {
            return { ok: true };
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
                input.value = `grid=${start}:${end}:${step}`;
                return { ok: true };
            }
            if (isNumeric && stepNum === 0 && startNum !== endNum) {
                return { ok: false, message: `Parameter "${input.name}": step cannot be 0 when start and end differ.` };
            }
            if (!isNumeric && (step !== '0' || start !== end)) {
                return { ok: false, message: `Parameter "${input.name}": non-numeric values must use start=end and step=0.` };
            }
            input.value = start;
            return { ok: true };
        }

        const baseValue = parseStructured(controls.dataset.baseValue || input.value || '');
        const leaves = collectLeaves(baseValue);
        const candidatePerLeaf = [];
        const startStructure = JSON.parse(JSON.stringify(baseValue));

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

        const totalCombos = candidatePerLeaf.reduce((acc, arr) => acc * arr.length, 1);
        if (totalCombos > 256) {
            return { ok: false, message: `Parameter "${input.name}" expands to ${totalCombos} combinations (max 256).` };
        }

        if (totalCombos === 1) {
            input.value = toPythonLiteral(startStructure);
            return { ok: true };
        }

        const combos = cartesianProduct(candidatePerLeaf).map(comboValues => {
            const out = JSON.parse(JSON.stringify(startStructure));
            leaves.forEach((leaf, idx) => {
                setAtPath(out, leaf.path, comboValues[idx]);
            });
            return out;
        });
        input.value = `grid=${toPythonLiteral(combos)}`;
        return { ok: true };
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
                input.value = preset;
            }

            if (!simulationOnlyParams.has(paramName)) {
                addGridControls(input, preset);
            }
        });

        // Append the cloned section to the container
        elements.container.appendChild(section);
        
        return section;
    }
    
    // Initialize UI
    createParameterSection();

    function updateModeUi() {
        const selected = document.querySelector('input[name="sim_run_mode"]:checked');
        const isGrid = selected && selected.value === 'grid';
        if (elements.buttonLabel) {
            elements.buttonLabel.textContent = isGrid
                ? 'Run parameter grid simulation'
                : 'Run trial simulation';
        }
        if (elements.gridHelp) {
            elements.gridHelp.classList.toggle('hidden', !isGrid);
        }
        document.querySelectorAll('.param-input').forEach(input => {
            const paramName = input.dataset.param || input.name;
            const keepSingle = simulationOnlyParams.has(paramName);
            input.classList.toggle('hidden', isGrid && !keepSingle);
        });
        document.querySelectorAll('.grid-parameter-controls').forEach(ctrl => {
            ctrl.classList.toggle('hidden', !isGrid);
        });
    }

    elements.runModeInputs.forEach(input => {
        input.addEventListener('change', updateModeUi);
    });
    updateModeUi();

    if (elements.form) {
        elements.form.addEventListener('submit', (event) => {
            const selected = document.querySelector('input[name="sim_run_mode"]:checked');
            const isGrid = selected && selected.value === 'grid';
            if (!isGrid) {
                return;
            }

            const params = Array.from(elements.form.querySelectorAll('.param-input'));
            for (const input of params) {
                const paramName = input.dataset.param || input.name;
                if (simulationOnlyParams.has(paramName)) {
                    continue;
                }
                const result = normalizeGridInput(input);
                if (!result.ok) {
                    event.preventDefault();
                    window.alert(result.message);
                    return;
                }
            }
        });
    }
});
