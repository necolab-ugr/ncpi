// static/js/simulation_four_area.js
document.addEventListener('DOMContentLoaded', function() {
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
        inter_area_scale: 0.15,
        inter_area_p: 0.02,
        inter_area_delay: 10.0,
        "inter_area.C_YX": "[[0.02, 0.02], [0.0, 0.0]]",
        "inter_area.J_YX": "[[0.23835, 0.303], [0.0, 0.0]]",
        "inter_area.delay_YX": "[[10.0, 10.0], [0.0, 0.0]]"
    };

    // Get DOM Elements
    const elements = {
        // Parameters Section of the form
        paramSection: document.getElementById('parameter-section'),
        // Add a container where to append the cloned section
        container: document.getElementById('parameter-container')
    };

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
        });

        // Append the cloned section to the container
        elements.container.appendChild(section);

        return section;
    }

    // Initialize UI
    createParameterSection();
});
