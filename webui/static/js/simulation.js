// static/js/simulation.js
document.addEventListener('DOMContentLoaded', function() {
    // Parameters preset values of ncpi simulation configuration option
    const ncpiPresets = {
        simulation_time: 12000,
        simulation_resolution: 0.0625,
        random_seed: 442597,
        num_threads: 32,
        population_name: "Excitatory (E), Inhibitory (I)",
        population_size: "8192, 1024",
        neuron_model_name: "iaf_psc_exp",
        connection_rule: "pairwise_bernouilli",
        connection_probability: "20",
        target_indegree: "20",
        synaptic_weight: "465, 160",
        synaptic_delay: "20",
        allow_autapses: true
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
            console.log(paramName);
            
            // NCPI mode: preset values, readonly
            input.value = ncpiPresets[paramName];
            input.readOnly = false;
        });

        // Append the cloned section to the container
        elements.container.appendChild(section);
        
        return section;
    }
    
    // Initialize UI
    createParameterSection();
});