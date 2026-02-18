// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get all radio buttons and the dropdown
    const modelRadios = document.querySelectorAll('input[name="model-radio"]');
    const sklearnDropdown = document.getElementById('sklearn-dropdown');
    
    // Function to enable/disable dropdown based on selection
    function updateDropdownState() {
        const selectedModel = document.querySelector('input[name^="model"]:checked').id; // select all inputs whose names start with model*
        const hiddenModelInput = document.getElementById('hidden-model-input');
        
        if (selectedModel === 'model-sklearn') {
            // Enable dropdown
            sklearnDropdown.disabled = false;
            sklearnDropdown.classList.remove('opacity-50', 'cursor-not-allowed');
            hiddenModelInput.value = sklearnDropdown.value;
        } else {
            // Disable dropdown
            sklearnDropdown.disabled = true;
            sklearnDropdown.classList.add('opacity-50', 'cursor-not-allowed');
            hiddenModelInput.value = document.querySelector('input[name="model-radio"]:checked').value;
        }
    }
    
    // Add event listeners to all radio buttons
    modelRadios.forEach(radio => {
        radio.addEventListener('change', updateDropdownState);
    });
    
    // Add event listener to dropdown as well
    sklearnDropdown.addEventListener('change', function() {
        if (!this.disabled) {
            document.getElementById('hidden-model-input').value = this.value;
        }
    });
    
    // Initialize state on page load
    updateDropdownState();


    // TRAIN SECTION TOGGLE PARAMETERS
    // Get radio buttons and option sections
    const trainRadio = document.getElementById('train-model');
    const loadRadio = document.getElementById('load-model');
    const trainOptions = document.getElementById('train-options');
    const loadOptions = document.getElementById('load-options');
    
    // Function to toggle visibility
    function toggleTrainingOptions() {
        if (trainRadio.checked) {
            // Show train options, hide load options
            trainOptions.classList.remove('hidden');
            loadOptions.classList.add('hidden');
        } else if (loadRadio.checked) {
            // Show load options, hide train options
            trainOptions.classList.add('hidden');
            loadOptions.classList.remove('hidden');
        }
    }
    
    // Add event listeners to radio buttons
    trainRadio.addEventListener('change', toggleTrainingOptions);
    loadRadio.addEventListener('change', toggleTrainingOptions);
    
    // Initialize on page load
    toggleTrainingOptions();
});


