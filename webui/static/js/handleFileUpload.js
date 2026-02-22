function handleFileUpload(event) {
    const file = event.target.files[0];
    // Get the id of the uploaded file
    const inputId = event.target.id;

    if (file) {
        // Update Alpine.js state (if using global Alpine component)
        // if (window.Alpine && Alpine.$data) {
            // This is a simplified approach - you might need to adjust based on your Alpine setup
            const component = Alpine.$data(event.target.closest('[x-data]'));

            if (component) {
                // Initialize uploads if needed
                if (!component.uploads) {
                    component.uploads = {};
                }

                // Store info about uploaded files
                component.uploads[inputId] = {
                    name: file.name,
                    uploaded: false
                };

                // Allow page-specific hooks (no-op unless defined)
                if (typeof component.onFileSelected === 'function') {
                    component.onFileSelected(file, inputId);
                }

                // Force Alpine to re-evaluate (if needed)
                component.$nextTick && component.$nextTick();
            }
        
        console.log('File uploaded:', file.name, 'Size: ', file.size, 'Type: ', file.type, 'id: ', inputId, 'component.uploads{}: ', component.uploads);
    }
}

// Make functions globally available
window.handleFileUpload = handleFileUpload;
