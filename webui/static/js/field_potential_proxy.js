
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

function setupUploadZones() {
    const zones = document.querySelectorAll('[data-upload]');
    zones.forEach((zone) => {
        const inputId = zone.getAttribute('data-upload');
        const input = document.getElementById(inputId);
        const filename = zone.querySelector('[data-filename]');
        const uploadedBox = zone.querySelector('[data-uploaded]');
        const uploadedName = zone.querySelector('[data-uploaded-name]');
        const uploadedLabel = zone.querySelector('[data-uploaded-label]');
        const defaultName = uploadedBox ? uploadedBox.dataset.defaultName : '';

        if (!input) {
            return;
        }

        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (event) => {
            event.preventDefault();
            zone.classList.add('border-primary', 'bg-primary/5');
        });
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('border-primary', 'bg-primary/5');
        });
        zone.addEventListener('drop', (event) => {
            event.preventDefault();
            zone.classList.remove('border-primary', 'bg-primary/5');
            if (event.dataTransfer.files && event.dataTransfer.files[0]) {
                input.files = event.dataTransfer.files;
                if (filename) {
                    filename.textContent = event.dataTransfer.files[0].name;
                }
                if (uploadedBox && uploadedName) {
                    uploadedName.textContent = event.dataTransfer.files[0].name;
                    if (uploadedLabel) {
                        uploadedLabel.textContent = 'Uploaded';
                    }
                    uploadedBox.classList.remove('hidden');
                }
            }
        });

        input.addEventListener('change', () => {
            if (filename) {
                if (input.files && input.files[0]) {
                    filename.textContent = input.files[0].name;
                } else {
                    filename.textContent = 'Not selected';
                }
            }
            if (uploadedBox && uploadedName) {
                if (input.files && input.files[0]) {
                    uploadedName.textContent = input.files[0].name;
                    if (uploadedLabel) {
                        uploadedLabel.textContent = 'Uploaded';
                    }
                    uploadedBox.classList.remove('hidden');
                } else {
                    if (defaultName) {
                        uploadedName.textContent = defaultName;
                        if (uploadedLabel) {
                            uploadedLabel.textContent = 'Loaded';
                        }
                        uploadedBox.classList.remove('hidden');
                    } else {
                        uploadedName.textContent = '';
                        uploadedBox.classList.add('hidden');
                    }
                }
            }
        });
    });

    const extInput = document.getElementById('extBackRateFile');
    const extButton = document.getElementById('extBackRateButton');
    const extFilename = document.getElementById('extBackRateFilename');
    const extUploadedBox = document.getElementById('extBackRateUploaded');
    const extUploadedName = document.getElementById('extBackRateUploadedName');
    const extUploadedLabel = document.getElementById('extBackRateUploadedLabel');
    const extDefaultName = extUploadedBox ? extUploadedBox.dataset.defaultName : '';
    if (extInput && extButton) {
        extButton.addEventListener('click', () => extInput.click());
        extInput.addEventListener('change', () => {
            if (extFilename) {
                if (extInput.files && extInput.files[0]) {
                    extFilename.textContent = extInput.files[0].name;
                } else {
                    extFilename.textContent = 'Not selected';
                }
            }
            if (extUploadedBox && extUploadedName) {
                if (extInput.files && extInput.files[0]) {
                    extUploadedName.textContent = extInput.files[0].name;
                    if (extUploadedLabel) {
                        extUploadedLabel.textContent = 'Uploaded';
                    }
                    extUploadedBox.classList.remove('hidden');
                } else {
                    if (extDefaultName) {
                        extUploadedName.textContent = extDefaultName;
                        if (extUploadedLabel) {
                            extUploadedLabel.textContent = 'Loaded';
                        }
                        extUploadedBox.classList.remove('hidden');
                    } else {
                        extUploadedName.textContent = '';
                        extUploadedBox.classList.add('hidden');
                    }
                }
            }
        });
    }
}

// Ejecutar al cargar la página para establecer estado inicial
document.addEventListener('DOMContentLoaded', () => {
    const hasProxyMethod = document.getElementById('proxyMethod');
    if (hasProxyMethod) {
        updateFormVisibility();
    }
    setupUploadZones();
});
