
function updateFormVisibility() {
    const method = document.getElementById('proxyMethod').value;
    
    // Simulation Step: LRWS, ERWS1, ERWS2
    const simulationStep = document.getElementById('simulationStepContainer');
    simulationStep.classList.toggle('hidden', !['LRWS', 'ERWS1', 'ERWS2'].includes(method));
    
    // FR Upload: Method = FR
    const frUpload = document.getElementById('frUploadContainer');
    frUpload.classList.toggle('hidden', method !== 'FR');
    
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

// Ejecutar al cargar la página para establecer estado inicial
document.addEventListener('DOMContentLoaded', updateFormVisibility);