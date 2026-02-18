// Add close functionality to alert messages
document.querySelectorAll('.close-button').forEach(button => {
    button.addEventListener('click', function() {
    // Find the parent alert div and remove it
    this.closest('.flash-container').remove();
    });
});