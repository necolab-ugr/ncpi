// DARK MODE TOGGLE
// Get elements
const darkModeToggle = document.getElementById('darkModeToggle');
const darkModeIcon = document.getElementById('darkModeIcon');
const htmlElement = document.documentElement;

// Check for saved theme preference or respect OS preference
const getCurrentTheme = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        return savedTheme;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

// Apply theme on page load
const currentTheme = getCurrentTheme();
if (currentTheme === 'dark') {
    htmlElement.classList.add('dark');
    darkModeIcon.textContent = 'light_mode'; // Change icon to light mode
}

// Toggle dark mode
darkModeToggle.addEventListener('click', () => {
    if (htmlElement.classList.contains('dark')) {
        // Switch to light mode
        htmlElement.classList.remove('dark');
        darkModeIcon.textContent = 'dark_mode';
        localStorage.setItem('theme', 'light');
    } else {
        // Switch to dark mode
        htmlElement.classList.add('dark');
        darkModeIcon.textContent = 'light_mode';
        localStorage.setItem('theme', 'dark');
    }
});

// Optional: Listen for OS theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) { // Only if user hasn't set a preference
        if (e.matches) {
            htmlElement.classList.add('dark');
            darkModeIcon.textContent = 'light_mode';
        } else {
            htmlElement.classList.remove('dark');
            darkModeIcon.textContent = 'dark_mode';
        }
    }
});