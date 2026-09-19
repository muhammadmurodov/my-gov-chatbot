// My.gov.uz — Open WebUI runtime hook (/static/loader.js, loaded on every page).
// Purpose: default the interface to a ChatGPT-style dark theme.
// The stock theme init sets `localStorage.theme = 'system'` on first visit, so a
// user on a light OS lands in light mode. We make dark the first-visit default,
// while still letting the user switch themes afterwards (we only seed once).
(function () {
  try {
    if (!localStorage.getItem('mygov_theme_default_applied')) {
      localStorage.theme = 'dark';
      localStorage.setItem('mygov_theme_default_applied', '1');
    }
    // Re-assert the class in case the inline init resolved 'system' -> light.
    if (localStorage.theme === 'dark' || localStorage.theme === 'oled-dark') {
      document.documentElement.classList.remove('light');
      document.documentElement.classList.add('dark');
    }
  } catch (e) {
    /* localStorage may be blocked; ignore and fall back to stock behavior */
  }
})();
