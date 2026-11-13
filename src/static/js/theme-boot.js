/*
 * theme-boot.js: a classic, blocking script in <head>, before the stylesheets. ES5, no imports.
 * It applies the theme and motion choices before first paint so the page never flashes the wrong theme:
 *   - open-speech-theme: 'system' | 'light' | 'dark' (a legacy 'dark' / 'light' value is honoured)
 *   - open-speech-prefs.reduceMotion: 'system' | 'reduce' | 'full'
 *   - for this page load only (screenshots): ?theme=light|dark|system, ?motion=reduce|full, ?spacing=wcag
 * It sets <html data-theme> (removed for System), <html data-motion> (removed for System),
 * <html data-spacing="wcag">, and rewrites both theme-color metas to the forced theme's --bg.
 */
(function () {
  try {
    var root = document.documentElement;
    var COLORS = { light: '#F7F7F4', dark: '#0E0F13' };
    var theme = 'system';
    var motion = 'system';
    var spacing = '';

    try {
      var stored = window.localStorage.getItem('open-speech-theme');
      if (stored === 'light' || stored === 'dark' || stored === 'system') theme = stored;
      var prefs = JSON.parse(window.localStorage.getItem('open-speech-prefs') || 'null');
      if (prefs && (prefs.reduceMotion === 'reduce' || prefs.reduceMotion === 'full' || prefs.reduceMotion === 'system')) {
        motion = prefs.reduceMotion;
      }
    } catch (storageError) {
      // Storage can be unavailable (private mode, blocked site data): keep the defaults.
    }

    var query = window.location.search.replace(/^\?/, '').split('&');
    for (var i = 0; i < query.length; i += 1) {
      var pair = query[i].split('=');
      var key = decodeURIComponent(pair[0] || '');
      var value = decodeURIComponent((pair[1] || '').replace(/\+/g, ' '));
      if (key === 'theme' && (value === 'light' || value === 'dark' || value === 'system')) theme = value;
      if (key === 'motion' && (value === 'reduce' || value === 'full')) motion = value;
      if (key === 'spacing' && value === 'wcag') spacing = 'wcag';
    }

    if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
    else root.removeAttribute('data-theme');
    if (motion === 'reduce' || motion === 'full') root.setAttribute('data-motion', motion);
    else root.removeAttribute('data-motion');
    if (spacing) root.setAttribute('data-spacing', spacing);

    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var m = 0; m < metas.length; m += 1) {
      var media = metas[m].getAttribute('media') || '';
      var scheme = media.indexOf('dark') >= 0 ? 'dark' : 'light';
      metas[m].setAttribute('content', COLORS[theme === 'system' ? scheme : theme]);
    }
  } catch (error) {
    // Never block the page: the stylesheets follow the OS colour scheme without this script.
  }
}());
