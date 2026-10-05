/*
 * F.O.R.G.E. AI Challenge - light / dark mode switch
 *
 * Add to the <head> of every page, after the stylesheet:
 *   <script src="../../../js/theme.js"></script>
 *
 * The switch is placed in .site-header__inner, or in an element marked
 * data-theme-slot (the home page). Without a saved choice the page follows
 * the system setting; flipping the switch saves "light" or "dark" and sets
 * <html data-theme="...">, which css/forge.css uses to pick the colors.
 */
(function () {
  var KEY = 'forge-theme';
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  // Apply a saved choice before the page paints, so it never flashes the wrong theme.
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* storage blocked */ }
  if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved);

  function current() {
    var t = root.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return media && media.matches ? 'dark' : 'light';
  }

  var STYLE = [
    '.theme-toggle{display:inline-flex;align-items:center;align-self:center;margin:0 0 0 auto;padding:.2rem .1rem;',
    'border:0;background:none;color:var(--muted,#5b6672);font:inherit;font-size:.85rem;cursor:pointer}',
    '.theme-toggle:hover{color:var(--text,#1b1f24);border:0}',
    '.theme-toggle:focus-visible{outline:2px solid var(--accent,#1f5fbf);outline-offset:3px;border-radius:999px}',
    '.theme-toggle__track{position:relative;width:2.6rem;height:1.4rem;border-radius:999px;',
    'background:var(--border,#dde1e6);transition:background .2s}',
    '.theme-toggle__thumb{position:absolute;top:.15rem;left:.15rem;width:1.1rem;height:1.1rem;border-radius:50%;',
    'background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.3);transition:transform .2s;',
    'display:flex;align-items:center;justify-content:center;font-size:.7rem;line-height:1}',
    '.theme-toggle[aria-checked="true"] .theme-toggle__track{background:var(--accent,#1f5fbf)}',
    '.theme-toggle[aria-checked="true"] .theme-toggle__thumb{transform:translateX(1.2rem)}',
    '.site-header__inner .crumbs{margin-left:auto}',
    '.site-header__inner .theme-toggle{order:3;margin-left:1rem}',
    '@media (max-width:40rem){.site-header__inner .theme-toggle{order:2;margin-left:auto}',
    '.site-header__inner .crumbs{order:3;flex-basis:100%;margin-left:0}}',
    '@media (prefers-reduced-motion:reduce){.theme-toggle__track,.theme-toggle__thumb{transition:none}}'
  ].join('');

  function mount() {
    var style = document.createElement('style');
    style.textContent = STYLE;
    document.head.appendChild(style);

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'theme-toggle';
    btn.setAttribute('role', 'switch');
    btn.setAttribute('aria-label', 'Dark mode');
    btn.innerHTML =
      '<span class="theme-toggle__track" aria-hidden="true"><span class="theme-toggle__thumb"></span></span>';

    function sync() {
      var dark = current() === 'dark';
      btn.setAttribute('aria-checked', dark ? 'true' : 'false');
      btn.querySelector('.theme-toggle__thumb').textContent = dark ? '☾' : '☀';
      btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
    }

    btn.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(KEY, next); } catch (e) { /* storage blocked */ }
      sync();
    });

    // Follow system changes until the reader picks a theme.
    if (media) {
      var onChange = function () { if (!root.getAttribute('data-theme')) sync(); };
      if (media.addEventListener) media.addEventListener('change', onChange);
      else if (media.addListener) media.addListener(onChange);
    }

    var slot = document.querySelector('[data-theme-slot]') || document.querySelector('.site-header__inner');
    if (slot) {
      slot.appendChild(btn);
    } else {
      btn.style.cssText = 'position:fixed;top:.75rem;right:1rem;z-index:10';
      document.body.appendChild(btn);
    }
    sync();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
