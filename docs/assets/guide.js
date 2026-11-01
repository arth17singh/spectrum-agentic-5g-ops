/* SPECTRUM presenter guide: shared behaviour.
   Injects sidebar nav, mobile top bar, on-this-page TOC, prev/next pager,
   theme toggle, screenshot lightbox and pin <-> component-card linking.
   No dependencies. Works from file://. Each page needs:
     <link rel="stylesheet" href="assets/guide.css">
     <script src="assets/guide.js" defer></script>
     <body data-page="<filename>.html">  ...  <main class="content"> ... </main>
*/
(function () {
  'use strict';

  var NAV = [
    { group: 'Start', items: [
      ['index.html', 'Start here'],
      ['demo-playbook.html', 'Demo playbook'],
      ['concepts.html', 'Concepts explained simply']] },
    { group: 'The console', items: [
      ['shell.html', 'Console frame & controls'],
      ['mission.html', 'Mission Control']] },
    { group: 'Learn', plane: 'learn', items: [['federation.html', 'Federation Center']] },
    { group: 'Verify', plane: 'verify', items: [['proving.html', 'Proving Ground']] },
    { group: 'Release', plane: 'release', items: [['release.html', 'Release']] },
    { group: 'Operate', plane: 'operate', items: [
      ['swarm.html', 'Swarm Tower'],
      ['oversight.html', 'Human oversight & phone approvals']] },
    { group: 'Prove', plane: 'prove', items: [['audit.html', 'Audit, provenance & dossier']] },
    { group: 'Cross-cutting', items: [
      ['trust.html', 'Trust Center'],
      ['observability.html', 'Observability'],
      ['settings.html', 'Settings, feature flags & presenter view']] },
    { group: 'Build', items: [
      ['feature-map.html', 'Feature map & parallel build plan'],
      ['architecture.html', 'Architecture']] }
  ];
  var PLANES = ['learn', 'verify', 'release', 'operate', 'prove'];
  var THEME_KEY = 'spectrum-guide-theme';

  var root = document.documentElement;
  var body = document.body;
  root.classList.add('js');

  /* ---------- theme ---------- */
  var memTheme = null; /* fallback when storage is blocked (some browsers on file://) */
  function storedTheme() { try { return localStorage.getItem(THEME_KEY) || memTheme; } catch (e) { return memTheme; } }
  function systemDark() { return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; }
  function effectiveTheme() { var t = storedTheme(); return t === 'light' || t === 'dark' ? t : (systemDark() ? 'dark' : 'light'); }
  function applyTheme() {
    var t = storedTheme();
    if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t); else root.removeAttribute('data-theme');
    var label = effectiveTheme() === 'dark' ? 'Light theme' : 'Dark theme';
    document.querySelectorAll('.theme-btn').forEach(function (b) {
      b.querySelector('span').textContent = label;
      b.setAttribute('aria-label', 'Switch to ' + label.toLowerCase());
    });
  }
  function toggleTheme() {
    var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    memTheme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage blocked: keep it for this page view */ }
    applyTheme();
  }
  applyTheme();

  /* ---------- helpers ---------- */
  function el(tag, attrs, html) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (html != null) n.innerHTML = html;
    return n;
  }
  function slug(s) {
    return s.toLowerCase().replace(/&/g, ' and ').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'section';
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function ribbon(on, cls) {
    return '<div class="ribbon-mini' + (cls ? ' ' + cls : '') + '" aria-hidden="true">' + PLANES.map(function (p) {
      return '<span data-p="' + p + '"' + (on === p ? ' class="on"' : '') + '></span>';
    }).join('') + '</div>';
  }
  var ICON_MOON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 10.2A6 6 0 0 1 5.8 2.5a6 6 0 1 0 7.7 7.7z" fill="currentColor"/></svg>';
  var ICON_MENU = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';

  /* ---------- locate current page ---------- */
  var page = body.getAttribute('data-page') || location.pathname.split('/').pop() || 'index.html';
  var flat = [];
  NAV.forEach(function (g) { g.items.forEach(function (it) { flat.push({ href: it[0], title: it[1], plane: g.plane || null, group: g.group }); }); });
  var idx = -1;
  flat.forEach(function (f, i) { if (f.href === page) idx = i; });
  var current = idx >= 0 ? flat[idx] : null;
  if (!body.getAttribute('data-plane') && current && current.plane) body.setAttribute('data-plane', current.plane);
  var plane = body.getAttribute('data-plane');

  var main = document.querySelector('main.content') || document.querySelector('main');
  if (!main) return;
  if (!main.id) main.id = 'main';

  /* ---------- sidebar ---------- */
  var side = el('nav', { class: 'side', id: 'side-nav', 'aria-label': 'Guide pages' });
  var html = '<a class="side-brand" href="index.html"><b>SPECTRUM</b><span>Presenter guide</span>' + ribbon(null, 'all') + '</a>';
  NAV.forEach(function (g) {
    var dot = g.plane ? ' style="--dot:var(--' + g.plane + ')"' : '';
    html += '<div class="side-group"' + dot + '><h2>' + (g.plane ? '<span class="dot" aria-hidden="true"></span>' : '') + esc(g.group) + '</h2><ul>';
    g.items.forEach(function (it) {
      html += '<li><a href="' + it[0] + '"' + (it[0] === page ? ' aria-current="page"' : '') + '>' + esc(it[1]) + '</a></li>';
    });
    html += '</ul></div>';
  });
  html += '<div class="side-foot"><button type="button" class="theme-btn">' + ICON_MOON + '<span>Dark theme</span></button></div>';
  side.innerHTML = html;
  body.insertBefore(side, body.firstChild);
  var skip = el('a', { class: 'skip-link', href: '#' + main.id }, 'Skip to content');
  body.insertBefore(skip, side);

  /* mobile top bar */
  var top = el('header', { class: 'topbar' });
  var h1 = main.querySelector('h1');
  top.innerHTML = '<button type="button" class="menu-btn" aria-controls="side-nav" aria-expanded="false">' + ICON_MENU + '<span>Pages</span></button>' +
    '<span class="title">' + esc(h1 ? h1.textContent : (current ? current.title : 'SPECTRUM guide')) + '</span>' +
    '<button type="button" class="theme-btn">' + ICON_MOON + '<span>Dark theme</span></button>';
  body.insertBefore(top, side.nextSibling);
  var menuBtn = top.querySelector('.menu-btn');
  function setNav(open) {
    body.classList.toggle('nav-open', open);
    menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { var a = side.querySelector('a[aria-current]') || side.querySelector('a'); if (a) a.focus(); }
  }
  menuBtn.addEventListener('click', function (e) { e.stopPropagation(); setNav(!body.classList.contains('nav-open')); });
  document.addEventListener('click', function (e) { if (body.classList.contains('nav-open') && !side.contains(e.target)) setNav(false); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && body.classList.contains('nav-open')) { setNav(false); menuBtn.focus(); } });
  document.querySelectorAll('.theme-btn').forEach(function (b) { b.addEventListener('click', toggleTheme); });
  applyTheme();

  /* ---------- page header ribbon (signature) ---------- */
  var head = main.querySelector('.page-head');
  if (head && !head.querySelector('.ribbon-mini')) head.insertAdjacentHTML('afterbegin', ribbon(plane, plane ? '' : 'all'));

  /* ---------- on-this-page TOC ---------- */
  var heads = Array.prototype.filter.call(main.querySelectorAll('h2, h3'), function (h) {
    return !h.closest('[data-toc="skip"]') && h.getAttribute('data-toc') !== 'skip' && !h.closest('.page-head') && !h.closest('.callout') && !h.closest('details');
  });
  var used = {};
  document.querySelectorAll('[id]').forEach(function (n) { used[n.id] = true; });
  var toc = el('aside', { class: 'toc', 'aria-label': 'On this page' });
  if (heads.length > 1) {
    var list = '';
    heads.forEach(function (h) {
      var target = h.closest('.comp, .beat, .term') || h;
      if (!target.id) {
        var base = slug(h.textContent), id = base, n = 2;
        while (used[id]) id = base + '-' + n++;
        used[id] = true;
        target.id = id;
      }
      var label = h.getAttribute('data-toc-label') || h.textContent.replace(/\s+/g, ' ').trim();
      var num = target.classList.contains('comp') ? (target.querySelector('.comp-num') || {}).textContent : null;
      if (num) label = num + '. ' + label;
      list += '<li><a class="lvl-' + h.tagName.charAt(1) + '" href="#' + target.id + '">' + esc(label) + '</a></li>';
    });
    toc.innerHTML = '<h2>On this page</h2><ol>' + list + '</ol>';
  }
  body.appendChild(toc);

  if (heads.length > 1 && 'IntersectionObserver' in window) {
    var links = {};
    toc.querySelectorAll('a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting; });
      var firstVisible = null;
      heads.forEach(function (h) { var t = h.closest('.comp, .beat, .term') || h; if (!firstVisible && visible[t.id]) firstVisible = t.id; });
      if (firstVisible) {
        toc.querySelectorAll('a.is-active').forEach(function (a) { a.classList.remove('is-active'); });
        if (links[firstVisible]) links[firstVisible].classList.add('is-active');
      }
    }, { rootMargin: '0px 0px -55% 0px' });
    heads.forEach(function (h) { io.observe(h.closest('.comp, .beat, .term') || h); });
  }

  /* ---------- prev / next ---------- */
  if (idx >= 0) {
    var prev = flat[idx - 1], next = flat[idx + 1];
    var pager = el('nav', { class: 'pager', 'aria-label': 'Previous and next page' });
    pager.innerHTML =
      (prev ? '<a class="prev" href="' + prev.href + '" rel="prev"><span>Previous</span><b>' + esc(prev.title) + '</b></a>' : '') +
      (next ? '<a class="next" href="' + next.href + '" rel="next"><span>Next</span><b>' + esc(next.title) + '</b></a>' : '');
    main.appendChild(pager);
  }

  /* ---------- pins <-> component cards ---------- */
  function cardFor(pin) { var h = pin.getAttribute('href'); return h && h.charAt(0) === '#' ? document.getElementById(h.slice(1)) : null; }
  function related(id) {
    return Array.prototype.slice.call(document.querySelectorAll(
      'a.pin[href="#' + id + '"], .box[data-target="' + id + '"], .shot-legend a[href="#' + id + '"]'));
  }
  function setHot(id, on) {
    var card = document.getElementById(id);
    if (card) card.classList.toggle('is-hot', on);
    related(id).forEach(function (n) { n.classList.toggle('is-hot', on); });
  }
  var cardIds = {};
  document.querySelectorAll('a.pin, .shot-legend a').forEach(function (pin) {
    var card = cardFor(pin);
    if (!card) return;
    var id = card.id;
    cardIds[id] = true;
    if (pin.classList.contains('pin') && !pin.getAttribute('aria-label')) {
      var name = card.querySelector('h3');
      pin.setAttribute('aria-label', pin.textContent.trim() + ': ' + (name ? name.textContent.trim() : id));
      pin.setAttribute('title', name ? name.textContent.trim() : id);
    }
    pin.addEventListener('mouseenter', function () { setHot(id, true); });
    pin.addEventListener('mouseleave', function () { setHot(id, false); });
    pin.addEventListener('focus', function () { setHot(id, true); });
    pin.addEventListener('blur', function () { setHot(id, false); });
    pin.addEventListener('click', function () {
      setHot(id, false);
      card.classList.remove('is-flash'); void card.offsetWidth; card.classList.add('is-flash');
    });
  });
  Object.keys(cardIds).forEach(function (id) {
    var card = document.getElementById(id);
    card.addEventListener('mouseenter', function () { setHot(id, true); });
    card.addEventListener('mouseleave', function () { setHot(id, false); });
    card.addEventListener('focusin', function () { setHot(id, true); });
    card.addEventListener('focusout', function () { setHot(id, false); });
  });

  /* ---------- lightbox ---------- */
  var box = el('dialog', { class: 'lightbox', 'aria-label': 'Enlarged screenshot' });
  box.innerHTML = '<button type="button" aria-label="Close">&times;</button><img alt=""><p></p>';
  body.appendChild(box);
  var boxImg = box.querySelector('img'), boxCap = box.querySelector('p');
  function closeBox() { if (box.open) box.close(); }
  box.addEventListener('click', closeBox);
  document.querySelectorAll('figure.shot img, img[data-zoom]').forEach(function (img) {
    img.setAttribute('tabindex', '0');
    img.setAttribute('role', 'button');
    if (!img.getAttribute('aria-label')) img.setAttribute('aria-label', 'Enlarge: ' + (img.alt || 'screenshot'));
    function open() {
      boxImg.src = img.currentSrc || img.src;
      boxImg.alt = img.alt;
      var cap = img.closest('figure') && img.closest('figure').querySelector('figcaption');
      boxCap.textContent = cap ? cap.textContent.trim() : '';
      if (box.showModal) box.showModal(); else box.setAttribute('open', '');
    }
    img.addEventListener('click', open);
    img.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });

  /* ---------- flows: hide the arrow on a box that wrapped to a new row ---------- */
  var flows = document.querySelectorAll('ol.flow');
  function markRows() {
    flows.forEach(function (f) {
      var prevTop = null;
      Array.prototype.forEach.call(f.children, function (li) {
        var t = li.offsetTop;
        li.classList.toggle('row-start', prevTop !== null && t > prevTop + 4 && getComputedStyle(f).flexDirection !== 'column');
        prevTop = t;
      });
    });
  }
  if (flows.length) {
    markRows();
    var rowTimer = 0;
    window.addEventListener('resize', function () { clearTimeout(rowTimer); rowTimer = setTimeout(markRows, 120); });
  }

  /* ---------- print: expand collapsibles ---------- */
  window.addEventListener('beforeprint', function () {
    document.querySelectorAll('details').forEach(function (d) { if (!d.open) { d.setAttribute('data-was-closed', ''); d.open = true; } });
  });
  window.addEventListener('afterprint', function () {
    document.querySelectorAll('details[data-was-closed]').forEach(function (d) { d.open = false; d.removeAttribute('data-was-closed'); });
  });

  /* follow OS theme changes when no explicit choice is stored */
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', applyTheme);
  }
})();
