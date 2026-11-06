/* elementfromgaze — browser side.  <script src="http://127.0.0.1:8770/efg.js"></script>
 *
 *   efg.elementFromGaze()   -> {element, p, dwellMs} | null   the clickable you are looking at
 *   efg.point()             -> {x, y} | null                   gaze in viewport (client) pixels
 *   efg.history             -> [{element, label, start, end}]  what you looked at, and for how long
 *   events on the element   'gazeenter', 'gazeleave', 'gazedwell' (detail: {p, dwellMs})
 *   efg.debug(true)         dot + outline overlay (also: Alt+G, or ?efgdebug in the URL)
 *
 * How it decides (UI-aware decoding): the tracker's error is ~1° (a Gaussian of `sd` px). We drop
 * 48 fixed sample points of that Gaussian around the gaze, ask the page which clickable sits under
 * each (document.elementFromPoint, so real shapes count: pie slices, rounded buttons, overlaps), and
 * the share of points each element gets is its probability. Those shares are smoothed over time
 * (~0.25 s), and the winner must hold > 40%. Gaze in empty space -> null.
 *
 * Screen -> page mapping: the server also sends the OS cursor; whenever the mouse rests inside the
 * page we know exactly where the viewport sits on the screen (and its zoom), no setup needed.
 */
(function () {
  if (window.efg) return;
  const ORIGIN = (document.currentScript && new URL(document.currentScript.src).origin) || "http://127.0.0.1:8770";
  const SELECTOR = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],' +
    '[role=menuitem],[role=tab],[role=option],[role=checkbox],[role=radio],[role=slider],[onclick],[data-gaze],' +
    '[tabindex]:not([tabindex="-1"])';
  const opt = { sdScale: 1.0, tau: 0.25, minP: 0.4, dwellMs: 600, selector: SELECTOR };

  // 48 fixed points of a unit 2-D Gaussian (sunflower pattern: even, deterministic -> no flicker)
  const NORM = [];
  for (let i = 0; i < 48; i++) {
    const u = (i + 0.5) / 48, r = Math.sqrt(-2 * Math.log(1 - u * 0.97)), a = i * 2.39996323;
    NORM.push([r * Math.cos(a), r * Math.sin(a)]);
  }

  // ---- screen (physical px) -> viewport (CSS px) ---------------------------------------
  const map = { scale: window.devicePixelRatio || 1, ox: null, oy: null, pairs: [], fitted: false };
  function guessOffset() {
    const bx = Math.max(0, (window.outerWidth - window.innerWidth) / 2);
    map.ox = window.screenX + bx;
    map.oy = window.screenY + Math.max(0, window.outerHeight - window.innerHeight - bx);
  }
  guessOffset();
  let lastMouse = null, cursorMovedAt = 0, prevCursor = null;
  addEventListener("mousemove", (e) => { lastMouse = { x: e.clientX, y: e.clientY, t: performance.now() }; }, { passive: true });
  addEventListener("resize", () => { map.pairs = []; map.fitted = false; map.scale = window.devicePixelRatio || 1; guessOffset(); });
  function learnMapping(m) {
    const now = performance.now();
    if (!prevCursor || prevCursor[0] !== m.cx || prevCursor[1] !== m.cy) { cursorMovedAt = now; prevCursor = [m.cx, m.cy]; return; }
    // cursor at rest, and the page saw the mouse after it last moved -> an exact pair
    if (!lastMouse || lastMouse.t < cursorMovedAt - 60 || now - cursorMovedAt < 80) return;
    const P = map.pairs, last = P[P.length - 1];
    if (last && last.cx === m.cx && last.cy === m.cy) return;
    P.push({ cx: m.cx, cy: m.cy, x: lastMouse.x, y: lastMouse.y });
    if (P.length > 20) P.shift();
    // scale from the spread of pairs when it is wide enough, else trust devicePixelRatio
    if (P.length >= 3) {
      const mx = P.reduce((s, p) => s + p.x, 0) / P.length, mcx = P.reduce((s, p) => s + p.cx, 0) / P.length;
      let num = 0, den = 0;
      for (const p of P) { num += (p.cx - mcx) * (p.x - mx); den += (p.x - mx) ** 2; }
      if (den > 300 * 300 * P.length / 4) map.scale = num / den;
    }
    const ox = P.map((p) => p.cx / map.scale - p.x).sort((a, b) => a - b);
    const oy = P.map((p) => p.cy / map.scale - p.y).sort((a, b) => a - b);
    map.ox = ox[ox.length >> 1]; map.oy = oy[oy.length >> 1]; map.fitted = true;
  }
  const toClient = (x, y) => ({ x: x / map.scale - map.ox, y: y / map.scale - map.oy });

  // ---- what is clickable -------------------------------------------------------------
  const pointerCache = new WeakMap();
  function clickableOf(el) {
    for (let e = el, d = 0; e && e !== document.body && d < 8; e = e.parentElement, d++) {
      if (e.matches && e.matches(opt.selector)) return e;
      let c = pointerCache.get(e);
      if (c === undefined) { c = getComputedStyle(e).cursor === "pointer"; pointerCache.set(e, c); }
      if (c) {   // JS-bound handlers (pie slices, cards): cursor:pointer; take the outermost such ancestor
        let top = e;
        for (let a = e.parentElement; a && a !== document.body; a = a.parentElement) {
          if (getComputedStyle(a).cursor === "pointer" && !(a.matches && a.matches("body,html"))) top = a; else break;
        }
        return top;
      }
    }
    return null;
  }
  const label = (el) => (el.getAttribute("aria-label") || el.getAttribute("data-gaze") || el.textContent || el.tagName)
    .trim().replace(/\s+/g, " ").slice(0, 40);

  // ---- decoding -------------------------------------------------------------------------
  let gaze = null, lastT = 0, scores = new Map(), current = null, since = 0, dwellFired = false;
  const history = [];
  function step(m) {
    const now = performance.now();
    const dt = lastT ? Math.min((now - lastT) / 1000, 0.5) : 0.033;
    lastT = now;
    const decay = Math.exp(-dt / opt.tau);
    for (const [k, v] of scores) { const nv = v * decay; if (nv < 0.01) scores.delete(k); else scores.set(k, nv); }
    if (!m.ok || map.ox === null) { gaze = null; return decide(now); }
    const g = toClient(m.x, m.y), sd = (m.sd / map.scale) * opt.sdScale;
    gaze = g;
    const hit = new Map();
    for (const [nx, ny] of NORM) {
      const px = g.x + nx * sd, py = g.y + ny * sd;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py);
      const c = el && clickableOf(el);
      if (c) hit.set(c, (hit.get(c) || 0) + 1 / NORM.length);
    }
    for (const [k, w] of hit) scores.set(k, (scores.get(k) || 0) + w * (1 - decay));
    decide(now);
  }
  function decide(now) {
    let best = null, bp = 0;
    for (const [k, v] of scores) if (v > bp) { best = k; bp = v; }
    const next = bp >= opt.minP && best.isConnected ? best : null;
    if (next !== current) {
      if (current) {
        current.removeAttribute("data-gazed");
        current.dispatchEvent(new CustomEvent("gazeleave", { bubbles: true, detail: { dwellMs: now - since } }));
        history.push({ element: current, label: label(current), start: since, end: now });
        if (history.length > 500) history.shift();
      }
      current = next; since = now; dwellFired = false;
      if (current) {
        current.setAttribute("data-gazed", "");
        current.dispatchEvent(new CustomEvent("gazeenter", { bubbles: true, detail: { p: bp } }));
      }
    } else if (current && !dwellFired && now - since >= opt.dwellMs) {
      dwellFired = true;
      current.dispatchEvent(new CustomEvent("gazedwell", { bubbles: true, detail: { p: bp, dwellMs: now - since } }));
    }
    current_p = bp;
    if (dbg.on) drawDebug();
  }
  let current_p = 0;

  // ---- debug overlay ------------------------------------------------------------------
  const dbg = { on: /[?&]efgdebug/.test(location.search) || (document.currentScript && document.currentScript.hasAttribute("data-debug")) };
  let dotEl, boxEl, tagEl;
  function ensureDebug() {
    if (dotEl) return;
    const mk = (css) => { const d = document.createElement("div"); d.style.cssText = "position:fixed;pointer-events:none;z-index:2147483647;" + css; document.documentElement.appendChild(d); return d; };
    dotEl = mk("width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;background:#ff2d20;opacity:.75;");
    boxEl = mk("border:3px solid #4fc3ff;border-radius:6px;box-sizing:border-box;transition:all 60ms linear;");
    tagEl = mk("left:8px;bottom:8px;font:12px/1.3 Consolas,monospace;color:#fff;background:#000a;padding:4px 8px;border-radius:4px;");
  }
  function drawDebug() {
    ensureDebug();
    dotEl.style.display = gaze ? "block" : "none";
    if (gaze) { dotEl.style.left = gaze.x + "px"; dotEl.style.top = gaze.y + "px"; }
    if (current) {
      const r = current.getBoundingClientRect();
      Object.assign(boxEl.style, { display: "block", left: r.left - 4 + "px", top: r.top - 4 + "px", width: r.width + 8 + "px", height: r.height + 8 + "px" });
    } else boxEl.style.display = "none";
    tagEl.textContent = (connected ? "efg ✓" : "efg: sin conexión con " + ORIGIN) + (map.fitted ? " · ventana localizada" : " · mueve el ratón dentro de la página") +
      (current ? ` · ${label(current)} ${(current_p * 100).toFixed(0)}% ${((performance.now() - since) / 1000).toFixed(1)} s` : "");
  }
  function debug(on) {
    dbg.on = on === undefined ? !dbg.on : !!on;
    if (!dbg.on && dotEl) [dotEl, boxEl, tagEl].forEach((e) => (e.style.display = "none"));
    return dbg.on;
  }
  addEventListener("keydown", (e) => { if (e.altKey && (e.key === "g" || e.key === "G")) debug(); });

  // ---- connection -----------------------------------------------------------------------
  let connected = false;
  function connect() {
    const es = new EventSource(ORIGIN + "/gaze");
    es.onopen = () => { connected = true; };
    es.onmessage = (ev) => { const m = JSON.parse(ev.data); learnMapping(m); step(m); };
    es.onerror = () => { connected = false; if (dbg.on) drawDebug(); };
  }
  connect();

  window.efg = {
    elementFromGaze: () => (current ? { element: current, p: current_p, dwellMs: performance.now() - since } : null),
    point: () => (gaze ? { x: gaze.x, y: gaze.y } : null),
    history, options: opt, debug,
    get mapping() { return { ...map, pairs: map.pairs.length }; },
  };
})();
