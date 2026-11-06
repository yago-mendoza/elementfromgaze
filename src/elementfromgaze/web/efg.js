/* elementfromgaze — browser side.  <script src="http://127.0.0.1:8770/efg.js" data-show></script>
 *
 *   efg.elementFromGaze()   -> {element, p, dwellMs, attractor} | null   what you are looking at
 *   efg.point()             -> {x, y} | null                   raw gaze in viewport (client) pixels
 *   efg.history             -> [{element, label, start, end}]  what you looked at, and for how long
 *   events on the element   'gazeenter', 'gazeleave', 'gazedwell' (detail: {p, dwellMs})
 *   efg.show(true)          the gaze pointer (also: data-show on the script tag)
 *   efg.debug(true)         pointer + raw dot + status line (Alt+G, or ?efgdebug in the URL)
 *
 * ATTRACTORS — <button data-gaze>  (optional data-gaze-weight="3", data-gaze-learn="off")
 * The pointer snaps onto an attractor when that is the likely explanation of the measurement.
 *   Measurement model: g = true gaze + e,  e ~ N(0, diag(sx², sy²))  (sx, sy sent by the server;
 *   measured per-fixation error after warm-up, ~0.7° / 1.0°). Mostly a slowly varying BIAS, so
 *   looking longer does not shrink it: we use the full error, not error/√frames.
 *   Hypothesis k, "looking somewhere inside attractor k" (uniform over its rectangle R_k):
 *     p(g | k) = [Φ((x1-gx)/sx) - Φ((x0-gx)/sx)] · [Φ((y1-gy)/sy) - Φ((y0-gy)/sy)] / |R_k|
 *   Hypothesis 0, "looking at anything else": p(g | 0) = 1 / |viewport|
 *   Posterior P(k | g) = w_k p(g|k) / (w_0 p(g|0) + Σ_j w_j p(g|j)), w = data-gaze-weight.
 *   Snap when P > 0.8 (stay while P > 0.5). No radius to tune: for an isolated 200×60 button this
 *   snaps within ~2σ of its edge; two attractors side by side split the posterior, neither passes
 *   0.8, and the pointer does not pretend to know which one you mean.
 *
 * LEARNING FROM ATTRACTORS: an unambiguous snap (P > 0.95, runner-up < 0.03, ≥ 0.4 s, mapping known)
 * is posted to the engine as a weak label (weight 0.35 vs a click's 1.0, at most once per element
 * every 3 s). The engine treats it like a click: global correction + eye-image fine-tuning.
 *
 * Other clickables (no data-gaze) are still decoded by elementFromGaze(): 48 fixed points of the
 * error Gaussian, document.elementFromPoint on each (real shapes: pie slices, rounded corners),
 * share of points = probability, smoothed ~0.25 s, with hysteresis.
 *
 * Screen -> page mapping: the server also sends the OS cursor; whenever the mouse rests inside the
 * page we know exactly where the viewport sits on the screen (and its zoom), no setup needed.
 */
(function () {
  if (window.efg) return;
  const SCRIPT = document.currentScript;
  const ORIGIN = (SCRIPT && new URL(SCRIPT.src).origin) || "http://127.0.0.1:8770";
  const SELECTOR = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],' +
    '[role=menuitem],[role=tab],[role=option],[role=checkbox],[role=radio],[role=slider],[onclick],[data-gaze],' +
    '[tabindex]:not([tabindex="-1"])';
  const opt = {
    sdScale: 1.0, tau: 0.25, minP: 0.4, stayP: 0.22, switchP: 0.5, lead: 1.3, dwellMs: 600, selector: SELECTOR,
    snapP: 0.8, unsnapP: 0.5, bgWeight: 1.0,                                  // attractors
    learn: true, learnP: 0.95, learnRunnerUp: 0.03, learnMinMs: 400, learnGapMs: 3000, learnConf: 0.35,
  };

  // ---- maths -------------------------------------------------------------------------
  function erf(x) {   // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  }
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  // 48 fixed points of a unit 2-D Gaussian (sunflower pattern: even, deterministic -> no flicker)
  const NORM = [];
  for (let i = 0; i < 48; i++) {
    const u = (i + 0.5) / 48, r = Math.sqrt(-2 * Math.log(1 - u * 0.97)), a = i * 2.39996323;
    NORM.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  // critically damped spring, exact step: still on fixations, ~150 ms glide on saccades, no overshoot
  function spring(s, target, dt, w) {
    if (!s.p) { s.p = target.slice(); s.v = target.map(() => 0); return s.p; }
    const ex = Math.exp(-w * dt);
    for (let i = 0; i < target.length; i++) {
      const e0 = s.p[i] - target[i], c = s.v[i] + w * e0;
      s.p[i] = target[i] + (e0 + c * dt) * ex;
      s.v[i] = (s.v[i] - w * c * dt) * ex;
    }
    return s.p;
  }

  // ---- screen (physical px) <-> viewport (CSS px) ---------------------------------------
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
    if (P.length >= 3) {   // scale from the spread of pairs when wide enough, else trust devicePixelRatio
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
  const toScreen = (x, y) => ({ x: (x + map.ox) * map.scale, y: (y + map.oy) * map.scale });

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

  // ---- attractors: closed-form posterior ------------------------------------------------
  let attr = { list: [], at: 0 };
  function attractors(now) {
    if (now - attr.at > 400) attr = { list: [...document.querySelectorAll("[data-gaze]")], at: now };
    return attr.list;
  }
  function attractorPosterior(g, sx, sy, now) {
    const area = innerWidth * innerHeight, out = [];
    let Z = opt.bgWeight / area;
    for (const el of attractors(now)) {
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
      const w = parseFloat(el.getAttribute("data-gaze-weight")) || 1;
      const px = Phi((r.right - g.x) / sx) - Phi((r.left - g.x) / sx);
      const py = Phi((r.bottom - g.y) / sy) - Phi((r.top - g.y) / sy);
      const L = (w * px * py) / (r.width * r.height);
      if (L > 0) { out.push([el, L]); Z += L; }
    }
    return out.map(([el, L]) => [el, L / Z]).sort((a, b) => b[1] - a[1]);
  }

  // ---- decoding -------------------------------------------------------------------------
  let gaze = null, lastT = 0, scores = new Map(), current = null, since = 0, dwellFired = false, current_p = 0;
  let snap = null, snapP = 0, snapSince = 0, snapServerT = 0, runnerUp = 0, serverT = 0;
  const learnedAt = new WeakMap(), history = [], stats = { labels: 0 };
  let lastSd = [50, 70], lastSdScreen = 55;
  function step(m) {
    const now = performance.now();
    serverT = m.t;
    const dt = lastT ? Math.min((now - lastT) / 1000, 0.5) : 0.033;
    lastT = now;
    const decay = Math.exp(-dt / opt.tau);
    for (const [k, v] of scores) { const nv = v * decay; if (nv < 0.01) scores.delete(k); else scores.set(k, nv); }
    if (!m.ok || map.ox === null) { gaze = null; setSnap(null, 0, now, m); return decide(now); }
    const g = toClient(m.x, m.y);
    const sx = ((m.sdx || m.sd) / map.scale) * opt.sdScale, sy = ((m.sdy || m.sd) / map.scale) * opt.sdScale;
    gaze = g; lastSd = [sx, sy]; lastSdScreen = m.sd || 55;
    // attractors first: closed form
    const post = attractorPosterior(g, sx, sy, now);
    const cur = snap && post.find((p) => p[0] === snap);
    if (cur && cur[1] >= opt.unsnapP && !(post[0][0] !== snap && post[0][1] >= opt.snapP)) setSnap(snap, cur[1], now, m);
    else if (post.length && post[0][1] >= opt.snapP) setSnap(post[0][0], post[0][1], now, m);
    else setSnap(null, 0, now, m);
    runnerUp = snap ? (post.find((p) => p[0] !== snap) || [null, 0])[1] : 0;
    maybeLearn(now);
    // every clickable: Monte Carlo over the error Gaussian
    const hit = new Map();
    for (const [nx, ny] of NORM) {
      const px = g.x + nx * sx, py = g.y + ny * sy;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py);
      const c = el && clickableOf(el);
      if (c) hit.set(c, (hit.get(c) || 0) + 1 / NORM.length);
    }
    for (const [k, w] of hit) scores.set(k, (scores.get(k) || 0) + w * (1 - decay));
    decide(now);
  }
  function setSnap(el, p, now, m) {
    if (el !== snap) { snap = el; snapSince = now; snapServerT = m.t; }
    snapP = p;
  }
  function maybeLearn(now) {
    if (!opt.learn || !snap || !map.fitted || snap.getAttribute("data-gaze-learn") === "off") return;
    if (snapP < opt.learnP || runnerUp > opt.learnRunnerUp || now - snapSince < opt.learnMinMs) return;
    const last = learnedAt.get(snap) || -1e9;
    if (now - last < opt.learnGapMs) return;
    learnedAt.set(snap, now);
    const r = snap.getBoundingClientRect(), c = toScreen(r.left + r.width / 2, r.top + r.height / 2);
    const t1 = serverT, t0 = Math.max(snapServerT + 0.15, t1 - 1.2);
    // label noise per axis, screen px: where inside the element you looked (uniform: w²/12) on top
    // of a small base; a wide, short button pins y far better than x
    const base = 0.5 * lastSdScreen, ws = r.width * map.scale, hs = r.height * map.scale;
    const sdx = Math.sqrt(base * base + ws * ws / 12), sdy = Math.sqrt(base * base + hs * hs / 12);
    stats.labels++;
    fetch(ORIGIN + "/label", { method: "POST", body: JSON.stringify({ t0, t1, x: c.x, y: c.y, sdx, sdy, conf: opt.learnConf, p: snapP, label: label(snap) }) }).catch(() => {});
  }
  function decide(now) {
    let next, np;
    if (snap) { next = snap; np = snapP; }       // an attractor wins over the generic decoding
    else {
      let best = null, bp = 0;
      for (const [k, v] of scores) if (v > bp) { best = k; bp = v; }
      const cp = current && current.isConnected && !current.hasAttribute("data-gaze") ? scores.get(current) || 0 : 0;
      if (cp >= opt.stayP) next = best !== current && bp >= opt.switchP && bp > cp * opt.lead ? best : current;
      else next = best && bp >= opt.minP && best.isConnected ? best : null;
      if (next && next.hasAttribute("data-gaze")) next = null;    // attractors only through the posterior
      np = next ? scores.get(next) || 0 : 0;
    }
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
        current.dispatchEvent(new CustomEvent("gazeenter", { bubbles: true, detail: { p: np } }));
      }
    } else if (current && !dwellFired && now - since >= opt.dwellMs) {
      dwellFired = true;
      current.dispatchEvent(new CustomEvent("gazedwell", { bubbles: true, detail: { p: np, dwellMs: now - since } }));
    }
    current_p = np;
  }

  // ---- drawing: the pointer morphs from a soft circle into the attractor it snaps to ---------
  const view = {
    show: !!(SCRIPT && SCRIPT.hasAttribute("data-show")),
    debug: /[?&]efgdebug/.test(location.search) || !!(SCRIPT && SCRIPT.hasAttribute("data-debug")),
  };
  let lens, dot, tag;
  function ensureEls() {
    if (lens) return;
    const mk = (css) => { const d = document.createElement("div"); d.style.cssText = "position:fixed;pointer-events:none;z-index:2147483647;display:none;" + css; document.documentElement.appendChild(d); return d; };
    lens = mk("box-sizing:border-box;border:2px solid rgba(79,195,255,.9);background:rgba(79,195,255,.16);box-shadow:0 0 22px rgba(79,195,255,.45);");
    dot = mk("width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;background:#ff2d20;opacity:.8;");
    tag = mk("left:8px;bottom:8px;font:12px/1.3 Consolas,monospace;color:#fff;background:#000a;padding:4px 8px;border-radius:4px;");
  }
  const sLens = {}, sDot = {};
  let rafT = 0, lensOn = false, radius = 0;
  function frame(t) {
    const dt = rafT ? Math.min((t - rafT) / 1000, 0.1) : 0.016;
    rafT = t;
    if (view.show || view.debug) draw(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  function draw(dt) {
    ensureEls();
    const R = 13;
    let target = null, round = 50;
    const box = current && current.isConnected ? current.getBoundingClientRect() : null;
    if (box && (snap || view.debug)) { target = [box.left - 4, box.top - 4, box.width + 8, box.height + 8]; round = 10; }
    else if (gaze) target = [gaze.x - R, gaze.y - R, 2 * R, 2 * R];
    if (target) {
      if (!lensOn) { sLens.p = null; lensOn = true; }
      const [l, tp, w, h] = spring(sLens, target, dt, 30);
      radius += (round - radius) * Math.min(1, dt * 18);
      Object.assign(lens.style, { display: "block", left: l + "px", top: tp + "px", width: w + "px", height: h + "px",
        borderRadius: round === 50 ? "50%" : radius.toFixed(1) + "px" });
    } else { lens.style.display = "none"; lensOn = false; }
    if (view.debug && gaze) {
      const [x, y] = spring(sDot, [gaze.x, gaze.y], dt, 40);
      Object.assign(dot.style, { display: "block", left: x + "px", top: y + "px" });
    } else dot.style.display = "none";
    tag.style.display = view.debug ? "block" : "none";
    if (view.debug) tag.textContent = (connected ? "efg ✓" : "efg: sin conexión con " + ORIGIN) +
      (map.fitted ? " · ventana localizada" : " · mueve el ratón dentro de la página y déjalo quieto") +
      (current ? ` · ${label(current)} ${(current_p * 100).toFixed(0)}%${snap ? " (imán)" : ""} ${((performance.now() - since) / 1000).toFixed(1)} s` : "") +
      ` · etiquetas por mirada ${stats.labels}`;
  }
  function show(on) { view.show = on === undefined ? !view.show : !!on; if (!view.show && !view.debug && lens) [lens, dot, tag].forEach((e) => (e.style.display = "none")); return view.show; }
  function debug(on) { view.debug = on === undefined ? !view.debug : !!on; if (!view.show && !view.debug && lens) [lens, dot, tag].forEach((e) => (e.style.display = "none")); return view.debug; }
  addEventListener("keydown", (e) => { if (e.altKey && (e.key === "g" || e.key === "G")) debug(); });

  // ---- connection -----------------------------------------------------------------------
  let connected = false;
  function connect() {
    const es = new EventSource(ORIGIN + "/gaze");
    es.onopen = () => { connected = true; };
    es.onmessage = (ev) => { const m = JSON.parse(ev.data); learnMapping(m); step(m); };
    es.onerror = () => { connected = false; };
  }
  connect();

  window.efg = {
    elementFromGaze: () => (current ? { element: current, p: current_p, dwellMs: performance.now() - since, attractor: current === snap } : null),
    point: () => (gaze ? { x: gaze.x, y: gaze.y } : null),
    posterior: () => (gaze ? attractorPosterior(gaze, lastSd[0], lastSd[1], performance.now()).slice(0, 5).map(([el, p]) => ({ label: label(el), p })) : []),
    history, options: opt, show, debug, origin: ORIGIN, stats,
    get mapping() { return { ...map, pairs: map.pairs.length }; },
  };
})();
