/* elementfromgaze — browser side.  <script src="http://127.0.0.1:8770/efg.js" data-show></script>
 *
 *   efg.elementFromGaze()   -> {element, p, dwellMs, attractor} | null   what you are looking at
 *   efg.point()             -> {x, y} | null                   raw gaze in viewport (client) pixels
 *   efg.history             -> [{element, label, start, end}]  what you looked at, and for how long
 *   efg.reading()           -> {p, line, word, offset, ...} | null      reading: which line / word (see READING)
 *   efg.textLines(el)       -> [{left, right, top, bottom, cy, index}]   text lines of a block, reading order
 *   events on the element   'gazeenter', 'gazeleave', 'gazedwell' (detail: {p, dwellMs})
 *   events on window        'efgfixation' (every fixation, with the reading filter's view), 'efglabel'
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
 * READING — text is read line by line, left to right; the return sweep to the next line is a big
 * leftward saccade. The scanpath's SHAPE is precise (fixation-to-fixation ~0.1°) while its position
 * carries the ~1° bias, so a joint filter tracks, per fixation:
 *   mode m ∈ {reading, not}, line ℓ (DOM text lines near the gaze), vertical gain g (5-point grid)
 *   and vertical offset δ (2 px grid):   y | ℓ,g,δ ~ N(Yc + g(c_ℓ - Yc) + slope·(x - Xc) + δ, σy²)
 *   transitions: same line moving right / return sweep to the next line (landing ~3.5 chars in) /
 *   rare regressions and skips; a "posture jump" term resets δ to its prior (change-point); not
 *   reading = an isotropic saccade model. Identifiability: in a uniform paragraph, (ℓ+k, δ-k·spacing)
 *   explains the data equally well, so the absolute line comes from layout anchors (short
 *   paragraph-final lines, paragraph gaps, block edges). Two copies run: DISPLAY (prior on δ = the
 *   engine's error) and LABEL (prior twice as wide, forgets in 10 s): the label copy must find the
 *   line from the layout, not from where the engine already thinks you look.
 * LEARNING FROM READING: a fully read line (≥ 4 fixations, start to end, reading-sized forward
 * saccades) whose line the label copy's fixed-lag smoother is ≥ 99.5 % sure of is posted as a
 * VERTICAL-only label (axis "y", sd 30 px, weight 0.3; ≤ 1 per 1.5 s and per line). Guards: engine
 * correction changes (ey) shift δ, engine discontinuities (gen) reset it, and a click on another
 * line than the one being "read" resets δ and pauses reading labels for 60 s.
 * Simulated (scratchpad reading/): line 27 % → 86 %, offset ~2 px after ~5-8 lines, 0-0.2 % wrong labels.
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
  const learnedAt = new WeakMap(), history = [], stats = { labels: 0, readLabels: 0, readGuards: 0, readResets: 0, readMs: 0 };
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

  // =======================================================================================
  // ---- READING: fixations -> (mode, line, gain, offset) filter -> highlight + vertical labels
  // =======================================================================================
  // Units: SCREEN px for tracker quantities (converted with map.scale), CLIENT px for the page.
  const RD = {
    // learn: OFF until validated on real reading. First real test (2026-11-06): the user sat 37%
    // closer than in the warm-up, the vertical error was ~3.5° (5 lines), and one label at p=0.997
    // named line 15 for line 19. Highlight only; turn on with efg.readingOptions.learn = true.
    on: true, learn: false,
    fixT: 36, fixWin: 8, fixMinN: 2, fixMaxGap: 0.25,            // fixations: 3σ of 0.2° jitter (screen px), s
    jitter: 12, rw: 3, sx: 35, margin: 15, dxRw: 3, dxSd0: 30,   // screen px (rw, dxRw per √s)
    gains: [0.88, 0.94, 1, 1.06, 1.12], gainSd: 0.05, gainMix: 0.01,
    tauShow: 120, tauLabel: 10, pRN: 0.03, pNR: 0.03, pBack: 0.015, pSkip: 0.01, eps: 0.03,
    maxCells: 161, maxLines: 40, probeMs: 300, window: 32,
    labelP: 0.995, labelSdy: 30, labelConf: 0.3, labelGap: 1.5, lineRelabelS: 30, maxLabelAge: 12, pauseMs: 60000,
    showP: 0.8, keepP: 0.6,
    // Don't highlight a line that only fits by assuming a big vertical error: on the first real
    // tests the tracker was 3-3.5° off (4-5 lines) and the filter, confident, highlighted lines far
    // above the one being read. |dy| above maxDy x the tracker's vertical sd = not plausible = hide.
    maxDy: 1.5,
  };
  const NOREAD = 'input,textarea,select,button,script,style,noscript,svg,[contenteditable=""],[contenteditable="true"],[data-gaze-read="off"]';

  // ---- text lines from the DOM (cached per block, re-anchored with one getBoundingClientRect)
  let layoutVer = 0, layoutBump = 0;
  const bumpLayout = () => { if (!layoutBump) layoutBump = requestAnimationFrame(() => { layoutBump = 0; layoutVer++; }); };
  addEventListener("resize", bumpLayout);
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", bumpLayout);
  const mo = new MutationObserver((recs) => {
    for (const r of recs) { const t = r.target.nodeType === 1 ? r.target : r.target.parentElement; if (t && !t.hasAttribute("data-efg")) { bumpLayout(); return; } }
  });
  if (document.documentElement) mo.observe(document.documentElement, { subtree: true, childList: true, characterData: true });
  const scrollTimes = [];
  addEventListener("scroll", () => { scrollTimes.push(performance.now()); if (scrollTimes.length > 64) scrollTimes.shift(); }, { capture: true, passive: true });
  const scrolledDuring = (a, b) => scrollTimes.some((t) => t >= a - 150 && t <= b);
  const BLOCKY = new Set(["block", "list-item", "table-cell", "flow-root", "table-caption", "flex", "grid", "inline-block"]);
  const dispCache = new WeakMap(), blockIds = new WeakMap(), lineCache = new WeakMap(), byId = new Map();
  let nextBlockId = 1;
  function isBlock(e) {
    let d = dispCache.get(e);
    if (!d || d[0] !== layoutVer) { d = [layoutVer, getComputedStyle(e).display]; dispCache.set(e, d); }
    return BLOCKY.has(d[1]);
  }
  function blockOf(node) {
    for (let e = node.nodeType === 1 ? node : node.parentElement; e && e !== document.body && e !== document.documentElement; e = e.parentElement) {
      if (isBlock(e)) return e;
    }
    return null;
  }
  function blockId(b) {
    let id = blockIds.get(b);
    if (!id) { id = nextBlockId++; blockIds.set(b, id); }
    byId.set(id, new WeakRef(b));
    return id;
  }
  const rng = document.createRange();
  function measureBlock(block, br) {
    const rects = [];
    const tw = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (/\S/.test(n.data) && blockOf(n) === block && !(n.parentElement && n.parentElement.closest(NOREAD)) ? 1 : 3),
    });
    for (let n; (n = tw.nextNode());) { rng.selectNodeContents(n); for (const r of rng.getClientRects()) if (r.width > 0.5 && r.height > 2) rects.push(r); }
    const fs = parseFloat(getComputedStyle(block).fontSize) || 16;
    rects.sort((a, b) => (a.top + a.bottom) - (b.top + b.bottom) || a.left - b.left);
    const lines = [];
    for (const r of rects) {   // merge fragments of the same line box
      let L = null;
      for (const q of lines) {
        const ov = Math.min(q.b, r.bottom) - Math.max(q.t, r.top);
        if (ov > 0.5 * Math.min(q.b - q.t, r.height) && r.left < q.r + 3 * fs && r.right > q.l - 3 * fs) { L = q; break; }
      }
      if (L) { L.l = Math.min(L.l, r.left); L.r = Math.max(L.r, r.right); L.t = Math.min(L.t, r.top); L.b = Math.max(L.b, r.bottom); }
      else lines.push({ l: r.left, r: r.right, t: r.top, b: r.bottom });
    }
    const cols = [];   // multi-column blocks: columns by horizontal overlap, LTR, each top to bottom
    for (const L of lines) {
      let c = cols.find((c) => Math.min(c.r, L.r) - Math.max(c.l, L.l) > 0.3 * Math.min(c.r - c.l, L.r - L.l));
      if (!c) { c = { l: L.l, r: L.r, lines: [] }; cols.push(c); }
      c.l = Math.min(c.l, L.l); c.r = Math.max(c.r, L.r); c.lines.push(L);
    }
    cols.sort((a, b) => a.l - b.l);
    const out = [];
    for (const c of cols) { c.lines.sort((a, b) => a.t - b.t); for (const L of c.lines) out.push(L); }
    return { ver: layoutVer, bl: br.left, bt: br.top, w: br.width, h: br.height, fs,
      lines: out.map((L, i) => ({ index: i, l: L.l, r: L.r, t: L.t, b: L.b })) };
  }
  // lines of a block, CLIENT px now, reading order
  function blockLines(block) {
    const br = block.getBoundingClientRect();
    let c = lineCache.get(block);
    if (!c || c.ver !== layoutVer || Math.abs(c.w - br.width) > 0.5 || Math.abs(c.h - br.height) > 0.5) { c = measureBlock(block, br); lineCache.set(block, c); }
    const dx = br.left - c.bl, dy = br.top - c.bt, id = blockId(block), cw = 0.5 * c.fs;
    return c.lines.map((L) => ({ id: id + ":" + L.index, block, index: L.index, l: L.l + dx, r: L.r + dx, t: L.t + dy, b: L.b + dy,
      c: (L.t + L.b) / 2 + dy, cw }));
  }
  function textNodeAt(x, y) {
    if (document.caretPositionFromPoint) { const p = document.caretPositionFromPoint(x, y); return p && p.offsetNode; }
    if (document.caretRangeFromPoint) { const r = document.caretRangeFromPoint(x, y); return r && r.startContainer; }
    return null;
  }
  function nextTextBlock(block) {   // the next block in document order that has text (reading continues there)
    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (/\S/.test(n.data) && !block.contains(n) && !(n.parentElement && n.parentElement.closest(NOREAD)) ? 1 : 3),
    });
    tw.currentNode = block;
    for (let n, k = 0; (n = tw.nextNode()) && k < 50; k++) {
      if (block.compareDocumentPosition(n) & Node.DOCUMENT_POSITION_FOLLOWING) { const b = blockOf(n); if (b) return b; }
    }
    return null;
  }
  let probe = { at: -1e9, x: 0, y: 0, blocks: [] };
  function probeBlocks(g, reach, now) {
    if (now - probe.at < RD.probeMs && Math.hypot(g.x - probe.x, g.y - probe.y) < 80) return probe.blocks;
    const found = new Set();
    for (const fy of [-1, -0.5, 0, 0.5, 1]) for (const fx of [-250, 0, 250]) {
      const x = g.x + fx, y = g.y + fy * reach;
      if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
      const n = textNodeAt(x, y);
      if (!n || n.nodeType !== 3 || !/\S/.test(n.data) || (n.parentElement && n.parentElement.closest(NOREAD))) continue;
      const b = blockOf(n);
      if (b) found.add(b);
    }
    probe = { at: now, x: g.x, y: g.y, blocks: [...found] };
    return probe.blocks;
  }

  // ---- the filter --------------------------------------------------------------------------
  const K = RD.gains.length;
  let G = null, NG = 0, gstep = 2, sigB = 0, gridScale = 0;
  const copies = { show: null, label: null };
  function npdf(x, m, s) { const z = (x - m) / s; return Math.exp(-0.5 * z * z) / (s * 2.5066283); }
  function makeCopy(priorSd, tau) {
    const pi0 = new Float64Array(K * NG);
    let Z = 0;
    for (let k = 0; k < K; k++) {
      const pg = Math.exp(-0.5 * ((RD.gains[k] - 1) / RD.gainSd) ** 2);
      for (let j = 0; j < NG; j++) { const v = pg * Math.exp(-0.5 * (G[j] / priorSd) ** 2); pi0[k * NG + j] = v; Z += v; }
    }
    for (let i = 0; i < pi0.length; i++) pi0[i] /= Z;
    return { pi0, tau, ids: [], aR: new Float64Array(0), aN: null, prevFix: null, lastT: null, ring: [] };
  }
  function setupGrid(m) {
    const sb = (m.sdy || m.sd || 61) / map.scale;
    if (G && Math.abs(sb - sigB) < 0.1 * sigB && gridScale === map.scale) return;
    sigB = sb; gridScale = map.scale;
    const half = 4 * sb;
    gstep = Math.max(2, (2 * half) / (RD.maxCells - 1));
    NG = Math.round((2 * half) / gstep) + 1;
    G = new Float64Array(NG);
    for (let j = 0; j < NG; j++) G[j] = -half + j * gstep;
    copies.show = makeCopy(sb, RD.tauShow);
    copies.label = makeCopy(2 * sb, RD.tauLabel);
    segs.length = 0; curSeg = null;
    st0.dx = 0; st0.dxVar = (RD.dxSd0 / map.scale) ** 2; st0.dxT = null;
  }
  // shared per-fixation context: transitions + emission parameters (identical for both copies)
  function buildCtx(f, prev, lines, sc) {
    const L = lines.length, cs = new Float64Array(L);
    const m = RD.margin / sc, dxVar = st0.dxVar, sxe = Math.sqrt((RD.sx / sc) ** 2 + dxVar);
    const xq = f.x - st0.dx, B = new Float64Array(L), hany = new Float64Array(L);
    for (let l = 0; l < L; l++) {
      const Ln = lines[l];
      B[l] = Phi((Ln.r + m - xq) / sxe) - Phi((Ln.l - m - xq) / sxe);
      hany[l] = B[l] / (Ln.r - Ln.l + 2 * m + 2 * sxe);
      cs[l] = Ln.c;
    }
    const same = new Float64Array(L), next = new Float64Array(L), back = new Float64Array(L), skipW = new Float64Array(L), hn2r = new Float64Array(L);
    let dx = 0, dy = 0, pRN = RD.pRN, pNR = RD.pNR, f2 = 1 / (innerWidth * innerHeight);
    if (prev) {
      dx = f.x - prev.x; dy = f.y - prev.y;
      if (f.t0 - prev.t1 > 0.6) { pRN = 0.2; pNR = 0.2; }
      const xp = prev.x - st0.dx;
      for (let l = 0; l < L; l++) {
        const cw = lines[l].cw;
        const hs = 0.85 * npdf(dx, 8 * cw, 4 * cw) + 0.15 * npdf(dx, -6 * cw, 6 * cw);
        const prs = 0.02 + 0.93 * Phi((xp - (lines[l].r - 12 * cw)) / (5 * cw));
        same[l] = (1 - prs) * (1 - RD.pBack - RD.pSkip) * hs * B[l];
        if (l + 1 < L) { const c2 = lines[l + 1].cw; next[l] = prs * npdf(xq, lines[l + 1].l + 3.5 * c2, Math.sqrt(9 * c2 * c2 + dxVar)); }
        if (l > 0) back[l] = RD.pBack * hany[l - 1];
        skipW[l] = (RD.pSkip / 6) * hany[l];
      }
      // not reading: isotropic saccades (short + long) + some horizontal scanning (rows, toolbars)
      const a = Math.sqrt(dx * dx + dy * dy + 25 / (sc * sc)), la = Math.log(a);
      const m1 = Math.log(100 / sc), m2 = Math.log(300 / sc);
      const pa = 0.5 * Math.exp(-0.5 * ((la - m1) / 0.8) ** 2) / (a * 0.8 * 2.5066) + 0.5 * Math.exp(-0.5 * ((la - m2) / 0.6) ** 2) / (a * 0.6 * 2.5066);
      const h1 = 15 / sc, h2 = 150 / sc;
      f2 = 0.8 * pa / (2 * Math.PI * a) + 0.2 * Math.exp(-0.5 * (dy / h1) ** 2) / (h1 * 2.5066) * Math.exp(-Math.abs(dx) / h2) / (2 * h2);
    }
    for (let l = 0; l < L; l++) hn2r[l] = hany[l] / L;
    const Yc = toClient(0, (lastMsg.H || 1200) / 2).y, Xc = toClient((lastMsg.W || 1920) / 2, 0).x;
    const mu = new Float64Array(L * K);
    for (let l = 0; l < L; l++) for (let k = 0; k < K; k++) mu[l * K + k] = Yc + RD.gains[k] * (cs[l] - Yc) + st0.slope * (f.x - Xc);
    const jit = RD.jitter / sc, sig = Math.sqrt(jit * jit * 3 / Math.max(f.n, 1) + 25 / (sc * sc));
    return { lines, ids: lines.map((L) => L.id), L, same, next, back, skipW, hn2r, mu, sig, y: f.y, f2, pRN, pNR, hany,
      floor: RD.eps / (400 / sc), Yc, Xc };
  }
  function emission(ctx, out) {   // E[l,k,j] = (1-eps) N(y - mu - G_j; sig) + floor
    const { L, mu, sig, y, floor } = ctx, c0 = (1 - RD.eps) / (sig * 2.5066283), inv = 1 / sig;
    for (let lk = 0; lk < L * K; lk++) {
      const base = lk * NG, d0 = y - mu[lk];
      for (let j = 0; j < NG; j++) {
        const z = (d0 - G[j]) * inv;
        out[base + j] = z > 7 || z < -7 ? floor : c0 * Math.exp(-0.5 * z * z) + floor;
      }
    }
    return out;
  }
  function blurKernel(sdc) {
    if (sdc <= 0.3) return null;
    const r = Math.ceil(3 * sdc), k = new Float64Array(2 * r + 1);
    let s = 0;
    for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sdc) ** 2); s += k[i + r]; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    return k;
  }
  const tmpRow = new Float64Array(1024);
  // blur along δ, mix gains, then the posture-jump (change-point) term; rows = L (aR) or 1 (aN)
  function diffuse(a, rows, kern, rho, pi0, perRowSum) {
    const KN = K * NG;
    if (kern) {
      const r = (kern.length - 1) >> 1;
      for (let row = 0; row < rows * K; row++) {
        const base = row * NG;
        for (let j = 0; j < NG; j++) tmpRow[j] = a[base + j];
        for (let j = 0; j < NG; j++) {
          let s = 0;
          const lo = Math.max(0, j - r), hi = Math.min(NG - 1, j + r);
          for (let q = lo; q <= hi; q++) s += kern[q - j + r] * tmpRow[q];
          a[base + j] = s;
        }
      }
    }
    if (RD.gainMix && K > 1) {
      const e = RD.gainMix, tmp = new Float64Array(KN);
      for (let row = 0; row < rows; row++) {
        const base = row * KN;
        for (let i = 0; i < KN; i++) tmp[i] = a[base + i];
        for (let k = 0; k < K; k++) {
          const up = k > 0 ? k - 1 : k, dn = k < K - 1 ? k + 1 : k;
          for (let j = 0; j < NG; j++) a[base + k * NG + j] = (1 - e) * tmp[k * NG + j] + 0.5 * e * (tmp[up * NG + j] + tmp[dn * NG + j]);
        }
      }
    }
    if (rho > 0) {
      for (let row = 0; row < rows; row++) {
        const base = row * KN;
        let s = 0;
        if (perRowSum) for (let i = 0; i < KN; i++) s += a[base + i];
        else for (let i = 0; i < KN; i++) s += a[base + i] * pi0[i];
        for (let i = 0; i < KN; i++) a[base + i] = (1 - rho) * a[base + i] + rho * s * (perRowSum ? pi0[i] : 1);
      }
    }
  }
  function remap(cp, ids) {   // carry the belief over to a new candidate line set, by line id
    const KN = K * NG, out = new Float64Array(ids.length * KN);
    if (cp.ids.length) {
      const old = new Map(cp.ids.map((id, i) => [id, i]));
      for (let l = 0; l < ids.length; l++) { const o = old.get(ids[l]); if (o !== undefined) out.set(cp.aR.subarray(o * KN, (o + 1) * KN), l * KN); }
    }
    cp.aR = out; cp.ids = ids;
  }
  function stepCopy(cp, ctx, E, f, store) {
    const KN = K * NG, L = ctx.L;
    if (!cp.prevFix) {   // first fixation, or a new chain after a gap / off-text: keep the offset belief
      const b = new Float64Array(KN);
      if (cp.aN) {
        for (let i = 0; i < KN; i++) b[i] = cp.aN[i];
        for (let l = 0; l < cp.ids.length; l++) for (let i = 0; i < KN; i++) b[i] += cp.aR[l * KN + i];
        const dt = Math.max(f.t1 - (cp.lastT === null ? f.t1 : cp.lastT), 0.01);
        diffuse(b, 1, blurKernel((RD.rw / map.scale) * Math.sqrt(dt) / gstep), 1 - Math.exp(-dt / cp.tau), cp.pi0, true);
        let s = 0;
        for (let i = 0; i < KN; i++) s += b[i];
        if (s > 0 && isFinite(s)) for (let i = 0; i < KN; i++) b[i] /= s; else b.set(cp.pi0);
      } else b.set(cp.pi0);
      cp.ids = ctx.ids; cp.aR = new Float64Array(L * KN); cp.aN = new Float64Array(KN);
      // a chain starts as "probably not reading": one fixation on text is not reading (needs the saccades)
      // (location alone is no evidence of reading: each part is normalised on its own)
      let zr = 0;
      for (let l = 0; l < L; l++) for (let i = 0; i < KN; i++) zr += (cp.aR[l * KN + i] = (ctx.hany[l] / L) * b[i] * E[l * KN + i]);
      for (let i = 0; i < KN; i++) cp.aN[i] = 0.85 * b[i];
      if (zr > 0) for (let q = 0; q < L * KN; q++) cp.aR[q] *= 0.15 / zr;
      if (!normalize(cp)) return;
      cp.prevFix = f; cp.lastT = f.t1;
      if (store) pushRing(cp, ctx, f, 0, 0, true);
      return;
    }
    remap(cp, ctx.ids);
    const dt = Math.max(f.t1 - cp.prevFix.t1, 0.01), sc = map.scale;
    const sdc = (RD.rw / sc) * Math.sqrt(dt) / gstep, kern = blurKernel(sdc), rho = 1 - Math.exp(-dt / cp.tau);
    diffuse(cp.aR, L, kern, rho, cp.pi0, true);
    diffuse(cp.aN, 1, kern, rho, cp.pi0, true);
    const aR = cp.aR, aN = cp.aN, nR = new Float64Array(L * KN), { same, next, back, skipW, hn2r, pRN, pNR, f2 } = ctx;
    let sumN = 0;
    const sumR = new Float64Array(KN);
    for (let l = 0; l < L; l++) for (let i = 0; i < KN; i++) sumR[i] += aR[l * KN + i];
    for (let l2 = 0; l2 < L; l2++) {
      const o = l2 * KN;
      for (let i = 0; i < KN; i++) {
        let v = same[l2] * aR[o + i];
        if (l2 > 0) v += next[l2 - 1] * aR[o - KN + i];
        if (l2 + 1 < L) v += back[l2 + 1] * aR[o + KN + i];
        let sk = 0;
        for (let d = 2; d <= 4; d++) { if (l2 - d >= 0) sk += aR[o - d * KN + i]; if (l2 + d < L) sk += aR[o + d * KN + i]; }
        v = (1 - pRN) * (v + skipW[l2] * sk) + pNR * hn2r[l2] * aN[i];
        nR[o + i] = v * E[o + i];
      }
    }
    for (let i = 0; i < KN; i++) { aN[i] = f2 * (pRN * sumR[i] + (1 - pNR) * aN[i]); sumN += aN[i]; }
    cp.aR = nR;
    if (!normalize(cp)) return;
    cp.prevFix = f; cp.lastT = f.t1;
    if (store) pushRing(cp, ctx, f, sdc, rho, false);
  }
  function normalize(cp) {
    let Z = 0;
    for (let i = 0; i < cp.aR.length; i++) Z += cp.aR[i];
    for (let i = 0; i < cp.aN.length; i++) Z += cp.aN[i];
    if (!(Z > 0) || !isFinite(Z)) { resetCopy(cp, true); return false; }
    for (let i = 0; i < cp.aR.length; i++) cp.aR[i] /= Z;
    for (let i = 0; i < cp.aN.length; i++) cp.aN[i] /= Z;
    return true;
  }
  function resetCopy(cp, all) {   // all: start over; else keep mode/line beliefs, forget the offset
    const KN = K * NG;
    if (all || !cp.aN) { cp.aN = null; cp.prevFix = null; cp.lastT = null; cp.ids = []; cp.aR = new Float64Array(0); cp.ring.length = 0; return; }
    for (let l = 0; l < cp.ids.length; l++) {
      let s = 0;
      for (let i = 0; i < KN; i++) s += cp.aR[l * KN + i];
      for (let i = 0; i < KN; i++) cp.aR[l * KN + i] = s * cp.pi0[i];
    }
    let s = 0;
    for (let i = 0; i < KN; i++) s += cp.aN[i];
    for (let i = 0; i < KN; i++) cp.aN[i] = s * cp.pi0[i];
    cp.ring.length = 0;
  }
  function pushRing(cp, ctx, f, sdc, rho, first) {
    cp.ring.push({ fi: f.fi, f, ctx, sdc, rho, first, aR: Float32Array.from(cp.aR), aN: Float64Array.from(cp.aN), ids: ctx.ids });
    if (cp.ring.length > RD.window) cp.ring.shift();
  }
  function shiftArr(a, rows, d) {   // move the offset belief by d client px (interpolated)
    const c = d / gstep, i0 = Math.floor(c), fr = c - i0, KN = K * NG, tmp = new Float64Array(NG);
    for (let row = 0; row < rows * K; row++) {
      const base = row * NG;
      for (let j = 0; j < NG; j++) tmp[j] = 0;
      for (let j = 0; j < NG; j++) {
        const v = a[base + j];
        if (!v) continue;
        const j0 = j + i0, j1 = j0 + 1;
        if (j0 >= 0 && j0 < NG) tmp[j0] += (1 - fr) * v;
        if (j1 >= 0 && j1 < NG) tmp[j1] += fr * v;
      }
      for (let j = 0; j < NG; j++) a[base + j] = tmp[j];
    }
    return KN;
  }
  function shiftCopy(cp, d) {   // the engine moved the stream by d: delta_new = delta_old + d
    if (!cp || !cp.aN) return;
    shiftArr(cp.aR, cp.ids.length, d); shiftArr(cp.aN, 1, d);
    if (!normalize(cp)) return;
    for (const e of cp.ring) { shiftArr(e.aR, e.ids.length, d); shiftArr(e.aN, 1, d); e.ctx = Object.assign({}, e.ctx, { y: e.ctx.y + d }); }
  }
  // fixed-lag smoother (label copy): P(reading, line id) for the ring entries from index i0 on
  function smoothRing(cp, i0) {
    const ring = cp.ring, KN = K * NG, out = new Array(ring.length).fill(null);
    let n = ring.length - 1, e = ring[n];
    let bR = new Float64Array(e.ids.length * KN).fill(1), bN = new Float64Array(KN).fill(1);
    const E = new Float64Array(RD.maxLines * KN);
    for (let i = n; i >= i0; i--) {
      e = ring[i];
      const L = e.ids.length;
      let Z = 0;
      const pl = new Float64Array(L);
      for (let l = 0; l < L; l++) { let s = 0; for (let q = 0; q < KN; q++) s += e.aR[l * KN + q] * bR[l * KN + q]; pl[l] = s; Z += s; }
      for (let q = 0; q < KN; q++) Z += e.aN[q] * bN[q];
      const m = new Map();
      for (let l = 0; l < L; l++) m.set(e.ids[l], pl[l] / Z);
      out[i] = m;
      if (i === i0 || e.first) break;
      // β_{i-1} = R^T D^T T^T (E ⊙ β_i)
      const ctx = e.ctx;
      emission(ctx, E);
      const BR = new Float64Array(L * KN), BN = new Float64Array(KN);
      for (let q = 0; q < L * KN; q++) BR[q] = bR[q] * E[q];
      for (let q = 0; q < KN; q++) BN[q] = bN[q] * ctx.f2;
      const XR = new Float64Array(L * KN), XN = new Float64Array(KN);
      for (let l = 0; l < L; l++) {
        const o = l * KN;
        for (let q = 0; q < KN; q++) {
          let v = ctx.same[l] * BR[o + q];
          if (l + 1 < L) v += ctx.next[l] * BR[o + KN + q];
          if (l > 0) v += ctx.back[l] * BR[o - KN + q];
          let sk = 0;
          for (let d = 2; d <= 4; d++) { if (l - d >= 0) sk += ctx.skipW[l - d] * BR[o - d * KN + q]; if (l + d < L) sk += ctx.skipW[l + d] * BR[o + d * KN + q]; }
          XR[o + q] = (1 - ctx.pRN) * (v + sk) + ctx.pRN * BN[q];
        }
      }
      for (let q = 0; q < KN; q++) { let s = 0; for (let l = 0; l < L; l++) s += ctx.hn2r[l] * BR[l * KN + q]; XN[q] = ctx.pNR * s + (1 - ctx.pNR) * BN[q]; }
      // D^T: (1-ρ)·blur(mix(X)) + ρ·<X, π0>
      const dots = new Float64Array(L);
      for (let l = 0; l < L; l++) { let s = 0; for (let q = 0; q < KN; q++) s += XR[l * KN + q] * cp.pi0[q]; dots[l] = s; }
      let dotN = 0;
      for (let q = 0; q < KN; q++) dotN += XN[q] * cp.pi0[q];
      const kern = blurKernel(e.sdc);
      diffuse(XR, L, kern, 0, cp.pi0, true); diffuse(XN, 1, kern, 0, cp.pi0, true);
      for (let l = 0; l < L; l++) for (let q = 0; q < KN; q++) XR[l * KN + q] = (1 - e.rho) * XR[l * KN + q] + e.rho * dots[l];
      for (let q = 0; q < KN; q++) XN[q] = (1 - e.rho) * XN[q] + e.rho * dotN;
      // R^T: back to the previous step's line set
      const prevIds = ring[i - 1].ids, idx = new Map(e.ids.map((id, k) => [id, k]));
      bR = new Float64Array(prevIds.length * KN);
      for (let l = 0; l < prevIds.length; l++) { const k = idx.get(prevIds[l]); if (k !== undefined) bR.set(XR.subarray(k * KN, (k + 1) * KN), l * KN); }
      bN = XN;
      let mx = 1e-300;
      for (let q = 0; q < bR.length; q++) if (bR[q] > mx) mx = bR[q];
      for (let q = 0; q < KN; q++) if (bN[q] > mx) mx = bN[q];
      for (let q = 0; q < bR.length; q++) bR[q] /= mx;
      for (let q = 0; q < KN; q++) bN[q] /= mx;
    }
    return out;
  }

  // ---- stage 0: slope (y depends on x) and horizontal offset, from line runs ---------------
  const st0 = { slope: 0, slopes: [], dx: 0, dxVar: 900, dxT: null };
  function stage0(seg, landing) {
    const sc = map.scale;
    const pts = seg.fixes.filter((f) => f.pRead > 0.8);
    if (pts.length >= 4) {
      let X = pts.map((f) => f.x), Y = pts.map((f) => f.y);
      if (Math.max(...X) - Math.min(...X) >= 250 / sc) {
        let b = 0;
        for (let it = 0; it < 2; it++) {
          const mx = X.reduce((s, v) => s + v, 0) / X.length, my = Y.reduce((s, v) => s + v, 0) / Y.length;
          let num = 0, den = 0;
          for (let i = 0; i < X.length; i++) { num += (X[i] - mx) * (Y[i] - my); den += (X[i] - mx) ** 2; }
          b = num / Math.max(den, 1e-9);
          const keep = X.map((x, i) => Math.abs(Y[i] - (my + b * (x - mx))) < 25 / sc);
          if (keep.every(Boolean) || keep.filter(Boolean).length < 4) break;
          X = X.filter((_, i) => keep[i]); Y = Y.filter((_, i) => keep[i]);
        }
        const mx = X.reduce((s, v) => s + v, 0) / X.length;
        const v = (81 / (sc * sc)) / Math.max(X.reduce((s, x) => s + (x - mx) ** 2, 0), 1);
        st0.slopes.push([b, v]);
        if (st0.slopes.length > 20) st0.slopes.shift();
        let num = 0, den = 1 / 0.01 ** 2;
        for (const [bb, vv] of st0.slopes) { num += bb / vv; den += 1 / vv; }
        st0.slope = num / den;
      }
    }
    // horizontal offset: the return sweep lands ~3.5 characters into the line (random walk + Kalman)
    const cap = (RD.dxSd0 / sc) ** 2;
    if (st0.dxT !== null) st0.dxVar = Math.min(st0.dxVar + (RD.dxRw / sc) ** 2 * Math.max(landing.t1 - st0.dxT, 0), cap);
    st0.dxT = landing.t1;
    if (landing.pRead > 0.8 && landing.line) {
      const o = landing.x - (landing.line.l + 3.5 * landing.line.cw), ov = (3 * landing.line.cw) ** 2 + (10 / sc) ** 2;
      const k = st0.dxVar / (st0.dxVar + ov);
      st0.dx += k * (o - st0.dx); st0.dxVar *= 1 - k;
    }
  }

  // ---- per-sample pipeline -------------------------------------------------------------------
  let lastMsg = {}, rawSeen = false, prevEy = null, prevGen = null, fixCount = 0, resets = 0;
  const fx = { cur: [], cand: [] };
  const segs = [];
  let curSeg = null, lastLabelT = -1e9, pausedUntil = 0;
  const lineLabelT = new Map(), readLog = [];
  const rview = { on: false, streak: 0, line: null, lineP: 0, pRead: 0, dy: 0, dySd: 0, gain: 1, word: null, wordIdx: -1, wordP: 0, at: 0, fix: null };
  function fixClose() {
    const c = fx.cur;
    fx.cur = []; fx.cand = [];
    if (c.length < RD.fixMinN) return null;
    let x = 0, y = 0, rx = 0, ry = 0;
    for (const s of c) { x += s.x; y += s.y; rx += s.sx; ry += s.sy; }
    const n = c.length;
    return { t0: c[0].t, t1: c[n - 1].t, lt0: c[0].lt, lt1: c[n - 1].lt, x: x / n, y: y / n, sx: rx / n, sy: ry / n, n };
  }
  // the frame noise is measured, not assumed: the agent's 0.2° (36 px threshold) cut real 250 ms
  // fixations into 50-90 ms pieces (real frame-to-frame step: median 30 px, p75 57 px)
  const noise = { steps: [], prev: null, T: RD.fixT };
  function fixPush(s) {
    if (noise.prev) {
      noise.steps.push(Math.hypot(s.x - noise.prev.x, s.y - noise.prev.y) * map.scale);
      if (noise.steps.length > 300) noise.steps.shift();
      if (noise.steps.length >= 60 && noise.steps.length % 15 === 0) {
        const st = noise.steps.slice().sort((a, b) => a - b), med = st[st.length >> 1];
        // step of a 2-D Gaussian with per-axis sd σ: Rayleigh(σ√2), median 1.665σ
        noise.T = Math.max(RD.fixT, Math.min(120, 3 * med / 1.665));
        RD.fixTLive = Math.round(noise.T);            // logged with the read test
      }
    }
    noise.prev = s;
    const T = noise.T / map.scale;
    let out = null;
    if (fx.cur.length && s.t - fx.cur[fx.cur.length - 1].t > RD.fixMaxGap) out = fixClose();
    if (!fx.cur.length) { fx.cur = [s]; return out; }
    const w = fx.cur.length > RD.fixWin ? fx.cur.slice(-RD.fixWin) : fx.cur;
    let mx = 0, my = 0;
    for (const p of w) { mx += p.x; my += p.y; }
    mx /= w.length; my /= w.length;
    if (Math.hypot(s.x - mx, s.y - my) <= T) { fx.cur.push(s); fx.cand = []; return out; }
    if (fx.cand.length) { const c = fx.cand[fx.cand.length - 1]; if (Math.hypot(s.x - c.x, s.y - c.y) > T) fx.cand = []; }
    fx.cand.push(s);
    if (fx.cand.length >= 2) { const cand = fx.cand; out = fixClose(); fx.cur = cand; }
    return out;
  }
  function resetOffsets(why) {
    for (const cp of [copies.show, copies.label]) if (cp) resetCopy(cp, false);
    st0.dxVar = (RD.dxSd0 / map.scale) ** 2;
    for (const s of segs) s.decided = true;
    if (curSeg) curSeg.decided = true;
    resets++; stats.readResets++;
    rview.on = false; rview.streak = 0;
    if (why === "guard") { pausedUntil = performance.now() + RD.pauseMs; stats.readGuards++; }
  }
  function readingSample(m) {
    if (!RD.on || map.ox === null) return;
    lastMsg = m;
    setupGrid(m);
    // engine-side changes of the stream: follow the correction, restart on discontinuities
    if (typeof m.gen === "number") { if (prevGen !== null && m.gen !== prevGen) resetOffsets("gen"); prevGen = m.gen; }
    if (typeof m.ey === "number") {
      if (prevEy !== null && Math.abs(m.ey - prevEy) > 0.5) { const d = (m.ey - prevEy) / map.scale; shiftCopy(copies.show, d); shiftCopy(copies.label, d); }
      prevEy = m.ey;
    }
    const hasRaw = typeof m.rx === "number" && typeof m.ry === "number";
    rawSeen = rawSeen || hasRaw;
    const ok = m.ok && (hasRaw || !rawSeen);
    if (!ok) return;
    const sx = hasRaw ? m.rx : m.x, sy = hasRaw ? m.ry : m.y, p = toClient(sx, sy);
    const f = fixPush({ t: m.t, lt: performance.now(), x: p.x, y: p.y, sx, sy });
    if (f) onFixation(f);
    else provisional();
  }
  function candidateLines(f, now) {
    const sc = map.scale, reach = Math.max(4 * sigB, 120 / sc);
    const blocks = new Set(probeBlocks(f, reach, now));
    const cp = copies.show;
    if (cp && cp.ids.length) {   // keep blocks that still hold belief, and the block reading continues into
      const KN = K * NG;
      let best = -1, bp = 0;
      for (let l = 0; l < cp.ids.length; l++) {
        let s = 0;
        for (let i = 0; i < KN; i++) s += cp.aR[l * KN + i];
        if (s > 1e-4) { const w = byId.get(+cp.ids[l].split(":")[0]); const b = w && w.deref(); if (b && b.isConnected) blocks.add(b); }
        if (s > bp) { bp = s; best = l; }
      }
      if (best >= 0) { const w = byId.get(+cp.ids[best].split(":")[0]); const b = w && w.deref(); if (b && b.isConnected) { const nb = nextTextBlock(b); if (nb) blocks.add(nb); } }
    }
    const list = [...blocks].sort((a, b) => (a === b ? 0 : a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
    let lines = [];
    for (const b of list) for (const L of blockLines(b)) if (L.c > -50 && L.c < innerHeight + 50 && Math.abs(L.c - f.y) < 4 * sigB + 120 / sc) lines.push(L);
    if (lines.length > RD.maxLines) {   // keep the nearest, in reading order
      const keep = new Set([...lines].sort((a, b) => Math.abs(a.c - f.y) - Math.abs(b.c - f.y)).slice(0, RD.maxLines));
      lines = lines.filter((L) => keep.has(L));
    }
    return lines;
  }
  function onFixation(f) {
    const now = performance.now();
    try { onFixation2(f, now); } finally {
      const ms = performance.now() - now;
      stats.readMs = 0.9 * (stats.readMs || 0) + 0.1 * ms; stats.readMsMax = Math.max(stats.readMsMax || 0, ms);
    }
  }
  function onFixation2(f, now) {
    if (scrolledDuring(f.lt0, f.lt1)) { rview.fix = null; return; }   // the page moved under the eye
    const lines = candidateLines(f, now);
    f.fi = fixCount++;
    const cs = copies.show, cl = copies.label;
    if (!lines.length) {   // no text near: not reading; restart the chain later, keeping the offset belief
      for (const cp of [cs, cl]) cp.prevFix = null;
      if (curSeg && curSeg.fixes.length) segs.push(curSeg);
      curSeg = null;
      rview.pRead = 0; rview.streak = 0; rview.on = false;
      emitFix(f, null);
      return;
    }
    const prev = cs.prevFix && f.t0 - cs.prevFix.t1 < 2 ? cs.prevFix : null;
    if (!prev) { cs.prevFix = null; cl.prevFix = null; }   // long gap: new chain, keep the offset belief
    const ctx = buildCtx(f, prev, lines, map.scale);
    const E = emission(ctx, new Float64Array(ctx.L * K * NG));
    stepCopy(cs, ctx, E, f, false);
    stepCopy(cl, ctx, E, f, true);
    // display view
    const KN = K * NG, L = ctx.L;
    let pRead = 0, best = 0, bp = 0;
    const pl = new Float64Array(L);
    for (let l = 0; l < L; l++) { let s = 0; for (let i = 0; i < KN; i++) s += cs.aR[l * KN + i]; pl[l] = s; pRead += s; if (s > bp) { bp = s; best = l; } }
    const lineP = pRead > 0 ? bp / pRead : 0, line = lines[best];
    // offset estimate on that line: E[(g-1)(c-Yc) + slope(x-Xc) + δ | line]
    let m1 = 0, m2 = 0, w = 0, gm = 0;
    for (let k = 0; k < K; k++) for (let j = 0; j < NG; j++) {
      const a = cs.aR[best * KN + k * NG + j];
      if (!a) continue;
      const e = (RD.gains[k] - 1) * (line.c - ctx.Yc) + st0.slope * (f.x - ctx.Xc) + G[j];
      m1 += a * e; m2 += a * e * e; w += a; gm += a * RD.gains[k];
    }
    const dy = w ? m1 / w : 0, dySd = w ? Math.sqrt(Math.max(m2 / w - dy * dy, 0)) : sigB;
    f.pRead = pRead; f.line = line; f.lineP = lineP;
    const plausible = Math.abs(dy) <= RD.maxDy * sigB;
    const conf = plausible && pRead > RD.showP && lineP > RD.showP;
    if (rview.on) rview.on = plausible && pRead > RD.keepP && lineP > RD.keepP && !(rview.line && rview.line.id !== line.id && !conf);
    rview.streak = conf ? (rview.line && rview.line.id === line.id ? rview.streak + 1 : 1) : 0;
    if (!rview.on && rview.streak >= 2) rview.on = true;
    Object.assign(rview, { line, lineP, pRead, dy, dySd, plausible, gain: w ? gm / w : 1, at: now, fix: f });
    pickWord(f.x);
    // segments (runs between return sweeps) -> stage 0 and labels
    const sweep = prev && isSweep(f, prev, line);
    if (sweep || !curSeg || !prev) {
      if (curSeg && curSeg.fixes.length) { segs.push(curSeg); stage0(curSeg, f); }
      curSeg = { fixes: [], epoch: resets, decided: false };
      if (sweep) scheduleLabels();
    }
    curSeg.fixes.push(f);
    while (segs.length > 20) segs.shift();
    emitFix(f, line);
  }
  function isSweep(f, prev, line) {
    const s = spacingNear(line), w = Math.max(line.r - line.l, 1);
    const dx = f.x - prev.x, dy = f.y - prev.y;
    return dx < -Math.max(15 * line.cw, 0.35 * w) && dy > -1.5 * s && dy < 4.5 * s;
  }
  function spacingNear(line) {
    const ls = blockLines(line.block);
    if (ls.length > 1) { const d = []; for (let i = 1; i < ls.length; i++) { const v = ls[i].c - ls[i - 1].c; if (v > 0) d.push(v); } if (d.length) { d.sort((a, b) => a - b); return d[d.length >> 1]; } }
    return Math.max(line.b - line.t, 1.4 * 2 * line.cw);
  }
  function emitFix(f, line) {
    const detail = { t0: f.t0, t1: f.t1, x: f.x, y: f.y, n: f.n, pRead: f.pRead || 0,
      line: line && f.pRead ? { element: line.block, index: line.index, p: f.lineP } : null, dy: rview.dy, shown: rview.on,
      ey: (typeof lastMsg.ey === "number" ? lastMsg.ey : 0) / map.scale };   // engine correction in this stream (client px)
    dispatchEvent(new CustomEvent("efgfixation", { detail }));
  }
  // the word under the gaze on the shown line (lazy word boxes, cached per line)
  const wordCache = new Map();
  function wordsOf(line) {
    const key = line.id + "@" + layoutVer, c = wordCache.get(key), br = line.block.getBoundingClientRect();
    if (c) return c.words.map((w) => ({ l: w.l + br.left - c.bl, r: w.r + br.left - c.bl, t: w.t + br.top - c.bt, b: w.b + br.top - c.bt, text: w.text }));
    const words = [];
    const tw = document.createTreeWalker(line.block, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (/\S/.test(n.data) && blockOf(n) === line.block ? 1 : 3) });
    for (let n; (n = tw.nextNode());) {
      const re = /\S+/g;
      for (let mm; (mm = re.exec(n.data));) {
        rng.setStart(n, mm.index); rng.setEnd(n, mm.index + mm[0].length);
        const r = rng.getBoundingClientRect(), cy = (r.top + r.bottom) / 2;
        if (r.width > 0 && cy >= line.t - 1 && cy <= line.b + 1) words.push({ l: r.left, r: r.right, t: r.top, b: r.bottom, text: mm[0] });
      }
    }
    if (wordCache.size > 200) wordCache.clear();
    wordCache.set(key, { bl: br.left, bt: br.top, words });
    return words.map((w) => ({ ...w }));
  }
  function pickWord(x) {
    rview.word = null; rview.wordP = 0; rview.wordIdx = -1;
    if (!rview.line) return;
    const ws = wordsOf(rview.line);
    if (!ws.length) return;
    const xq = x - st0.dx, s = Math.sqrt((RD.jitter / map.scale) ** 2 + st0.dxVar);
    let Z = 0, best = -1, bp = 0;
    ws.forEach((w, i) => { const p = Phi((w.r + 2 - xq) / s) - Phi((w.l - 2 - xq) / s); Z += p; if (p > bp) { bp = p; best = i; } });
    if (best >= 0 && Z > 0) { rview.word = ws[best]; rview.wordIdx = best; rview.wordP = bp / Z; }
  }
  function provisional() {   // during a fixation: move the word along the shown line, cheaply
    if (!rview.on || !rview.line || fx.cur.length < 3) return;
    let x = 0;
    for (const s of fx.cur) x += s.x;
    pickWord(x / fx.cur.length);
  }
  // ---- labels: smoother over the ring, decided after the next return sweep --------------------
  let labelPending = false;
  function scheduleLabels() {
    if (labelPending || !RD.learn || !opt.learn) return;
    labelPending = true;
    const run = () => { labelPending = false; try { evalLabels(); } catch (e) { /* never break the page */ } };
    if (window.requestIdleCallback) requestIdleCallback(run, { timeout: 500 }); else setTimeout(run, 0);
  }
  function evalLabels() {
    const cp = copies.label;
    if (!cp || !cp.ring.length || !map.fitted) return;
    const ring = cp.ring, oldest = ring[0].fi;
    for (const s of segs) if (!s.decided && (s.fixes[0].fi < oldest || s.epoch !== resets)) s.decided = true;
    const cand = segs.filter((s) => !s.decided);
    if (!cand.length) return;
    let firstFi = Math.min(...cand.map((s) => s.fixes[0].fi)), i0 = ring.findIndex((e) => e.fi >= firstFi);
    for (let i = ring.length - 1; i > i0; i--) if (ring[i].first) { i0 = i; break; }   // chains stop at a restart
    if (i0 < 0) return;
    const post = smoothRing(cp, i0), byFi = new Map();
    ring.forEach((e, i) => { if (post[i]) byFi.set(e.fi, post[i]); });
    const now = performance.now();
    for (const s of cand) {
      if (s.fixes.length < 4) { s.decided = true; continue; }
      const ps = s.fixes.map((f) => byFi.get(f.fi));
      if (ps.some((p) => !p)) continue;           // not covered (crosses a restart): dropped when it leaves the ring
      const tot = new Map();
      for (const p of ps) for (const [id, v] of p) tot.set(id, (tot.get(id) || 0) + v);
      let id = null, best = -1;
      for (const [k, v] of tot) if (v > best) { best = v; id = k; }
      const conf = Math.min(...ps.map((p) => p.get(id) || 0));
      const lastF = s.fixes[s.fixes.length - 1], line = (ring.find((e) => e.fi === lastF.fi) || {}).ctx?.lines.find((L) => L.id === id);
      if (!line || conf < RD.labelP) continue;
      // only lines read start to end, with reading-sized forward saccades (skimming jumps 2-3 words)
      const xs = s.fixes.map((f) => f.x - st0.dx), cw = line.cw;
      const fw = [];
      for (let i = 1; i < xs.length; i++) if (xs[i] > xs[i - 1]) fw.push(xs[i] - xs[i - 1]);
      fw.sort((a, b) => a - b);
      const full = xs[0] - line.l < 10 * cw && line.r - xs[xs.length - 1] < 15 * cw &&
        Math.max(...xs) - Math.min(...xs) >= 0.5 * (line.r - line.l) && fw.length >= 2 && fw[fw.length >> 1] <= 12 * cw;
      if (!full) continue;                        // may pass later if the line hypothesis changes
      s.decided = true;
      if (now < pausedUntil || lastF.t1 < serverT - RD.maxLabelAge) continue;
      if (lastF.t1 - lastLabelT < RD.labelGap || lastF.t1 - (lineLabelT.get(id) || -1e9) < RD.lineRelabelS) continue;
      lastLabelT = lastF.t1; lineLabelT.set(id, lastF.t1);
      const t1 = lastF.t1, t0 = Math.max(s.fixes[0].t0, t1 - 3);
      const xsS = s.fixes.map((f) => f.sx).sort((a, b) => a - b), ys = toScreen(0, line.c).y;
      const body = { t0, t1, x: xsS[xsS.length >> 1], y: ys, axis: "y", sdy: RD.labelSdy, conf: RD.labelConf, kind: "read", p: conf,
        label: "línea " + (line.index + 1) };
      stats.readLabels++;
      const rec = { t: t1, element: line.block, index: line.index, p: conf, y: ys };
      readLog.push(rec); if (readLog.length > 200) readLog.shift();
      dispatchEvent(new CustomEvent("efglabel", { detail: rec }));
      fetch(ORIGIN + "/label", { method: "POST", body: JSON.stringify(body) }).catch(() => {});
    }
  }
  // a click on another text line than the one we believe is being read: our lock is wrong
  addEventListener("pointerdown", (e) => {
    if (!rview.on || !rview.line || performance.now() - rview.at > 1500 || rview.lineP < 0.9) return;
    const L = rview.line, ls = blockLines(L.block), s = spacingNear(L);
    const hit = ls.find((q) => e.clientY >= q.t && e.clientY <= q.b && e.clientX >= q.l - 10 && e.clientX <= q.r + 10);
    const curL = ls.find((q) => q.id === L.id);
    if (hit && curL && hit.id !== L.id && Math.abs(hit.c - curL.c) <= 2.5 * s) resetOffsets("guard");
  }, { capture: true, passive: true });
  function readingInfo() {
    if (!rview.line || !rview.line.block.isConnected) return null;
    const L = blockLines(rview.line.block).find((q) => q.id === rview.line.id) || rview.line;
    const rect = { left: L.l, top: L.t, right: L.r, bottom: L.b, width: L.r - L.l, height: L.b - L.t };
    let word = null;
    if (rview.word) {
      const ws = wordsOf(L), w = ws[rview.wordIdx] || rview.word;
      word = { text: w.text, p: rview.wordP, rect: { left: w.l, top: w.t, right: w.r, bottom: w.b, width: w.r - w.l, height: w.b - w.t } };
    }
    return { p: rview.pRead, shown: rview.on, line: { element: L.block, index: L.index, p: rview.lineP, rect }, word,
      offset: { dy: rview.dy, sd: rview.dySd, gain: rview.gain, slope: st0.slope, dx: st0.dx, dxSd: Math.sqrt(st0.dxVar) },
      corrected: rview.fix ? { x: rview.fix.x - st0.dx, y: rview.fix.y - rview.dy } : null,
      raw: rawSeen, labels: stats.readLabels, paused: performance.now() < pausedUntil, ageMs: performance.now() - rview.at };
  }

  // ---- drawing: the pointer morphs from a soft circle into the attractor it snaps to ---------
  const view = {
    show: !!(SCRIPT && SCRIPT.hasAttribute("data-show")),
    debug: /[?&]efgdebug/.test(location.search) || !!(SCRIPT && SCRIPT.hasAttribute("data-debug")),
  };
  let lens, dot, tag, lineBar;
  function ensureEls() {
    if (lens) return;
    const mk = (css) => { const d = document.createElement("div"); d.setAttribute("data-efg", ""); d.style.cssText = "position:fixed;pointer-events:none;z-index:2147483647;display:none;" + css; document.documentElement.appendChild(d); return d; };
    lineBar = mk("box-sizing:border-box;border-bottom:2px solid rgba(79,195,255,.75);background:rgba(79,195,255,.08);border-radius:3px;");
    lens = mk("box-sizing:border-box;border:2px solid rgba(79,195,255,.9);background:rgba(79,195,255,.16);box-shadow:0 0 22px rgba(79,195,255,.45);");
    dot = mk("width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;background:#ff2d20;opacity:.8;");
    tag = mk("left:8px;bottom:8px;font:12px/1.3 Consolas,monospace;color:#fff;background:#000a;padding:4px 8px;border-radius:4px;");
  }
  const sLens = {}, sDot = {}, sLine = {};
  let rafT = 0, lensOn = false, lineOn = false, radius = 0;
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
    const rd = !snap && rview.on && performance.now() - rview.at < 1500 ? readingInfo() : null;
    if (box && (snap || view.debug) && !rd) { target = [box.left - 4, box.top - 4, box.width + 8, box.height + 8]; round = 10; }
    else if (rd && rd.word && rd.word.p >= 0.5) { const w = rd.word.rect; target = [w.left - 3, w.top - 2, w.width + 6, w.height + 4]; round = 6; }
    else if (gaze) target = [gaze.x - R, gaze.y - R, 2 * R, 2 * R];
    if (target) {
      if (!lensOn) { sLens.p = null; lensOn = true; }
      const [l, tp, w, h] = spring(sLens, target, dt, 30);
      radius += (round - radius) * Math.min(1, dt * 18);
      Object.assign(lens.style, { display: "block", left: l + "px", top: tp + "px", width: w + "px", height: h + "px",
        borderRadius: round === 50 ? "50%" : radius.toFixed(1) + "px" });
    } else { lens.style.display = "none"; lensOn = false; }
    if (rd) {   // the line being read: a faint band with an underline
      const r = rd.line.rect;
      if (!lineOn) { sLine.p = null; lineOn = true; }
      const [l, tp, w, h] = spring(sLine, [r.left - 6, r.top - 2, r.width + 12, r.height + 4], dt, 26);
      Object.assign(lineBar.style, { display: "block", left: l + "px", top: tp + "px", width: w + "px", height: h + "px" });
    } else { lineBar.style.display = "none"; lineOn = false; }
    if (view.debug && gaze) {
      const [x, y] = spring(sDot, [gaze.x, gaze.y], dt, 40);
      Object.assign(dot.style, { display: "block", left: x + "px", top: y + "px" });
    } else dot.style.display = "none";
    tag.style.display = view.debug ? "block" : "none";
    if (view.debug) tag.textContent = (connected ? "efg ✓" : "efg: sin conexión con " + ORIGIN) +
      (map.fitted ? " · ventana localizada" : " · mueve el ratón dentro de la página y déjalo quieto") +
      (current ? ` · ${label(current)} ${(current_p * 100).toFixed(0)}%${snap ? " (imán)" : ""} ${((performance.now() - since) / 1000).toFixed(1)} s` : "") +
      ` · etiquetas por mirada ${stats.labels}` +
      (RD.on ? ` · lectura ${(rview.pRead * 100).toFixed(0)}%` + (rview.line ? ` línea ${rview.line.index + 1} ${(rview.lineP * 100).toFixed(0)}%` : "") +
        ` dy ${rview.dy >= 0 ? "+" : ""}${rview.dy.toFixed(0)}±${rview.dySd.toFixed(0)} px · etiq. lectura ${stats.readLabels}` +
        (rawSeen ? "" : " · (sin rx/ry: flujo suavizado)") + (performance.now() < pausedUntil ? " · pausa" : "") : "");
  }
  function show(on) { view.show = on === undefined ? !view.show : !!on; if (!view.show && !view.debug && lens) [lens, dot, tag, lineBar].forEach((e) => (e.style.display = "none")); return view.show; }
  function debug(on) { view.debug = on === undefined ? !view.debug : !!on; if (!view.show && !view.debug && lens) [lens, dot, tag, lineBar].forEach((e) => (e.style.display = "none")); return view.debug; }
  addEventListener("keydown", (e) => { if (e.altKey && (e.key === "g" || e.key === "G")) debug(); });

  // ---- connection -----------------------------------------------------------------------
  let connected = false;
  function connect() {
    const es = new EventSource(ORIGIN + "/gaze");
    es.onopen = () => { connected = true; };
    es.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      learnMapping(m); step(m);
      try { readingSample(m); } catch (e) { if (view.debug) console.warn("efg reading:", e); }
    };
    es.onerror = () => { connected = false; };
  }
  connect();

  window.efg = {
    elementFromGaze: () => (current ? { element: current, p: current_p, dwellMs: performance.now() - since, attractor: current === snap } : null),
    point: () => (gaze ? { x: gaze.x, y: gaze.y } : null),
    posterior: () => (gaze ? attractorPosterior(gaze, lastSd[0], lastSd[1], performance.now()).slice(0, 5).map(([el, p]) => ({ label: label(el), p })) : []),
    reading: readingInfo,
    textLines: (el) => blockLines(el).map((L) => ({ index: L.index, left: L.l, right: L.r, top: L.t, bottom: L.b, cy: L.c })),
    readLog, readingOptions: RD,
    history, options: opt, show, debug, origin: ORIGIN, stats,
    get mapping() { return { ...map, pairs: map.pairs.length }; },
  };
})();
