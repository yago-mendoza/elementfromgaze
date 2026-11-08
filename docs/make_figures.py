"""Draws the README figures as SVG (run: python docs/make_figures.py). Numbers are measured, and each
says where it comes from; nothing here is sketched to look plausible."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "assets"
BG, PANEL, RULE, INK, DIM, FAINT = "#120E0F", "#1E1617", "#302426", "#F0E8E6", "#9A8D8E", "#5C4C4E"
ACCENT, STEEL, GREEN, VIOLET, GOLD = "#E5322D", "#6FA8C9", "#5DBB73", "#9B8AE0", "#D9A441"
FONT = "ui-monospace, SFMono-Regular, Consolas, Menlo, monospace"
SANS = "-apple-system, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">\n<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>\n{body}\n</svg>\n')


def box(x, y, w, h, title, lines=(), color=INK, fill=PANEL, dash=False):
    d = ' stroke-dasharray="4 4"' if dash else ""
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{color}" stroke-opacity=".8"{d}/>',
         f'<text x="{x + 12}" y="{y + 22}" font-size="12.5" fill="{color}" font-weight="600">{title}</text>']
    for i, ln in enumerate(lines):
        s.append(f'<text x="{x + 12}" y="{y + 42 + 16 * i}" font-size="11" fill="{DIM}">{ln}</text>')
    return "\n".join(s)


def arrow(x1, y1, x2, y2, color=FAINT, label="", lx=0, ly=-6):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 8 * math.cos(a), y2 - 8 * math.sin(a)
    p1 = (hx + 4 * math.sin(a), hy - 4 * math.cos(a))
    p2 = (hx - 4 * math.sin(a), hy + 4 * math.cos(a))
    s = [f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{color}" stroke-width="1.4"/>',
         f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>']
    if label:
        s.append(f'<text x="{(x1 + x2) / 2 + lx}" y="{(y1 + y2) / 2 + ly}" font-size="10.5" fill="{color}" text-anchor="middle">{label}</text>')
    return "\n".join(s)


def pipeline():
    W, H = 960, 560
    b = []
    b.append(f'<text x="28" y="40" font-size="11" fill="{ACCENT}" letter-spacing="2">ELEMENTFROMGAZE · ONE FRAME, 33 MS</text>')
    # row 1: sensing
    b.append(box(28, 64, 200, 92, "webcam", ["colour 1920×1080, 30 fps", "IR camera: recorded only"], color=FAINT))
    b.append(box(268, 64, 210, 92, "MediaPipe face", ["478 landmarks + iris", "head pose · 52 blendshapes"], color=FAINT))
    b.append(box(518, 64, 196, 92, "eye patches", ["2 × 36×60 px, grey", "warped: corners fixed"], color=GREEN))
    b.append(box(754, 64, 178, 92, "eye network", ["2 small CNNs, averaged", "eye-in-head yaw, pitch"], color=VIOLET))
    b.append(arrow(228, 110, 268, 110))
    b.append(arrow(478, 110, 518, 110))
    b.append(arrow(714, 110, 754, 110))
    # row 2: geometry and correction
    b.append(box(754, 214, 178, 106, "gaze ray", ["head pose turns and", "places it; it hits the", "screen plane"], color=INK))
    b.append(box(518, 214, 196, 106, "correction", ["Kalman shift from", "clicks, typing and", "the 9-button check"], color=GOLD))
    b.append(box(268, 214, 210, 106, "decoding", ["snap: posterior > 0.8", "reading: line HMM", "spring for the pointer"], color=ACCENT))
    b.append(arrow(843, 156, 843, 214))
    b.append(arrow(754, 267, 714, 267))
    b.append(arrow(518, 267, 478, 267))
    b.append(box(28, 214, 200, 106, "you", ["pointer on the button", "long blink: click", "efg.js events in pages"], color=ACCENT, fill="#2A1314"))
    b.append(arrow(268, 267, 228, 267))
    # row 3: learning
    b.append(box(268, 378, 446, 110, "learning on the ride", [
        "warm-up (~80 s): fine-tunes the network's top to today's posture",
        "posture bank: reuses a warm-up when you sit like that again",
        "clicks and the text caret: free labels (you look where you click)",
        "efg train: retrains on every saved session"], color=GREEN, dash=True))
    b.append(arrow(491, 378, 491, 322, GREEN, "labels", 24, 4))
    b.append(f'<polyline points="714,410 946,410 946,140" fill="none" stroke="{GREEN}" stroke-width="1.4" stroke-dasharray="3 4"/>')
    b.append(arrow(946, 140, 934, 140, GREEN))
    b.append(f'<text x="938" y="398" font-size="10.5" fill="{GREEN}" text-anchor="end">fine-tune the network</text>')
    b.append(arrow(128, 322, 128, 433, FAINT))
    b.append(f'<line x1="128" y1="433" x2="268" y2="433" stroke="{FAINT}" stroke-width="1.4"/>')
    b.append(f'<text x="138" y="392" font-size="10.5" fill="{FAINT}">clicks, keys</text>')
    # key strip
    b.append(f'<rect x="0" y="512" width="{W}" height="48" fill="#181213"/><line x1="0" y1="512" x2="{W}" y2="512" stroke="{RULE}"/>')
    keys = [(FAINT, "hardware and outside libraries"), (GREEN, "data, labels"), (VIOLET, "trained weights"),
            (GOLD, "what is measured now"), (ACCENT, "what you see")]
    x = 28
    for col, lab in keys:
        b.append(f'<rect x="{x}" y="530" width="10" height="10" rx="2" fill="{col}"/>')
        b.append(f'<text x="{x + 16}" y="539" font-size="10.5" fill="{DIM}">{lab}</text>')
        x += 32 + 7 * len(lab)
    (OUT / "pipeline.svg").write_text(svg(W, H, "\n".join(b)), encoding="utf-8")


# Button test, 2026-11-06 14:48 (efg targets, 56 trials, the author, warmed up): share of 1 s dwell
# decisions that land in the right button of a regular grid of h x 2h buttons, by h in degrees.
SIZE_CURVE = {0.75: 0.232, 1.0: 0.393, 1.25: 0.571, 1.5: 0.643, 2.0: 0.875, 2.5: 0.875, 3.0: 0.946, 4.0: 0.982}
CLICK_CURVE = {0.75: 0.451, 1.0: 0.608, 1.25: 0.627, 1.5: 0.725, 2.0: 0.843, 2.5: 0.902, 3.0: 0.941, 4.0: 0.98}
PX_PER_DEG = 61.2          # 1920x1200 screen, 301 mm wide, eyes at ~55 cm


def size_chart():
    W, H = 960, 470
    x0, x1, y0, y1 = 90, 920, 380, 70
    X = lambda d: x0 + (x1 - x0) * d / 4.2
    Y = lambda p: y0 - (y0 - y1) * p
    b = [f'<text x="28" y="40" font-size="11" fill="{ACCENT}" letter-spacing="2">HOW BIG MUST A BUTTON BE</text>']
    for p in (0, .25, .5, .75, 1):
        b.append(f'<line x1="{x0}" y1="{Y(p)}" x2="{x1}" y2="{Y(p)}" stroke="{RULE}"/>')
        b.append(f'<text x="{x0 - 12}" y="{Y(p) + 4}" font-size="11" fill="{DIM}" text-anchor="end">{p:.0%}</text>')
    for d in (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4):
        b.append(f'<line x1="{X(d)}" y1="{y0}" x2="{X(d)}" y2="{y0 + 5}" stroke="{FAINT}"/>')
        b.append(f'<text x="{X(d)}" y="{y0 + 20}" font-size="11" fill="{DIM}" text-anchor="middle">{d:g}°</text>')
        b.append(f'<text x="{X(d)}" y="{y0 + 35}" font-size="9.5" fill="{FAINT}" text-anchor="middle">{d * PX_PER_DEG:.0f} px</text>')
    # Windows controls: menu items and toolbar buttons are ~24-32 px tall at 100-125% scaling
    b.append(f'<rect x="{X(24 / PX_PER_DEG)}" y="{y1}" width="{X(32 / PX_PER_DEG) - X(24 / PX_PER_DEG)}" height="{y0 - y1}" fill="{BAD_BG}"/>')
    b.append(f'<text x="{X(28 / PX_PER_DEG)}" y="{y1 - 8}" font-size="10.5" fill="{DIM}" text-anchor="middle">Windows menus</text>')
    b.append(f'<line x1="{X(1.5)}" y1="{y1}" x2="{X(1.5)}" y2="{y0}" stroke="{FAINT}" stroke-dasharray="3 4"/>')
    b.append(f'<text x="{X(1.5) + 6}" y="{y1 + 12}" font-size="10.5" fill="{DIM}">the warm-up\'s check buttons</text>')

    def line(curve, col, dash=""):
        pts = " ".join(f"{X(d):.1f},{Y(p):.1f}" for d, p in curve.items())
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out = [f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"{d}/>']
        out += [f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="3.2" fill="{col}"/>' for k, v in curve.items()]
        return "\n".join(out)
    b.append(line(CLICK_CURVE, STEEL, "5 4"))
    b.append(line(SIZE_CURVE, ACCENT))
    b.append(f'<text x="{X(3.05)}" y="{Y(0.946) + 22}" font-size="11.5" fill="{GOLD}">95% from 3° (184 px)</text>')
    b.append(f'<rect x="0" y="422" width="{W}" height="48" fill="#181213"/><line x1="0" y1="422" x2="{W}" y2="422" stroke="{RULE}"/>')
    b.append(f'<line x1="28" y1="446" x2="48" y2="446" stroke="{ACCENT}" stroke-width="2.2"/>'
             f'<text x="56" y="450" font-size="10.5" fill="{DIM}">1 s look, after the warm-up</text>')
    b.append(f'<line x1="300" y1="446" x2="320" y2="446" stroke="{STEEL}" stroke-width="2.2" stroke-dasharray="5 4"/>'
             f'<text x="328" y="450" font-size="10.5" fill="{DIM}">after learning from 5 clicks (replayed)</text>')
    b.append(f'<text x="640" y="450" font-size="10.5" fill="{FAINT}">grid of h × 2h buttons; x = h</text>')
    (OUT / "button-size.svg").write_text(svg(W, H, "\n".join(b)), encoding="utf-8")


BAD_BG = "#3A1A1C"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    size_chart()
    print("wrote", OUT / "pipeline.svg", OUT / "button-size.svg")
