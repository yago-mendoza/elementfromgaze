# elementfromgaze

Webcam eye tracking whose end goal is one call: **`elementFromGaze()`** — the UI element (button, line, word) you are looking at, in a web page or in any Windows app. A riff on the DOM's `document.elementFromPoint`.

Laptop webcam only: no infrared hardware, no GPU, nothing leaves the machine.

> **Status: experimental, Windows, single user.** The eye network is trained on one person (the author). A base that works for anyone, with a short calibration for each new person, is the next milestone (see Roadmap).

## What it does today

| | |
|---|---|
| **Accuracy** | ~0.9–1.3° per fixation after the posture warm-up (≈1 cm on screen at 55 cm); 0.89° in the button-picking test |
| **Without warm-up, in a new posture** | ~2° (was 5.75° before training on everyday use) |
| **Cost** | ~5 ms per frame on CPU, on top of MediaPipe |
| **Desktop** | `efg control`: the pointer snaps to the buttons of any app (UI Automation); a long blink clicks |
| **Web** | `efg.js`: gaze events for any page, magnetic `data-gaze` buttons, a reading (line/word) layer |
| **Learns while you use it** | every mouse click and every typed character is a free label (you look where you click and type) |

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --only-binary :all: -e .
# run it as `python -m elementfromgaze`: Smart App Control blocks the efg.exe launcher
.\.venv\Scripts\python -m elementfromgaze --hd calibrate        # ~40 s, follow the red dot (repeat in a few postures)
.\.venv\Scripts\python -m elementfromgaze --hd track            # warm-up (~80 s) + follow a moving ring, logs the error
.\.venv\Scripts\python -m elementfromgaze --hd gesture          # record your long blink once
.\.venv\Scripts\python -m elementfromgaze --hd control --dot --serve --click
```

Python 3.14 is not supported yet (missing wheels; building them from source trips Smart App Control).

Keys while running: **Ctrl+Alt+R** recalibrate (~45 s) · **Ctrl+Alt+arrows** nudge the dot · **Ctrl+Alt+G** quit.

## Commands

| Command | |
|---|---|
| `probe` | camera + face check, prints speed |
| `calibrate --tag` | full-screen calibration session; each run is kept and the model retrains |
| `train` | retrain from every saved session |
| `track` / `targets` | accuracy tests: follow a ring / pick lit buttons of shrinking size |
| `dot [--serve] [--no-dot]` | gaze over the whole screen; `--serve` streams it to web pages on `http://127.0.0.1:8770` |
| `control [--click] [--dot] [--serve]` | gaze control of the desktop |
| `gesture` | long-blink calibration for `--click` |
| `ir` | live view of the integrated infrared camera |

`--hd` uses the colour camera at 1920×1080 (more eye detail, ~17 fps). `--rewarm` redoes the posture warm-up.

## How it works

```
camera ─► MediaPipe Face Landmarker (478 landmarks, blendshapes, head pose)
       ─► two eye patches, warped so the eye corners sit at fixed points (removes roll and distance)
       ─► small CNN (2 networks averaged) ─► eye-in-head yaw/pitch
       ─► gaze ray: rotated and placed by the head pose, intersected with the screen plane ─► pixel
       ─► Kalman shift correction (clicks, typing, checks) ─► critically damped spring ─► pointer
```

- **Physics, not a black box**: head pose enters only through ray geometry, so turning the head does not drag the dot.
- **Posture warm-up (~80 s)**: smooth pursuit, row sweeps, corners, a guided head turn and a 9-button check fine-tune the top of the network to today's posture and light. Every part was ablated and helps.
- **Posture bank**: when the posture drifts (head pitch, distance, eyelids) it switches to a stored warm-up for that posture, or recalibrates in ~45 s and stores it.
- **Learning on the ride**: clicks and the text caret give labelled samples; only these external labels (never the tracker's own guesses) feed the correction and periodic fine-tuning. A click only shifts the correction — many clicks in one spot cannot rotate the rest of the screen.
- **Snapping**: on the desktop and in pages, the pointer snaps to an element when the Bayesian posterior that you look at it passes 0.8 (error model: the measured per-axis gaze error), and releases below 0.4–0.5.

Measured and rejected (kept in commit messages): MPIIGaze pretraining for the personal model, infrared fusion, larger eye crops, calibration-aware training, change-point correction, meta-learning, synthetic postures, a fixation-physics loss, binocular rays.

## Web: `efg.js`

Run `efg dot --serve` (or `control --serve`), then in any page:

```html
<script src="http://127.0.0.1:8770/efg.js" data-show></script>
<button data-gaze>Buy</button>          <!-- magnetic: the pointer snaps onto it -->
<script>
  efg.elementFromGaze()                  // {element, p, dwellMs, attractor} | null
  efg.reading()                          // {line, word, ...} while reading text
  addEventListener("efgfixation", e => …) // also gazeenter / gazeleave / gazedwell on elements
</script>
```

Demos: `http://127.0.0.1:8770/demo` and `/read`.

## Privacy

Everything runs locally. Sessions, models and logs live in `data/`, which is git-ignored. The text caret hook records positions only, never keys.

## Roadmap

1. ~~Core~~ · ~~head-pose robustness~~ · ~~appearance model~~ · ~~implicit recalibration, UI-aware decoding, reading prior~~ · ~~desktop control~~
2. **Works for anyone**: a base trained on many people (public gaze datasets + volunteers, with and without glasses), measured on people it never saw, plus a 2-minute first-run calibration.
3. Automatic periodic retraining on your own use.
4. Publish: GitHub + pip; a live in-browser demo (MediaPipe web + the eye network in ONNX).
