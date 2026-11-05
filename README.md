# elementfromgaze

Webcam gaze tracking whose end goal is one call: **`elementFromGaze()`** — the UI element (button, chart slice, line, word) you are looking at. A riff on the DOM's `document.elementFromPoint`.

Status: **temporary / experimental** (phase 1 of the roadmap below). Windows first.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --only-binary :all: mediapipe opencv-python numpy scikit-learn
.\.venv\Scripts\python -m pip install -e .
efg probe                    # camera + face check, prints speed, no windows
efg calibrate                # ~40 s full screen: follow the red dot
efg calibrate --tag lejos    # repeat from other head positions; every run is kept
efg eval                     # accuracy on random points, logged to data/eval_log.jsonl
efg dot                      # red dot over the whole screen where you look · Ctrl+Alt+G quits
```

Python 3.14 is not supported yet: some dependency wheels are missing and building them from source trips Windows Smart App Control.

## How it works (now)

```
camera (30 fps) → MediaPipe Face Landmarker (478 pts + blendshapes + head pose)
               → 24 features: iris in eye-local frame, eyelid opening, eyeLook* blendshapes, head rotation/translation
               → per-user regressor (degree-2 polynomial + ridge), trained from calibration sessions
               → One Euro filter → screen pixel
```

Calibration = 13 fixation points + 22 s of smooth pursuit. Every session also stores 24×48 eye crops for the appearance model in phase 3. Personal data (`data/`) never leaves the machine and is git-ignored.

## Roadmap

1. **Core** – capture, features, calibration, filter, red-dot overlay, eval harness. ← *here*
2. **Robustness to head pose** – sessions from several positions/distances, pose-normalised features.
3. **Light appearance model** – tiny CNN on eye crops (+ landmarks), personal fine-tune; optional pretrained backbone (L2CS-Net / ETH-XGaze) as a feature source. Try the integrated **IR camera**.
4. **Beyond pure regression** – the ideas that could beat plain webcam baselines:
   - *implicit recalibration*: every mouse click is a free labelled sample (you look where you click);
   - *UI-aware decoding*: snap gaze to the most likely element given the layout (hit-test with a prior), not to a raw pixel;
   - *reading prior* for text: lines are read left→right, top→bottom, so line/word tracking can use that structure.
5. **Adapters** – WebSocket gaze stream; browser client (`elementFromGaze()`, word under gaze via caret ranges); Windows UI Automation (any clickable on the desktop).
6. Package, publish.
