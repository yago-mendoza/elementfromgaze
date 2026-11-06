"""efg — command line for elementfromgaze.

  efg probe        camera + landmarker check, prints speed (no windows)
  efg calibrate    full-screen calibration; each run is saved as a new session and the model retrains
  efg train        retrain the model from every saved session
  efg eval         full-screen accuracy test on random points
  efg targets      buttons light up, you look; % of buttons picked right per size
  efg dot          red dot over the whole screen where you look (Ctrl+Alt+G quits)
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def probe(seconds: float = 5.0) -> None:
    from .tracker import FeatureWorker

    got = {"n": 0, "last": None}

    def on_sample(s):
        got["n"] += 1
        got["last"] = s

    w = FeatureWorker(on_sample)
    time.sleep(seconds)
    w.close()
    s = got["last"]
    print(f"{got['n']} muestras con cara en {seconds:.0f}s · {w.stats()}")
    if s is not None:
        print("features:", s.features.round(3).tolist())
        print("crops:", s.crops.shape, "blink:", round(s.blink, 2))


def main() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap =argparse.ArgumentParser(prog="efg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hd", action="store_true", help="colour camera at 1920x1080 (more eye detail, ~17 fps)")
    ap.add_argument("--no-train", action="store_true", help="record only; don't retrain after a session")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    c = sub.add_parser("calibrate")
    c.add_argument("--tag", default="normal", help="condition label, e.g. 'lejos', 'ladeado', 'lampara-off'")
    c.add_argument("--note", default="", help="free text: light, glasses, tiredness, time of day…")
    sub.add_parser("train")
    e = sub.add_parser("eval")
    e.add_argument("-n", type=int, default=12)
    e.add_argument("--tag", default="normal")
    e.add_argument("--note", default="")
    mv = sub.add_parser("motion", help="fixate points while moving the laptop or the head")
    mv.add_argument("mode", nargs="?", default="portatil", choices=["portatil", "cabeza", "ladeo"])
    mv.add_argument("--note", default="")
    fl = sub.add_parser("follow", help="calibrate by looking at your own mouse cursor")
    fl.add_argument("--note", default="")
    ed = sub.add_parser("edges", help="follow the screen border, then look away on cue")
    ed.add_argument("--note", default="")
    sub.add_parser("bench")
    tk_ = sub.add_parser("track", help="follow a moving ring with the live dot shown; logs the errors")
    tk_.add_argument("--note", default="")
    tk_.add_argument("--no-warmup", action="store_true", help="measure the base model, without warm-up")
    tk_.add_argument("--rewarm", action="store_true", help="redo the warm-up even if one is <30 min old")
    tg = sub.add_parser("targets", help="look at lit buttons of shrinking size; measures which you can pick by gaze")
    tg.add_argument("--note", default="")
    tg.add_argument("--show", action="store_true", help="outline the button the tracker thinks you look at")
    tg.add_argument("--no-warmup", action="store_true")
    tg.add_argument("--rewarm", action="store_true")
    sub.add_parser("ir", help="live view of the infrared camera")
    ct = sub.add_parser("control", help="gaze over the whole desktop: snaps to buttons of any app; --click: squint to click")
    ct.add_argument("--click", action="store_true", help="squint (~0.3 s) to click the highlighted button")
    ct.add_argument("--dot", action="store_true", help="also draw the red dot")
    ct.add_argument("--serve", action="store_true")
    ct.add_argument("--no-warmup", action="store_true")
    ct.add_argument("--rewarm", action="store_true")
    ct.add_argument("--guest", action="store_true", help="someone else: no learning, nothing saved, no warm-up")
    sub.add_parser("gesture", help="20 s: record normal / squint / blink / smile to set the squint-click threshold")
    dt = sub.add_parser("dot")
    dt.add_argument("--no-learn", action="store_true", help="do not learn from clicks/typing")
    dt.add_argument("--no-warmup", action="store_true", help="skip the ~45 s posture warm-up")
    dt.add_argument("--rewarm", action="store_true", help="redo the warm-up even if one is <30 min old")
    dt.add_argument("--serve", action="store_true", help="stream the gaze to web pages (http://127.0.0.1:8770, demo at /demo)")
    dt.add_argument("--no-dot", action="store_true", help="don't draw the red dot (e.g. with --serve)")
    dt.add_argument("--guest", action="store_true", help="someone else: no learning, nothing saved, no warm-up")
    a = ap.parse_args()
    import os
    if a.hd:
        os.environ["EFG_RES"] = "1920x1080"
    if a.no_train:
        os.environ["EFG_NO_TRAIN"] = "1"

    if a.cmd == "probe":
        probe()
        return
    if a.cmd == "track":
        from .track import track
        track(a.note, warm=not a.no_warmup, rewarm=a.rewarm)
        return
    if a.cmd == "targets":
        from .targets import targets
        targets(a.note, warm=not a.no_warmup, rewarm=a.rewarm, show=a.show)
        return
    if a.cmd == "control":
        from . import app as _app
        _app.dot(warm=not (a.no_warmup or a.guest), rewarm=a.rewarm and not a.guest, serve=a.serve, show=a.dot,
                 control=True, click=a.click, guest=a.guest)
        return
    if a.cmd == "gesture":
        from .gesture import gesture
        gesture()
        return
    if a.cmd == "ir":
        from .ircam import view
        view()
        return
    if a.cmd == "bench":
        from .bench import bench
        bench()
        return
    from . import app

    {"calibrate": lambda: app.calibrate(a.tag, a.note), "train": lambda: app.train(force=True),
     "eval": lambda: app.evaluate(a.n, a.tag, a.note), "dot": lambda: app.dot(learn=not a.no_learn, warm=not (a.no_warmup or a.guest), rewarm=a.rewarm and not a.guest,
                             serve=a.serve, show=not a.no_dot, guest=a.guest),
     "motion": lambda: app.motion(a.mode, note=a.note), "edges": lambda: app.edges(a.note),
     "follow": lambda: __import__("elementfromgaze.follow", fromlist=["follow"]).follow(a.note)}[a.cmd]()


if __name__ == "__main__":
    main()
