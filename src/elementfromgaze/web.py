"""Local gaze server for web pages: `efg dot --serve`.

  GET /gaze     Server-Sent Events, one JSON message per camera frame:
                {"t", "x", "y", "ok", "cx", "cy", "sd", "W", "H"}  (physical screen pixels;
                cx/cy = OS cursor, which the page uses to learn its own screen -> viewport mapping)
  GET /efg.js   the browser side: window.efg.elementFromGaze(), gaze events, debug overlay
  GET /demo     a test page with buttons, a pie menu and text
  POST /log     the demo's test results -> data/web_log.jsonl
  POST /label   {t0, t1, x, y, conf}: the page is confident you looked at (x, y) between t0 and t1
                (engine clock, screen px) -> a weak implicit label, like a click

Only binds 127.0.0.1. Standard library only: no extra dependency.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8770
STATIC = Path(__file__).with_name("web")


class GazeServer:
    def __init__(self, port: int = PORT, log_path: Path | None = None):
        self.clients: list[queue.Queue] = []
        self.log_path = log_path
        self.labels: queue.Queue = queue.Queue()      # gaze labels posted by pages (attractor snaps)
        self.lock = threading.Lock()
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _head(self, code, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

            def do_OPTIONS(self):
                self._head(204, "text/plain")

            def do_POST(self):
                path = self.path.split("?")[0]
                n = min(int(self.headers.get("Content-Length") or 0), 1_000_000)
                body = self.rfile.read(n)
                try:
                    rec = json.loads(body)
                except ValueError:
                    self._head(400, "text/plain")
                    return
                if path == "/label":
                    try:
                        lab = {k: float(rec[k]) for k in ("t0", "t1", "x", "y", "conf")}
                    except (KeyError, TypeError, ValueError):
                        self._head(400, "text/plain")
                        return
                    lab["conf"] = min(max(lab["conf"], 0.05), 0.5)      # a page can never outweigh a click
                    for k in ("sdx", "sdy"):                              # optional per-axis noise (px)
                        if k in rec:
                            try:
                                lab[k] = min(max(float(rec[k]), 10.0), 400.0)
                            except (TypeError, ValueError):
                                pass
                    if ("sdx" in lab) != ("sdy" in lab):
                        lab.pop("sdx", None), lab.pop("sdy", None)
                    lab["label"] = str(rec.get("label", ""))[:40]
                    srv.labels.put(lab)
                    self._head(204, "text/plain")
                    return
                if path != "/log" or srv.log_path is None:
                    self._head(404, "text/plain")
                    return
                with srv.lock, open(srv.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._head(204, "text/plain")

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/gaze":
                    self._head(200, "text/event-stream")
                    q: queue.Queue = queue.Queue(maxsize=8)
                    with srv.lock:
                        srv.clients.append(q)
                    try:
                        while True:
                            try:
                                msg = q.get(timeout=5)
                            except queue.Empty:
                                msg = ": ping"           # keeps proxies / the browser from timing out
                            self.wfile.write((msg + "\n\n").encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        pass
                    finally:
                        with srv.lock:
                            srv.clients.remove(q)
                    return
                files = {"/efg.js": ("efg.js", "application/javascript; charset=utf-8"),
                         "/demo": ("demo.html", "text/html; charset=utf-8"), "/": ("demo.html", "text/html; charset=utf-8")}
                if path in files:
                    name, ctype = files[path]
                    self._head(200, ctype)
                    self.wfile.write((STATIC / name).read_bytes())
                    return
                self._head(404, "text/plain")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{port}"

    def push(self, msg: dict) -> None:
        data = "data: " + json.dumps(msg, separators=(",", ":"))
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(data)
            except queue.Full:          # slow client: drop its oldest message, keep it live
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except (queue.Empty, queue.Full):
                    pass

    def close(self) -> None:
        self.httpd.shutdown()
