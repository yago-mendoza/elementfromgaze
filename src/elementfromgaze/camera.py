"""Webcam capture on a background thread that always holds the newest frame."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, warmup: int = 20):
        hd = width > 640
        # MSMF hands OpenCV a broken stride above 640x480 on this webcam; DirectShow + MJPG works
        # (1920x1080 at ~17 fps here) and keeps USB bandwidth low.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hd else cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")
        if hd:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # The first frames come out overexposed while auto-exposure settles.
        for _ in range(warmup):
            self.cap.read()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def next(self, last_seq: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than `last_seq` exists; return (seq, frame)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or not self._running, timeout)
            return self._seq, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
