"""The laptop's Windows Hello infrared camera, read through Windows.Media.Capture.

OpenCV can't open it (it isn't exposed as a regular webcam), but the Windows Runtime frame API
can. On this Lenovo it streams 640x360 8-bit at 15 fps, alternating emitter-lit frames with
blanked ones, so ~7 useful frames per second. In the lit frames each eye shows a corneal glint
(the reflection of the IR emitter): the reference point commercial eye trackers measure the pupil
against, and it doesn't care about room light.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np


class IRCamera:
    """Background thread that keeps the latest emitter-lit IR frame."""

    def __init__(self, min_brightness: float = 2.0, record: bool = False):
        self.min_brightness = min_brightness
        self.record = record
        self.recorded: list[tuple[float, np.ndarray, bool]] = []   # (monotonic t, JPEG, emitter lit)
        self.frame: np.ndarray | None = None
        self.seq = 0
        self.t = 0.0
        self.error: str | None = None
        self.lit = self.dark = 0
        self._cond = threading.Condition()
        self._running = True
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()

    async def _open(self):
        from winrt.windows.media.capture import (MediaCapture, MediaCaptureInitializationSettings,
                                                 MediaCaptureMemoryPreference, MediaCaptureSharingMode,
                                                 StreamingCaptureMode)
        from winrt.windows.media.capture.frames import MediaFrameSourceGroup, MediaFrameSourceKind

        groups = await MediaFrameSourceGroup.find_all_async()
        ir_groups = [g for g in groups if any(i.source_kind == MediaFrameSourceKind.INFRARED for i in g.source_infos)]
        if not ir_groups:
            raise RuntimeError("no hay cámara infrarroja")
        # prefer the IR-only group; the combined vendor group also works
        g = min(ir_groups, key=lambda g: len(list(g.source_infos)))
        last_err = None
        for mode in (MediaCaptureSharingMode.EXCLUSIVE_CONTROL, MediaCaptureSharingMode.SHARED_READ_ONLY):
            s = MediaCaptureInitializationSettings()
            s.source_group = g
            s.sharing_mode = mode
            s.memory_preference = MediaCaptureMemoryPreference.CPU
            s.streaming_capture_mode = StreamingCaptureMode.VIDEO
            cap = MediaCapture()
            try:
                await cap.initialize_async(s)
            except Exception as e:  # noqa: BLE001 - try the next sharing mode
                last_err = e
                continue
            src = next(v for v in cap.frame_sources.values() if v.info.source_kind == MediaFrameSourceKind.INFRARED)
            return cap, src
        raise RuntimeError(f"no se pudo abrir la cámara IR: {last_err}")

    async def _main(self) -> None:
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import Buffer

        try:
            cap, src = await self._open()
            reader = await cap.create_frame_reader_async(src)
            await reader.start_async()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            return
        last_ts = None
        prev_mean = 0.0
        try:
            while self._running:
                fr = reader.try_acquire_latest_frame()
                if fr is None or fr.video_media_frame is None or fr.system_relative_time == last_ts:
                    await asyncio.sleep(0.004)
                    continue
                last_ts = fr.system_relative_time
                bmp = fr.video_media_frame.software_bitmap
                if bmp is None:
                    continue
                if bmp.bitmap_pixel_format != BitmapPixelFormat.GRAY8:
                    bmp = SoftwareBitmap.convert(bmp, BitmapPixelFormat.GRAY8)
                w, h = bmp.pixel_width, bmp.pixel_height
                buf = Buffer(w * h)
                bmp.copy_to_buffer(buf)
                img = np.frombuffer(bytes(buf), np.uint8)[: w * h].reshape(h, w).copy()
                # The emitter flashes every other frame. At night the unlit frames come out black, but
                # in daylight ambient IR makes them merely dimmer, so compare with the previous frame.
                mean = float(img[::4, ::4].mean())
                lit = mean > self.min_brightness and mean > 1.3 * prev_mean
                prev_mean = mean
                now = time.monotonic()
                if self.record:   # unlit frames too: lit minus unlit isolates the emitter's light
                    import cv2
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ok:
                        self.recorded.append((now, jpg.ravel(), lit))
                if not lit:
                    self.dark += 1
                    continue
                self.lit += 1
                with self._cond:
                    self.frame, self.t, self.seq = img, now, self.seq + 1
                    self._cond.notify_all()
        finally:
            await reader.stop_async()
            cap.close()

    def next(self, last_seq: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        with self._cond:
            self._cond.wait_for(lambda: self.seq != last_seq or not self._running or self.error, timeout)
            return self.seq, (self.frame if self.seq != last_seq else None)

    def fps(self) -> float:
        return self.lit / max(time.monotonic() - self._t0, 1e-6)

    def take(self) -> list[tuple[float, np.ndarray, bool]]:
        out, self.recorded = self.recorded, []
        return out

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=3)


def open_ir(record: bool = False) -> "IRCamera | None":
    """The IR camera if this machine has one and it opens; None otherwise (never raises)."""
    try:
        import winrt.windows.media.capture  # noqa: F401
    except Exception:  # noqa: BLE001 - not Windows / bindings missing
        return None
    cam = IRCamera(record=record)
    t0 = time.monotonic()
    while cam.seq == 0 and cam.error is None and time.monotonic() - t0 < 3.0:
        time.sleep(0.05)
    if cam.error or cam.seq == 0:
        cam.close()
        return None
    return cam


def view() -> None:
    """Live view: the IR frame plus both eyes magnified, so you can see the pupil and the glint."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from .features import ensure_model
    from .tracker import LANDMARKER

    cam = IRCamera()
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(LANDMARKER))), num_faces=1))
    print("cámara IR: ESC o q para salir")
    seq, t0 = 0, time.monotonic()
    try:
        while True:
            seq, img = cam.next(seq)
            if cam.error:
                print("error:", cam.error)
                return
            if img is None:
                continue
            h, w = img.shape
            big = cv2.cvtColor(cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))))
            strip = np.zeros((240, w * 2, 3), np.uint8)
            if r.face_landmarks:
                L = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
                for k, (a, b) in enumerate(((33, 133), (362, 263))):
                    c = (L[a] + L[b]) / 2
                    half = max(np.linalg.norm(L[b] - L[a]) * 0.9, 6)
                    x0, x1 = int(c[0] - half), int(c[0] + half)
                    y0, y1 = int(c[1] - half / 2), int(c[1] + half / 2)
                    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
                        eye = cv2.resize(img[y0:y1, x0:x1], (w, 240), interpolation=cv2.INTER_CUBIC)
                        strip[:, k * w:(k + 1) * w] = cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(big, (x0 * 2, y0 * 2), (x1 * 2, y1 * 2), (80, 200, 80), 1)
                width = np.linalg.norm(L[133] - L[33])
                msg = f"ancho del ojo {width:.0f} px"
            else:
                msg = "no veo la cara"
            fps = cam.lit / max(time.monotonic() - t0, 1e-6)
            cv2.putText(big, f"IR {w}x{h} · {fps:.1f} frames iluminados/s · {msg}".replace("·", "-"), (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("elementfromgaze - IR", np.vstack([big, strip]))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cam.close()
        lm.close()
        cv2.destroyAllWindows()
