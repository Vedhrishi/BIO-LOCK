"""Camera access. Everything slow (opening a device, waiting for frames, per-frame processing,
key derivation) happens on worker threads; the UI thread only picks up the newest result.

    scan_cameras()            list the cameras that deliver frames (run it on a thread)
    open_camera(index)        try DirectShow, Media Foundation, then OpenCV's default backend
    CameraWorker(...)         runs one Session: grabber thread keeps only the newest frame,
                              processor thread runs Session.process on every new frame
"""

import os
import time
import threading

import cv2
import numpy as np

if os.name == 'nt':
    BACKENDS = (('DirectShow', cv2.CAP_DSHOW), ('Media Foundation', cv2.CAP_MSMF), ('default', cv2.CAP_ANY))
else:
    BACKENDS = (('default', cv2.CAP_ANY),)

WARMUP_FRAMES = 25
WARMUP_SECONDS = 3.0


class CameraInfo:
    def __init__(self, index, backend, width, height, dark=False, grey=False):
        self.index, self.backend = index, backend
        self.width, self.height = width, height
        self.dark = dark            # only black frames so far (covered lens / privacy shutter?)
        self.grey = grey            # no colour: often an infrared (Windows Hello) camera

    def describe(self):
        text = f'Camera {self.index}  -  {self.width}x{self.height}  ({self.backend})'
        if self.dark:
            text += '  -  black image'
        elif self.grey:
            text += '  -  greyscale / infrared'
        return text

    def __repr__(self):
        return f'<{self.describe()}>'


def is_black(frame):
    """True for the all-dark frames some drivers deliver while starting, or with the lens covered."""
    small = frame[::8, ::8]
    return float(small.mean()) < 6.0 and float(small.std()) < 4.0


def _is_grey(frame):
    if frame.ndim < 3 or frame.shape[2] < 3:
        return True
    s = frame[::8, ::8].astype(np.int16)
    return float(np.abs(s[..., 0] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 2]).mean()) < 1.5


def _warm_up(cap):
    """(first non-black frame or None, any frame delivered?) within the warm-up budget."""
    t0, got, frames, misses = time.monotonic(), None, 0, 0
    while frames < WARMUP_FRAMES and misses < 100 and time.monotonic() - t0 < WARMUP_SECONDS:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            misses += 1                         # some cameras need a second before the first frame arrives
            time.sleep(0.03)
            continue
        frames += 1
        got = frame
        if not is_black(frame):
            return frame, True
    return None, got is not None


def _open(index, api):
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # a request: the session accepts any size and aspect ratio
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # avoid stale queued frames where the driver supports it
    return cap


def open_camera(index, prefer=None):
    """(capture, CameraInfo) from the first backend that delivers real frames, else (None, reason).

    A backend that only delivers black frames is used as a last resort (flagged `dark`), so a
    covered lens is reported as such rather than as a missing camera."""
    order = sorted(BACKENDS, key=lambda b: b[0] != prefer)
    reason, dark_backend = 'No camera answered at this position.', None
    for name, api in order:
        cap = None
        try:
            cap = _open(index, api)
            if cap is None:
                continue
            frame, delivered = _warm_up(cap)
            if frame is not None:
                return cap, CameraInfo(index, name, frame.shape[1], frame.shape[0], grey=_is_grey(frame))
            if delivered and dark_backend is None:
                dark_backend = (name, api)
            reason = 'The camera opened but delivered no usable frames.'
        except Exception as e:                      # driver errors surface as cv2.error
            reason = f'{type(e).__name__}: {e}'
        if cap is not None:
            cap.release()
    if dark_backend is not None:
        cap = _open(index, dark_backend[1])
        if cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, CameraInfo(index, dark_backend[0], frame.shape[1], frame.shape[0], dark=True)
            cap.release()
        reason = 'The camera only delivers black frames. Is the lens covered or the privacy shutter closed?'
    return None, reason


def scan_cameras(max_index=5):
    """CameraInfo of every camera that delivers frames. Slow (seconds): call it from a thread."""
    found, misses = [], 0
    for i in range(max_index):
        cap, info = open_camera(i)
        if cap is None:
            misses += 1
            if misses >= 2 and i >= 1:
                break
            continue
        misses = 0
        cap.release()
        found.append(info)
    return found


class CameraWorker(threading.Thread):
    """Runs one enrolment or verification session off the UI thread.

    source     camera index, or an object with read()/release() (a fake camera in the tests)
    session    biolock.session.Session
    finalize   optional callable(Result): template training / key derivation / state changes, run here
               so the UI thread never waits for PBKDF2 or for the drive
    render     optional callable(Status) -> image for the UI (drawn and scaled on this thread)

    The UI polls `latest(last_seq)`; `state` is one of opening, running, finalizing, done, failed."""

    def __init__(self, source, session, finalize=None, render=None, prefer=None):
        super().__init__(daemon=True, name='biolock-camera')
        self.source = source
        self.session = session
        self.finalize = finalize
        self.render = render
        self.prefer = prefer
        self.state = 'opening'
        self.error = ''
        self.info = None
        self.result = None
        self._stop_evt = threading.Event()
        self._cancel_evt = threading.Event()
        self._lock = threading.Lock()
        self._packet = None             # (seq, status, image)
        self._seq = 0
        self._frame = None              # newest raw frame from the grabber
        self._frame_no = 0
        self._new = threading.Event()
        self._grab_failed = False

    # ---- called from the UI thread ------------------------------------------------
    def latest(self, last_seq=0):
        """Newest (seq, status, image) or None if nothing new since `last_seq`. Never blocks."""
        with self._lock:
            p = self._packet
        return p if p is not None and p[0] != last_seq else None

    def cancel(self):
        """Ask for the attempt to be abandoned (the session decides what that means)."""
        self._cancel_evt.set()
        self._new.set()

    def stop(self):
        """Stop without finalising (application is closing)."""
        self._stop_evt.set()
        self._new.set()

    @property
    def finished(self):
        return self.state in ('done', 'failed')

    # ---- worker threads -------------------------------------------------------------
    def _grab(self, cap):
        errors = 0
        while not self._stop_evt.is_set():
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None or frame.size == 0:
                errors += 1
                if errors > 40:                         # more than a second without a frame
                    self._grab_failed = True
                    self._new.set()
                    return
                time.sleep(0.03)
                continue
            errors = 0
            with self._lock:
                self._frame = frame
                self._frame_no += 1
            self._new.set()

    def _publish(self, status):
        image = None
        if self.render is not None:
            try:
                image = self.render(status)
            except Exception:
                image = None
        with self._lock:
            self._seq += 1
            self._packet = (self._seq, status, image)

    def run(self):
        cap, grabber = None, None
        try:
            if isinstance(self.source, int):
                cap, info = open_camera(self.source, self.prefer)
                if cap is None:
                    self.error = info
                    self.state = 'failed'
                    return
                self.info = info
            else:
                cap = self.source
            if self._stop_evt.is_set():
                return
            self.state = 'running'
            grabber = threading.Thread(target=self._grab, args=(cap,), daemon=True, name='biolock-grabber')
            grabber.start()
            seen = 0
            while not self._stop_evt.is_set():
                self._new.wait(0.25)
                self._new.clear()
                if self._stop_evt.is_set():
                    break
                if self._cancel_evt.is_set() or self._grab_failed:
                    if self._grab_failed:
                        self.error = 'The camera stopped delivering frames.'
                    self.session.cancel()
                    break
                with self._lock:
                    frame, n = self._frame, self._frame_no
                if frame is None or n == seen:
                    continue
                seen = n
                status = self.session.process(frame)
                self._publish(status)
                if status.done:
                    break
            result = self.session.result
            if result is not None and not self._stop_evt.is_set():
                if self.finalize is not None:
                    self.state = 'finalizing'
                    try:
                        self.finalize(result)
                    except Exception as e:              # never leave the UI waiting for a dead worker
                        result.outcome = 'error'
                        result.message = f'{type(e).__name__}: {e}'
                self.result = result
        except Exception as e:
            self.error = f'{type(e).__name__}: {e}'
            self.state = 'failed'
        finally:
            self._stop_evt.set()
            if grabber is not None:
                grabber.join(timeout=2.0)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if self.state != 'failed':
                self.state = 'failed' if (self._grab_failed and self.result is None) else 'done'
