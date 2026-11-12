"""Behavioural liveness cues: temporal motion (M), eye blinks (B) and a left-then-right head movement (H).

Each detector is a small state machine that is fed once per camera frame. They are
liveness *cues*: they stop a photograph that is standing still or being waved about,
but a replayed video of the owner that contains two blinks and the head movement
would satisfy them.

Both image measurements below compare the previous frame with the current one inside
the SAME face box, on block-averaged pixels. Averaging over blocks removes most of the
camera's sensor noise, which otherwise looks like movement.
"""

import cv2
import numpy as np

MOTION_BLOCK = 8        # pixels averaged per block side for the motion measure
CHANGE_COLS = 24        # the blink "where did the image change" measure uses a 24-column grid

# Parts of the face box (fractions of its width/height) compared when the eyes disappear.
EYE_ROWS = (0.30, 0.50)
REST_ROWS = (0.55, 0.95)
SIDE = (0.12, 0.88)


def _small(img, cols, rows):
    return cv2.resize(img, (max(1, cols), max(1, rows)), interpolation=cv2.INTER_AREA)


def box_motion(prev_eq, eq, box, block=MOTION_BLOCK):
    """Mean absolute difference (0-1) between two frames inside `box`, on block-averaged pixels."""
    x, y, w, h = box
    a, b = prev_eq[y:y + h, x:x + w], eq[y:y + h, x:x + w]
    if a.size == 0 or a.shape != b.shape:
        return None
    cols, rows = a.shape[1] // block, a.shape[0] // block
    if cols < 2 or rows < 2:
        return None
    return cv2.mean(cv2.absdiff(_small(a, cols, rows), _small(b, cols, rows)))[0] / 255.0


def local_change(prev_eq, eq, box):
    """(change in the eye band, change in the rest of the face) between two frames, each 0-1.

    A real blink changes the eye band and little else. A photograph that is moved, or a
    blurred frame, changes both; sensor noise changes neither."""
    x, y, w, h = box
    x0, x1 = x + int(SIDE[0] * w), x + int(SIDE[1] * w)
    out = []
    for r0, r1 in (EYE_ROWS, REST_ROWS):
        y0, y1 = y + int(r0 * h), y + int(r1 * h)
        a, b = prev_eq[y0:y1, x0:x1], eq[y0:y1, x0:x1]
        if a.size == 0 or a.shape != b.shape:
            return None
        rows = max(2, int(round(CHANGE_COLS * a.shape[0] / float(max(a.shape[1], 1)))))
        out.append(cv2.mean(cv2.absdiff(_small(a, CHANGE_COLS, rows), _small(b, CHANGE_COLS, rows)))[0] / 255.0)
    return out[0], out[1]


class MotionLive:
    """Presence check: frame-to-frame motion inside the face box, averaged over a window.

    Passes when the average of the last `frames` observations reaches `thresh`
    (at least half a window must have been observed)."""

    def __init__(self, thresh=0.009, frames=20):
        self.thresh = float(thresh)
        self.frames = int(frames)
        self.reset()

    def reset(self):
        self._m = []
        self.is_live = False
        self.score = 0.0

    @property
    def average(self):
        return sum(self._m) / len(self._m) if self._m else 0.0

    @property
    def progress(self):
        if self.is_live:
            return 1.0
        fill = min(1.0, len(self._m) / max(self.frames // 2, 1))
        return fill * min(1.0, self.average / self.thresh)

    def update(self, motion):
        """motion: box_motion() of this frame, or None when it cannot be measured."""
        if self.is_live or motion is None:
            return self.is_live
        self.score = float(motion)
        self._m.append(self.score)
        if len(self._m) > self.frames:
            self._m.pop(0)
        if len(self._m) >= max(1, self.frames // 2) and self.average >= self.thresh:
            self.is_live = True
        return self.is_live


class BlinkDetector:
    """Counts blinks from the per-frame answer to "is an open eye visible?".

    A blink is: eyes visible for at least `open_stable` frames, then not visible for
    `min_closed`..`max_closed` frames, then visible again, provided that

      * the face box stayed where it was while the eyes were not visible (`max_shift`), and
      * at the moment the eyes disappeared the image changed mainly in the eye band: by at
        least `min_change`, and at least `local_ratio` times as much as the rest of the face.

    The two conditions matter because the eye detector also loses the eyes when the face
    (or a photograph of it) moves or blurs, and occasionally through sensor noise. Those
    drop-outs change the whole face or nothing at all, so they are not counted."""

    ONSET = 2       # the eye-band change is looked for in the frames around the eyes disappearing

    def __init__(self, needed=2, min_closed=2, max_closed=24, open_stable=3, cooldown=6, max_shift=0.08,
                 local_ratio=2.5, min_change=0.011):
        self.needed = int(needed)
        self.min_closed = int(min_closed)
        self.max_closed = int(max_closed)
        self.open_stable = int(open_stable)
        self.cooldown = int(cooldown)
        self.max_shift = float(max_shift)
        self.local_ratio = float(local_ratio)
        self.min_change = float(min_change)
        self.reset()

    def reset(self):
        self.blinks = 0
        self.is_done = False
        self.rejected = {'moved': 0, 'not_local': 0, 'length': 0}   # eye drop-outs that were not counted
        self._open_run = 0          # consecutive frames with eyes visible
        self._closed_run = 0        # consecutive frames without
        self._armed = False         # eyes were stably open before the current closed run
        self._cool = 0
        self._anchor = None         # face box when the eyes closed
        self._moved = False
        self._local = False         # the eye band changed more than the rest of the face
        self._prev_change = None

    def interrupt(self):
        """The face was lost (or a second face appeared): whatever was in progress is not a blink."""
        self._open_run = 0
        self._closed_run = 0
        self._armed = False
        self._anchor = None
        self._moved = False
        self._local = False
        self._prev_change = None

    @staticmethod
    def _shift(box, anchor):
        (x, y, w, h), (ax, ay, aw, ah) = box, anchor
        ref = float(max(aw, 1))
        centre = max(abs((x + w / 2.0) - (ax + aw / 2.0)), abs((y + h / 2.0) - (ay + ah / 2.0))) / ref
        size = abs(w - aw) / ref
        return max(centre, size)

    def _is_local(self, change):
        if change is None:
            return False
        eye, rest = change
        return eye >= self.min_change and eye >= self.local_ratio * max(rest, 1e-4)

    def update(self, eyes_open, box, change=None):
        """Feed one frame. `change` is local_change() for this frame; without it the eye-band
        condition is not applied (used by callers that have no images, e.g. unit tests).
        Returns True when a blink was counted on this frame."""
        if self.is_done:
            return False
        self._cool = max(0, self._cool - 1)
        check_local = change is not None or self._prev_change is not None
        counted = False
        if eyes_open:
            if self._closed_run:
                length_ok = self.min_closed <= self._closed_run <= self.max_closed
                if self._armed and self._cool == 0:
                    if not length_ok:
                        self.rejected['length'] += 1
                    elif self._moved:
                        self.rejected['moved'] += 1
                    elif not self._local:
                        self.rejected['not_local'] += 1
                    else:
                        self.blinks += 1
                        self._cool = self.cooldown
                        counted = True
                self._closed_run = 0
                self._open_run = 0
                self._armed = False
            self._open_run += 1
        else:
            if self._closed_run == 0:
                self._armed = self._open_run >= self.open_stable
                self._anchor = box
                self._moved = False
                # the eyes may start to close one frame before the detector loses them
                self._local = (not check_local) or self._is_local(self._prev_change) or self._is_local(change)
            else:
                if self._closed_run < self.ONSET and self._is_local(change):
                    self._local = True
                if box is not None and self._anchor is not None \
                        and self._shift(box, self._anchor) > self.max_shift:
                    self._moved = True
            self._closed_run += 1
            self._open_run = 0
        self._prev_change = change
        if self.blinks >= self.needed:
            self.is_done = True
        return counted


class HeadTurn:
    """Left-then-right head movement, measured as the shift of the face-box centre in % of face width.

    The baseline is the mean centre over the first `calib_obs` observations."""

    L, R, D = 'left', 'right', 'done'

    def __init__(self, pct=10, calib_obs=10):
        self.pct = float(pct)
        self.calib_obs = int(calib_obs)
        self.reset()

    def reset(self):
        self.phase = self.L
        self.baseline = None
        self._calib = []
        self.is_done = False
        self.delta_pct = 0.0

    @property
    def calibrating(self):
        return self.baseline is None

    def update(self, box):
        if self.is_done or box is None:
            return self.is_done
        x, _y, w, _h = box
        cx = x + w / 2.0
        if self.baseline is None:
            self._calib.append(cx)
            if len(self._calib) < self.calib_obs:
                return False
            self.baseline = float(np.mean(self._calib))
        self.delta_pct = (cx - self.baseline) / max(w, 1) * 100.0
        if self.phase == self.L:
            if self.delta_pct <= -self.pct:
                self.phase = self.R
        elif self.delta_pct >= self.pct:
            self.phase = self.D
            self.is_done = True
        return self.is_done
