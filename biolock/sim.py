"""Synthetic camera sessions made from still face photographs.

The tests and experiments/simulate.py use this to drive the real `Session` state machine
without a camera. A photograph is placed in a 640x480 frame at webcam scale and animated:
small movements, sensor noise, "closed eyes" (the eye regions are painted over) and a
left/right shift. A scripted `Actor` reacts to the session's current stage the way a
person follows the on-screen prompts.

THIS IS A SIMULATION. An animated photograph performs the liveness cues by construction,
so these sessions show how the state machine and the matcher behave on inputs that do
(or do not) contain the cues. They are not measurements of live people or of physical
presentation attacks; those need the live trials described in docs/TESTING_GUIDE.md.
"""

import math

import cv2
import numpy as np

from . import session as S

FRAME_H, FRAME_W = 480, 640
FPS = 30.0


class SimClock:
    """Session clock that only advances when the simulation says so."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt=1.0 / FPS):
        self.t += dt


def render(base, dx=0.0, dy=0.0, rot=0.0, zoom=1.0, size=(FRAME_H, FRAME_W)):
    """Place `base` (grey image) in a frame: centred, then rotated / scaled / shifted. Borders are replicated."""
    bh, bw = base.shape[:2]
    M = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0), rot, zoom)
    M[0, 2] += size[1] / 2.0 - bw / 2.0 + dx
    M[1, 2] += size[0] / 2.0 - bh / 2.0 + dy
    return cv2.warpAffine(base, M, (size[1], size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def scale_photo(photo, scale):
    if scale == 1.0:
        return photo
    return cv2.resize(photo, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def to_camera(display_frame):
    """Display-space frame -> what the camera would deliver (the session mirrors every frame)."""
    return cv2.flip(display_frame, 1)


def jitter_params(rng, rot=4.0, zoom=0.04, shift=8.0):
    """Random small pose change, used to emulate the frame-to-frame variation of a live enrolment."""
    return dict(rot=float(rng.uniform(-rot, rot)), zoom=float(1.0 + rng.uniform(-zoom, zoom)),
                dx=float(rng.uniform(-shift, shift)), dy=float(rng.uniform(-shift, shift)))


def add_noise(frame, sigma, rng):
    if sigma <= 0:
        return frame
    n = rng.normal(0.0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)


def motion_blur(frame, dx, dy):
    """Linear blur along the displacement since the previous frame (what a moving photograph looks like)."""
    length = int(round(math.hypot(dx, dy)))
    if length < 2:
        return frame
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2.0
    ang = math.atan2(dy, dx)
    for i in range(length):
        t = i - c
        k[int(round(c + t * math.sin(ang))), int(round(c + t * math.cos(ang)))] = 1.0
    return cv2.filter2D(frame, -1, k / k.sum(), borderType=cv2.BORDER_REPLICATE)


def locate(engine, base):
    """(face box, eyes or None) of the single face in the neutral frame, in display coordinates; None if unusable."""
    _f, gray, eq = S.prepare(to_camera(render(base)))
    boxes = engine.detect_faces(eq)
    if not boxes:
        return None
    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    box = max(boxes, key=lambda b: b[2] * b[3] / (1.0 + math.hypot(b[0] + b[2] / 2 - cx, b[1] + b[3] / 2 - cy) / 40.0))
    return box, engine.find_eyes(eq, box), len(boxes)


def close_eyes(engine, base, checks=6, noise=2.0):
    """Copy of `base` with the eye regions painted over so that the eye detector no longer fires.

    Returns None when that cannot be achieved for this photograph (it is then not used for blink simulation)."""
    found = locate(engine, base)
    if found is None:
        return None
    box, eyes, _n = found
    x, y, w, h = box
    if eyes is None:
        from .engine import BOX_EYE_L, BOX_EYE_R
        eyes = ((x + BOX_EYE_L[0] * w, y + BOX_EYE_L[1] * h), (x + BOX_EYE_R[0] * w, y + BOX_EYE_R[1] * h))
    ox = FRAME_W / 2.0 - base.shape[1] / 2.0                # frame -> base coordinates
    oy = FRAME_H / 2.0 - base.shape[0] / 2.0
    rng = np.random.RandomState(4321)
    for grow in (1.0, 1.3, 1.6, 2.0, 2.5):                 # paint over more until the eye check stays quiet
        mask = np.zeros(base.shape[:2], np.uint8)
        for ex, ey in eyes:
            cv2.ellipse(mask, (int(round(ex - ox)), int(round(ey - oy))),
                        (int(0.13 * w * grow), int(0.075 * w * grow)), 0, 0, 360, 255, -1)
        closed = cv2.inpaint(base, mask, 4, cv2.INPAINT_TELEA)
        quiet = True
        for _ in range(checks):                             # with sensor noise and a small shift, as in a session
            frame = add_noise(render(closed, dx=float(rng.uniform(-2, 2)), dy=float(rng.uniform(-2, 2))), noise, rng)
            if engine.eyes_visible(S.prepare(to_camera(frame))[2], box):
                quiet = False
                break
        if quiet:
            return closed
    return None


def blink_check(engine, base, closed, box, rng, frames=6, noise=2.0):
    """(frames in which the eye check sees the open eyes, frames in which it still sees the painted-over
    eyes), out of `frames` noisy, slightly shifted renderings of each."""
    seen_open = seen_closed = 0
    for _ in range(frames):
        pose = dict(dx=float(rng.uniform(-2, 2)), dy=float(rng.uniform(-2, 2)))
        for img, is_open in ((base, True), (closed, False)):
            _f, _g, eq = S.prepare(to_camera(add_noise(render(img, **pose), noise, rng)))
            if engine.eyes_visible(eq, box):
                seen_open += is_open
                seen_closed += not is_open
    return seen_open, seen_closed


class Actor:
    """A photograph that behaves like a person following the prompts.

    kind:
      'live'    moves gently, blinks when asked, moves left then right, then holds still
      'static'  does nothing at all (a photograph on a stand): only sensor noise changes
      'shaken'  a hand-held photograph: trembles, is moved left/right when asked, never blinks
    """

    def __init__(self, engine, photo, scale, rng, kind='live', noise=2.0, sway=10.0, period=24,
                 blink_len=4, blink_gap=12, react=8, speed=3.0, tremor=2.5, blinks=True):
        self.engine = engine
        self.base = scale_photo(photo, scale)
        self.rng = rng
        self.kind = kind
        self.noise = noise
        self.sway, self.period = sway, period
        self.blink_len, self.blink_gap = blink_len, blink_gap
        self.react, self.speed, self.tremor = react, speed, tremor
        # A photograph can play this role only if exactly one face is detected in it (a session refuses
        # frames with several) and, for a blinking actor, if the eye check sees its open eyes steadily
        # and stops seeing them once they are painted over. `why` says what was missing.
        self.closed = None
        found = locate(engine, self.base)
        self.why = ''
        if found is None:
            self.why = 'no face detected'
        elif found[2] != 1:
            self.why = 'several faces detected'
        elif kind == 'live' and blinks:
            self.closed = close_eyes(engine, self.base)
            if self.closed is None:
                self.why = 'painted-over eyes still detected'
            else:
                seen_open, seen_closed = blink_check(engine, self.base, self.closed, found[0], rng)
                if seen_open < 5:
                    self.why = 'open eyes not detected steadily'
                elif seen_closed:
                    self.why = 'painted-over eyes still detected'
        self.usable = not self.why
        self.t = 0
        self.x = self.y = 0.0
        self._px = self._py = 0.0
        self._stage = None
        self._since = 0             # frames since the stage last changed
        self._bank = [rng.normal(0.0, noise, (FRAME_H, FRAME_W)).astype(np.float32) for _ in range(6)] \
            if noise > 0 else []

    def _noise(self, frame):
        if not self._bank:
            return frame
        n = self._bank[self.t % len(self._bank)]
        return np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)

    def frame(self, status=None):
        """Next display-space frame, given the session's last Status (None before the first frame)."""
        stage = status.stage if status is not None else S.ST_PRESENCE
        if stage != self._stage:
            self._stage, self._since = stage, 0
        self._since += 1
        self.t += 1
        base = self.base
        if self.kind == 'static':
            return self._noise(render(base))

        if self.kind == 'shaken':
            self.x += float(self.rng.normal(0.0, self.tremor)) - 0.08 * self.x
            self.y += float(self.rng.normal(0.0, self.tremor)) - 0.08 * self.y
        target_x = 0.0
        if stage == S.ST_PRESENCE and self.kind == 'live':
            target_x = self.sway * math.sin(2 * math.pi * self.t / self.period)
            self.x, self.y = target_x, 0.4 * self.sway * math.sin(2 * math.pi * self.t / (1.7 * self.period))
        elif stage == S.ST_BLINK and self.kind == 'live':
            self.x *= 0.8
            self.y *= 0.8
            cycle = self.blink_gap + self.blink_len
            if self.closed is not None and self._since > self.react                     and (self._since - self.react) % cycle >= self.blink_gap:
                base = self.closed
        elif stage == S.ST_HEAD:
            if status is not None and not status.head_calibrating and self._since > self.react:
                step = -self.speed if status.head_phase == 'left' else self.speed
                self.x = float(np.clip(self.x + step, -0.3 * FRAME_W, 0.3 * FRAME_W))
            elif self.kind == 'live':
                self.x *= 0.8
        elif stage == S.ST_FACE and self.kind == 'live':
            self.x *= 0.85
            self.y *= 0.85
            self.x += float(self.rng.normal(0.0, 0.4))
            self.y += float(self.rng.normal(0.0, 0.4))
        out = render(base, self.x, self.y)
        if self.kind == 'shaken':
            out = motion_blur(out, self.x - self._px, self.y - self._py)
        self._px, self._py = self.x, self.y
        return self._noise(out)


def run(session, clock, next_frame, max_frames=4000, fps=FPS):
    """Feed frames from `next_frame(status)` (display space) until the session finishes. Returns the Result."""
    status = None
    for _ in range(max_frames):
        frame = next_frame(status)
        clock.tick(1.0 / fps)
        status = session.process(to_camera(frame))
        if status.done:
            return status.result
    return session.cancel()
