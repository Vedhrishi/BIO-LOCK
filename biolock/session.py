"""One enrolment or verification attempt as a frame-by-frame state machine.

No GUI and no threads live here: `Session.process(frame)` is fed camera frames (by the
camera worker, a test, or a simulation) and returns a `Status` that says what to show.
The clock is injectable, so recorded or synthetic sessions can run faster than real time.

Stages, in order (the pipeline of the paper):

    presence  M   average frame-to-frame motion inside the face box reaches motion_thresh
    blink     B   blinks_needed blinks: eyes visible -> not visible -> visible, with the head
                  still and the image changing mainly in the eye band when the eyes disappear
    head      H   face-box centre moves head_turn_pct % of the face width left, then right
    face      F/R enrolment: capture enrol_frames quality-checked samples
                  verification: median fused score of match_frames frames >= min_conf

Two rules tie the stages together:
  * exactly one face must be in the frame for a frame to count;
  * with bind_liveness_identity (default) the face seen while blinking must also match:
    the median score of the blink-stage frames has to reach min_conf as well, so the
    liveness steps cannot be done by one face and the match by another (or by a photo).
"""

import time

import cv2
import numpy as np

from . import liveness
from .engine import lighting_quality, sharpness

FRAME_WIDTH = 640

MODE_ENROL = 'enrol'
MODE_VERIFY = 'verify'

ST_PRESENCE, ST_BLINK, ST_HEAD, ST_FACE, ST_DONE = 'presence', 'blink', 'head', 'face', 'done'
STAGES = (ST_PRESENCE, ST_BLINK, ST_HEAD, ST_FACE)

OUT_ENROLLED = 'enrolled'       # all samples captured (the controller then trains and saves)
OUT_ACCEPT = 'accept'
OUT_REJECT = 'reject'
OUT_TIMEOUT = 'timeout'
OUT_CANCEL = 'cancelled'
OUT_ERROR = 'error'

# prompts shown while the enrolment samples are captured, one block after another
ENROL_POSES = ('Look straight at the camera', 'Turn your head slightly LEFT', 'Turn your head slightly RIGHT',
               'Tilt your head slightly UP', 'Tilt your head slightly DOWN')


def prepare(frame, width=FRAME_WIDTH, mirror=True):
    """Camera frame -> (display BGR, raw grey, equalised grey), `width` pixels wide, aspect ratio kept.

    Detection and motion use the equalised image; face crops are cut from the raw grey image."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    if w != width:
        nh = max(2, int(round(h * width / float(w))))
        frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA if w > width else cv2.INTER_LINEAR)
    if mirror:
        frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray, cv2.equalizeHist(gray)


class Result:
    """Outcome of a finished session (one row of logs/attempts.csv is made from it)."""

    def __init__(self, mode):
        self.mode = mode
        self.outcome = None
        self.failed_stage = ''
        self.score = None               # median fused score of the match frames
        self.live_score = None          # median fused score of the blink-stage frames
        self.components = {}            # medians: lbph, hist, template, lbph_distance
        self.match_frames = 0
        self.passed = {'presence': False, 'blink': False, 'head': False}
        self.motion_avg = 0.0
        self.blinks = 0
        self.blink_rejects = {}         # eye drop-outs that were not counted as blinks, by reason
        self.blink_frames = 0           # frames of the blink step in which the eye check ran ...
        self.eyes_seen_frames = 0       # ... and in which it saw open eyes
        self.frames = 0
        self.faces = {0: 0, 1: 0, 2: 0}  # detection passes that found 0 / 1 / 2+ faces
        self.latency_s = 0.0            # first frame -> decision, on the session clock
        self.proc_ms = 0.0              # mean processing time per frame (real time)
        self.proc_ms_max = 0.0
        self.samples = []               # enrolment only: the normalised crops
        self.quality = 0                # enrolment only: lighting score of the last sample
        self.message = ''
        self.extra = {}                 # filled in by the controller (attempts left, wiped ...)

    @property
    def accepted(self):
        return self.outcome == OUT_ACCEPT

    @property
    def liveness_pass(self):
        return all(self.passed.values())


class Status:
    """What the UI needs to draw one frame. `frame` is the mirrored display image (no overlay)."""
    __slots__ = ('frame', 'mode', 'stage', 'boxes', 'box', 'multi', 'prompt', 'progress', 'passed',
                 'motion', 'motion_avg', 'blinks', 'head_phase', 'head_delta', 'head_calibrating', 'score',
                 'samples',
                 'lighting_ok', 'lighting', 'done', 'result', 'frame_no')

    def __init__(self):
        self.frame = None
        self.mode = MODE_VERIFY
        self.stage = ST_PRESENCE
        self.boxes = []
        self.box = None
        self.multi = False
        self.prompt = ''
        self.progress = 0.0
        self.passed = {}
        self.motion = 0.0
        self.motion_avg = 0.0
        self.blinks = 0
        self.head_phase = ''
        self.head_delta = 0.0
        self.head_calibrating = True
        self.score = None
        self.samples = 0
        self.lighting_ok = True
        self.lighting = 100
        self.done = False
        self.result = None
        self.frame_no = 0


class Session:

    def __init__(self, engine, cfg, mode, clock=time.monotonic):
        if mode not in (MODE_ENROL, MODE_VERIFY):
            raise ValueError(f'unknown mode {mode!r}')
        if mode == MODE_VERIFY and not engine.trained:
            raise ValueError('verification needs an enrolled face')
        self.engine = engine
        self.cfg = cfg
        self.mode = mode
        self.clock = clock
        self.motion = liveness.MotionLive(cfg['motion_thresh'], cfg['motion_frames'])
        self.blink = liveness.BlinkDetector(cfg['blinks_needed'], cfg['blink_min_cl'], cfg['blink_max_cl'],
                                            max_shift=cfg['blink_max_shift'],
                                            local_ratio=cfg['blink_local_ratio'],
                                            min_change=cfg['blink_min_change'])
        self.head = liveness.HeadTurn(cfg['head_turn_pct'], cfg['head_calib_obs'])
        self.stage = ST_PRESENCE
        self.result = None
        self.frame_no = 0
        self._res = Result(mode)
        self._t0 = None                 # session clock at the first frame
        self._t_face = None             # session clock when the face stage began
        self._prev_eq = None
        self._boxes = []
        self._box = None
        self._last_box = None           # last usable single-face box
        self._gap = 0                   # frames since the single face was last detected
        self._proc = []
        self._scores = []
        self._comps = []
        self._live_scores = []
        self._last_score = None
        self._motion_last = 0.0
        self._lighting = (100, '')
        # enrolment capture
        self._sharp = []
        self._last_raw = None
        self._last_take = None          # (frame number, clock) of the last accepted sample
        self._hint = ''

    # ------------------------------------------------------------------ public
    def cancel(self):
        """Abandon the attempt. Once matching has started this counts as a failed match,
        so an attempt cannot be backed out of after seeing the score."""
        if self.result is not None:
            return self.result
        if self.mode == MODE_VERIFY and self.stage == ST_FACE:
            return self._finish(OUT_REJECT, 'match', 'Cancelled during the face match.')
        return self._finish(OUT_CANCEL, self.stage, 'Cancelled.')

    def process(self, frame_bgr):
        """Advance the state machine by one camera frame and return the Status to display."""
        t_real = time.perf_counter()
        now = self.clock()
        frame, gray, eq = prepare(frame_bgr)
        if self.result is not None:
            return self._status(frame)
        if self._t0 is None:
            self._t0 = now
        self.frame_no += 1
        c = self.cfg

        if self._prev_eq is not None and self._prev_eq.shape != eq.shape:
            self._prev_eq = None                                   # camera changed resolution

        # ---- face detection: every frame while matching/capturing, every Nth during liveness
        every = 1 if self.stage == ST_FACE else max(1, int(c['process_every']))
        fresh = every == 1 or self._box is None or (self.frame_no - 1) % every == 0
        if fresh:
            boxes = self.engine.detect_faces(eq)
            self._boxes = boxes
            self._res.faces[min(len(boxes), 2)] += 1
            if len(boxes) == 1:
                self._box, self._gap = boxes[0], 0
            else:
                self._box = None
        boxes, box = self._boxes, self._box
        multi = len(boxes) > 1
        if box is None:
            self._gap += 1
        small = box is not None and box[2] < c['min_face_px']
        usable = box is not None and not small

        if usable:
            x, y, w, h = box
            self._lighting = lighting_quality(gray[y:y + h, x:x + w])

        # ---- timeouts
        if self.stage in (ST_PRESENCE, ST_BLINK, ST_HEAD):
            if now - self._t0 > c['liveness_timeout_s']:
                self._prev_eq = eq
                self._finish(OUT_TIMEOUT, self.stage,
                             f'The {self.stage} step was not completed in {c["liveness_timeout_s"]:g} s.')
                return self._status(frame, t_real)
        elif self.mode == MODE_VERIFY:
            if now - self._t_face > c['match_timeout_s']:
                self._decide(timed_out=True)
                return self._status(frame, t_real)
        elif now - self._t_face > c['enrol_timeout_s']:
            self._finish(OUT_TIMEOUT, 'capture', 'Enrolment took too long and was abandoned.')
            return self._status(frame, t_real)

        # ---- stage logic
        if self.stage == ST_PRESENCE:
            m = liveness.box_motion(self._prev_eq, eq, box) if usable and self._prev_eq is not None else None
            if m is not None:
                self._motion_last = m
                if self.motion.update(m):
                    self._res.passed['presence'] = True
                    self._res.motion_avg = self.motion.average
                    self.stage = ST_BLINK

        elif self.stage == ST_BLINK:
            # A blink can make the face detector miss a frame or two; the eye check then keeps
            # using the last face box. A longer gap, or a second face, cancels the blink in progress.
            look = box if usable else (self._last_box if (not multi and not small
                                                          and self._gap <= c['blink_face_gap']) else None)
            if look is None:
                self.blink.interrupt()
            else:
                eyes_open = self.engine.eyes_visible(eq, look)
                self._res.blink_frames += 1
                self._res.eyes_seen_frames += bool(eyes_open)
                change = liveness.local_change(self._prev_eq, eq, look) if self._prev_eq is not None else None
                self.blink.update(eyes_open, look, change or (0.0, 0.0))
                if eyes_open and usable and self.mode == MODE_VERIFY and c['bind_liveness_identity'] \
                        and (len(self._live_scores) < 3 or self.frame_no % 4 == 0) \
                        and len(self._live_scores) < 60:
                    # who is blinking? (every 4th frame is enough and keeps the frame rate up)
                    roi, _raw, _eyes = self.engine.extract(gray, eq, box)
                    self._live_scores.append(self.engine.score(roi)[0])
            if self.blink.is_done:
                self._res.passed['blink'] = True
                self.stage = ST_HEAD

        elif self.stage == ST_HEAD:
            if fresh and usable:
                self.head.update(box)
            if self.head.is_done:
                self._res.passed['head'] = True
                self.stage = ST_FACE
                self._t_face = now

        elif self.mode == MODE_VERIFY:                      # ST_FACE: match
            if usable:
                roi, _raw, _eyes = self.engine.extract(gray, eq, box)
                s, comp = self.engine.score(roi)
                self._scores.append(s)
                self._comps.append(comp)
                self._last_score = s
                if len(self._scores) >= c['match_frames']:
                    self._decide()

        else:                                               # ST_FACE: enrolment capture
            if usable:
                self._capture(gray, eq, box, now)
            else:
                self._hint = ''

        if usable:
            self._last_box = box
        self._prev_eq = eq
        return self._status(frame, t_real, multi=multi, small=small)

    # ------------------------------------------------------------------ internals
    def _capture(self, gray, eq, box, now):
        c = self.cfg
        q, msg = self._lighting
        if q < 50:
            self._hint = msg
            return
        if self._last_take is not None:
            if self.frame_no - self._last_take[0] <= 2 or now - self._last_take[1] < c['enrol_gap_s']:
                return                                      # spread the samples out in time
        roi, raw, _eyes = self.engine.extract(gray, eq, box)
        sharp = sharpness(raw)
        self._sharp.append(sharp)
        if len(self._sharp) > 30:
            self._sharp.pop(0)
        if len(self._sharp) >= 8 and sharp < 0.6 * float(np.median(self._sharp)):
            self._hint = 'Hold steady - the image is blurry'
            return
        if self._last_raw is not None and cv2.mean(cv2.absdiff(raw, self._last_raw))[0] < 1.0:
            self._hint = 'Move your head a little'
            return
        self._res.samples.append(roi)
        self._res.quality = q
        self._last_raw = raw
        self._last_take = (self.frame_no, now)
        self._hint = ''
        if len(self._res.samples) >= c['enrol_frames']:
            self._finish(OUT_ENROLLED, '', 'All samples captured.')

    def _decide(self, timed_out=False):
        c = self.cfg
        r = self._res
        r.match_frames = len(self._scores)
        if self._scores:
            r.score = float(np.median(self._scores))
            for k in ('lbph', 'hist', 'template', 'lbph_distance'):
                r.components[k] = float(np.median([x[k] for x in self._comps]))
        if self._live_scores:
            r.live_score = float(np.median(self._live_scores))
        if timed_out or r.match_frames < c['match_frames']:
            return self._finish(OUT_REJECT, 'match',
                                f'Only {r.match_frames} of {c["match_frames"]} usable frames within '
                                f'{c["match_timeout_s"]:g} s.')
        accepted, _med = self.engine.decide(self._scores)
        if not accepted:
            return self._finish(OUT_REJECT, 'match',
                                f'Match score {r.score:.0f}/100 is below {c["min_conf"]:g}.')
        if c['bind_liveness_identity'] and (len(self._live_scores) < 3 or r.live_score < c['min_conf']):
            shown = 'n/a' if r.live_score is None else f'{r.live_score:.0f}'
            return self._finish(OUT_REJECT, 'match',
                                f'The face seen during the blink step did not match (score {shown}/100).')
        return self._finish(OUT_ACCEPT, '', f'Match score {r.score:.0f}/100.')

    def _finish(self, outcome, failed_stage, message):
        r = self._res
        r.outcome, r.failed_stage, r.message = outcome, failed_stage, message
        r.frames = self.frame_no
        r.blinks = self.blink.blinks
        r.blink_rejects = dict(self.blink.rejected)
        if not r.passed['presence']:
            r.motion_avg = self.motion.average
        if self._t0 is not None:
            r.latency_s = max(0.0, self.clock() - self._t0)
        if self._proc:
            r.proc_ms = float(np.mean(self._proc))
            r.proc_ms_max = float(np.max(self._proc))
        self.stage = ST_DONE
        self.result = r
        return r

    def _prompt(self, multi, small):
        if self.result is not None:
            return self.result.message
        if multi:
            return 'Only ONE person may be in the frame'
        if self._box is None:
            return 'Face not detected - centre your face in the frame'
        if small:
            return 'Move closer to the camera'
        if self.stage == ST_PRESENCE:
            return 'Move your head gently from side to side'
        if self.stage == ST_BLINK:
            return f'Hold still and blink slowly  ({self.blink.blinks}/{self.blink.needed})'
        if self.stage == ST_HEAD:
            if self.head.calibrating:
                return 'Look straight at the camera'
            return 'Move your head to the LEFT' if self.head.phase == self.head.L \
                else 'Now move your head to the RIGHT'
        if self.mode == MODE_VERIFY:
            return 'Hold still - matching your face'
        n, total = len(self._res.samples), self.cfg['enrol_frames']
        if self._hint:
            return self._hint
        pose = ENROL_POSES[min(len(ENROL_POSES) - 1, n * len(ENROL_POSES) // max(total, 1))]
        return f'{pose}  ({n}/{total})'

    def _progress(self):
        if self.result is not None:
            return 1.0 if self.result.outcome in (OUT_ACCEPT, OUT_ENROLLED) else 0.0
        if self.stage == ST_PRESENCE:
            return 0.25 * self.motion.progress
        if self.stage == ST_BLINK:
            return 0.25 + 0.25 * min(1.0, self.blink.blinks / max(self.blink.needed, 1))
        if self.stage == ST_HEAD:
            return 0.50 + (0.0 if self.head.calibrating else 0.10 if self.head.phase == self.head.L else 0.18)
        if self.mode == MODE_VERIFY:
            return 0.75 + 0.25 * min(1.0, len(self._scores) / max(self.cfg['match_frames'], 1))
        return 0.75 + 0.25 * min(1.0, len(self._res.samples) / max(self.cfg['enrol_frames'], 1))

    def _status(self, frame, t_real=None, multi=False, small=False):
        if t_real is not None:
            self._proc.append((time.perf_counter() - t_real) * 1000.0)
            if len(self._proc) > 2000:
                del self._proc[:1000]
        s = Status()
        s.frame = frame
        s.mode = self.mode
        s.stage = self.stage
        s.boxes = self._boxes
        s.box = self._box
        s.multi = multi
        s.prompt = self._prompt(multi, small)
        s.progress = self._progress()
        s.passed = dict(self._res.passed)
        s.motion = self._motion_last
        s.motion_avg = self.motion.average
        s.blinks = self.blink.blinks
        s.head_phase = self.head.phase
        s.head_delta = self.head.delta_pct
        s.head_calibrating = self.head.calibrating
        s.score = self._last_score
        s.samples = len(self._res.samples)
        s.lighting, _ = self._lighting
        s.lighting_ok = s.lighting >= 50
        s.done = self.result is not None
        s.result = self.result
        s.frame_no = self.frame_no
        return s
