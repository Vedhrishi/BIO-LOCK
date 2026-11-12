"""Shared fixtures for the tests: drawn faces, a fast configuration, simulated sessions, fake cameras.

Nothing here needs a camera, a real photograph or a network connection."""
import os
import sys
import time
import shutil
import tempfile
import functools

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from biolock import sim                                    # noqa: E402
from biolock import session as S                           # noqa: E402
from biolock.config import load_config                     # noqa: E402
from biolock.engine import FaceEngine                      # noqa: E402
from tests.synthface import make_face                      # noqa: E402

SCALE = 1.6             # drawn faces are 250 px images, like the LFW photographs
OWNER, STRANGER = 6, 17  # seeds of the two drawn identities used throughout (see test_fixture_*)


def fast_cfg(**overrides):
    """The shipped configuration with shorter time-outs and a smaller enrolment, so tests stay quick."""
    base = dict(enrol_frames=24, liveness_timeout_s=8, match_timeout_s=3, enrol_timeout_s=40)
    base.update(overrides)
    return load_config(path='', use_env=False, overrides=base)


@functools.lru_cache(maxsize=None)
def face(seed):
    return make_face(seed)


def actor(engine, seed, kind='live', rng_seed=0, **kw):
    kw.setdefault('sway', 12.0)
    a = sim.Actor(engine, face(seed), SCALE, np.random.RandomState(1000 + rng_seed), kind=kind, **kw)
    assert a.usable, f'drawn face {seed} is not usable for a {kind} actor'
    return a


def play(engine, cfg, mode, next_frame, max_frames=4000):
    """Run one simulated session; returns its Result."""
    clock = sim.SimClock()
    return sim.run(S.Session(engine, cfg, mode, clock=clock), clock, next_frame, max_frames=max_frames)


def enrolled_engine(cfg, model_path=None, seed=OWNER):
    """An engine enrolled through a simulated enrolment session of the drawn face `seed`."""
    engine = FaceEngine(cfg, model_path)
    result = play(engine, cfg, S.MODE_ENROL, actor(engine, seed, rng_seed=77).frame)
    assert result.outcome == S.OUT_ENROLLED, (result.outcome, result.failed_stage, result.message)
    engine.train(result.samples, save=bool(model_path))
    return engine


class TempDirCase:
    """Mixin: self.tmp is a fresh folder that is removed afterwards."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='biolock_test_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class NoiseCap:
    """Camera stand-in that yields noise frames, optionally slowly or failing after a while."""

    def __init__(self, delay=0.0, fail_after=None, shape=(480, 640, 3)):
        self.n, self.delay, self.fail_after, self.shape = 0, delay, fail_after, shape
        self.released = False
        self._rng = np.random.RandomState(0)

    def read(self):
        time.sleep(self.delay)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, self._rng.randint(0, 255, self.shape, dtype=np.uint8)

    def release(self):
        self.released = True


class ActorCap:
    """Camera stand-in that shows a simulated person who follows the session's prompts.

    A new frame is only produced once the session has processed the previous one, so the simulated
    blink lasts the same number of processed frames on a slow machine as on a fast one."""

    class _View:
        def __init__(self, session):
            self.stage = session.stage
            self.head_calibrating = session.head.calibrating
            self.head_phase = session.head.phase

    def __init__(self, actor_, fps=60.0):
        self.actor = actor_
        self.session = None             # set once the worker exists
        self.period = 1.0 / fps
        self.released = False
        self._handed = 0

    def read(self):
        ses = self.session
        if ses is None:
            time.sleep(self.period)
            return True, sim.to_camera(self.actor.frame(None))
        deadline = time.time() + 1.0
        while ses.frame_no < self._handed and ses.result is None and time.time() < deadline:
            time.sleep(0.002)
        self._handed = ses.frame_no + 1
        return True, sim.to_camera(self.actor.frame(self._View(ses)))

    def release(self):
        self.released = True
