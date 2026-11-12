"""The session state machine, driven by simulated people made from drawn faces (no camera)."""
import unittest

import numpy as np
import cv2

from tests import helpers as H
from biolock import sim
from biolock import session as S
from biolock.engine import FaceEngine


class PrepareTests(unittest.TestCase):
    def test_keeps_the_aspect_ratio(self):
        for shape, want in (((480, 640, 3), (480, 640)), ((720, 1280, 3), (360, 640)),
                            ((1080, 1920, 3), (360, 640)), ((240, 320, 3), (480, 640)), ((600, 800, 3), (480, 640))):
            frame, gray, eq = S.prepare(np.zeros(shape, np.uint8))
            self.assertEqual(frame.shape[:2], want, shape)           # v9 squashed everything to 640x360
            self.assertEqual(gray.shape, want)
            self.assertEqual(eq.shape, want)

    def test_mirrors_and_accepts_grey_and_bgra(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :100] = 255
        frame, _g, _e = S.prepare(img)
        self.assertEqual(int(frame[:, :100].mean()), 0)
        self.assertEqual(int(frame[:, -100:].mean()), 255)
        self.assertEqual(S.prepare(np.zeros((480, 640), np.uint8))[0].shape, (480, 640, 3))
        self.assertEqual(S.prepare(np.zeros((480, 640, 4), np.uint8))[0].shape, (480, 640, 3))


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = H.fast_cfg()
        cls.engine = H.enrolled_engine(cls.cfg)

    def test_fixture_the_two_drawn_identities_are_far_apart(self):
        # everything below relies on this: the stranger must score well under the threshold
        other = FaceEngine(self.cfg)
        base = sim.scale_photo(H.face(H.STRANGER), H.SCALE)
        _f, gray, eq = S.prepare(sim.to_camera(sim.render(base)))
        box = other.detect_faces(eq)[0]
        score = self.engine.score(other.extract(gray, eq, box)[0])[0]
        self.assertLess(score, self.cfg['min_conf'] - 10)

    def play(self, next_frame, mode=S.MODE_VERIFY, cfg=None, **kw):
        return H.play(self.engine, cfg or self.cfg, mode, next_frame, **kw)

    def test_enrolment_collects_the_configured_number_of_samples(self):
        e = FaceEngine(self.cfg)
        r = H.play(e, self.cfg, S.MODE_ENROL, H.actor(e, H.OWNER, rng_seed=5).frame)
        self.assertEqual(r.outcome, S.OUT_ENROLLED)
        self.assertEqual(len(r.samples), self.cfg['enrol_frames'])
        self.assertTrue(all(s.shape == (100, 100) and s.dtype == np.uint8 for s in r.samples))
        self.assertTrue(r.liveness_pass)

    def test_enrolled_person_is_accepted_on_the_median_of_all_match_frames(self):
        r = self.play(H.actor(self.engine, H.OWNER, rng_seed=1).frame)
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_ACCEPT, ''))
        self.assertEqual(r.match_frames, self.cfg['match_frames'])
        self.assertGreaterEqual(r.score, self.cfg['min_conf'])
        self.assertGreaterEqual(r.live_score, self.cfg['min_conf'])
        self.assertEqual(r.blinks, self.cfg['blinks_needed'])
        self.assertTrue(r.passed['presence'] and r.passed['blink'] and r.passed['head'])
        self.assertEqual(set(r.components), {'lbph', 'hist', 'template', 'lbph_distance'})
        self.assertGreater(r.latency_s, 1.0)

    def test_another_person_is_rejected_at_the_match(self):
        r = self.play(H.actor(self.engine, H.STRANGER, rng_seed=2).frame)
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertTrue(r.liveness_pass)
        self.assertLess(r.score, self.cfg['min_conf'])

    def test_motionless_photograph_stops_at_presence(self):
        r = self.play(H.actor(self.engine, H.OWNER, kind='static', noise=2.0).frame)
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_TIMEOUT, 'presence'))
        self.assertFalse(r.passed['presence'])
        self.assertLess(r.motion_avg, self.cfg['motion_thresh'])
        self.assertGreaterEqual(r.latency_s, self.cfg['liveness_timeout_s'])

    def test_trembling_photograph_is_not_accepted(self):
        r = self.play(H.actor(self.engine, H.OWNER, kind='shaken', rng_seed=3).frame)
        self.assertNotEqual(r.outcome, S.OUT_ACCEPT)
        self.assertTrue(r.passed['presence'])                         # it does move ...
        self.assertFalse(r.passed['blink'])                           # ... but it never blinks
        self.assertEqual(r.failed_stage, 'blink')

    def test_liveness_by_one_face_and_match_by_another_is_rejected(self):
        def swap():
            a = H.actor(self.engine, H.STRANGER, rng_seed=4)          # does the liveness steps
            b = H.actor(self.engine, H.OWNER, rng_seed=5)             # shown for the match
            return lambda st: (b if st is not None and st.stage == S.ST_FACE else a).frame(st)
        r = self.play(swap())
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertGreaterEqual(r.score, self.cfg['min_conf'])        # the match frames alone would pass
        self.assertLess(r.live_score, self.cfg['min_conf'])
        unbound = self.play(swap(), cfg=H.fast_cfg(bind_liveness_identity=False))
        self.assertEqual(unbound.outcome, S.OUT_ACCEPT)               # which is why the binding exists

    def test_cancel_is_free_before_the_match_and_counts_once_matching_started(self):
        clock = sim.SimClock()
        ses = S.Session(self.engine, self.cfg, S.MODE_VERIFY, clock=clock)
        a = H.actor(self.engine, H.OWNER, rng_seed=6)
        st = None
        for _ in range(5):
            clock.tick()
            st = ses.process(sim.to_camera(a.frame(st)))
        self.assertEqual(ses.cancel().outcome, S.OUT_CANCEL)
        clock = sim.SimClock()
        ses = S.Session(self.engine, self.cfg, S.MODE_VERIFY, clock=clock)
        a = H.actor(self.engine, H.OWNER, rng_seed=6)
        st = None
        for _ in range(600):
            clock.tick()
            st = ses.process(sim.to_camera(a.frame(st)))
            if st.stage == S.ST_FACE:
                break
        self.assertEqual(st.stage, S.ST_FACE)
        r = ses.cancel()
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertIs(ses.cancel(), r)                                 # idempotent

    def test_hiding_the_face_during_the_match_is_a_failed_match(self):
        a = H.actor(self.engine, H.OWNER, rng_seed=7)
        blank = np.full((sim.FRAME_H, sim.FRAME_W), 120, np.uint8)
        r = self.play(lambda st: blank if st is not None and st.stage == S.ST_FACE else a.frame(st))
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertLess(r.match_frames, self.cfg['match_frames'])

    def test_two_faces_block_progress(self):
        base = sim.scale_photo(H.face(H.OWNER), 1.0)
        pair = np.hstack([base, sim.scale_photo(H.face(H.STRANGER), 1.0)])
        frame = sim.render(pair)
        self.assertEqual(len(self.engine.detect_faces(S.prepare(sim.to_camera(frame))[2])), 2)
        clock = sim.SimClock()
        ses = S.Session(self.engine, self.cfg, S.MODE_VERIFY, clock=clock)
        rng = np.random.RandomState(0)
        for i in range(60):
            clock.tick()
            st = ses.process(sim.to_camera(sim.add_noise(sim.render(pair, dx=8 * np.sin(i / 3.0)), 2.0, rng)))
        self.assertTrue(st.multi)
        self.assertIsNone(st.box)
        self.assertEqual(st.stage, S.ST_PRESENCE)
        self.assertIn('ONE', st.prompt)
        self.assertGreater(ses._res.faces[2], 0)

    def test_small_face_is_asked_to_come_closer(self):
        small = sim.scale_photo(H.face(H.OWNER), 0.42)
        clock = sim.SimClock()
        ses = S.Session(self.engine, self.cfg, S.MODE_VERIFY, clock=clock)
        for _ in range(4):
            clock.tick()
            st = ses.process(sim.to_camera(sim.render(small)))
        if st.box is not None:                                         # detected, but under min_face_px
            self.assertLess(st.box[2], self.cfg['min_face_px'])
            self.assertIn('closer', st.prompt)
        else:
            self.assertIn('not detected', st.prompt)

    def test_frozen_camera_gives_no_enrolment_samples(self):
        e = FaceEngine(self.cfg)
        a = H.actor(e, H.OWNER, rng_seed=8)
        still = sim.render(a.base)
        r = H.play(e, H.fast_cfg(enrol_timeout_s=6), S.MODE_ENROL,
                   lambda st: still if st is not None and st.stage == S.ST_FACE else a.frame(st))
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_TIMEOUT, 'capture'))
        self.assertLessEqual(len(r.samples), 1)                        # identical frames are not samples

    def test_status_reports_what_the_ui_needs(self):
        clock = sim.SimClock()
        ses = S.Session(self.engine, self.cfg, S.MODE_VERIFY, clock=clock)
        a = H.actor(self.engine, H.OWNER, rng_seed=9)
        st, stages = None, []
        for _ in range(600):
            clock.tick()
            st = ses.process(sim.to_camera(a.frame(st)))
            if not stages or stages[-1] != st.stage:
                stages.append(st.stage)
            self.assertEqual(st.frame.shape, (480, 640, 3))
            self.assertTrue(0.0 <= st.progress <= 1.0)
            self.assertTrue(st.prompt)
            if st.done:
                break
        self.assertEqual(stages, ['presence', 'blink', 'head', 'face', 'done'])
        self.assertIs(st.result, ses.result)
        again = ses.process(sim.to_camera(a.frame(st)))                # frames after the end change nothing
        self.assertTrue(again.done)
        self.assertEqual(ses.frame_no, st.frame_no)

    def test_modes_are_checked(self):
        with self.assertRaises(ValueError):
            S.Session(FaceEngine(self.cfg), self.cfg, S.MODE_VERIFY)   # nothing enrolled
        with self.assertRaises(ValueError):
            S.Session(self.engine, self.cfg, 'something')


if __name__ == '__main__':
    unittest.main(verbosity=2)
