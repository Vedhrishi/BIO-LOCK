"""Camera opening / worker threads with fake cameras, and a smoke test of the window."""
import os
import time
import inspect
import unittest
from unittest import mock

import numpy as np
import cv2

from tests import helpers as H
from biolock import camera as CAM
from biolock import controller as C
from biolock import session as S
from biolock import winutil
from biolock.engine import FaceEngine


class FakeCapture:
    """Stand-in for cv2.VideoCapture. `plan` maps (index, api) to none / black / colour / grey."""
    plan = {}
    opened = []

    def __init__(self, index, api):
        self.kind = FakeCapture.plan.get((index, api), 'none')
        self.released = False
        FakeCapture.opened.append((index, api))

    def isOpened(self):
        return self.kind != 'none'

    def set(self, *_a):
        return True

    def read(self):
        if self.kind == 'black':
            return True, np.zeros((480, 640, 3), np.uint8)
        frame = np.random.RandomState(1).randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if self.kind == 'grey':
            frame[:] = frame[:, :, :1]
        return True, frame

    def release(self):
        self.released = True


@unittest.skipUnless(os.name == 'nt', 'backend order is Windows specific')
class OpenCameraTests(unittest.TestCase):
    DSHOW, MSMF, ANY = cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY

    def open(self, plan, index=0, **kw):
        FakeCapture.plan, FakeCapture.opened = plan, []
        with mock.patch.object(CAM.cv2, 'VideoCapture', FakeCapture), mock.patch.object(CAM.time, 'sleep'):
            return CAM.open_camera(index, **kw)

    def test_first_backend_that_delivers_a_picture_wins(self):
        cap, info = self.open({(0, self.DSHOW): 'colour', (0, self.MSMF): 'colour'})
        self.assertEqual((info.backend, info.width, info.height, info.dark, info.grey),
                         ('DirectShow', 640, 480, False, False))
        self.assertEqual(FakeCapture.opened, [(0, self.DSHOW)])            # nothing else was even tried
        self.assertFalse(cap.released)

    def test_falls_back_when_directshow_only_gives_black_frames(self):
        cap, info = self.open({(0, self.DSHOW): 'black', (0, self.MSMF): 'colour'})
        self.assertEqual((info.backend, info.dark), ('Media Foundation', False))

    def test_falls_back_when_directshow_cannot_open(self):
        cap, info = self.open({(0, self.MSMF): 'colour'})
        self.assertEqual(info.backend, 'Media Foundation')

    def test_covered_lens_is_reported_as_dark_not_as_missing(self):
        cap, info = self.open({(0, self.DSHOW): 'black', (0, self.MSMF): 'black', (0, self.ANY): 'black'})
        self.assertIsNotNone(cap)
        self.assertTrue(info.dark)
        self.assertIn('black', info.describe())

    def test_no_camera(self):
        cap, reason = self.open({})
        self.assertIsNone(cap)
        self.assertIsInstance(reason, str)
        self.assertEqual(len(FakeCapture.opened), 3)

    def test_preferred_backend_is_tried_first(self):
        cap, info = self.open({(0, self.DSHOW): 'colour', (0, self.MSMF): 'colour'}, prefer='Media Foundation')
        self.assertEqual(info.backend, 'Media Foundation')

    def test_infrared_camera_is_labelled(self):
        cap, info = self.open({(1, self.DSHOW): 'grey'}, index=1)
        self.assertTrue(info.grey)
        self.assertIn('infrared', info.describe())

    def test_scan_lists_working_cameras_and_stops_after_two_gaps(self):
        FakeCapture.plan = {(0, self.DSHOW): 'colour', (1, self.DSHOW): 'grey'}
        FakeCapture.opened = []
        with mock.patch.object(CAM.cv2, 'VideoCapture', FakeCapture), mock.patch.object(CAM.time, 'sleep'):
            found = CAM.scan_cameras(max_index=8)
        self.assertEqual([(i.index, i.grey) for i in found], [(0, False), (1, True)])
        self.assertEqual(max(i for i, _api in FakeCapture.opened), 3)      # indices 4..7 were not probed


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = H.fast_cfg(liveness_timeout_s=60, match_timeout_s=30)
        cls.engine = H.enrolled_engine(cls.cfg)

    def session(self, mode=S.MODE_VERIFY):
        return S.Session(self.engine, self.cfg, mode)

    def wait(self, worker, seconds=30.0):
        worker.join(seconds)
        self.assertFalse(worker.is_alive(), 'worker did not finish')

    def test_publishes_new_packets_once_and_stops(self):
        w = CAM.CameraWorker(H.NoiseCap(delay=0.01), self.session(), render=lambda st: st.frame.shape)
        w.start()
        deadline, p = time.time() + 10, None
        while p is None and time.time() < deadline:
            p = w.latest(0)
            time.sleep(0.01)
        self.assertIsNotNone(p)
        seq, status, image = p
        self.assertEqual(image, (480, 640, 3))                             # whatever render returned
        self.assertEqual(status.frame.shape, (480, 640, 3))
        self.assertIsNone(w.latest(w._seq) if w._seq == seq else None)     # nothing new -> None
        self.assertEqual(w.state, 'running')
        w.stop()
        self.wait(w, 5)
        self.assertEqual(w.state, 'done')
        self.assertIsNone(w.result)                                        # stopped, not finalised
        self.assertTrue(w.source.released)

    def test_slow_camera_never_blocks_the_caller(self):
        w = CAM.CameraWorker(H.NoiseCap(delay=0.5), self.session())
        w.start()
        t = time.perf_counter()
        for _ in range(50):
            w.latest(0)
        self.assertLess(time.perf_counter() - t, 0.05)
        w.stop()
        self.wait(w, 5)

    def test_camera_loss_cancels_and_finalises(self):
        done = []
        w = CAM.CameraWorker(H.NoiseCap(fail_after=5), self.session(), finalize=done.append)
        w.start()
        self.wait(w, 20)
        self.assertEqual(w.state, 'done')
        self.assertIn('stopped delivering', w.error)
        self.assertEqual(len(done), 1)
        self.assertEqual(w.result.outcome, S.OUT_CANCEL)

    def test_cancel_request(self):
        done = []
        w = CAM.CameraWorker(H.NoiseCap(delay=0.01), self.session(), finalize=done.append)
        w.start()
        time.sleep(0.3)
        w.cancel()
        self.wait(w, 10)
        self.assertEqual((w.state, w.result.outcome, len(done)), ('done', S.OUT_CANCEL, 1))

    def test_failing_finalize_is_reported_not_swallowed(self):
        def boom(_result):
            raise OSError('drive unplugged')
        w = CAM.CameraWorker(H.NoiseCap(fail_after=5), self.session(), finalize=boom)
        w.start()
        self.wait(w, 20)
        self.assertEqual(w.result.outcome, 'error')
        self.assertIn('drive unplugged', w.result.message)

    def test_camera_that_cannot_be_opened(self):
        with mock.patch.object(CAM, 'open_camera', return_value=(None, 'No camera answered at this position.')):
            w = CAM.CameraWorker(7, self.session())
            w.start()
            self.wait(w, 5)
        self.assertEqual((w.state, w.error), ('failed', 'No camera answered at this position.'))
        self.assertTrue(w.finished)

    def test_whole_verification_through_the_worker(self):
        ses = self.session()
        cap = H.ActorCap(H.actor(self.engine, H.OWNER, rng_seed=31))
        cap.session = ses
        done = []
        w = CAM.CameraWorker(cap, ses, finalize=done.append)
        w.start()
        self.wait(w, 90)
        self.assertEqual(w.state, 'done')
        self.assertEqual((w.result.outcome, len(done)), (S.OUT_ACCEPT, 1))
        seq, status, _img = w.latest(0)
        self.assertTrue(status.done)


def _tk_root():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:                                   # no display
        return None
    return root


class WindowTests(H.TempDirCase, unittest.TestCase):
    """Drives the real window with a fake camera. Dialogs and beeps are patched out."""

    def setUp(self):
        super().setUp()
        from biolock import ui
        self.ui = ui
        self.root = _tk_root()
        if self.root is None:
            self.skipTest('no display available')
        self.shown = []
        box = ui.messagebox
        patches = [mock.patch.object(winutil, 'beep_sequence'),
                   mock.patch.object(box, 'showinfo', side_effect=lambda *a, **k: self.shown.append(('info', a))),
                   mock.patch.object(box, 'showwarning', side_effect=lambda *a, **k: self.shown.append(('warn', a))),
                   mock.patch.object(box, 'showerror', side_effect=lambda *a, **k: self.shown.append(('error', a))),
                   mock.patch.object(box, 'askyesno', return_value=True),
                   mock.patch.object(box, 'askyesnocancel', return_value=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.cfg = H.fast_cfg(liveness_timeout_s=60, match_timeout_s=30, enrol_timeout_s=120)
        self.ctl = C.Controller(os.path.join(self.tmp, 'drive'), self.cfg)
        self.cap = None
        self.app = ui.App(self.root, self.ctl, open_source=self.source, scan=lambda: [])
        self.root.geometry('+-5000+-5000')              # keep it off the visible desktop
        self.root.deiconify()
        self.root.update()

    def tearDown(self):
        self.app.shutdown()
        super().tearDown()

    def source(self):
        self.cap = H.ActorCap(H.actor(self.ctl.engine, self.who, rng_seed=41))
        return self.cap

    def run_attempt(self, who, seconds=120):
        self.who = who
        self.app.primary()
        self.assertIsNotNone(self.app.worker)
        self.cap.session = self.app.worker.session
        deadline = time.time() + seconds
        while self.app.worker is not None and time.time() < deadline:
            self.root.update()
            time.sleep(0.004)
        self.assertIsNone(self.app.worker, 'the attempt did not finish in time')
        return self.app.last_result

    def test_enrol_unlock_lock_through_the_window(self):
        app, ctl = self.app, self.ctl
        self.assertIn('Welcome', app.head.cget('text'))
        self.assertEqual(app.b_main.cget('text'), 'Begin face enrolment')
        r = self.run_attempt(H.OWNER)
        self.assertEqual(r.outcome, S.OUT_ENROLLED, r.message)
        self.assertEqual(ctl.condition(), C.LOCKED)
        self.assertIn('locked', app.head.cget('text'))
        self.assertEqual(self.shown[-1][0], 'info')
        self.assertIsNotNone(app._photo)                                   # frames reached the video label

        r = self.run_attempt(H.STRANGER)
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertEqual((ctl.condition(), ctl.attempts), (C.LOCKED, 1))
        self.assertIn('1 of 5', app.v_attempts.get())
        self.assertEqual(self.shown[-1][0], 'warn')

        r = self.run_attempt(H.OWNER)
        self.assertEqual(r.outcome, S.OUT_ACCEPT, r.message)
        self.assertEqual(ctl.condition(), C.UNLOCKED)
        self.assertIn('open', app.head.cget('text'))
        self.assertEqual(app.b_main.cget('text'), 'Open SecureVault')
        self.assertFalse(winutil.is_hidden(ctl.store.vault_dir))
        self.assertEqual(ctl.attempts, 0)

        app.lock()
        self.assertEqual(ctl.condition(), C.LOCKED)
        self.assertTrue(winutil.is_hidden(ctl.store.vault_dir))
        app.ctl.finalize(_accept())                                        # unlocked again ...
        app.refresh()
        app.quit()                                                         # ... closing locks (dialog answers Yes)
        self.assertFalse(ctl.unlocked)

    def test_cancel_button_and_escape_end_an_attempt(self):
        self.who = H.OWNER
        self.app.primary()
        self.cap.session = self.app.worker.session
        for _ in range(40):
            self.root.update()
            time.sleep(0.01)
        self.app._escape()
        deadline = time.time() + 20
        while self.app.worker is not None and time.time() < deadline:
            self.root.update()
            time.sleep(0.005)
        self.assertIsNone(self.app.worker)
        self.assertEqual(self.app.last_result.outcome, S.OUT_CANCEL)
        self.assertEqual(self.ctl.condition(), C.NEW)

    def test_camera_view_has_exactly_one_scheduler(self):
        # regression: a tick that re-armed itself twice doubled the event load every frame and froze the window
        poll, draw, start = (inspect.getsource(getattr(self.ui.App, n)) for n in ('_poll', '_draw', 'start'))
        self.assertEqual((poll.count("_later('poll'"), poll.count('.after(')), (1, 0))
        self.assertEqual((draw.count('_later('), draw.count('.after(')), (0, 0))
        self.assertEqual(start.count("_later('poll'"), 1)

    def test_sidebar_scrolls_on_a_short_window(self):
        app = self.app
        app.v_testing.set(True)
        app._testing_toggle()
        self.root.geometry('900x430')
        for _ in range(5):
            self.root.update()
        self.assertGreater(app.side.overflow(), 0)                         # content is taller than the window ...
        self.assertTrue(app.side._bar_shown)                               # ... so it can be scrolled
        app.side.canvas.yview_moveto(1.0)
        self.root.update()
        self.assertAlmostEqual(app.side.canvas.yview()[1], 1.0, places=2)  # the last control is reachable

    def test_wording_and_testing_mode_rules(self):
        app, ctl = self.app, self.ctl
        self.assertIn('not a probability', self.ui.App._score_note(self.cfg))
        src = inspect.getsource(self.ui)
        self.assertNotIn('confidence', src.lower())
        self.assertEqual(str(app.b_nolock.cget('state')), 'disabled')      # locked: lock-out cannot be switched off
        app.v_testing.set(True)
        app.v_nolock.set(True)
        app._testing_toggle()
        self.assertFalse(ctl.no_lockout)
        self.assertFalse(app.v_nolock.get())
        app.v_trial.set('attack_photo')
        app.v_light.set('dim')
        app._tags()
        self.assertEqual((ctl.testing, ctl.trial_label, ctl.lighting), (True, 'attack_photo', 'dim'))

    def test_unusual_drive_states_only_offer_a_wipe(self):
        os.makedirs(self.ctl.store.vault_dir)
        with open(os.path.join(self.ctl.store.vault_dir, 'x.txt'), 'w') as f:
            f.write('x')
        self.app.refresh()
        self.assertEqual(self.ctl.condition(), C.ORPHAN)
        self.assertIn('without', self.app.head.cget('text'))
        self.assertFalse(self.app.b_main.winfo_ismapped())
        self.assertTrue(self.app.b_wipe.winfo_manager())
        self.app.wipe_prompt()                                             # both confirmations answer Yes
        self.assertEqual(self.ctl.condition(), C.NEW)
        self.assertFalse(os.path.exists(self.ctl.store.vault_dir))

    def test_painter_scales_the_frame_to_the_video_area(self):
        p = self.ui.Painter()
        st = S.Status()
        st.frame = np.zeros((480, 640, 3), np.uint8)
        st.boxes, st.box = [(100, 100, 200, 200)], (100, 100, 200, 200)
        p.target = (1280, 700)
        self.assertEqual(p(st).shape, (700, 933, 3))                       # aspect ratio kept
        p.target = (320, 400)
        self.assertEqual(p(st).shape, (240, 320, 3))
        self.assertGreater(int(p(st).max()), 0)                            # the face box was drawn


class StartUpTests(H.TempDirCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        root = _tk_root()
        if root is None:
            self.skipTest('no display available')
        root.destroy()

    def test_run_builds_the_app_and_respects_the_drive_rule(self):
        from biolock import ui
        import tkinter as tk
        made = []

        class Quiet:                                    # no splash window during tests
            def __init__(self, root):
                pass

            def say(self, text):
                pass

            def close(self):
                pass

        real_app = ui.App

        def app(root, ctl, **kw):
            made.append(real_app(root, ctl, **kw))
            return made[-1]

        def loop(root):
            made[-1].shutdown()

        with mock.patch.object(ui, 'Splash', Quiet), mock.patch.object(ui, 'App', app), \
                mock.patch.object(tk.Tk, 'deiconify'), mock.patch.object(tk.Tk, 'mainloop', loop), \
                mock.patch.object(winutil, 'beep_sequence'):
            self.assertEqual(ui.run(os.path.join(self.tmp, 'drive')), 0)
            self.assertEqual(len(made), 1)
            self.assertEqual(made[0].ctl.condition(), C.NEW)
            with mock.patch.object(ui, 'enforce_external_drive', return_value=False):
                self.assertEqual(ui.run(os.path.join(self.tmp, 'internal')), 0)
            self.assertEqual(len(made), 1)                                  # refused: nothing was built ...
            self.assertFalse(os.path.exists(os.path.join(self.tmp, 'internal')))   # ... and nothing written

    def test_selftest_passes_and_writes_a_report(self):
        import io
        import json
        from contextlib import redirect_stdout
        from biolock.selftest import run_selftest
        path = os.path.join(self.tmp, 'selftest.json')
        with redirect_stdout(io.StringIO()), mock.patch.object(winutil, 'beep_sequence'):
            code = run_selftest(path)
        with open(path, encoding='utf-8') as f:
            report = json.load(f)
        self.assertEqual(code, 0, report)
        self.assertTrue(report['ok'])
        self.assertEqual(len(report['checks']), 7)
        self.assertFalse(report['frozen'])


def _accept():
    r = S.Result(S.MODE_VERIFY)
    r.outcome, r.score, r.live_score = S.OUT_ACCEPT, 60.0, 60.0
    r.passed = {'presence': True, 'blink': True, 'head': True}
    return r


if __name__ == '__main__':
    unittest.main(verbosity=2)
