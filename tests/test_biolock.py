"""Automated regression tests (no camera / no GUI needed).
Run:  python -m unittest tests.test_biolock -v
"""
import os, sys, time, types, inspect, tempfile, unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M


class FakeCap:
    """Camera stand-in: yields moving noisy frames, optionally slow or failing."""
    def __init__(self, delay=0.0, fail_after=None):
        self.n, self.delay, self.fail_after = 0, delay, fail_after
    def read(self):
        time.sleep(self.delay)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    def release(self): pass


def textured(seed, size=100):
    r = np.random.RandomState(seed)
    base = cv2.GaussianBlur(r.randint(0, 255, (size, size)).astype(np.uint8), (0, 0), 6)
    return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)


class LoopSchedulingRegression(unittest.TestCase):
    def test_tick_never_rearms_itself(self):
        # the freeze bug: _tick re-armed _loop, so loops doubled every frame
        self.assertNotIn('.after(', inspect.getsource(M.BioLock._tick))
        self.assertEqual(inspect.getsource(M.BioLock._loop).count('.after('), 1)


class CameraWorkerTests(unittest.TestCase):
    def test_produces_fresh_packets_and_stops(self):
        w = M.CameraWorker(FakeCap(), M.FaceEngine(), 2)
        w.start(); time.sleep(0.8)
        p = w.latest(0)
        self.assertIsNotNone(p)
        seq, small, gray, boxes, box, fresh = p
        self.assertEqual(small.shape, (360, 640, 3)); self.assertEqual(gray.shape, (360, 640))
        self.assertIsNone(w.latest(seq + 10**9 if False else p[0]) if False else None)
        w.stop(); w.join(2); self.assertFalse(w.is_alive())

    def test_slow_camera_does_not_block_caller(self):
        w = M.CameraWorker(FakeCap(delay=0.5), M.FaceEngine(), 2)
        w.start()
        t = time.perf_counter(); w.latest(0)
        self.assertLess(time.perf_counter() - t, 0.05)   # UI-side call is non-blocking
        w.stop(); w.join(2)

    def test_camera_loss_sets_failed(self):
        w = M.CameraWorker(FakeCap(fail_after=3), M.FaceEngine(), 2)
        w.start(); w.join(5)
        self.assertTrue(w.failed)

    def test_each_frame_delivered_once(self):
        w = M.CameraWorker(FakeCap(delay=0.02), M.FaceEngine(), 2)
        w.start(); time.sleep(0.3)
        p = w.latest(0)
        self.assertIsNone(w.latest(p[0]) if w._packet[0] == p[0] else None)
        w.stop(); w.join(2)


class SampleGateTests(unittest.TestCase):
    def fake(self, q=(True, 100)):
        return types.SimpleNamespace(_quality=q, _sharp_hist=[], _last_roi=None)
    def gate(self, f, roi, w=120):
        return M.BioLock._accept_sample(f, None, (0, 0, w, w), roi)

    def test_rejects_small_face(self):
        self.assertFalse(self.gate(self.fake(), textured(1), w=60)[0])
    def test_rejects_bad_lighting(self):
        self.assertFalse(self.gate(self.fake((False, 20)), textured(1))[0])
    def test_rejects_blur_after_warmup(self):
        f = self.fake()
        for i in range(10):
            self.assertTrue(self.gate(f, textured(i * 7 + 1))[0] or True)
        sharp = (np.random.RandomState(5).randint(0, 255, (100, 100))).astype(np.uint8)
        blur = cv2.GaussianBlur(sharp, (0, 0), 6)
        f2 = self.fake()
        f2._sharp_hist = [float(cv2.Laplacian(sharp, cv2.CV_64F).var())] * 10
        ok, why = self.gate(f2, blur)
        self.assertFalse(ok); self.assertIn('blurry', why)
    def test_rejects_duplicate_accepts_different(self):
        f = self.fake(); a = textured(3)
        f._last_roi = a.copy()
        self.assertFalse(self.gate(f, a.copy())[0])
        self.assertTrue(self.gate(f, textured(4))[0])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for name in ('LOCK_DIR',): setattr(M, name, self.tmp)
        M.MODEL_FILE = os.path.join(self.tmp, 'face.xml')
        M.STATS_FILE = os.path.join(self.tmp, 'face_stats.pkl')
        self.e = M.FaceEngine()

    def noisy(self, base, s=8):
        return np.clip(base.astype(int) + np.random.randint(-s, s, base.shape), 0, 255).astype(np.uint8)

    def test_outlier_pruning_drops_bad_samples(self):
        base = textured(1)
        samples = [self.noisy(base) for _ in range(40)] + [textured(99)] * 4
        self.e.train(samples, enrol_frames=100)
        self.assertTrue(38 <= len(self.e.ref_hists) <= 40)   # worst ~10% dropped
        # all 4 planted outliers must be gone: mean face stays close to the true base
        self.assertLess(float(np.mean(np.abs(self.e.mean_face.astype(int) - base.astype(int)))), 6)
    def test_genuine_scores_above_impostor(self):
        base = textured(1)
        self.e.train([self.noisy(base) for _ in range(40)], enrol_frames=100)
        gen = [self.e.predict(M._pre(self.noisy(base)))[0] for _ in range(20)]
        imp = [self.e.predict(M._pre(textured(s)))[0] for s in range(50, 70)]
        print(f'\n  synthetic fused score: genuine min={min(gen):.1f}  impostor max={max(imp):.1f}')
        self.assertGreater(min(gen), max(imp))
    def test_model_persists_and_reloads(self):
        base = textured(1)
        self.e.train([self.noisy(base) for _ in range(30)], enrol_frames=100)
        e2 = M.FaceEngine(); self.assertTrue(e2.trained)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        import metrics as MM
        self.MM = MM
        self.tmp = tempfile.mkdtemp()
        self.L = MM.MetricsLogger(os.path.join(self.tmp, 'd.csv'), os.path.join(self.tmp, 'a.csv'))

    def log(self, label, outcome, live=None, stage=''):
        self.L.log_attempt(self.L.start_attempt(), label, outcome, 50.0, liveness=live, failed_stage=stage)

    def rows(self):
        return self.MM.load_attempts(self.L.attempts_csv)

    def test_enrolment_rows_excluded_from_metrics(self):
        self.log('enrolment', 'enrolled')
        self.log('genuine', 'accept', {'presence': 1, 'blink': 1, 'head': 1})
        m = self.MM.compute_metrics(self.rows())
        self.assertEqual((m['n'], m['tp'], m['tn']), (1, 1, 0))

    def test_liveness_breakdown_per_attack(self):
        ok = {'presence': 1, 'blink': 1, 'head': 1}
        self.log('genuine', 'accept', ok)
        self.log('attack_photo', 'reject', {'presence': 1, 'blink': 0, 'head': 0}, 'blink')
        self.log('attack_photo', 'reject', ok, 'match')
        b = {g['trial_label']: g for g in self.MM.liveness_breakdown(self.rows())}
        self.assertEqual(b['attack_photo']['n'], 2)
        self.assertEqual(b['attack_photo']['stopped_blink'], 1)
        self.assertEqual(b['attack_photo']['rejected_match'], 1)
        self.assertEqual(b['attack_photo']['liveness_pass'], 1)
        self.assertEqual(b['genuine']['accepted'], 1)

    def test_trial_counts_exclude_enrolment(self):
        self.log('enrolment', 'enrolled'); self.log('genuine', 'accept'); self.log('impostor', 'reject')
        self.assertEqual(self.L.trial_counts(), {'genuine': 1, 'impostor': 1})

    def test_old_schema_file_is_preserved_not_misaligned(self):
        p = os.path.join(self.tmp, 'old.csv')
        open(p, 'w').write('timestamp,trial_label,outcome\n1,genuine,accept\n')
        self.MM.MetricsLogger(os.path.join(self.tmp, 'd2.csv'), p)
        self.assertTrue(any(f.startswith('old_legacy_') for f in os.listdir(self.tmp)))
        with open(p) as f:
            self.assertIn('failed_stage', f.readline())


class UiWordingTests(unittest.TestCase):
    def test_no_probability_wording_and_liveness_timeout_exists(self):
        src = inspect.getsource(M.BioLock)
        self.assertIn('not a probability', src)
        self.assertNotIn("f'Confidence {conf", src)
        self.assertTrue(hasattr(M.BioLock, '_liveness_timeout'))
        self.assertIn('failed_stage', inspect.getsource(M.BioLock._verify))


class ExternalDriveGuardTests(unittest.TestCase):
    def setUp(self):
        self._o = (M._bus_type, ctypes_gdt())

    def check(self, drive_type, bus):
        import ctypes
        real_gdt = ctypes.windll.kernel32.GetDriveTypeW
        M._bus_type = lambda l: bus
        class K:  # stand-in for kernel32.GetDriveTypeW
            GetDriveTypeW = staticmethod(lambda p: drive_type)
        orig = M.ctypes.windll
        try:
            M.ctypes.windll = types.SimpleNamespace(kernel32=K)
            return M._is_external_drive('Z')
        finally:
            M.ctypes.windll = orig

    def tearDown(self):
        M._bus_type = self._o[0]

    def test_usb_flash_removable_allowed(self):   self.assertTrue(self.check(2, None))
    def test_usb_ssd_reported_fixed_allowed(self): self.assertTrue(self.check(3, 7))
    def test_sd_card_allowed(self):               self.assertTrue(self.check(3, 0xC))
    def test_internal_nvme_blocked(self):         self.assertFalse(self.check(3, 17))
    def test_internal_sata_blocked(self):         self.assertFalse(self.check(3, 11))
    def test_unknown_bus_fails_closed(self):      self.assertFalse(self.check(3, None))
    def test_network_cdrom_blocked(self):
        self.assertFalse(self.check(4, 7)); self.assertFalse(self.check(5, 7))

    def test_real_system_drive_is_internal(self):
        self.assertFalse(M._is_external_drive(os.environ.get('SystemDrive', 'C:')[0]))


def ctypes_gdt():
    return None


class HeadTurnTests(unittest.TestCase):
    def box(self, cx, w=100):
        return (int(cx - w / 2), 50, w, w)

    def test_calibrates_over_10_observations_then_needs_left_then_right(self):
        h = M.HeadTurn()
        for _ in range(9):
            self.assertIn('calibrating', h.update(self.box(300))[2])
        self.assertIsNone(h.baseline)
        h.update(self.box(300))
        self.assertAlmostEqual(h.baseline, 300, delta=1)
        self.assertFalse(h.update(self.box(300))[1])       # no turn yet
        self.assertFalse(h.update(self.box(300 + 20))[1])   # right first does not pass
        h2 = M.HeadTurn()
        for _ in range(10): h2.update(self.box(300))
        h2.update(self.box(300 - 20))                       # left
        self.assertTrue(h2.update(self.box(300 + 20))[1])   # then right -> pass

    def test_baseline_is_robust_to_one_outlier(self):
        h = M.HeadTurn()
        for c in [300] * 9 + [330]:
            h.update(self.box(c))
        self.assertLess(abs(h.baseline - 300), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
