"""Face engine: cascades, alignment, template, matching, and the untrusted template file."""
import io
import os
import json
import shutil
import tempfile
import unittest

import numpy as np
import cv2

from tests import helpers as H
from biolock import engine as EN
from biolock import sim
from biolock import session as S
from biolock.engine import FaceEngine, tan_triggs


def textured(seed, size=100):
    r = np.random.RandomState(seed)
    base = cv2.GaussianBlur(r.randint(0, 255, (size, size)).astype(np.uint8), (0, 0), 6)
    return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)


def noisy(base, rng, s=8):
    return np.clip(base.astype(int) + rng.randint(-s, s + 1, base.shape), 0, 255).astype(np.uint8)


class CascadeTests(unittest.TestCase):
    def test_cascades_load_from_memory(self):
        for name in ('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml'):
            self.assertFalse(EN._cascade(name).empty())

    def test_cascades_load_from_a_non_ascii_path(self):
        # OpenCV's own loader cannot open such paths on Windows (user names with accents, non-Latin scripts)
        tmp = tempfile.mkdtemp(prefix='biolock_café_テスト_')
        old = cv2.data.haarcascades
        try:
            for name in ('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml'):
                shutil.copy(os.path.join(old, name), os.path.join(tmp, name))
            cv2.data.haarcascades = tmp + os.sep
            e = FaceEngine(H.fast_cfg())
            self.assertFalse(e.face_cc.empty())
            self.assertFalse(e.eye_cc.empty())
            found = sim.locate(e, sim.scale_photo(H.face(H.OWNER), H.SCALE))
            self.assertIsNotNone(found)
        finally:
            cv2.data.haarcascades = old
            shutil.rmtree(tmp, ignore_errors=True)

    def test_drawn_face_is_detected_with_both_eyes(self):
        e = FaceEngine(H.fast_cfg())
        for seed in (H.OWNER, H.STRANGER):
            box, eyes, n = sim.locate(e, sim.scale_photo(H.face(seed), H.SCALE))
            self.assertEqual(n, 1)
            self.assertIsNotNone(eyes)
            self.assertGreaterEqual(box[2], 80)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.e = FaceEngine(H.fast_cfg())

    def test_eyes_land_on_fixed_positions(self):
        img = np.full((480, 640), 128, np.uint8)
        left, right = (250.0, 200.0), (330.0, 214.0)                 # tilted, 81 px apart
        cv2.circle(img, (250, 200), 3, 255, -1)
        cv2.circle(img, (330, 214), 3, 0, -1)
        crop = self.e.crop(img, (200, 150, 180, 180), (left, right))
        self.assertEqual(crop.shape, (100, 100))
        _mn, _mx, lo, hi = cv2.minMaxLoc(cv2.GaussianBlur(crop, (0, 0), 1.0))
        want_l, want_r = (50 - 23, 36), (50 + 23, 36)                # EYE_IOD 0.46, EYE_Y 0.36 of 100 px
        self.assertLessEqual(abs(hi[0] - want_l[0]) + abs(hi[1] - want_l[1]), 3)
        self.assertLessEqual(abs(lo[0] - want_r[0]) + abs(lo[1] - want_r[1]), 3)

    def test_box_fallback_uses_the_same_geometry(self):
        img = np.full((480, 640), 128, np.uint8)
        box = (200, 100, 200, 200)
        cv2.circle(img, (int(200 + EN.BOX_EYE_L[0] * 200), int(100 + EN.BOX_EYE_L[1] * 200)), 3, 255, -1)
        crop = self.e.crop(img, box, None)
        _mn, _mx, _lo, hi = cv2.minMaxLoc(cv2.GaussianBlur(crop, (0, 0), 1.0))
        self.assertLessEqual(abs(hi[0] - 27) + abs(hi[1] - 36), 3)

    def test_tan_triggs_removes_a_brightness_change(self):
        base = textured(3)
        darker = (base * 0.45).astype(np.uint8)
        a, b = tan_triggs(base).astype(float), tan_triggs(darker).astype(float)
        self.assertEqual(a.shape, (100, 100))
        self.assertGreater(np.corrcoef(a.ravel(), b.ravel())[0, 1], 0.9)

    def test_eye_check_agrees_with_the_drawing(self):
        base = sim.scale_photo(H.face(H.OWNER), H.SCALE)
        box, _eyes, _n = sim.locate(self.e, base)
        _f, _g, eq = S.prepare(sim.to_camera(sim.render(base)))
        self.assertTrue(self.e.eyes_visible(eq, box))
        closed = sim.close_eyes(self.e, base)
        _f, _g, eq = S.prepare(sim.to_camera(sim.render(closed)))
        self.assertFalse(self.e.eyes_visible(eq, box))


class TemplateTests(H.TempDirCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg = H.fast_cfg()
        self.path = os.path.join(self.tmp, '.biolock', 'face_model.npz')
        self.e = FaceEngine(self.cfg, self.path)
        self.rng = np.random.RandomState(0)
        self.base = textured(1)

    def rois(self, n=40):
        return [tan_triggs(noisy(self.base, self.rng)) for _ in range(n)]

    def test_outlier_pruning_drops_the_planted_bad_samples(self):
        bad = tan_triggs(textured(99))
        kept = self.e.train(self.rois(40) + [bad] * 4, save=False)
        self.assertTrue(38 <= kept <= 40)
        self.assertGreater(self.e.lbph_distance(bad), 20)            # no enrolled sample is the bad crop

    def test_numpy_distance_equals_opencv_predict(self):
        rois = self.rois(12)
        self.e.train(rois, save=False)
        ref = cv2.face.LBPHFaceRecognizer_create(radius=5, neighbors=8, grid_x=6, grid_y=6)
        ref.train(rois, np.zeros(len(rois), np.int32))
        for seed in (5, 6, 7):
            probe = tan_triggs(textured(seed))
            self.assertAlmostEqual(self.e.lbph_distance(probe), ref.predict(probe)[1], delta=2e-3)

    def test_genuine_scores_above_impostor_scores(self):
        self.e.train(self.rois(40), save=False)
        gen = [self.e.score(tan_triggs(noisy(self.base, self.rng)))[0] for _ in range(20)]
        imp = [self.e.score(tan_triggs(textured(s)))[0] for s in range(50, 70)]
        self.assertGreater(min(gen), max(imp))

    def test_components_are_consistent(self):
        rois = self.rois(12)
        self.e.train(rois, save=False)
        c = self.e.components(rois[0])
        self.assertEqual(c, self.e.compare(self.e.features(rois[0])))
        for k in ('lbph', 'hist', 'template'):
            self.assertTrue(0.0 <= c[k] <= 100.0)
        cfg = self.cfg
        want = cfg['w_lbph'] * c['lbph'] + cfg['w_hist'] * c['hist'] + cfg['w_template'] * c['template']
        self.assertAlmostEqual(self.e.fuse(c), want, places=6)
        d_mid = (cfg['lbph_d_lo'] + cfg['lbph_d_hi']) / 2
        self.assertAlmostEqual(float(EN.calibrate(cfg, d_mid, 0, 0)[0]), 50.0, places=6)

    def test_untrained_engine_scores_zero(self):
        self.assertEqual(self.e.score(tan_triggs(self.base))[0], 0.0)
        self.assertEqual(self.e.decide([]), (False, 0.0))

    def test_decision_is_the_median_not_the_best_frame(self):
        t = self.cfg['min_conf']
        self.assertFalse(self.e.decide([100.0] + [0.0] * 14)[0])     # one lucky frame must not unlock
        self.assertFalse(self.e.decide([100.0] * 7 + [t - 1] * 8)[0])
        self.assertTrue(self.e.decide([0.0] * 7 + [t] * 8)[0])

    def test_template_survives_save_and_load(self):
        rois = self.rois(30)
        self.e.train(rois)
        again = FaceEngine(self.cfg, self.path)
        self.assertTrue(again.trained)
        self.assertEqual(again.digest(), self.e.digest())
        self.assertAlmostEqual(again.score(rois[0])[0], self.e.score(rois[0])[0], places=5)
        self.assertEqual(again.meta['samples'], self.e.meta['samples'])

    def test_template_file_holds_features_only_and_no_pickle(self):
        self.e.train(self.rois(30))
        with np.load(self.path, allow_pickle=False) as z:
            self.assertEqual(sorted(z.files), ['hog', 'lbph', 'meta', 'tmpl'])
            self.assertEqual(z['lbph'].shape[1], 6 * 6 * 256)
            self.assertEqual(z['tmpl'].size, 100 * 100)              # one mean face, not the samples
            self.assertEqual(json.loads(str(z['meta']))['version'], EN.MODEL_VERSION)

    def test_digest_changes_with_any_stored_value(self):
        self.e.train(self.rois(30))
        before = self.e.digest()
        g = self.e._G.copy()
        g[0, 0] += 1e-3
        self.assertNotEqual(EN.template_digest(self.e.meta, g, self.e._hog_ref, self.e._tmpl_ref), before)
        self.assertNotEqual(EN.template_digest(dict(self.e.meta, samples=1), self.e._G, self.e._hog_ref,
                                               self.e._tmpl_ref), before)
        self.assertEqual(EN.template_digest(self.e.meta, self.e._G, self.e._hog_ref, self.e._tmpl_ref), before)

    def _write(self, **arrays):
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            f.write(buf.getvalue())

    def _good(self):
        self.e.train(self.rois(30), save=False)
        return dict(meta=np.array(json.dumps(self.e.meta)), lbph=self.e._G, hog=self.e._hog_ref,
                    tmpl=self.e._tmpl_ref)

    def assertRejected(self):
        e = FaceEngine(self.cfg, self.path)
        self.assertFalse(e.trained)
        self.assertIsNotNone(e.load_error)
        self.assertIsNone(e.digest())
        return e

    def test_garbage_file_is_rejected(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            f.write(b'not a template' * 50)
        self.assertRejected()

    def test_pickled_object_array_is_never_loaded(self):
        good = self._good()
        buf = io.BytesIO()
        np.savez(buf, **dict(good, meta=np.array([{'version': 2}], dtype=object)), allow_pickle=True)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            f.write(buf.getvalue())
        self.assertRejected()

    def test_wrong_version_shape_and_values_are_rejected(self):
        good = self._good()
        meta = json.loads(str(good['meta']))
        self._write(**dict(good, meta=np.array(json.dumps(dict(meta, version=1)))))
        self.assertRejected()
        self._write(**dict(good, lbph=good['lbph'][:, :100]))
        self.assertRejected()
        bad = good['lbph'].copy()
        bad[0, 0] = np.nan
        self._write(**dict(good, lbph=bad))
        self.assertRejected()
        self._write(**dict(good, meta=np.array(json.dumps(dict(meta, radius=10 ** 6)))))
        self.assertRejected()
        self._write(**good)                                           # and the untouched file loads
        self.assertTrue(FaceEngine(self.cfg, self.path).trained)

    def test_oversized_file_is_rejected_before_it_is_unpacked(self):
        good = self._good()
        self._write(**dict(good, lbph=np.zeros((4000, 6 * 6 * 256), np.float32)))   # compresses to almost nothing
        e = self.assertRejected()
        self.assertIn('too much', e.load_error)

    def test_rejected_template_leaves_a_usable_engine(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            f.write(b'x')
        e = FaceEngine(self.cfg, self.path)
        self.assertEqual(e.train(self.rois(12), save=False), 12)
        self.assertTrue(e.trained)


class NoPickleAnywhereTests(unittest.TestCase):
    def test_package_never_unpickles(self):
        pkg = os.path.join(H.ROOT, 'biolock')
        for name in os.listdir(pkg):
            if name.endswith('.py'):
                with open(os.path.join(pkg, name), encoding='utf-8') as f:
                    src = f.read()
                self.assertNotIn('import pickle', src, name)
                self.assertNotIn('allow_pickle=True', src, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
