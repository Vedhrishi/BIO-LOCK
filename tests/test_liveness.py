"""Liveness detectors: presence (motion), blink, head movement."""
import unittest

import numpy as np
import cv2

from tests import helpers as H
from biolock import liveness as L


def scene(seed=0, shape=(240, 320)):
    r = np.random.RandomState(seed)
    return cv2.GaussianBlur(r.randint(0, 255, shape).astype(np.uint8), (0, 0), 4)


BOX = (80, 40, 160, 160)


class MotionMeasureTests(unittest.TestCase):
    def test_sensor_noise_is_not_motion_but_a_shift_is(self):
        base = cv2.equalizeHist(scene(1))
        r = np.random.RandomState(2)
        noisy = [np.clip(base.astype(int) + r.normal(0, 5, base.shape), 0, 255).astype(np.uint8) for _ in range(2)]
        noise = L.box_motion(noisy[0], noisy[1], BOX)
        shifted = np.roll(base, 3, axis=1)
        moved = L.box_motion(base, shifted, BOX)
        self.assertLess(noise, 0.01)
        self.assertGreater(moved, 3 * noise)

    def test_unmeasurable_inputs_give_none(self):
        base = scene(1)
        self.assertIsNone(L.box_motion(base, base, (0, 0, 8, 8)))            # smaller than two blocks
        self.assertIsNone(L.box_motion(base, base, (5000, 5000, 100, 100)))  # outside the frame

    def test_presence_needs_half_a_window_above_the_threshold(self):
        m = L.MotionLive(thresh=0.01, frames=20)
        for _ in range(9):
            self.assertFalse(m.update(0.05))                 # enough motion, too few observations
        self.assertTrue(m.update(0.05))
        m = L.MotionLive(thresh=0.01, frames=20)
        for _ in range(40):
            self.assertFalse(m.update(0.004))                # a still scene never passes
        self.assertFalse(m.update(None))
        self.assertAlmostEqual(m.average, 0.004)
        self.assertLess(m.progress, 0.5)

    def test_average_uses_only_the_last_window(self):
        m = L.MotionLive(thresh=0.01, frames=20)
        for _ in range(20):
            m.update(0.0)
        for _ in range(11):
            m.update(0.02)
        self.assertTrue(m.is_live)                           # 11 of the last 20 at 0.02 -> average 0.011


class LocalChangeTests(unittest.TestCase):
    def test_change_in_the_eye_band_only(self):
        a = cv2.equalizeHist(scene(3))
        b = a.copy()
        x, y, w, h = BOX
        b[y + int(0.36 * h):y + int(0.44 * h), x + int(0.2 * w):x + int(0.45 * w)] = 0     # one "eye" goes dark
        eye, rest = L.local_change(a, b, BOX)
        self.assertGreater(eye, 0.012)
        self.assertLess(rest, 1e-6)

    def test_a_moved_image_changes_both_regions(self):
        a = cv2.equalizeHist(scene(3))
        eye, rest = L.local_change(a, np.roll(a, 4, axis=1), BOX)
        self.assertGreater(eye, 0.012)
        self.assertLess(eye, 2.5 * rest)


class BlinkTests(unittest.TestCase):
    B0 = (100, 100, 200, 200)

    def feed(self, det, pattern, box=None, change=None):
        """pattern: string of 'o' (eyes visible) and '-' (not visible)."""
        counted = 0
        for ch in pattern:
            counted += det.update(ch == 'o', box or self.B0, change)
        return counted

    def test_open_closed_open_counts_once(self):
        d = L.BlinkDetector()
        self.assertEqual(self.feed(d, 'ooooo---ooooo'), 1)
        self.assertEqual(d.blinks, 1)

    def test_two_blinks_finish_and_cooldown_separates_them(self):
        d = L.BlinkDetector(needed=2, cooldown=6)
        self.assertEqual(self.feed(d, 'oooo--ooo--oooo'), 1)         # second one comes too soon
        self.assertFalse(d.is_done)
        self.assertEqual(self.feed(d, 'oooooo--o'), 1)
        self.assertTrue(d.is_done)
        self.assertEqual(self.feed(d, 'ooo---ooo'), 0)               # nothing is counted once done

    def test_too_short_too_long_and_unarmed_runs_are_not_blinks(self):
        d = L.BlinkDetector(min_closed=2, max_closed=24)
        self.assertEqual(self.feed(d, 'ooooo-ooooo'), 0)             # a single-frame drop-out
        self.assertEqual(self.feed(d, 'ooooo' + '-' * 25 + 'ooooo'), 0)
        self.assertEqual(d.rejected['length'], 2)
        self.assertEqual(self.feed(L.BlinkDetector(), '---ooooo'), 0)   # eyes never seen open first
        self.assertEqual(self.feed(L.BlinkDetector(), 'oo---ooo'), 0)   # open for fewer than 3 frames

    def test_moving_face_box_is_not_a_blink(self):
        d = L.BlinkDetector(max_shift=0.08)
        self.feed(d, 'ooooo')
        d.update(False, self.B0)
        d.update(False, (130, 100, 200, 200))                        # 15 % of the face width sideways
        self.assertFalse(d.update(True, (130, 100, 200, 200)))
        self.assertEqual((d.blinks, d.rejected['moved']), (0, 1))
        d = L.BlinkDetector(max_shift=0.08)
        self.feed(d, 'ooooo')
        d.update(False, self.B0)
        d.update(False, (104, 100, 200, 200))                        # 2 %: normal detector jitter
        self.assertTrue(d.update(True, self.B0))

    def test_interrupt_cancels_the_blink_in_progress(self):
        d = L.BlinkDetector()
        self.feed(d, 'ooooo--')
        d.interrupt()                                                # face lost / second face
        self.assertEqual(self.feed(d, 'ooo'), 0)
        self.assertEqual(self.feed(d, 'oo---ooo'), 1)                # and it recovers afterwards

    def test_eye_band_condition(self):
        local, everywhere, noise = (0.05, 0.004), (0.05, 0.04), (0.004, 0.004)
        quiet = (0.002, 0.002)

        def run(onset):
            d = L.BlinkDetector(local_ratio=2.5, min_change=0.01)
            self.feed(d, 'ooooo', change=quiet)
            d.update(False, self.B0, onset)
            d.update(False, self.B0, quiet)
            return d.update(True, self.B0, quiet), d
        counted, d = run(local)
        self.assertTrue(counted)
        counted, d = run(everywhere)                                 # the whole face changed: moved / blurred
        self.assertFalse(counted)
        self.assertEqual(d.rejected['not_local'], 1)
        counted, d = run(noise)                                      # nothing changed: detector flicker
        self.assertFalse(counted)

    def test_eye_band_change_one_frame_before_or_after_the_drop_out_counts(self):
        quiet, local = (0.002, 0.002), (0.05, 0.004)
        d = L.BlinkDetector(local_ratio=2.5, min_change=0.01)
        self.feed(d, 'oooo', change=quiet)
        d.update(True, self.B0, local)                               # eyes start to close, still detected
        d.update(False, self.B0, quiet)
        d.update(False, self.B0, quiet)
        self.assertTrue(d.update(True, self.B0, quiet))
        d = L.BlinkDetector(local_ratio=2.5, min_change=0.01)
        self.feed(d, 'ooooo', change=quiet)
        d.update(False, self.B0, quiet)
        d.update(False, self.B0, local)                              # change arrives on the second closed frame
        self.assertTrue(d.update(True, self.B0, quiet))


class HeadTurnTests(unittest.TestCase):
    def box(self, cx, w=100):
        return (int(cx - w / 2), 50, w, w)

    def test_calibrates_over_ten_observations(self):
        h = L.HeadTurn(pct=10, calib_obs=10)
        for _ in range(9):
            h.update(self.box(300))
            self.assertTrue(h.calibrating)
        h.update(self.box(300))
        self.assertFalse(h.calibrating)
        self.assertAlmostEqual(h.baseline, 300, delta=1)

    def test_needs_left_then_right(self):
        h = L.HeadTurn()
        for _ in range(10):
            h.update(self.box(300))
        self.assertFalse(h.update(self.box(320)))                    # right first does not pass
        self.assertEqual(h.phase, h.L)
        self.assertFalse(h.update(self.box(292)))                    # 8 % left: not enough
        self.assertFalse(h.update(self.box(288)))                    # 12 % left
        self.assertEqual(h.phase, h.R)
        self.assertFalse(h.update(self.box(305)))
        self.assertTrue(h.update(self.box(312)))
        self.assertTrue(h.is_done)

    def test_baseline_is_robust_to_one_outlier_and_ignores_missing_boxes(self):
        h = L.HeadTurn()
        for c in [300] * 9 + [330]:
            h.update(self.box(c))
        self.assertLess(abs(h.baseline - 300), 5)
        self.assertFalse(h.update(None))


if __name__ == '__main__':
    unittest.main(verbosity=2)
