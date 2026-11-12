"""Controller: what the outcome of an attempt does to the drive (no window, no camera)."""
import os
import csv
import shutil
import tempfile
import unittest

import numpy as np

from tests import helpers as H
from biolock import controller as C
from biolock import session as S
from biolock import winutil
from biolock.engine import FaceEngine


def verify_result(outcome, stage='', score=50.0, live=50.0):
    r = S.Result(S.MODE_VERIFY)
    r.outcome, r.failed_stage, r.score, r.live_score = outcome, stage, score, live
    r.match_frames = 15 if stage in ('', 'match') else 0
    r.passed = {k: stage in ('', 'match') for k in ('presence', 'blink', 'head')}
    r.frames, r.latency_s = 100, 3.3
    return r


class ControllerTests(unittest.TestCase):
    """One simulated enrolment is made once; every test works on a copy of the resulting drive folder."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = H.fast_cfg()
        cls.master = tempfile.mkdtemp(prefix='biolock_master_')
        ctl = C.Controller(cls.master, cls.cfg)
        assert ctl.condition() == C.NEW
        ses = ctl.begin(S.MODE_ENROL)
        result = H.play(ctl.engine, cls.cfg, S.MODE_ENROL, H.actor(ctl.engine, H.OWNER, rng_seed=21).frame)
        del ses
        ctl.finalize(result)
        assert result.outcome == S.OUT_ENROLLED, result.message
        cls.enrol_result = result

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.master, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='biolock_test_')
        self.base = os.path.join(self.tmp, 'drive')
        shutil.copytree(self.master, self.base)
        self.now = [1000.0]
        self.ctl = C.Controller(self.base, self.cfg, clock=lambda: self.now[0])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self):
        with open(os.path.join(self.base, 'logs', 'attempts.csv'), newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    # ---- enrolment -----------------------------------------------------------------
    def test_enrolment_leaves_a_locked_drive_and_keeps_no_images(self):
        ctl = self.ctl
        self.assertEqual(ctl.condition(), C.LOCKED)
        self.assertEqual(self.enrol_result.samples, [])                   # crops are dropped after training
        names = sorted(os.listdir(ctl.store.lock_dir))
        self.assertEqual(names, ['audit.log', 'drive.id', 'face_model.npz', 'state.json', 'vault.key'])
        self.assertTrue(winutil.is_hidden(ctl.store.lock_dir))
        # checked on the folder the enrolment was written to: a copy only keeps file attributes on Python 3.12+
        self.assertTrue(winutil.is_hidden(os.path.join(self.master, '.biolock', 'face_model.npz')))
        self.assertIn('ENROLLED', ctl.store.read_log())
        self.assertEqual(self.rows()[0]['trial_label'], 'enrolment')
        with self.assertRaises(C.Refused):
            ctl.begin(S.MODE_ENROL)                                       # one enrolment per drive

    def test_simulated_owner_unlocks_and_stranger_does_not(self):
        ctl = self.ctl
        r = H.play(ctl.engine, self.cfg, S.MODE_VERIFY, H.actor(ctl.engine, H.OWNER, rng_seed=22).frame)
        ctl.finalize(r)
        self.assertEqual(r.outcome, S.OUT_ACCEPT)
        self.assertEqual(ctl.condition(), C.UNLOCKED)
        ctl.lock()
        r = H.play(ctl.engine, self.cfg, S.MODE_VERIFY, H.actor(ctl.engine, H.STRANGER, rng_seed=23).frame)
        ctl.finalize(r)
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'match'))
        self.assertEqual((ctl.condition(), ctl.attempts), (C.LOCKED, 1))

    # ---- unlock / lock ---------------------------------------------------------------
    def test_accept_opens_the_vault_and_lock_hides_it(self):
        ctl = self.ctl
        with self.assertRaises(C.Refused):
            ctl.open_vault()
        r = ctl.finalize(verify_result(S.OUT_ACCEPT))
        self.assertEqual(r.outcome, S.OUT_ACCEPT)
        self.assertTrue(ctl.unlocked)
        vault = ctl.open_vault()
        self.assertTrue(os.path.isdir(vault))
        self.assertFalse(winutil.is_hidden(vault))
        with open(os.path.join(vault, 'note.txt'), 'w') as f:
            f.write('secret')
        ctl.lock()
        self.assertEqual(ctl.condition(), C.LOCKED)
        self.assertTrue(winutil.is_hidden(vault))
        self.assertTrue(os.path.exists(os.path.join(vault, 'note.txt')))  # hidden, not deleted
        self.assertEqual(ctl.state['total_unlocks'], 1)
        log = ctl.store.read_log()
        self.assertIn('UNLOCKED', log)
        self.assertIn('LOCKED', log)

    def test_vault_left_open_is_hidden_again_at_the_next_start(self):
        self.ctl.finalize(verify_result(S.OUT_ACCEPT))
        vault = self.ctl.open_vault()
        self.assertFalse(winutil.is_hidden(vault))
        again = C.Controller(self.base, self.cfg)                         # e.g. the drive was pulled out
        self.assertEqual(again.condition(), C.LOCKED)
        self.assertTrue(winutil.is_hidden(vault))

    def test_auto_lock_countdown(self):
        ctl = self.ctl
        self.assertIsNone(ctl.auto_lock_in())
        ctl.finalize(verify_result(S.OUT_ACCEPT))
        self.assertAlmostEqual(ctl.auto_lock_in(), self.cfg['auto_lock_mins'] * 60, delta=1)
        self.now[0] += 200
        self.assertAlmostEqual(ctl.auto_lock_in(), self.cfg['auto_lock_mins'] * 60 - 200, delta=1)
        ctl.touch()
        self.assertAlmostEqual(ctl.auto_lock_in(), self.cfg['auto_lock_mins'] * 60, delta=1)

    # ---- failed matches ----------------------------------------------------------------
    def test_failed_matches_count_and_the_fifth_wipes(self):
        ctl = self.ctl
        ctl.finalize(verify_result(S.OUT_ACCEPT))
        with open(os.path.join(ctl.open_vault(), 'note.txt'), 'w') as f:
            f.write('secret')
        ctl.lock()
        for n in range(1, self.cfg['max_attempts']):
            r = ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
            self.assertEqual((ctl.attempts, r.extra['attempts_left']), (n, self.cfg['max_attempts'] - n))
            self.assertTrue(r.extra['counted'])
            self.assertNotIn('wiped', r.extra)
        r = ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
        self.assertTrue(r.extra['wiped'])
        self.assertEqual(ctl.condition(), C.NEW)
        self.assertFalse(os.path.exists(ctl.store.vault_dir))
        self.assertFalse(os.path.exists(ctl.store.model_file))
        self.assertFalse(os.path.exists(ctl.store.vault_key))
        self.assertFalse(ctl.engine.trained)
        self.assertIn('WIPE', ctl.store.read_log())
        ctl.begin(S.MODE_ENROL)                                           # and a new enrolment may start

    def test_a_successful_unlock_resets_the_counter(self):
        ctl = self.ctl
        for _ in range(3):
            ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
        self.assertEqual(ctl.attempts, 3)
        ctl.finalize(verify_result(S.OUT_ACCEPT))
        self.assertEqual(ctl.attempts, 0)
        self.assertEqual(C.Controller(self.base, self.cfg).attempts, 0)   # persisted

    def test_counter_survives_a_restart(self):
        self.ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
        self.assertEqual(C.Controller(self.base, self.cfg).attempts, 1)

    def test_testing_mode_can_switch_the_lock_out_off(self):
        ctl = self.ctl
        ctl.testing, ctl.no_lockout, ctl.trial_label, ctl.lighting = True, True, 'impostor', 'dim'
        for _ in range(self.cfg['max_attempts'] + 2):
            r = ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
            self.assertFalse(r.extra['counted'])
        self.assertEqual((ctl.attempts, ctl.condition()), (0, C.LOCKED))
        last = self.rows()[-1]
        self.assertEqual((last['trial_label'], last['lighting_condition'], last['counted']), ('impostor', 'dim', '0'))
        ctl.no_lockout = False                                            # tags without testing mode are ignored
        ctl.testing = False
        ctl.finalize(verify_result(S.OUT_REJECT, 'match', score=5.0))
        self.assertEqual(self.rows()[-1]['trial_label'], '')
        self.assertEqual(ctl.attempts, 1)

    # ---- liveness time-outs ----------------------------------------------------------------
    def test_liveness_timeouts_are_not_failed_matches_but_lead_to_a_cooldown(self):
        ctl, c = self.ctl, self.cfg
        for n in range(1, c['liveness_fail_limit']):
            r = ctl.finalize(verify_result(S.OUT_TIMEOUT, 'blink'))
            self.assertFalse(r.extra['counted'])
            self.assertEqual(ctl.cooldown_left(), 0)
        ctl.finalize(verify_result(S.OUT_TIMEOUT, 'presence'))
        self.assertEqual(ctl.attempts, 0)
        self.assertAlmostEqual(ctl.cooldown_left(), c['liveness_cooldown_s'], delta=1)
        with self.assertRaises(C.Refused):
            ctl.begin(S.MODE_VERIFY)
        self.now[0] += c['liveness_cooldown_s'] + 1
        ctl.begin(S.MODE_VERIFY)
        ctl.finalize(verify_result(S.OUT_TIMEOUT, 'head'))                # the next one waits twice as long
        self.assertAlmostEqual(ctl.cooldown_left(), 2 * c['liveness_cooldown_s'], delta=1)
        self.now[0] += 1000
        ctl.finalize(verify_result(S.OUT_ACCEPT))                         # an unlock clears it
        self.assertEqual((ctl.state['live_fails'], ctl.cooldown_left()), (0, 0))

    def test_cancel_changes_nothing(self):
        r = self.ctl.finalize(verify_result(S.OUT_CANCEL, 'blink'))
        self.assertFalse(r.extra['counted'])
        self.assertEqual((self.ctl.attempts, self.ctl.cooldown_left()), (0, 0))

    # ---- the key ties template, drive id and volume together -------------------------------------
    def test_swapped_template_matches_but_does_not_unlock(self):
        ctl = self.ctl
        other = os.path.join(self.tmp, 'other.npz')
        e = H.enrolled_engine(self.cfg, other, seed=H.STRANGER)           # an attacker's own enrolment
        self.assertTrue(e.trained)
        with open(other, 'rb') as f:
            winutil.atomic_write(ctl.store.model_file, f.read())
        attacked = C.Controller(self.base, self.cfg)
        self.assertEqual(attacked.condition(), C.LOCKED)
        r = attacked.finalize(verify_result(S.OUT_ACCEPT))                # their face now "matches" ...
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'key'))   # ... but the key does not
        self.assertFalse(attacked.unlocked)
        self.assertFalse(r.extra['counted'])
        self.assertIn('KEY_MISMATCH', attacked.store.read_log())

    def test_enrolment_copied_to_another_drive_does_not_unlock(self):
        real_serial = self.ctl.store.volume_serial
        self.ctl.store.volume_serial = lambda: 123456                      # same files on another volume
        r = self.ctl.finalize(verify_result(S.OUT_ACCEPT))
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'key'))
        self.ctl.store.volume_serial = real_serial
        self.assertEqual(self.ctl.finalize(verify_result(S.OUT_ACCEPT)).outcome, S.OUT_ACCEPT)
        self.ctl.lock()
        winutil.atomic_write(self.ctl.store.drive_id_file, b'{"id": "0123456789abcdef0123456789abcdef"}')
        moved = C.Controller(self.base, self.cfg)                          # another drive identifier
        r = moved.finalize(verify_result(S.OUT_ACCEPT))
        self.assertEqual((r.outcome, r.failed_stage), (S.OUT_REJECT, 'key'))

    # ---- drives in an unusual state ------------------------------------------------------------------
    def test_legacy_enrolment_is_recognised_and_never_loaded(self):
        base = os.path.join(self.tmp, 'old')
        os.makedirs(os.path.join(base, '.biolock'))
        for name in ('face.xml', 'face_stats.pkl'):
            with open(os.path.join(base, '.biolock', name), 'wb') as f:
                f.write(b'\x80\x04cos\nsystem\n')                         # would be dangerous to unpickle
        ctl = C.Controller(base, self.cfg)
        self.assertEqual(ctl.condition(), C.LEGACY)
        self.assertFalse(ctl.engine.trained)
        for mode in (S.MODE_ENROL, S.MODE_VERIFY):
            with self.assertRaises(C.Refused):
                ctl.begin(mode)
        ctl.wipe()
        self.assertEqual(ctl.condition(), C.NEW)

    def test_damaged_and_orphaned_drives_only_offer_a_wipe(self):
        with open(self.ctl.store.model_file, 'r+b') as f:                  # corrupt the template in place
            f.seek(40)
            f.write(b'\x00' * 400)
        ctl = C.Controller(self.base, self.cfg)
        self.assertEqual(ctl.condition(), C.DAMAGED)
        self.assertIsNotNone(ctl.engine.load_error)
        with self.assertRaises(C.Refused):
            ctl.begin(S.MODE_VERIFY)
        with self.assertRaises(C.Refused):
            ctl.begin(S.MODE_ENROL)

        base = os.path.join(self.tmp, 'orphan')
        os.makedirs(os.path.join(base, 'SecureVault'))
        with open(os.path.join(base, 'SecureVault', 'someone_elses.txt'), 'w') as f:
            f.write('x')
        orphan = C.Controller(base, self.cfg)
        self.assertEqual(orphan.condition(), C.ORPHAN)
        with self.assertRaises(C.Refused):
            orphan.begin(S.MODE_ENROL)                                    # no enrolling over somebody's vault
        orphan.wipe()
        self.assertEqual(orphan.condition(), C.NEW)

    def test_deleting_the_state_file_does_not_unenrol_the_drive(self):
        os.remove(self.ctl.store.state_file)
        again = C.Controller(self.base, self.cfg)
        self.assertEqual(again.condition(), C.LOCKED)                     # v9 would have offered a new enrolment

    def test_manual_wipe(self):
        self.ctl.finalize(verify_result(S.OUT_ACCEPT))
        self.ctl.open_vault()
        self.ctl.wipe('test')
        self.assertEqual(self.ctl.condition(), C.NEW)
        self.assertFalse(self.ctl.unlocked)
        self.assertFalse(os.path.exists(self.ctl.store.vault_dir))

    def test_one_log_row_per_attempt_with_scores(self):
        self.ctl.finalize(verify_result(S.OUT_ACCEPT, score=61.5, live=58.25))
        self.ctl.finalize(verify_result(S.OUT_TIMEOUT, 'blink', score=None, live=None))
        rows = self.rows()
        self.assertEqual(len(rows), 3)                                    # enrolment + 2
        self.assertEqual((rows[1]['outcome'], rows[1]['score'], rows[1]['live_score']), ('accept', '61.5', '58.25'))
        self.assertEqual((rows[2]['outcome'], rows[2]['failed_stage'], rows[2]['liveness_pass']), ('timeout', 'blink', '0'))
        self.assertEqual(rows[1]['min_conf'], str(float(self.cfg['min_conf'])))

    def test_camera_choice_is_remembered_per_drive(self):
        self.ctl.set_camera(1, 'Media Foundation', 'Camera 1')
        again = C.Controller(self.base, self.cfg)
        self.assertEqual((again.local['cam_index'], again.local['cam_backend']), (1, 'Media Foundation'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
