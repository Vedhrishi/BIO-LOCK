"""Configuration rules and the experiment log / error-rate arithmetic."""
import os
import csv
import json
import math
import unittest
from unittest import mock

import numpy as np

from tests import helpers as H
from biolock import config as CFG
from biolock import metrics as M
from biolock import session as S


class ConfigTests(H.TempDirCase, unittest.TestCase):
    def load(self, **kw):
        kw.setdefault('path', '')
        kw.setdefault('use_env', False)
        return CFG.load_config(**kw)

    def test_defaults_are_valid_and_are_the_documented_matcher(self):
        c = self.load()
        self.assertEqual((c['lbph_radius'], c['lbph_neighbors'], c['lbph_grid'], c['roi_size']), (5, 8, 6, 100))
        self.assertEqual((c['w_lbph'], c['w_hist'], c['w_template']), (1.0, 0.0, 0.0))
        self.assertEqual((c['match_frames'], c['enrol_frames'], c['max_attempts']), (15, 100, 5))
        self.assertTrue(c['bind_liveness_identity'])

    def test_invalid_values_are_refused(self):
        for bad in (dict(w_lbph=0.5), dict(w_lbph=1.2, w_hist=-0.2), dict(min_conf=120), dict(motion_thresh=0),
                    dict(match_frames=0), dict(match_frames=7.5), dict(blink_min_cl=5, blink_max_cl=3),
                    dict(lbph_d_lo=80.0), dict(min_conf='high'), dict(bind_liveness_identity=1),
                    dict(liveness_timeout_s=0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.load(overrides=bad)
        with self.assertRaises(ValueError):
            self.load(overrides={'no_such_key': 1})

    def test_file_then_environment_then_overrides(self):
        path = os.path.join(self.tmp, 'config.json')
        with open(path, 'w') as f:
            json.dump({'min_conf': 30, 'auto_lock_mins': 9, 'unknown': 1}, f)
        self.assertEqual(self.load(path=path)['min_conf'], 30)
        with mock.patch.dict(os.environ, {'BIOLOCK_MIN_CONF': '41', 'BIOLOCK_TRIAL_LABEL': 'impostor'}):
            c = self.load(path=path, use_env=True)
            self.assertEqual((c['min_conf'], c['auto_lock_mins'], c['trial_label']), (41, 9, 'impostor'))
            self.assertEqual(self.load(path=path, use_env=True, overrides={'min_conf': 50})['min_conf'], 50)
        self.assertNotIn('unknown', self.load(path=path))

    def test_packaged_build_cannot_be_weakened_from_outside(self):
        path = os.path.join(self.tmp, 'config.json')
        with open(path, 'w') as f:
            json.dump({'min_conf': 0, 'max_attempts': 9999, 'blinks_needed': 1, 'bind_liveness_identity': False,
                       'w_lbph': 0.0, 'w_hist': 1.0, 'auto_lock_mins': 9}, f)
        env = {'BIOLOCK_MIN_CONF': '0', 'BIOLOCK_MOTION_THRESH': '0.0001', 'BIOLOCK_MATCH_FRAMES': '1'}
        with mock.patch.dict(os.environ, env):
            c = self.load(path=path, use_env=True, frozen=True)
        d = CFG.DEFAULTS
        for k in ('min_conf', 'max_attempts', 'blinks_needed', 'bind_liveness_identity', 'w_lbph', 'w_hist',
                  'motion_thresh', 'match_frames'):
            self.assertEqual(c[k], d[k], k)
        self.assertEqual(c['auto_lock_mins'], 9)                           # a user preference is still honoured

    def test_broken_file_is_loud_from_source_and_ignored_when_packaged(self):
        path = os.path.join(self.tmp, 'config.json')
        with open(path, 'w') as f:
            f.write('{ nope')
        with self.assertRaises(ValueError):
            self.load(path=path)
        self.assertEqual(self.load(path=path, frozen=True)['min_conf'], CFG.DEFAULTS['min_conf'])

    def test_save_round_trip(self):
        path = os.path.join(self.tmp, 'c.json')
        c = self.load(overrides={'min_conf': 44})
        CFG.save_config(c, path)
        self.assertEqual(self.load(path=path)['min_conf'], 44)


def result(outcome='accept', stage='', score=50.0, live=52.0, passed=(True, True, True)):
    r = S.Result(S.MODE_VERIFY)
    r.outcome, r.failed_stage, r.score, r.live_score = outcome, stage, score, live
    r.components = {'lbph': score, 'hist': 40.0, 'template': 30.0, 'lbph_distance': 55.5}
    r.passed = dict(zip(('presence', 'blink', 'head'), passed))
    r.match_frames, r.frames, r.latency_s, r.proc_ms, r.blinks = 15, 120, 4.0, 12.5, 2
    r.faces = {0: 3, 1: 110, 2: 1}
    return r


class LoggerTests(H.TempDirCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp, 'logs', 'attempts.csv')
        self.log = M.MetricsLogger(self.path)
        self.cfg = H.fast_cfg()

    def rows(self):
        return M.load_attempts(self.path)

    def test_one_row_per_attempt_and_nothing_until_then(self):
        self.assertFalse(os.path.exists(self.path))
        row = self.log.log_result(result(), self.cfg, trial_label='genuine', lighting='dim', camera='Camera 0')
        self.assertEqual(len(self.rows()), 1)
        r = self.rows()[0]
        self.assertEqual(list(r.keys()), M.FIELDS)
        self.assertEqual((r['trial_label'], r['outcome'], r['score'], r['live_score']), ('genuine', 'accept', '50.0', '52.0'))
        self.assertEqual((r['presence_pass'], r['blink_pass'], r['head_pass'], r['liveness_pass']), ('1', '1', '1', '1'))
        self.assertEqual((r['faces_0'], r['faces_1'], r['faces_multi']), ('3', '110', '1'))
        self.assertEqual((r['latency_ms'], r['fps'], r['lighting_condition']), ('4000.0', '30.0', 'dim'))
        self.assertEqual(float(r['min_conf']), self.cfg['min_conf'])
        self.assertEqual(row['outcome'], 'accept')

    def test_missing_values_are_empty_not_text(self):
        self.log.log_result(result('timeout', 'blink', score=None, live=None, passed=(True, False, False)), self.cfg)
        r = self.rows()[0]
        self.assertEqual((r['score'], r['live_score'], r['liveness_pass'], r['blink_pass']), ('', '', '0', '0'))

    def test_file_with_other_columns_is_kept_aside(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'w', newline='') as f:
            f.write('timestamp,trial_label,outcome\n1,genuine,accept\n')
        self.log.log_result(result(), self.cfg, trial_label='genuine')
        names = os.listdir(os.path.dirname(self.path))
        self.assertTrue(any(n.startswith('attempts_legacy_') for n in names), names)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(list(self.rows()[0].keys()), M.FIELDS)

    def test_unwritable_log_does_not_raise(self):
        with mock.patch('builtins.open', side_effect=PermissionError):
            self.assertIsNone(M.MetricsLogger(os.path.join(self.tmp, 'x', 'a.csv')).log_result(result(), self.cfg))

    def test_trial_counts_ignore_unlabelled_and_enrolment_rows(self):
        self.log.log_result(result(), self.cfg, trial_label='genuine')
        self.log.log_result(result('reject', 'match', 10.0), self.cfg, trial_label='impostor')
        self.log.log_result(result(), self.cfg, trial_label='')
        enrol = result('enrolled')
        enrol.mode = S.MODE_ENROL
        self.log.log_result(enrol, self.cfg, trial_label='enrolment')
        self.assertEqual(self.log.trial_counts(), {'genuine': 1, 'impostor': 1})


def trial(label, outcome, stage='', live=1, score=''):
    return {'trial_label': label, 'outcome': outcome, 'failed_stage': stage, 'liveness_pass': str(live),
            'mode': 'verify', 'score': str(score), 'live_score': str(score), 'latency_ms': '4000', 'proc_ms': '10'}


class ErrorRateTests(unittest.TestCase):
    def test_far_frr_from_logged_trials(self):
        rows = ([trial('genuine', 'accept')] * 8 + [trial('genuine', 'reject', 'match')] * 1
                + [trial('genuine', 'timeout', 'blink', live=0)] * 1
                + [trial('impostor', 'reject', 'match')] * 18 + [trial('impostor', 'accept')] * 2
                + [trial('attack_photo', 'timeout', 'presence', live=0)] * 10
                + [trial('enrolment', 'enrolled')] + [trial('', 'accept')])
        m = M.compute_metrics(rows)
        self.assertEqual((m['n'], m['tp'], m['fn'], m['fp'], m['tn']), (40, 8, 2, 2, 28))
        self.assertAlmostEqual(m['FRR'], 0.2)                              # a time-out of a genuine user is a false reject
        self.assertAlmostEqual(m['FAR'], 2 / 30)
        self.assertAlmostEqual(m['accuracy'], 36 / 40)
        self.assertTrue(m['FAR_ci'][0] < m['FAR'] < m['FAR_ci'][1])
        by = {r['trial_label']: r for r in M.summarize_by(rows, ('trial_label',))}
        self.assertAlmostEqual(by['impostor']['FAR'], 0.1)
        self.assertEqual(by['attack_photo']['FAR'], 0.0)
        self.assertTrue(math.isnan(by['attack_photo']['FRR']))

    def test_wilson_interval(self):
        lo, hi = M.wilson(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1135, places=3)                       # 0 of 30 still allows about 11 %
        lo, hi = M.wilson(15, 30)
        self.assertAlmostEqual((lo + hi) / 2, 0.5, places=6)
        self.assertTrue(all(math.isnan(v) for v in M.wilson(0, 0)))

    def test_liveness_breakdown_per_attack(self):
        rows = [trial('genuine', 'accept'), trial('attack_photo', 'timeout', 'blink', live=0),
                trial('attack_photo', 'timeout', 'presence', live=0), trial('attack_photo', 'reject', 'match'),
                trial('attack_replay', 'accept'), trial('impostor', 'reject', 'key')]
        b = {g['trial_label']: g for g in M.liveness_breakdown(rows)}
        p = b['attack_photo']
        self.assertEqual((p['n'], p['stopped_presence'], p['stopped_blink'], p['rejected_match'], p['accepted']),
                         (3, 1, 1, 1, 0))
        self.assertAlmostEqual(p['liveness_pass_rate'], 1 / 3)
        self.assertEqual(b['attack_replay']['accept_rate'], 1.0)
        self.assertEqual(b['impostor']['rejected_key'], 1)
        self.assertEqual(list(b), ['genuine', 'impostor', 'attack_photo', 'attack_replay'])

    def test_threshold_sweep_from_logged_scores(self):
        rows = [trial('genuine', 'accept', score=60), trial('genuine', 'reject', 'match', score=30),
                trial('impostor', 'reject', 'match', score=20), trial('impostor', 'reject', 'match', score=33),
                trial('attack_photo', 'timeout', 'blink', live=0)]
        sweep = {r['threshold']: r for r in M.threshold_sweep(rows, [25, 35, 70])}
        self.assertEqual((sweep[35]['FAR'], sweep[35]['FRR']), (0.0, 0.5))
        self.assertEqual((sweep[25]['FAR'], sweep[25]['FRR']), (1 / 3, 0.0))
        self.assertEqual((sweep[70]['FAR'], sweep[70]['FRR']), (0.0, 1.0))  # the photo never reached the match


class SummariserTests(H.TempDirCase, unittest.TestCase):
    def run_script(self, path, *extra):
        import io
        import sys
        from contextlib import redirect_stdout
        sys.path.insert(0, os.path.join(H.ROOT, 'experiments'))
        import summarize_live_logs as SL
        buf = io.StringIO()
        with mock.patch.object(sys, 'argv', ['summarize_live_logs.py', '--csv', path, *extra]), redirect_stdout(buf):
            code = SL.main()
        return code, buf.getvalue()

    def test_summary_of_a_small_log(self):
        path = os.path.join(self.tmp, 'logs', 'attempts.csv')
        log, cfg = M.MetricsLogger(path), H.fast_cfg()
        for _ in range(4):
            log.log_result(result(), cfg, trial_label='genuine', lighting='normal')
        log.log_result(result('reject', 'match', 20.0, 22.0), cfg, trial_label='genuine', lighting='dim')
        for _ in range(5):
            log.log_result(result('reject', 'match', 12.0, 11.0), cfg, trial_label='impostor', lighting='normal')
        log.log_result(result('timeout', 'blink', None, None, (True, False, False)), cfg, trial_label='attack_photo')
        log.log_result(result(), cfg, trial_label='')                      # an ordinary unlock, not a trial
        out_json = os.path.join(self.tmp, 'summary.json')
        code, text = self.run_script(path, '--json', out_json)
        self.assertEqual(code, 0)
        self.assertIn('11 labelled trials', text)
        self.assertIn('attack_photo', text)
        self.assertIn('20.0%', text)                                       # FRR: 1 of 5 genuine
        with open(out_json) as f:
            summary = json.load(f)
        self.assertAlmostEqual(summary['overall']['FRR'], 0.2)
        self.assertEqual(summary['overall']['FAR'], 0.0)
        sweep = {s['threshold']: s for s in summary['threshold_sweep']}
        self.assertEqual(sweep[20]['FRR'], 0.0)                            # at 20 the rejected genuine would pass

    def test_empty_log(self):
        code, text = self.run_script(os.path.join(self.tmp, 'missing.csv'))
        self.assertEqual(code, 1)
        self.assertIn('No labelled trials', text)


class RocTests(unittest.TestCase):
    def test_perfect_and_useless_scores(self):
        y = np.r_[np.ones(50), np.zeros(200)]
        perfect = M.roc_summary(np.r_[np.full(50, 80.0), np.full(200, 20.0)], y)
        self.assertEqual((perfect['eer'], perfect['auc'], perfect['tar1']), (0.0, 1.0, 1.0))
        rng = np.random.RandomState(0)
        useless = M.roc_summary(rng.rand(250), y)
        self.assertTrue(0.4 < useless['auc'] < 0.6)
        self.assertTrue(0.4 < useless['eer'] < 0.6)
        self.assertEqual((useless['n_gen'], useless['n_imp']), (50, 200))

    def test_known_overlap(self):
        s = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], float)
        r = M.roc_summary(s, y)
        self.assertAlmostEqual(r['auc'], 23 / 24)                          # one inverted pair out of 24
        self.assertAlmostEqual(r['eer'], (1 / 6 + 1 / 4) / 2, places=6)
        far, frr, fa, ni, fr, ng = M.far_frr(s, y, 6)
        self.assertEqual((fa, ni, fr, ng), (1, 6, 1, 4))
        self.assertAlmostEqual(far, 1 / 6)
        self.assertAlmostEqual(frr, 0.25)

    def test_ties_are_not_split(self):
        r = M.roc_summary(np.zeros(20), np.r_[np.ones(10), np.zeros(10)])
        self.assertAlmostEqual(r['auc'], 0.5)
        self.assertAlmostEqual(r['eer'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
