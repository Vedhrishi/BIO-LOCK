"""Experiment logging and the arithmetic that turns logged trials into error rates.

One row per finished attempt is appended to logs/attempts.csv next to the app. Nothing is
written per camera frame. Logging only observes: no value here feeds back into a decision.

Conventions:
  * trial_label is what the experimenter says the attempt was (genuine, impostor,
    attack_photo, attack_phone, attack_replay; set in the app's Testing mode panel).
  * outcome is what the system decided. Only 'accept' opens the vault; 'reject',
    'timeout' and 'cancelled' all leave it locked.
  * FAR = accepted non-genuine trials / non-genuine trials,
    FRR = genuine trials that were not accepted / genuine trials.
"""

import os
import csv
import math
import time
import datetime
import threading

import numpy as np

try:
    import psutil
    _PS = psutil.Process(os.getpid())
except Exception:                                   # psutil is optional
    psutil = None
    _PS = None

FIELDS = [
    'timestamp', 'app_version', 'mode', 'trial_label', 'outcome', 'failed_stage',
    'score', 'live_score', 'lbph', 'hist', 'template', 'lbph_distance', 'match_frames',
    'presence_pass', 'blink_pass', 'head_pass', 'liveness_pass', 'motion_avg', 'blinks',
    'blink_frames', 'eyes_seen_frames', 'dropouts_moved', 'dropouts_not_local', 'dropouts_length',
    'frames', 'faces_0', 'faces_1', 'faces_multi',
    'latency_ms', 'proc_ms', 'proc_ms_max', 'fps', 'cpu_percent', 'mem_mb',
    'min_conf', 'w_lbph', 'w_hist', 'w_template', 'match_frames_cfg', 'motion_thresh', 'enrol_frames',
    'lighting_condition', 'movement_condition', 'camera', 'counted', 'note',
]

NEGATIVE_LABELS = ('impostor', 'attack_photo', 'attack_phone', 'attack_replay')
TRIAL_LABELS = ('genuine',) + NEGATIVE_LABELS


def resource_usage():
    """(cpu_percent since the previous call, resident memory in MB) of this process; NaN without psutil.
    100 % means one fully used logical CPU."""
    if _PS is None:
        return float('nan'), float('nan')
    try:
        return _PS.cpu_percent(interval=None), _PS.memory_info().rss / 1048576.0
    except Exception:
        return float('nan'), float('nan')


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return '' if math.isnan(v) else round(v, 4)
    return v


class MetricsLogger:

    def __init__(self, attempts_csv):
        self.attempts_csv = attempts_csv
        self._lock = threading.Lock()
        self._ready = False

    def _ensure(self):
        """Create the file with a header. A file written with a different set of columns is kept
        under another name instead of receiving misaligned rows."""
        if self._ready:
            return
        path = self.attempts_csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    header = next(csv.reader(f), [])
            except Exception:
                header = []
            if header != FIELDS:
                root, ext = os.path.splitext(path)
                os.replace(path, f'{root}_legacy_{int(time.time())}{ext}')
        if not os.path.exists(path):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        self._ready = True

    def start_attempt(self):
        """Call when an attempt begins, so that cpu_percent covers just this attempt."""
        resource_usage()

    def log_result(self, result, cfg, trial_label='', lighting='', movement='', camera='', counted=None,
                   note='', version=''):
        """Append one attempt. `result` is a session.Result. Returns the row, or None if it could not be written."""
        cpu, mem = resource_usage()
        comp = result.components or {}
        rej = result.blink_rejects or {}
        lat = result.latency_s
        row = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'app_version': version, 'mode': result.mode,
            'trial_label': trial_label or '', 'outcome': result.outcome, 'failed_stage': result.failed_stage,
            'score': result.score, 'live_score': result.live_score,
            'lbph': comp.get('lbph'), 'hist': comp.get('hist'), 'template': comp.get('template'),
            'lbph_distance': comp.get('lbph_distance'), 'match_frames': result.match_frames,
            'presence_pass': result.passed['presence'], 'blink_pass': result.passed['blink'],
            'head_pass': result.passed['head'], 'liveness_pass': result.liveness_pass,
            'motion_avg': result.motion_avg, 'blinks': result.blinks, 'frames': result.frames,
            'blink_frames': result.blink_frames, 'eyes_seen_frames': result.eyes_seen_frames,
            'dropouts_moved': rej.get('moved', 0), 'dropouts_not_local': rej.get('not_local', 0),
            'dropouts_length': rej.get('length', 0),
            'faces_0': result.faces[0], 'faces_1': result.faces[1], 'faces_multi': result.faces[2],
            'latency_ms': lat * 1000.0, 'proc_ms': result.proc_ms, 'proc_ms_max': result.proc_ms_max,
            'fps': (result.frames / lat) if lat > 0 else None,
            'cpu_percent': cpu, 'mem_mb': mem,
            'min_conf': cfg['min_conf'], 'w_lbph': cfg['w_lbph'], 'w_hist': cfg['w_hist'],
            'w_template': cfg['w_template'], 'match_frames_cfg': cfg['match_frames'],
            'motion_thresh': cfg['motion_thresh'], 'enrol_frames': cfg['enrol_frames'],
            'lighting_condition': lighting or '', 'movement_condition': movement or '', 'camera': camera,
            'counted': counted, 'note': note,
        }
        row = {k: _fmt(row.get(k)) for k in FIELDS}
        try:
            with self._lock:
                self._ensure()
                with open(self.attempts_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        except Exception:                           # a full or read-only drive must not break an unlock
            return None
        return row

    def trial_counts(self):
        """{trial_label: number of logged verification trials}."""
        counts = {}
        for r in load_attempts(self.attempts_csv):
            if r.get('trial_label') in TRIAL_LABELS and r.get('mode') != 'enrol':
                counts[r['trial_label']] = counts.get(r['trial_label'], 0) + 1
        return counts


def load_attempts(path):
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


# ----------------------------------------------------------------------
# Error rates from logged trials
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (as fractions). Treats the trials as independent."""
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def _trial_rows(rows):
    return [r for r in rows if r.get('trial_label') in TRIAL_LABELS and r.get('mode', 'verify') != 'enrol']


def confusion_counts(rows):
    """(tp, fn, fp, tn) with "positive" = the system accepted the attempt."""
    tp = fn = fp = tn = 0
    for r in _trial_rows(rows):
        genuine = r['trial_label'] == 'genuine'
        accepted = r.get('outcome') == 'accept'
        if genuine:
            tp, fn = tp + accepted, fn + (not accepted)
        else:
            fp, tn = fp + accepted, tn + (not accepted)
    return tp, fn, fp, tn


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r.get(key, ''))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out.append(v)
    return out


def compute_metrics(rows):
    """FAR, FRR (with 95 % Wilson intervals), accuracy, precision, recall, F1 and mean cost of the given trials."""
    rows = _trial_rows(rows)
    tp, fn, fp, tn = confusion_counts(rows)
    n_gen, n_neg, total = tp + fn, fp + tn, tp + fn + fp + tn
    nan = float('nan')
    precision = tp / (tp + fp) if (tp + fp) else nan
    recall = tp / n_gen if n_gen else nan
    f1 = 2 * precision * recall / (precision + recall) if (tp + fp) and n_gen and (precision + recall) else nan

    def avg(key):
        xs = _floats(rows, key)
        return sum(xs) / len(xs) if xs else nan
    return {
        'n': total, 'n_genuine': n_gen, 'n_negative': n_neg, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'FAR': fp / n_neg if n_neg else nan, 'FAR_ci': wilson(fp, n_neg),
        'FRR': fn / n_gen if n_gen else nan, 'FRR_ci': wilson(fn, n_gen),
        'accuracy': (tp + tn) / total if total else nan, 'precision': precision, 'recall': recall, 'f1': f1,
        'avg_latency_ms': avg('latency_ms'), 'avg_proc_ms': avg('proc_ms'), 'avg_cpu_percent': avg('cpu_percent'),
        'avg_mem_mb': avg('mem_mb'),
    }


def summarize_by(rows, group_fields):
    """compute_metrics() per distinct value of the given column(s)."""
    groups = {}
    for r in _trial_rows(rows):
        groups.setdefault(tuple(r.get(f, '') for f in group_fields), []).append(r)
    out = []
    for key in sorted(groups):
        entry = dict(zip(group_fields, key))
        entry.update(compute_metrics(groups[key]))
        out.append(entry)
    return out


def liveness_breakdown(rows):
    """Per trial label: how many trials passed all three liveness steps, where the others stopped, and how
    many were finally accepted. This shows what the liveness steps (not the matcher) do to each attack."""
    out = {}
    for r in _trial_rows(rows):
        g = out.setdefault(r['trial_label'], {'trial_label': r['trial_label'], 'n': 0, 'liveness_pass': 0,
                                              'stopped_presence': 0, 'stopped_blink': 0, 'stopped_head': 0,
                                              'rejected_match': 0, 'rejected_key': 0, 'accepted': 0})
        g['n'] += 1
        if str(r.get('liveness_pass')) == '1':
            g['liveness_pass'] += 1
        stage = r.get('failed_stage', '')
        if stage in ('presence', 'blink', 'head'):
            g['stopped_' + stage] += 1
        elif stage in ('match', 'key'):
            g['rejected_' + stage] += 1
        if r.get('outcome') == 'accept':
            g['accepted'] += 1
    for g in out.values():
        g['liveness_pass_rate'] = g['liveness_pass'] / g['n']
        g['accept_rate'] = g['accepted'] / g['n']
        g['accept_ci'] = wilson(g['accepted'], g['n'])
    return [out[k] for k in TRIAL_LABELS if k in out]


def threshold_sweep(rows, thresholds):
    """FAR/FRR that the logged trials would give at other thresholds, recomputed from the logged median
    scores. Trials that never reached the match stage stay rejected at every threshold."""
    rows = _trial_rows(rows)
    out = []
    for t in thresholds:
        fp = fn = n_gen = n_neg = 0
        for r in rows:
            live = str(r.get('liveness_pass')) == '1' and r.get('failed_stage') in ('', 'match')
            scores = _floats([r], 'score')
            bound = _floats([r], 'live_score')
            ok = live and bool(scores) and scores[0] >= t and (not bound or bound[0] >= t)
            if r['trial_label'] == 'genuine':
                n_gen += 1
                fn += not ok
            else:
                n_neg += 1
                fp += ok
        out.append({'threshold': t, 'n_genuine': n_gen, 'n_negative': n_neg,
                    'FAR': fp / n_neg if n_neg else float('nan'), 'FRR': fn / n_gen if n_gen else float('nan')})
    return out


# ----------------------------------------------------------------------
# Score-level statistics (used by the dataset benchmarks)
# ----------------------------------------------------------------------
def roc_summary(scores, labels):
    """AUC, equal-error rate and TAR at 1 % / 5 % FAR for scores where higher means "more genuine"."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-scores, kind='stable')
    s, g = scores[order], y[order]
    P, N = g.sum(), (1 - g).sum()
    if P == 0 or N == 0:
        nan = float('nan')
        return dict(auc=nan, eer=nan, eer_threshold=nan, tar1=nan, tar5=nan, n_gen=int(P), n_imp=int(N))
    # one ROC point per distinct score value, so that ties are not split arbitrarily
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tpr = np.r_[0.0, np.cumsum(g)[last] / P]
    fpr = np.r_[0.0, np.cumsum(1 - g)[last] / N]
    thr = np.r_[np.inf, s[last]]
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    trap = getattr(np, 'trapezoid', None) or np.trapz

    def tar_at(far):
        return float(tpr[max(int(np.searchsorted(fpr, far, side='right')) - 1, 0)])
    return dict(auc=float(trap(tpr, fpr)), eer=float((fpr[i] + 1 - tpr[i]) / 2), eer_threshold=float(thr[i]),
                tar1=tar_at(0.01), tar5=tar_at(0.05), n_gen=int(P), n_imp=int(N))


def far_frr(scores, labels, threshold):
    """(FAR, FRR, false accepts, impostor trials, false rejects, genuine trials) for accept = score >= threshold."""
    scores = np.asarray(scores)
    y = np.asarray(labels).astype(bool)
    acc = scores >= threshold
    fa, ni = int(np.sum(acc & ~y)), int(np.sum(~y))
    fr, ng = int(np.sum(~acc & y)), int(np.sum(y))
    return (fa / ni if ni else float('nan')), (fr / ng if ng else float('nan')), fa, ni, fr, ng
