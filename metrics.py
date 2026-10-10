"""
Logging + metric computation for BioLock experiments.

Two kinds of log this produces:

1. detections.csv  — one row per processed camera frame:
       timestamp, mode, stage, n_faces, lighting_condition, movement_condition
   n_faces is 0 / 1 / 2+ so you can report how often multiple faces or no
   face were detected under each test condition.

2. attempts.csv     — one row per completed authentication attempt:
       timestamp, trial_label, outcome, confidence, lbph_conf, hist_conf,
       template_conf, latency_ms, cpu_percent, mem_mb, min_conf,
       motion_thresh, enrol_frames, w_lbph, w_hist, w_template,
       lighting_condition, movement_condition

   trial_label is what YOU tell BioLock this attempt actually is
   ('genuine', 'impostor', 'attack_photo', 'attack_phone', 'attack_replay')
   — set it via the Testing Mode panel in the app, or pass it into the
   offline experiment runner. outcome is what the system decided
   ('accept' / 'reject'). Comparing the two is what gives you FAR/FRR.

Nothing here changes BioLock's authentication logic — this only
observes and records it.
"""

import os
import sys
import csv
import time
import threading

try:
    import psutil
    _PS = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PS = None


def _root_dir():
    # frozen exe: __file__ points into a temp extraction dir that is deleted on exit,
    # so logs must live next to the executable to survive between runs
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


DETECTIONS_CSV = os.path.join(_root_dir(), 'logs', 'detections.csv')
ATTEMPTS_CSV = os.path.join(_root_dir(), 'logs', 'attempts.csv')

_DET_FIELDS = ['timestamp', 'mode', 'stage', 'n_faces', 'lighting_condition', 'movement_condition']
_ATT_FIELDS = ['timestamp', 'trial_label', 'outcome', 'confidence', 'lbph_conf', 'hist_conf',
               'template_conf', 'latency_ms', 'cpu_percent', 'mem_mb', 'min_conf', 'motion_thresh',
               'enrol_frames', 'w_lbph', 'w_hist', 'w_template', 'lighting_condition', 'movement_condition',
               'failed_stage', 'presence_pass', 'blink_pass', 'head_pass', 'liveness_pass']


def _ensure_csv(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        # schema changed (new columns)? keep the old file instead of writing misaligned rows
        try:
            with open(path, 'r', newline='') as f:
                header = next(csv.reader(f), [])
            if header != list(fields):
                os.replace(path, path.replace('.csv', f'_legacy_{int(time.time())}.csv'))
        except Exception:
            pass
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def resource_usage():
    """Returns (cpu_percent, mem_mb) for the current process. cpu_percent
    needs a prior call to 'prime' the internal psutil counter — see
    MetricsLogger.start_attempt()."""
    if _PS is None:
        return (float('nan'), float('nan'))
    try:
        cpu = _PS.cpu_percent(interval=None)
        mem = _PS.memory_info().rss / (1024 * 1024)
        return (cpu, mem)
    except Exception:
        return (float('nan'), float('nan'))


class MetricsLogger:

    def __init__(self, detections_csv=DETECTIONS_CSV, attempts_csv=ATTEMPTS_CSV):
        self.detections_csv = detections_csv
        self.attempts_csv = attempts_csv
        _ensure_csv(self.detections_csv, _DET_FIELDS)
        _ensure_csv(self.attempts_csv, _ATT_FIELDS)
        self._lock = threading.Lock()
        self._t0 = None
        if _PS is not None:
            _PS.cpu_percent(interval=None)  # prime it; first real reading comes next call

    # ---- per-frame face-count logging ---------------------------------
    def log_detection(self, mode, stage, n_faces, lighting_condition=None, movement_condition=None):
        row = {
            'timestamp': time.time(),
            'mode': mode,
            'stage': stage,
            'n_faces': n_faces,
            'lighting_condition': lighting_condition or '',
            'movement_condition': movement_condition or '',
        }
        with self._lock:
            with open(self.detections_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=_DET_FIELDS).writerow(row)

    # ---- per-attempt timing -------------------------------------------
    def start_attempt(self):
        """Call once when a verification/enrolment attempt begins (camera
        opens / first frame of the 4-step pipeline). Returns a token to
        pass to log_attempt()."""
        if _PS is not None:
            _PS.cpu_percent(interval=None)  # reset the averaging window
        return time.perf_counter()

    def log_attempt(self, start_token, trial_label, outcome, confidence,
                     lbph_conf=None, hist_conf=None, template_conf=None,
                     cfg=None, lighting_condition=None, movement_condition=None,
                     liveness=None, failed_stage=''):
        """Call once when an attempt finishes (accept or reject decided).
        liveness: dict with optional bool keys presence/blink/head (each liveness
        step logged on its own); failed_stage: '' | presence | blink | head | match."""
        lv = liveness or {}
        _b = lambda k: '' if k not in lv else int(bool(lv[k]))
        lv_all = '' if not lv else int(all(bool(lv.get(k)) for k in ('presence', 'blink', 'head')))
        latency_ms = (time.perf_counter() - start_token) * 1000.0
        cpu, mem = resource_usage()
        cfg = cfg or {}
        row = {
            'timestamp': time.time(),
            'trial_label': trial_label or '',
            'outcome': outcome,
            'confidence': confidence,
            'lbph_conf': lbph_conf,
            'hist_conf': hist_conf,
            'template_conf': template_conf,
            'latency_ms': round(latency_ms, 2),
            'cpu_percent': cpu,
            'mem_mb': round(mem, 2) if mem == mem else mem,  # NaN-safe round
            'min_conf': cfg.get('min_conf'),
            'motion_thresh': cfg.get('motion_thresh'),
            'enrol_frames': cfg.get('enrol_frames'),
            'w_lbph': cfg.get('w_lbph'),
            'w_hist': cfg.get('w_hist'),
            'w_template': cfg.get('w_template'),
            'lighting_condition': lighting_condition or cfg.get('lighting_condition') or '',
            'movement_condition': movement_condition or cfg.get('movement_condition') or '',
            'failed_stage': failed_stage,
            'presence_pass': _b('presence'), 'blink_pass': _b('blink'), 'head_pass': _b('head'),
            'liveness_pass': lv_all,
        }
        with self._lock:
            with open(self.attempts_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=_ATT_FIELDS).writerow(row)
        return row

    def trial_counts(self):
        """{trial_label: count} of logged verification trials (enrolment excluded)."""
        counts = {}
        try:
            with self._lock, open(self.attempts_csv, 'r', newline='') as f:
                for r in csv.DictReader(f):
                    lab = r.get('trial_label', '')
                    if lab and lab != 'enrolment':
                        counts[lab] = counts.get(lab, 0) + 1
        except Exception:
            pass
        return counts


# ----------------------------------------------------------------------
# Aggregation: turn logged attempts into FAR / FRR / accuracy / etc.
# ----------------------------------------------------------------------
# Convention used throughout this project for the confusion matrix:
#   "positive" = the system ACCEPTED the attempt
#   "should accept" = trial_label == 'genuine'
#   anything else ('impostor', 'attack_photo', 'attack_phone', 'attack_replay')
#     should be REJECTED
#
#   TP = genuine, accepted          FN = genuine, rejected   (-> FRR)
#   FP = non-genuine, accepted (-> FAR)   TN = non-genuine, rejected

NEGATIVE_LABELS = {'impostor', 'attack_photo', 'attack_phone', 'attack_replay'}


def _trial_rows(rows):
    # enrolment (and any untagged) rows are not authentication trials
    return [r for r in rows if r.get('trial_label') == 'genuine' or r.get('trial_label') in NEGATIVE_LABELS]


def confusion_counts(rows):
    tp = fn = fp = tn = 0
    for r in _trial_rows(rows):
        is_genuine = r['trial_label'] == 'genuine'
        accepted = r['outcome'] == 'accept'
        if is_genuine and accepted:
            tp += 1
        elif is_genuine and not accepted:
            fn += 1
        elif not is_genuine and accepted:
            fp += 1
        elif not is_genuine and not accepted:
            tn += 1
    return tp, fn, fp, tn


def compute_metrics(rows):
    """rows: list of dicts with at least 'trial_label' and 'outcome'.
    Returns a dict with FAR, FRR, accuracy, precision, recall, f1, and
    the raw counts, computed only from the rows given — so call this
    per parameter-combination / per condition subset."""
    rows = _trial_rows(rows)
    tp, fn, fp, tn = confusion_counts(rows)
    n_genuine = tp + fn
    n_negative = fp + tn
    total = tp + fn + fp + tn

    far = fp / n_negative if n_negative else float('nan')
    frr = fn / n_genuine if n_genuine else float('nan')
    accuracy = (tp + tn) / total if total else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) else float('nan'))

    lat = [float(r['latency_ms']) for r in rows if r.get('latency_ms') not in (None, '')]
    cpu = [float(r['cpu_percent']) for r in rows if r.get('cpu_percent') not in (None, '', 'nan')]
    mem = [float(r['mem_mb']) for r in rows if r.get('mem_mb') not in (None, '', 'nan')]

    def _avg(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    return {
        'n': total, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'FAR': far, 'FRR': frr, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1,
        'avg_latency_ms': _avg(lat), 'avg_cpu_percent': _avg(cpu), 'avg_mem_mb': _avg(mem),
    }


def load_attempts(path=ATTEMPTS_CSV):
    if not os.path.exists(path):
        return []
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def summarize_by(path=ATTEMPTS_CSV, group_fields=('min_conf',)):
    """Groups logged attempts by the given field(s) (e.g. ('min_conf',) or
    ('enrol_frames', 'w_lbph')) and computes compute_metrics() for each
    group. Returns a list of {**group_values, **metrics} dicts, handy to
    dump straight into a pandas DataFrame or write out as a results table."""
    rows = load_attempts(path)
    groups = {}
    for r in rows:
        key = tuple(r.get(f, '') for f in group_fields)
        groups.setdefault(key, []).append(r)

    out = []
    for key, grp_rows in groups.items():
        m = compute_metrics(grp_rows)
        entry = dict(zip(group_fields, key))
        entry.update(m)
        out.append(entry)
    return out


def liveness_breakdown(rows):
    """Per trial_label: how many trials, how many passed all three liveness
    steps, where the rest were stopped, and the final accept count. This is
    what shows whether liveness (not just the face matcher) stops each attack."""
    out = {}
    for r in _trial_rows(rows):
        g = out.setdefault(r['trial_label'], {'trial_label': r['trial_label'], 'n': 0,
                                              'liveness_pass': 0, 'stopped_presence': 0,
                                              'stopped_blink': 0, 'stopped_head': 0,
                                              'rejected_match': 0, 'accepted': 0})
        g['n'] += 1
        fs = r.get('failed_stage', '')
        if r.get('liveness_pass') == '1':
            g['liveness_pass'] += 1
        if fs in ('presence', 'blink', 'head'):
            g['stopped_' + fs] += 1
        elif fs == 'match':
            g['rejected_match'] += 1
        if r.get('outcome') == 'accept':
            g['accepted'] += 1
    for g in out.values():
        g['liveness_pass_rate'] = round(g['liveness_pass'] / g['n'], 3) if g['n'] else float('nan')
        g['accept_rate'] = round(g['accepted'] / g['n'], 3) if g['n'] else float('nan')
    return list(out.values())
