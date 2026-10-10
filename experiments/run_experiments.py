#!/usr/bin/env python3
"""
Offline experiment runner for BioLock.

This does NOT drive the live Tkinter app or a real webcam — it works on
a folder of images/video frames YOU supply (see dataset layout below),
re-running the face-match + liveness-relevant scoring under every
parameter combination requested, and writing one CSV row of metrics
per combination. That CSV is what you paste into your paper's tables.

----------------------------------------------------------------------
WHY THIS NEEDS YOUR OWN DATA
----------------------------------------------------------------------
FAR/FRR/accuracy/etc. are properties of real face images and real
presentation-attack samples (a printed photo, a phone screen, a replay
video) measured against a real enrolled face — there is no way to
compute genuine numbers without that data. This script does the sweep
and the arithmetic; you supply the captures. See "Capturing the
dataset" below for exactly what to record.

----------------------------------------------------------------------
Dataset layout expected under --dataset DIR
----------------------------------------------------------------------
DIR/
  enrol/                     10-100+ frames of the real subject's face,
                              used to train the model for each
                              enrol_frames value in the sweep.
                              (.jpg/.png, already cropped-ish is fine —
                              a face will be detected in each)

  genuine/<lighting>/<movement>/*.jpg
                              More frames of the SAME real subject,
                              NOT used for training, organised by the
                              lighting/movement condition they were
                              shot under, e.g.:
                              genuine/bright/still/*.jpg
                              genuine/dim/head_turn/*.jpg

  impostor/*.jpg              Frames of OTHER people's faces (false-accept
                              test against a real, non-attack face).

  attack_photo/*.jpg          Camera pointed at a PRINTED PHOTO of the
                              enrolled subject.
  attack_phone/*.jpg          Camera pointed at the enrolled subject's
                              face shown on a PHONE SCREEN.
  attack_replay/*.jpg         Frames extracted from a RECORDED VIDEO of
                              the enrolled subject, played back at the
                              camera. (Use ffmpeg to pull frames from
                              your .mp4: see README in this folder.)

Every image must contain exactly one face (this script uses the same
Haar cascade as the app to find and crop it — images where 0 or 2+
faces are detected are counted and reported, matching the "0/1/
multiple-face detection" logging requirement, and excluded from the
match-score calculations).

----------------------------------------------------------------------
Capturing the dataset (what to actually go and record)
----------------------------------------------------------------------
- enrol/            : 100 frames of yourself, good lighting, neutral
                       expression, slight natural movement — same as a
                       normal enrolment session.
- genuine/bright/*   : ~30 frames, strong front lighting
- genuine/normal/*   : ~30 frames, normal room lighting
- genuine/dim/*      : ~30 frames, low light
- genuine/backlit/*  : ~30 frames, light source behind you
  (repeat each lighting folder for movement conditions you care about:
   still / head_turn / talking, if you want that cross-tab too)
- impostor/          : ~30 frames each of 3-5 other people
- attack_photo/      : print a clear photo of the enrolled face, ~30
                       frames of the camera pointed at the print
- attack_phone/      : display a photo of the enrolled face on a phone,
                       ~30 frames of the camera pointed at the screen
- attack_replay/     : record a 10-20s video of the enrolled subject,
                       play it back at the camera, extract frames:
                         ffmpeg -i replay.mp4 -vf fps=5 attack_replay/frame_%03d.jpg

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------
    pip install -r ../requirements.txt
    python run_experiments.py --dataset /path/to/dataset --out results

Produces, under --out:
    results/min_conf_sweep.csv
    results/motion_thresh_sweep.csv       (see note on motion below)
    results/enrol_frames_sweep.csv
    results/weight_combo_sweep.csv
    results/detection_counts.csv          (0/1/multi-face tally per folder)
    results/condition_breakdown.csv       (per lighting/movement folder)
    results/attack_breakdown.csv          (per attack type)
    results/all_runs_raw.csv              (every individual image score, for
                                            your own further analysis/plots)

A NOTE ON motion_thresh: motion liveness needs a *sequence* of frames
from the same moment (to diff against each other), not independent
photos. If your attack_replay/attack_phone captures are saved as a
sequence of consecutively-numbered frames (frame_001.jpg, frame_002.jpg,
...) this script will diff consecutive frames within each folder to
evaluate the motion threshold sweep; genuine/impostor/attack_photo
folders (independent stills) are skipped for that specific sweep and
noted as such in motion_thresh_sweep.csv.
"""

import os
import sys
import csv
import glob
import time
import argparse
import itertools

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULTS
import metrics as metrics_mod

try:
    import psutil
    _PS = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PS = None


# --------------------------------------------------------------------------
# Face finding / preprocessing — mirrors main.py's FaceEngine exactly, but
# standalone so this script has no Tkinter/camera dependency.
# --------------------------------------------------------------------------
def _find_cc(name):
    cands = [cv2.data.haarcascades + name]
    d = os.path.dirname(cv2.__file__)
    for sub in ('data', os.path.join('..', 'data'), ''):
        cands.append(os.path.normpath(os.path.join(d, sub, name)))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


FACE_XML = _find_cc('haarcascade_frontalface_default.xml')
_CC = cv2.CascadeClassifier(FACE_XML) if FACE_XML else None


def _pre(img):
    return cv2.resize(cv2.GaussianBlur(cv2.equalizeHist(img), (3, 3), 0), (100, 100))


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def detect_faces(gray):
    if _CC is None or gray is None:
        return []
    f = _CC.detectMultiScale(gray, 1.05, 5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    return list(f)


def face_roi(gray, box):
    x, y, w, h = box
    return _pre(gray[y:y + h, x:x + w])


def raw_roi(gray, box):
    x, y, w, h = box
    return gray[y:y + h, x:x + w]


def load_and_crop_folder(folder):
    """Returns (rois, raw_rois, detection_tally) for every image in folder.
    detection_tally counts how many images had 0 / 1 / 2+ faces."""
    rois, raws = [], []
    tally = {'0': 0, '1': 0, '2+': 0}
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    for p in paths:
        gray = load_gray(p)
        if gray is None:
            continue
        boxes = detect_faces(gray)
        n = len(boxes)
        tally['0' if n == 0 else ('1' if n == 1 else '2+')] += 1
        if n == 1:
            rois.append(face_roi(gray, boxes[0]))
            raws.append(raw_roi(gray, boxes[0]))
    return rois, raws, tally


# --------------------------------------------------------------------------
# Minimal fused-score face engine (train/predict only — no liveness state,
# that's evaluated separately via the motion-threshold frame-diff pass)
# --------------------------------------------------------------------------
class OfflineFaceEngine:
    def __init__(self):
        self.lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.mean_face = None
        self._fhc = None
        self.trained = False

    def train(self, rois):
        arr = np.array(rois, dtype=np.uint8)
        lbl = np.zeros(len(rois), dtype=np.int32)
        self.lbph.train(arr, lbl)
        self.mean_face = np.mean(arr, axis=0).astype(np.uint8)
        hists = [cv2.calcHist([s], [0], None, [256], [0, 256]).flatten() for s in rois]
        self._fhc = np.mean(hists, axis=0).astype(np.float32)
        self.trained = True

    def scores(self, roi):
        """Returns (lbph_conf, hist_conf, template_conf) for one ROI,
        independent of fusion weights — fusion is applied afterwards so
        the whole weight-combination sweep can reuse these same three
        numbers instead of re-running LBPH/hist/template per combo."""
        _label, dist = self.lbph.predict(roi)
        lbph_conf = max(0.0, min(100.0, 100.0 - dist))

        h = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten().astype(np.float32)
        corr = cv2.compareHist(h, self._fhc, cv2.HISTCMP_CORREL)
        hist_conf = max(0.0, min(100.0, (corr + 1) / 2 * 100))

        a = roi.astype(np.float32)
        b = self.mean_face.astype(np.float32)
        res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
        template_conf = max(0.0, min(100.0, (float(res.max()) + 1) / 2 * 100))

        return lbph_conf, hist_conf, template_conf


def fuse(lbph_conf, hist_conf, template_conf, w_lbph, w_hist, w_template):
    return w_lbph * lbph_conf + w_hist * hist_conf + w_template * template_conf


# --------------------------------------------------------------------------
# Motion-threshold evaluation (needs ordered frame sequences, not stills)
# --------------------------------------------------------------------------
def motion_scores_for_sequence(folder):
    """Returns a list of per-consecutive-frame-pair motion scores
    (mean abs diff / 255) for a folder of sequentially-named frames —
    mirrors MotionLive.update()'s scoring exactly."""
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    scores = []
    prev = None
    for p in paths:
        gray = load_gray(p)
        if gray is None:
            continue
        boxes = detect_faces(gray)
        if len(boxes) != 1:
            prev = None
            continue
        roi = raw_roi(gray, boxes[0])
        if prev is not None and prev.shape == roi.shape:
            diff = cv2.absdiff(roi, prev)
            scores.append(float(np.mean(diff)) / 255.0)
        prev = roi
    return scores


# --------------------------------------------------------------------------
# Main sweep logic
# --------------------------------------------------------------------------
NEGATIVE = metrics_mod.NEGATIVE_LABELS


def run_match_sweep(engine, genuine_rois, impostor_rois, attack_rois_by_type,
                     min_conf_values, weight_combos, label='min_conf_sweep'):
    """Core match-score sweep shared by the min_conf, weight-combo, and
    (indirectly) enrol_frames sweeps: score every held-out face once,
    then just re-threshold/re-weight in memory for every combo — this
    is why latency here is reported separately as 'per-decision scoring
    latency', not end-to-end camera latency (see docs/EXPERIMENTS.md)."""

    if _PS is not None:
        _PS.cpu_percent(interval=None)
    t_start = time.perf_counter()

    scored = {'genuine': [], 'impostor': []}
    for name, rois in attack_rois_by_type.items():
        scored[name] = []

    for roi in genuine_rois:
        scored['genuine'].append(engine.scores(roi))
    for roi in impostor_rois:
        scored['impostor'].append(engine.scores(roi))
    for name, rois in attack_rois_by_type.items():
        for roi in rois:
            scored[name].append(engine.scores(roi))

    n_scored = sum(len(v) for v in scored.values())
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    avg_latency_ms = elapsed_ms / n_scored if n_scored else float('nan')
    cpu = _PS.cpu_percent(interval=None) if _PS is not None else float('nan')
    mem = (_PS.memory_info().rss / (1024 * 1024)) if _PS is not None else float('nan')

    rows_out = []
    for min_conf, (w_lbph, w_hist, w_template) in itertools.product(min_conf_values, weight_combos):
        attempt_rows = []
        for trial_label, score_list in scored.items():
            for (lbph_c, hist_c, tmpl_c) in score_list:
                conf = fuse(lbph_c, hist_c, tmpl_c, w_lbph, w_hist, w_template)
                outcome = 'accept' if conf >= min_conf else 'reject'
                attempt_rows.append({'trial_label': trial_label, 'outcome': outcome})
        m = metrics_mod.compute_metrics(attempt_rows)
        rows_out.append({
            'min_conf': min_conf, 'w_lbph': w_lbph, 'w_hist': w_hist, 'w_template': w_template,
            **m,
            'avg_scoring_latency_ms': round(avg_latency_ms, 4),
            'cpu_percent': cpu, 'mem_mb': round(mem, 2),
        })
    return rows_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='Path to the dataset folder (see layout in this file\'s docstring)')
    ap.add_argument('--out', default='results', help='Output folder for CSVs')
    ap.add_argument('--min-conf-range', default='20,25,30,35,40,45,50,55,60',
                     help='Comma-separated min_conf values to sweep (requested range: 20-60)')
    ap.add_argument('--motion-thresh-range', default='0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10',
                     help='Comma-separated motion_thresh values to sweep (requested range: 0.01-0.10)')
    ap.add_argument('--enrol-frames-list', default='10,20,40,60,80,100',
                     help='Comma-separated enrol_frames values to sweep')
    ap.add_argument('--weight-step', type=float, default=0.25,
                     help='Granularity for the LBPH/Hist/Template weight-combination grid (default 0.25 -> 15 combos; use 0.1 for a finer 66-combo grid)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    min_conf_values = [float(x) for x in args.min_conf_range.split(',')]
    motion_values = [float(x) for x in args.motion_thresh_range.split(',')]
    enrol_frames_list = [int(x) for x in args.enrol_frames_list.split(',')]

    # weight grid: all (w_lbph, w_hist, w_template) on the step grid that sum to 1
    step = args.weight_step
    n_steps = round(1 / step)
    weight_combos = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            weight_combos.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))

    # ---- load dataset, tallying 0/1/multi-face detections per folder ----
    detection_rows = []

    def _load(folder_rel):
        folder = os.path.join(args.dataset, folder_rel)
        if not os.path.isdir(folder):
            print(f'[warn] missing folder, skipping: {folder}')
            return [], [], {'0': 0, '1': 0, '2+': 0}
        rois, raws, tally = load_and_crop_folder(folder)
        detection_rows.append({'folder': folder_rel, **tally,
                                'total_images': sum(tally.values())})
        return rois, raws, tally

    enrol_rois, _enrol_raws, _ = _load('enrol')
    if len(enrol_rois) < max(enrol_frames_list):
        print(f'[warn] only {len(enrol_rois)} usable enrol images found, but enrol_frames '
              f'sweep asks for up to {max(enrol_frames_list)}. Larger values will reuse all '
              f'available frames (results for those will be identical).')

    impostor_rois, _, _ = _load('impostor')

    attack_rois = {}
    for name in ('attack_photo', 'attack_phone', 'attack_replay'):
        rois, _, _ = _load(name)
        attack_rois[name] = rois

    genuine_cond_rois = {}
    genuine_root = os.path.join(args.dataset, 'genuine')
    if os.path.isdir(genuine_root):
        for lighting in sorted(os.listdir(genuine_root)):
            lroot = os.path.join(genuine_root, lighting)
            if not os.path.isdir(lroot):
                continue
            for movement in sorted(os.listdir(lroot)):
                mdir = os.path.join(lroot, movement)
                if not os.path.isdir(mdir):
                    continue
                rois, _, tally = load_and_crop_folder(mdir)
                detection_rows.append({'folder': f'genuine/{lighting}/{movement}', **tally,
                                        'total_images': sum(tally.values())})
                genuine_cond_rois[(lighting, movement)] = rois

    all_genuine_rois = [r for rs in genuine_cond_rois.values() for r in rs]

    with open(os.path.join(args.out, 'detection_counts.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['folder', '0', '1', '2+', 'total_images'])
        w.writeheader()
        w.writerows(detection_rows)
    print(f"Wrote {os.path.join(args.out, 'detection_counts.csv')}")

    if not enrol_rois or not all_genuine_rois:
        print('\n[error] No usable enrol/ or genuine/ images found — cannot run the match-score '
              'sweeps without real captured data. See this script\'s docstring for the dataset '
              'layout and what to capture.')
        return

    # ================= enrol_frames sweep =================
    enrol_rows = []
    for n in enrol_frames_list:
        engine = OfflineFaceEngine()
        engine.train(enrol_rois[:n] if n <= len(enrol_rois) else enrol_rois)
        combo_rows = run_match_sweep(engine, all_genuine_rois, impostor_rois, attack_rois,
                                      min_conf_values=[DEFAULTS['min_conf']],
                                      weight_combos=[(DEFAULTS['w_lbph'], DEFAULTS['w_hist'], DEFAULTS['w_template'])])
        for r in combo_rows:
            r['enrol_frames'] = n
            enrol_rows.append(r)
    _write_csv(os.path.join(args.out, 'enrol_frames_sweep.csv'), enrol_rows)

    # ================= min_conf sweep (default enrol_frames/weights) =================
    engine = OfflineFaceEngine()
    engine.train(enrol_rois[:DEFAULTS['enrol_frames']] if len(enrol_rois) >= DEFAULTS['enrol_frames'] else enrol_rois)
    min_conf_rows = run_match_sweep(engine, all_genuine_rois, impostor_rois, attack_rois,
                                     min_conf_values=min_conf_values,
                                     weight_combos=[(DEFAULTS['w_lbph'], DEFAULTS['w_hist'], DEFAULTS['w_template'])])
    _write_csv(os.path.join(args.out, 'min_conf_sweep.csv'), min_conf_rows)

    # ================= weight-combination sweep (default min_conf) =================
    weight_rows = run_match_sweep(engine, all_genuine_rois, impostor_rois, attack_rois,
                                   min_conf_values=[DEFAULTS['min_conf']],
                                   weight_combos=weight_combos)
    _write_csv(os.path.join(args.out, 'weight_combo_sweep.csv'), weight_rows)

    # ================= per lighting/movement condition breakdown =================
    condition_rows = []
    for (lighting, movement), rois in genuine_cond_rois.items():
        attempt_rows = []
        for roi in rois:
            lbph_c, hist_c, tmpl_c = engine.scores(roi)
            conf = fuse(lbph_c, hist_c, tmpl_c, DEFAULTS['w_lbph'], DEFAULTS['w_hist'], DEFAULTS['w_template'])
            outcome = 'accept' if conf >= DEFAULTS['min_conf'] else 'reject'
            attempt_rows.append({'trial_label': 'genuine', 'outcome': outcome})
        m = metrics_mod.compute_metrics(attempt_rows)
        condition_rows.append({'lighting': lighting, 'movement': movement, 'n_images': len(rois), **m})
    _write_csv(os.path.join(args.out, 'condition_breakdown.csv'), condition_rows)

    # ================= per attack-type breakdown (at default threshold) =================
    attack_rows_out = []
    for name, rois in attack_rois.items():
        attempt_rows = []
        for roi in rois:
            lbph_c, hist_c, tmpl_c = engine.scores(roi)
            conf = fuse(lbph_c, hist_c, tmpl_c, DEFAULTS['w_lbph'], DEFAULTS['w_hist'], DEFAULTS['w_template'])
            outcome = 'accept' if conf >= DEFAULTS['min_conf'] else 'reject'
            attempt_rows.append({'trial_label': name, 'outcome': outcome})
        accepted = sum(1 for r in attempt_rows if r['outcome'] == 'accept')
        attack_rows_out.append({
            'attack_type': name, 'n_images': len(rois), 'accepted': accepted,
            'attack_success_rate': (accepted / len(rois)) if rois else float('nan'),
        })
    _write_csv(os.path.join(args.out, 'attack_breakdown.csv'), attack_rows_out)

    # ================= motion threshold sweep (sequence folders only) =================
    motion_rows = []
    for name in ('attack_phone', 'attack_replay'):
        folder = os.path.join(args.dataset, name)
        if not os.path.isdir(folder):
            continue
        seq_scores = motion_scores_for_sequence(folder)
        for thr in motion_values:
            would_pass_liveness = any(
                sum(seq_scores[max(0, i - DEFAULTS['motion_frames']):i + 1]) /
                max(1, len(seq_scores[max(0, i - DEFAULTS['motion_frames']):i + 1])) >= thr
                for i in range(len(seq_scores))
            ) if seq_scores else False
            motion_rows.append({
                'source': name, 'motion_thresh': thr, 'n_frame_pairs': len(seq_scores),
                'liveness_would_pass': would_pass_liveness,
                'note': 'liveness_would_pass=True means this threshold is too LOW to '
                        'block this attack type based on motion alone',
            })
    if not motion_rows:
        motion_rows.append({'note': 'No sequential-frame folders (attack_phone/attack_replay) found — '
                                     'motion_thresh sweep needs consecutively-captured frames, not '
                                     'independent stills. See docstring.'})
    _write_csv(os.path.join(args.out, 'motion_thresh_sweep.csv'), motion_rows)

    print(f'\nDone. Real, measured results (from the data under {args.dataset}) are in {args.out}/*.csv')


def _write_csv(path, rows):
    if not rows:
        print(f'[warn] no rows for {path}, skipping')
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
