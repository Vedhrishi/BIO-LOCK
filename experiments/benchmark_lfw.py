#!/usr/bin/env python3
"""
Measured benchmark of BioLock's face-match stage on a public dataset (LFW).

WHAT THIS MEASURES
    Only the identity-verification stage R (Haar detection -> preprocessing -> LBPH /
    histogram / template fusion -> threshold), using BioLock's own FaceEngine code, on real
    photographs of real people (Labeled Faces in the Wild, people with >= 20 images).
WHAT IT DOES NOT MEASURE
    Liveness (motion / blink / head turn), presentation attacks, webcam conditions, the
    vault-key stage, or latency on the target hardware. Those need live trials
    (docs/TESTING_GUIDE.md). LFW photos are unconstrained (pose, light, expression, age), so
    this is a harder, different setting from enrolling on a webcam.

PROTOCOL
    * Identities are split in half (seeded): a DEV half to choose weights/threshold, a TEST half
      for the numbers that are reported. Nothing reported is tuned on the test half.
    * For every identity: K random photos are used for enrolment (same FaceEngine.train as the
      app, including its outlier pruning); the identity's other photos are genuine trials; up to
      N_IMP photos of other identities (same half) are impostor trials.
    * The three component scores of every trial are stored, so weights/thresholds can be swept
      offline without re-running recognition.
Usage:  python experiments/benchmark_lfw.py [--k 10] [--seed 0] [--out results/lfw]
"""
import os
import sys
import json
import time
import argparse
import tempfile
import itertools

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M


def load_lfw(data_home=None):
    from sklearn.datasets import fetch_lfw_people
    d = fetch_lfw_people(data_home=data_home, min_faces_per_person=20, color=False, slice_=None, resize=1.0)
    imgs = d.images
    if imgs.max() <= 1.0 + 1e-6:          # scikit-learn returns float images scaled to [0, 1]
        imgs = imgs * 255.0
    return np.clip(np.rint(imgs), 0, 255).astype(np.uint8), d.target, list(d.target_names)


def extract_rois(images, engine):
    """Detect one face per image with BioLock's own detector; return ROIs + detection counts."""
    rois, keep, counts = [], [], {'0': 0, '1': 0, '2+': 0}
    for i, img in enumerate(images):
        gray = cv2.equalizeHist(img)
        boxes = engine.detect_all(gray)
        n = len(boxes)
        counts['0' if n == 0 else ('1' if n == 1 else '2+')] += 1
        if n == 0:
            continue
        # LFW photos are centred on one person: take the largest detection if several
        b = max(boxes, key=lambda x: x[2] * x[3])
        rois.append(engine.roi(gray, b))
        keep.append(i)
    return rois, np.array(keep), counts


def component_scores(engine, roi):
    l, _ = engine._lbph_confidence(roi)
    return l, engine._hist_confidence(roi), engine._template_confidence(roi)


def run_trials(rois, labels, ids, k, n_imp, rng, tmpdir):
    M.MODEL_FILE = os.path.join(tmpdir, 'face.xml')
    M.STATS_FILE = os.path.join(tmpdir, 'face_stats.pkl')
    M.LOCK_DIR = tmpdir
    engine = M.FaceEngine()
    rows = []
    for pid in ids:
        idx = np.where(labels == pid)[0]
        if len(idx) < k + 5:
            continue
        idx = rng.permutation(idx)
        enrol, genuine = idx[:k], idx[k:]
        others = np.where(np.isin(labels, [i for i in ids if i != pid]))[0]
        imp = rng.choice(others, size=min(n_imp, len(others)), replace=False)
        engine.train([rois[j] for j in enrol], enrol_frames=k)
        for j in genuine:
            rows.append((pid, 1) + component_scores(engine, rois[j]))
        for j in imp:
            rows.append((pid, 0) + component_scores(engine, rois[j]))
    return np.array(rows, dtype=np.float64)   # pid, is_genuine, lbph, hist, tmpl


def fuse(rows, w):
    return w[0] * rows[:, 2] + w[1] * rows[:, 3] + w[2] * rows[:, 4]


def far_frr(scores, y, thr):
    acc = scores >= thr
    far = float(np.mean(acc[y == 0])) if np.any(y == 0) else float('nan')
    frr = float(np.mean(~acc[y == 1])) if np.any(y == 1) else float('nan')
    return far, frr


def eer_auc(scores, y):
    """(EER, threshold at EER, AUC) from one sorted sweep over all thresholds (accept if score >= t)."""
    order = np.argsort(-scores, kind='stable')
    s, g = scores[order], y[order]
    P, N = g.sum(), (1 - g).sum()
    tpr = np.concatenate([[0.0], np.cumsum(g) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - g) / N])
    frr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - frr)))
    thr = float(s[max(i - 1, 0)])
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float((fpr[i] + frr[i]) / 2), thr, float(trapz(tpr, fpr))


def wilson(p, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10, help='enrolment photos per identity')
    ap.add_argument('--n-imp', type=int, default=150, help='impostor photos per enrolled identity')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/lfw')
    ap.add_argument('--data-home', default=None, help='where scikit-learn caches LFW')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    t0 = time.time()
    images, target, names = load_lfw(a.data_home)
    engine0 = M.FaceEngine()
    rois, keep, counts = extract_rois(images, engine0)
    labels = target[keep]
    print(f'images={len(images)} identities={len(names)}  face detections per image: {counts}  usable={len(rois)}')

    ids = rng.permutation(np.unique(labels))
    dev_ids, test_ids = ids[:len(ids) // 2], ids[len(ids) // 2:]
    with tempfile.TemporaryDirectory() as td:
        dev = run_trials(rois, labels, dev_ids, a.k, a.n_imp, rng, td)
        test = run_trials(rois, labels, test_ids, a.k, a.n_imp, rng, td)
    print(f'trials: dev={len(dev)}  test={len(test)}  ({time.time() - t0:.0f}s)')

    # choose weights + threshold on DEV only: weight grid (step 0.05), minimise EER
    grid = [(i / 20, j / 20, round(1 - i / 20 - j / 20, 10)) for i, j in itertools.product(range(21), range(21))
            if i + j <= 20]
    best = None
    for w in grid:
        eer, thr, auc = eer_auc(fuse(dev, w), dev[:, 1])
        if best is None or eer < best[0]:
            best = (eer, thr, w)
    sel_w, sel_thr = best[2], best[1]

    paper_w = (M.W_LBPH, M.W_HIST, M.W_TEMPLATE)
    configs = {
        f'BioLock default weights {paper_w}, threshold {M.MIN_CONF}': (paper_w, M.MIN_CONF),
        f'Paper weights (0.45, 0.20, 0.35), threshold {M.MIN_CONF}': ((0.45, 0.20, 0.35), M.MIN_CONF),
        'LBPH only, threshold 40': ((1, 0, 0), M.MIN_CONF),
        f'dev-selected weights {sel_w}, dev threshold {sel_thr:.1f}': (sel_w, sel_thr),
    }
    report = {'dataset': 'LFW (people with >=20 images)', 'k_enrol': a.k, 'seed': a.seed,
              'detections': counts, 'dev_identities': int(len(dev_ids)), 'test_identities': int(len(test_ids)),
              'n_genuine_test': int(test[:, 1].sum()), 'n_impostor_test': int((1 - test[:, 1]).sum()),
              'configs': {}}
    y = test[:, 1]
    print(f'\nTEST half: {int(y.sum())} genuine trials, {int((1 - y).sum())} impostor trials')
    print(f'{"configuration":70s} FAR [95% CI]            FRR [95% CI]            balanced-acc')
    for name, (w, thr) in configs.items():
        s = fuse(test, w)
        far, frr = far_frr(s, y, thr)
        fl, fh = wilson(far, int((1 - y).sum()))
        rl, rh = wilson(frr, int(y.sum()))
        eer, _, auc = eer_auc(s, y)
        report['configs'][name] = dict(weights=list(w), threshold=float(thr), FAR=far, FRR=frr,
                                       FAR_CI=[fl, fh], FRR_CI=[rl, rh], EER=eer, AUC=auc,
                                       balanced_accuracy=1 - (far + frr) / 2)
        print(f'{name:70s} {far:5.3f} [{fl:.3f},{fh:.3f}]   {frr:5.3f} [{rl:.3f},{rh:.3f}]   {1 - (far + frr) / 2:.3f}   '
              f'(EER {eer:.3f}, AUC {auc:.3f})')
    for comp, col in (('LBPH', 2), ('histogram', 3), ('template', 4)):
        eer, _, auc = eer_auc(test[:, col], y)
        report['configs'][f'component only: {comp}'] = dict(EER=eer, AUC=auc)
        print(f'{"component only: " + comp:70s} EER {eer:.3f}  AUC {auc:.3f}')
    json.dump(report, open(os.path.join(a.out, 'lfw_benchmark.json'), 'w'), indent=2)
    np.savetxt(os.path.join(a.out, 'test_scores.csv'), test, delimiter=',',
               header='identity,is_genuine,lbph,hist,template', comments='')
    print(f'\nSaved {a.out}/lfw_benchmark.json and test_scores.csv')


if __name__ == '__main__':
    main()
