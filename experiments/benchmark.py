#!/usr/bin/env python3
"""Measured benchmark of the identity-verification stage (the matcher) on two public datasets.

    python experiments/benchmark.py --split dev     parameter selection: calibration, fusion weights, threshold
    python experiments/benchmark.py --split test    the reported numbers, with the shipped configuration

WHAT IS MEASURED
    Stages F and R of the pipeline with the app's own code (biolock/session.prepare, biolock/engine): frame
    preparation, Haar face detection, eye detection, alignment, Tan-Triggs normalisation, LBPH / histogram /
    template scores, fusion, threshold. Real photographs of real people (see experiments/datasets.py).
WHAT IS NOT MEASURED
    Liveness, presentation attacks, a webcam, the vault-key stage, timing. A photograph is not a live session:
    "median of m" below means m different photographs of one person, not 15 consecutive camera frames.

PROTOCOL
    * Identities at even index (sorted order) form the DEV half, odd index the TEST half. Parameters are
      chosen on dev only; test is used only to report.
    * Per identity and seed: K random photographs enrol (LFW K=10, Olivetti K=5), up to 30 / 5 of the others
      are genuine probes, 150 / 120 impostor probes are drawn uniformly over the other identities of the half.
    * Two galleries: "photos" = the K photographs; "live" = each photograph plus jittered, noisy copies passed
      through detection again (about 90 samples), which emulates the 100-frame enrolment of the app. The
      acceptance threshold is set for the "live" gallery, because a larger gallery lowers impostor distances.
    * A photograph in which no face is detected is left out (counted in the report); if several are detected
      the one nearest the centre is used (the app itself refuses frames with more than one face).

SELECTION RULES (applied on the dev halves, stated before the test halves were run)
    calibration  distance mapped to 100 / 0 = 1st percentile of genuine / 99th of impostor distances, pooled;
                 correlations mapped to 0 / 100 = 1st percentile of impostor / 99th of genuine values
    weights      minimum mean equal-error rate over the two dev sets on a 0.05 grid; if LBPH alone is not
                 measurably worse (95 % identity-bootstrap interval of the difference contains 0), LBPH alone
    threshold    smallest whole number T such that at most 1 % of single-photo impostor probes score >= T on
                 each dev set, "live" gallery
"""
import os
import sys
import json
import time
import math
import argparse
import tempfile
import itertools
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import datasets as DS                                               # noqa: E402
from biolock import __version__                                     # noqa: E402
from biolock.config import load_config, DEFAULTS                    # noqa: E402
from biolock.engine import FaceEngine, calibrate, fuse              # noqa: E402
from biolock.metrics import roc_summary, far_frr, wilson            # noqa: E402

PAPER_W = (0.45, 0.20, 0.35)        # weights printed in the manuscript
V9_W = (0.5, 0.3, 0.2)              # weights shipped in version 9
GALLERIES = ('photos', 'live')
RAW = ('dist', 'hist', 'tmpl')
_W = {}


# --------------------------------------------------------------------------------------------------------
# worker: enrol every identity of one (dataset, seed) and score every other photograph of the half
# --------------------------------------------------------------------------------------------------------
def _worker(info, folder, split):
    key = (info['name'], folder, split)
    if key not in _W:
        cv2.setNumThreads(1)
        cfg = load_config(path='', use_env=False)
        E = FaceEngine(cfg)
        E9 = FaceEngine(load_config(path='', use_env=False, overrides={'lbph_radius': 1, 'lbph_grid': 8}))
        D, F = DS.unstage(info, folder)
        ids = DS.split_ids(D, split)
        pool = np.flatnonzero(F['valid'] & np.isin(D['target'], ids))
        feats = {int(i): E.features(F['rois'][i]) for i in pool}
        h9 = {int(i): E9.lbph_hist(F['rois_extra'][i]) for i in pool}
        _W.clear()
        _W[key] = (cfg, E, E9, D, F, ids, pool, feats, h9)
    return _W[key]


def seed_job(job):
    info, folder, split, seed, only = job
    cfg, E, E9, D, F, ids, pool, feats, h9 = _worker(info, folder, split)
    trials = DS.make_trials(F['valid'], D['target'], ids, info['k'], info['n_imp'], info['max_gen'], seed)
    rng = np.random.RandomState(1000 + seed)
    models = []
    for pid, gal, gen, imp in trials:
        if only is not None and pid not in only:
            continue
        probes = np.setdiff1d(pool, gal)
        m = dict(pid=int(pid), gal=gal, gen=gen, imp=imp, probes=probes, probe_pid=D['target'][probes])
        for g in GALLERIES:
            rois = [F['rois'][i] for i in gal] if g == 'photos' else DS.live_gallery(E, D, F, gal, rng)
            m['n_' + g] = E.train(rois, save=False)
            comp = [E.compare(feats[int(i)]) for i in probes]
            m[g] = {'dist': np.array([c['lbph_distance'] for c in comp], np.float32),
                    'hist': np.array([c['hist_corr'] for c in comp], np.float32),
                    'tmpl': np.array([c['tmpl_corr'] for c in comp], np.float32)}
        E9.train([F['rois_extra'][i] for i in gal], save=False)
        m['v9'] = {'dist': np.array([E9._nearest(h9[int(i)]) for i in probes], np.float32)}
        models.append(m)
    return info['name'], seed, models


# --------------------------------------------------------------------------------------------------------
# trials sampled from the scored models
# --------------------------------------------------------------------------------------------------------
def single_trials(models, gallery):
    """One row per (model, single probe photograph): identity of the model, label, raw measurements."""
    pid, y, raw = [], [], collections.defaultdict(list)
    for m in models:
        pos = {int(i): j for j, i in enumerate(m['probes'])}
        idx = np.array([pos[int(i)] for i in np.r_[m['gen'], m['imp']]])
        pid.append(np.full(len(idx), m['pid']))
        y.append(np.r_[np.ones(len(m['gen'])), np.zeros(len(m['imp']))])
        for k in m[gallery]:
            raw[k].append(m[gallery][k][idx])
    return dict(pid=np.concatenate(pid), y=np.concatenate(y), **{k: np.concatenate(v) for k, v in raw.items()})


def multi_trials(models, score_of, m_photos, seed, reps=10, max_imp_ids=30):
    """Decisions on the median score of m different photographs of one person.
    Genuine: `reps` random m-subsets of the identity's own probes. Impostor: m random photographs of each of
    up to `max_imp_ids` other identities."""
    rng = np.random.RandomState(100 + seed)
    pid, y, s = [], [], []
    for m in models:
        sc = score_of(m)
        own = np.flatnonzero(m['probe_pid'] == m['pid'])
        if len(own) >= m_photos:
            for _ in range(reps):
                s.append(np.median(sc[rng.choice(own, m_photos, replace=False)]))
                y.append(1)
                pid.append(m['pid'])
        others = np.unique(m['probe_pid'][m['probe_pid'] != m['pid']])
        for q in rng.permutation(others)[:max_imp_ids]:
            qi = np.flatnonzero(m['probe_pid'] == q)
            if len(qi) >= m_photos:
                s.append(np.median(sc[rng.choice(qi, m_photos, replace=False)]))
                y.append(0)
                pid.append(m['pid'])
    return dict(pid=np.array(pid), y=np.array(y, float), score=np.array(s))


def fused(cfg, t, weights=None):
    return fuse(cfg, *calibrate(cfg, t['dist'], t['hist'], t['tmpl']), weights=weights)


# --------------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------------
def mean_roc(runs):
    """runs: [(scores, labels)] one per seed -> mean of the ROC summaries."""
    ms = [roc_summary(s, y) for s, y in runs]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}
    out['n_gen'] = int(sum(m['n_gen'] for m in ms))
    out['n_imp'] = int(sum(m['n_imp'] for m in ms))
    out['seeds'] = len(ms)
    return out


def operating_point(runs, pids, threshold, B=2000, seed=12345):
    """FAR/FRR at `threshold`, pooled over seeds, with a 95 % interval from resampling the enrolled
    identities (trials of one identity are correlated, so a per-trial interval would be too narrow)."""
    s = np.concatenate([r[0] for r in runs])
    y = np.concatenate([r[1] for r in runs]).astype(bool)
    p = np.concatenate(pids)
    acc = s >= threshold
    ids = np.unique(p)
    fa = np.array([np.sum(acc & ~y & (p == i)) for i in ids], float)
    ni = np.array([np.sum(~y & (p == i)) for i in ids], float)
    fr = np.array([np.sum(~acc & y & (p == i)) for i in ids], float)
    ng = np.array([np.sum(y & (p == i)) for i in ids], float)
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(ids), size=(B, len(ids)))
    far_b = fa[pick].sum(1) / np.maximum(ni[pick].sum(1), 1)
    frr_b = fr[pick].sum(1) / np.maximum(ng[pick].sum(1), 1)
    return dict(threshold=float(threshold), far=float(fa.sum() / max(ni.sum(), 1)), frr=float(fr.sum() / max(ng.sum(), 1)),
                false_accepts=int(fa.sum()), impostor_trials=int(ni.sum()),
                false_rejects=int(fr.sum()), genuine_trials=int(ng.sum()), identities=int(len(ids)),
                far_ci=[float(v) for v in np.percentile(far_b, [2.5, 97.5])],
                frr_ci=[float(v) for v in np.percentile(frr_b, [2.5, 97.5])],
                far_wilson=list(wilson(int(fa.sum()), int(ni.sum()))),
                frr_wilson=list(wilson(int(fr.sum()), int(ng.sum()))))


def boot_eer_diff(per_dataset, wa, wb, cfg, B=500, seed=777):
    """Bootstrap (over enrolled identities) of mean-over-datasets EER(wa) - EER(wb)."""
    rng = np.random.RandomState(seed)
    diffs = np.zeros(B)
    for trials_by_seed in per_dataset:
        ids = np.unique(np.concatenate([t['pid'] for t in trials_by_seed]))
        sa = [fused(cfg, t, wa) for t in trials_by_seed]
        sb = [fused(cfg, t, wb) for t in trials_by_seed]
        where = [{i: np.flatnonzero(t['pid'] == i) for i in ids} for t in trials_by_seed]
        for b in range(B):
            pick = ids[rng.randint(0, len(ids), len(ids))]
            d = 0.0
            for t, a_, b_, w in zip(trials_by_seed, sa, sb, where):
                rows = np.concatenate([w[i] for i in pick])
                d += roc_summary(a_[rows], t['y'][rows])['eer'] - roc_summary(b_[rows], t['y'][rows])['eer']
            diffs[b] += d / len(trials_by_seed) / len(per_dataset)
    return diffs


def pct(x, q):
    return [round(float(v), 2) for v in np.percentile(x, q)] if len(x) else []


# --------------------------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------------------------
def dataset_report(cfg, runs, T, weightings):
    """runs: {seed: models} of one dataset. Everything reported for it with configuration `cfg`."""
    seeds = sorted(runs)
    out = {'galleries': {}, 'weightings': {}, 'components': {}}
    for g in GALLERIES:
        st = [single_trials(runs[s], g) for s in seeds]
        sc = [fused(cfg, t) for t in st]
        y = np.concatenate([t['y'] for t in st]).astype(bool)
        d = np.concatenate([t['dist'] for t in st])
        res = {'gallery_samples_mean': float(np.mean([m['n_' + g] for s in seeds for m in runs[s]])),
               'single': dict(mean_roc([(a, t['y']) for a, t in zip(sc, st)]),
                              at_threshold=operating_point([(a, t['y']) for a, t in zip(sc, st)],
                                                           [t['pid'] for t in st], T)),
               'distance_percentiles': {'q': [1, 5, 50, 95, 99], 'genuine': pct(d[y], [1, 5, 50, 95, 99]),
                                        'impostor_q': [0.1, 0.5, 1, 5, 50],
                                        'impostor': pct(d[~y], [0.1, 0.5, 1, 5, 50])}}
        for m_photos in (3, 5):
            mt = [multi_trials(runs[s], lambda m, g=g: fused(cfg, m[g]), m_photos, s) for s in seeds]
            mt = [t for t in mt if t['y'].sum() > 0 and (1 - t['y']).sum() > 0]
            if mt:
                res[f'median{m_photos}'] = dict(mean_roc([(t['score'], t['y']) for t in mt]),
                                                at_threshold=operating_point([(t['score'], t['y']) for t in mt],
                                                                             [t['pid'] for t in mt], T))
        out['galleries'][g] = res
    st = [single_trials(runs[s], 'photos') for s in seeds]
    for name, w in weightings.items():
        runs_w = [(fused(cfg, t, w), t['y']) for t in st]
        out['weightings'][name] = dict(mean_roc(runs_w), weights=list(w),
                                       at_threshold=operating_point(runs_w, [t['pid'] for t in st], T))
    for name, key, sign in (('LBPH distance', 'dist', -1), ('histogram (HOG) correlation', 'hist', 1),
                            ('template correlation', 'tmpl', 1)):
        out['components'][name] = mean_roc([(sign * t[key], t['y']) for t in st])
    v9 = [single_trials(runs[s], 'v9') for s in seeds]
    out['v9_baseline'] = {'single': mean_roc([(-t['dist'], t['y']) for t in v9])}
    mt = [multi_trials(runs[s], lambda m: -m['v9']['dist'], 5, s) for s in seeds]
    mt = [t for t in mt if t['y'].sum() > 0 and (1 - t['y']).sum() > 0]
    if mt:
        out['v9_baseline']['median5'] = mean_roc([(t['score'], t['y']) for t in mt])
    out['models'] = int(sum(len(runs[s]) for s in seeds))
    out['identities'] = int(len({m['pid'] for s in seeds for m in runs[s]}))
    return out


def select(cfg, all_runs):
    """The three selection rules of the module docstring, applied to the dev runs."""
    names = sorted(all_runs)
    st = {n: [single_trials(all_runs[n][s], 'photos') for s in sorted(all_runs[n])] for n in names}
    pool = lambda key, lab: np.concatenate([t[key][t['y'] == lab] for n in names for t in st[n]])  # noqa: E731
    cal = {'lbph_d_lo': float(np.percentile(pool('dist', 1), 1)), 'lbph_d_hi': float(np.percentile(pool('dist', 0), 99)),
           'hist_c_lo': float(np.percentile(pool('hist', 0), 1)), 'hist_c_hi': float(np.percentile(pool('hist', 1), 99)),
           'tmpl_c_lo': float(np.percentile(pool('tmpl', 0), 1)), 'tmpl_c_hi': float(np.percentile(pool('tmpl', 1), 99))}
    sel = {'calibration': {'derived_through_engine': cal, 'shipped': {k: cfg[k] for k in cal}}}

    grid = [(round(1 - i / 20 - j / 20, 2), i / 20, j / 20) for i in range(21) for j in range(21) if i + j <= 20]
    rows = []
    for w in grid:
        e = [float(np.mean([roc_summary(fused(cfg, t, w), t['y'])['eer'] for t in st[n]])) for n in names]
        rows.append((float(np.mean(e)), w, e))
    rows.sort(key=lambda r: (r[0], -r[1][0]))
    best = rows[0]
    lbph = next(r for r in rows if r[1] == (1.0, 0.0, 0.0))
    diff = boot_eer_diff([st[n] for n in names], (1.0, 0.0, 0.0), best[1], cfg)
    lo, hi = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
    chosen = (1.0, 0.0, 0.0) if lo <= 0.0 <= hi or best[1] == (1.0, 0.0, 0.0) else best[1]
    per = {}
    for j, n in enumerate(names):                       # the best grid point for each dataset on its own
        _m, w, e = min(rows, key=lambda r: (r[2][j], -r[1][0]))
        per[n] = {'weights': list(w), 'eer': e[j], 'lbph_alone_eer': lbph[2][j]}
    sel['weights'] = {
        'datasets': names, 'grid_step': 0.05, 'per_dataset_minimum': per,
        'top': [{'weights': list(w), 'mean_eer': m, 'eer': e} for m, w, e in rows[:8]],
        'grid_minimum': {'weights': list(best[1]), 'mean_eer': best[0], 'eer': best[2]},
        'lbph_alone': {'weights': [1.0, 0.0, 0.0], 'mean_eer': lbph[0], 'eer': lbph[2]},
        'paper': next({'weights': list(w), 'mean_eer': m, 'eer': e} for m, w, e in rows if w == PAPER_W),
        'v9': next({'weights': list(w), 'mean_eer': m, 'eer': e} for m, w, e in rows if w == V9_W),
        'lbph_minus_minimum': {'mean_eer_difference': lbph[0] - best[0], 'ci95': [lo, hi],
                               'bootstrap': 'identities resampled 500 times'},
        'chosen_by_rule': list(chosen), 'shipped': [cfg['w_lbph'], cfg['w_hist'], cfg['w_template']]}

    thr = {}
    for n in names:
        lt = [single_trials(all_runs[n][s], 'live') for s in sorted(all_runs[n])]
        s = np.concatenate([fused(cfg, t) for t in lt])
        y = np.concatenate([t['y'] for t in lt]).astype(bool)
        imp = np.sort(s[~y])[::-1]
        allowed = int(math.floor(0.01 * len(imp)))                  # impostors that may score >= T
        t_exact = float(imp[allowed]) if allowed < len(imp) else float(imp[-1])
        T_n = int(math.floor(t_exact)) + 1                          # smallest whole T above that score
        thr[n] = {'score_with_1pct_far': t_exact, 'threshold': T_n, 'impostor_probes': int(len(imp)),
                  'far_at_threshold': float(np.mean(imp >= T_n)), 'frr_at_threshold': float(np.mean(s[y] < T_n))}
    T = max(v['threshold'] for v in thr.values())
    sel['threshold'] = {'per_dataset': thr, 'chosen_by_rule': T, 'shipped': cfg['min_conf'],
                        'lbph_distance_at_threshold': cfg['lbph_d_hi'] - T / 100.0 * (cfg['lbph_d_hi'] - cfg['lbph_d_lo'])}
    return sel


def show(name, rep, T):
    print(f"\n=== {name}: {rep['identities']} enrolled identities, {rep['models']} enrolments over the seeds ===")
    print('  %-9s %-9s %7s %7s %6s %6s %9s   FAR at T=%g [95%% CI]        FRR at T [95%% CI]' % (
        'gallery', 'decision', 'genuine', 'impost.', 'EER', 'AUC', 'TAR@1%FAR', T))
    for g in GALLERIES:
        for d in ('single', 'median3', 'median5'):
            m = rep['galleries'][g].get(d)
            if not m:
                continue
            a = m['at_threshold']
            print('  %-9s %-9s %7d %7d %5.1f%% %6.3f %8.1f%%   %5.2f%% [%5.2f-%5.2f]         %5.1f%% [%5.1f-%5.1f]' % (
                g, d, m['n_gen'], m['n_imp'], 100 * m['eer'], m['auc'], 100 * m['tar1'], 100 * a['far'],
                100 * a['far_ci'][0], 100 * a['far_ci'][1], 100 * a['frr'], 100 * a['frr_ci'][0], 100 * a['frr_ci'][1]))
    print('  fusion weightings (single photograph, "photos" gallery):')
    for n, m in rep['weightings'].items():
        print('    %-34s EER %5.1f%%  AUC %.3f  TAR@1%%FAR %5.1f%%   at T: FAR %5.2f%%  FRR %5.1f%%' % (
            n, 100 * m['eer'], m['auc'], 100 * m['tar1'], 100 * m['at_threshold']['far'], 100 * m['at_threshold']['frr']))
    for n, m in rep['components'].items():
        print('    component alone: %-28s EER %5.1f%%  AUC %.3f' % (n, 100 * m['eer'], m['auc']))
    b = rep['v9_baseline']
    print('  version-9 preprocessing + LBPH r=1, 8x8:  single EER %5.1f%% TAR@1%%FAR %5.1f%%%s' % (
        100 * b['single']['eer'], 100 * b['single']['tar1'],
        '' if 'median5' not in b else '   median of 5: EER %5.1f%% TAR@1%%FAR %5.1f%%' % (
            100 * b['median5']['eer'], 100 * b['median5']['tar1'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--split', default='dev', choices=('dev', 'test'))
    ap.add_argument('--datasets', default='olivetti,lfw')
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) * 2 // 5)))
    ap.add_argument('--quick', action='store_true', help='6 identities and 1 seed per dataset (smoke test)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cfg = load_config(path='', use_env=False)
    E = FaceEngine(cfg)
    T = cfg['min_conf']
    report = {'meta': dict(split=args.split, version=__version__, opencv=cv2.__version__, numpy=np.__version__,
                           python=sys.version.split()[0], quick=args.quick,
                           config={k: cfg[k] for k in ('min_conf', 'w_lbph', 'w_hist', 'w_template', 'lbph_radius',
                                                       'lbph_neighbors', 'lbph_grid', 'roi_size', 'lbph_d_lo',
                                                       'lbph_d_hi', 'hist_c_lo', 'hist_c_hi', 'tmpl_c_lo',
                                                       'tmpl_c_hi')}),
              'datasets': {}}
    all_runs = {}
    with tempfile.TemporaryDirectory() as folder:
        jobs = []
        for name in args.datasets.split(','):
            D = DS.LOADERS[name]()
            ids = DS.split_ids(D, args.split)
            mine = np.isin(D['target'], ids)
            info = DS.stage_photos(D, folder)
            F = DS.extract_parallel(info, folder, len(D['photos']), np.flatnonzero(mine), args.workers,
                                    with_v9=True, size=E.size)
            DS.stage_features(info, F, folder)
            v = F['valid']
            report['datasets'][name] = {
                'title': D['title'], 'photographs': int(mine.sum()), 'identities_in_half': int(len(ids)),
                'k_enrol': D['k'], 'seeds': list(D['seeds'][:1] if args.quick else D['seeds']),
                'detection': {'face_found': float(v[mine].mean()), 'no_face': int((~v[mine]).sum()),
                              'several_faces': int((F['ndet'][mine] > 1).sum()),
                              'eyes_found_of_faces': float(F['eyes'][mine & v].mean()),
                              'face_width_px_mean': float(F['boxw'][mine & v].mean())}}
            only = set(int(i) for i in ids[:6]) if args.quick else None
            jobs += [(info, folder, args.split, s, only) for s in report['datasets'][name]['seeds']]
            print(f"{name}: {int(mine.sum())} photographs in the {args.split} half, faces found in "
                  f"{100 * v[mine].mean():.1f} %, both eyes in {100 * F['eyes'][mine & v].mean():.1f} % of those "
                  f"({time.time() - t0:.0f}s)", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for name, seed, models in ex.map(seed_job, jobs):
                all_runs.setdefault(name, {})[seed] = models
                print(f'  scored {name} seed {seed}: {len(models)} enrolments ({time.time() - t0:.0f}s)', flush=True)

    shipped_w = (cfg['w_lbph'], cfg['w_hist'], cfg['w_template'])
    weightings = {'shipped (%g, %g, %g)' % shipped_w: shipped_w}
    dev_json = os.path.join(args.out, 'benchmark_dev.json')
    dev_sel = None
    if args.split == 'dev':
        report['selection'] = dev_sel = select(cfg, all_runs)
    elif os.path.exists(dev_json):
        with open(dev_json, encoding='utf-8') as f:
            dev_sel = json.load(f).get('selection')
    if dev_sel:                                 # the best weightings of the development grid, for reference
        for rank, r in enumerate(dev_sel['weights']['top'][:3], 1):
            w = tuple(r['weights'])
            if w != shipped_w and w not in weightings.values():
                weightings['development grid, rank %d (%g, %g, %g)' % ((rank,) + w)] = w
    if shipped_w != (1.0, 0.0, 0.0):
        weightings['LBPH alone (1, 0, 0)'] = (1.0, 0.0, 0.0)
    weightings['manuscript (0.45, 0.2, 0.35)'] = PAPER_W
    weightings['version 9 (0.5, 0.3, 0.2)'] = V9_W
    for name, runs in all_runs.items():
        report['datasets'][name].update(dataset_report(cfg, runs, T, weightings))
        show(report['datasets'][name]['title'], report['datasets'][name], T)

    if args.split == 'dev':
        s = report['selection']
        print('\n=== selection on the dev halves ===')
        print('  calibration through the engine:', {k: round(v, 3) for k, v in s['calibration']['derived_through_engine'].items()})
        print('  calibration shipped:           ', s['calibration']['shipped'])
        w = s['weights']
        for r in w['top'][:5]:
            print('  weights %-20s mean EER %.2f%%  (%s)' % (r['weights'], 100 * r['mean_eer'],
                                                            ', '.join('%.1f%%' % (100 * e) for e in r['eer'])))
        for k in ('lbph_alone', 'paper', 'v9'):
            print('  %-10s %-17s mean EER %.2f%%  (%s)' % (k, w[k]['weights'], 100 * w[k]['mean_eer'],
                                                           ', '.join('%.1f%%' % (100 * e) for e in w[k]['eer'])))
        d = w['lbph_minus_minimum']
        print('  LBPH alone minus grid minimum: %+.2f pp, 95%% CI [%+.2f, %+.2f] pp -> rule chooses %s (shipped %s)' % (
            100 * d['mean_eer_difference'], 100 * d['ci95'][0], 100 * d['ci95'][1], w['chosen_by_rule'], w['shipped']))
        for n, v in s['threshold']['per_dataset'].items():
            print('  threshold %-9s 1%% impostor score %.2f -> T=%d  (dev FAR %.2f%%, FRR %.1f%%, %d impostor probes)' % (
                n, v['score_with_1pct_far'], v['threshold'], 100 * v['far_at_threshold'], 100 * v['frr_at_threshold'],
                v['impostor_probes']))
        print('  rule chooses T = %d (LBPH distance %.1f); shipped min_conf = %g' % (
            s['threshold']['chosen_by_rule'], s['threshold']['lbph_distance_at_threshold'], s['threshold']['shipped']))
        ok = (list(w['chosen_by_rule']) == list(w['shipped'])
              and s['threshold']['chosen_by_rule'] == s['threshold']['shipped'])
        report['selection']['shipped_matches_rules'] = bool(ok)
        print('  shipped configuration matches the selection rules:', ok)

    report['meta']['seconds'] = round(time.time() - t0, 1)
    path = os.path.join(args.out, f'benchmark_{args.split}{"_quick" if args.quick else ""}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'\nSaved {path}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
