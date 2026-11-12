"""Experiment 1: LBPH-only, DEV split. Which crop/alignment + normalisation is best?"""
import os, sys, time, json, itertools
import numpy as np
import cv2
from multiprocessing import Pool
import rlib

R = rlib.R
EYES = os.path.join(R, 'lfw_eyes.npz')


def eyes_cache():
    if os.path.exists(EYES):
        return np.load(EYES)['eyes']
    d = rlib.data()
    out = np.full((len(d['images']), 4), np.nan, np.float32)
    for i, (g, b) in enumerate(zip(d['images'], d['boxes'])):
        if b[2] <= 0:
            continue
        # look for eyes at ~webcam scale (x1.6), on the equalised image
        up = cv2.resize(g, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        e = rlib.find_eyes(cv2.equalizeHist(up), [int(v * 1.6) for v in b])
        if e is not None:
            out[i] = [e[0][0] / 1.6, e[0][1] / 1.6, e[1][0] / 1.6, e[1][1] / 1.6]
    np.savez(EYES, eyes=out)
    return out


def build_rois(cfg):
    d = rlib.data()
    eyes = eyes_cache()
    rois, valid = [], []
    mask = rlib.ellipse_mask(cfg.get('out', 100)) if cfg.get('mask') else None
    for g, b, e in zip(d['images'], d['boxes'], eyes):
        if b[2] <= 0:
            rois.append(np.zeros((cfg.get('out', 100),) * 2, np.uint8)); valid.append(False); continue
        ee = None if np.isnan(e[0]) else ((e[0], e[1]), (e[2], e[3]))
        src = g
        if cfg.get('pre_eq'):
            src = cv2.equalizeHist(g)
        r = rlib.crop(src, b, cfg['crop'], cfg.get('out', 100), cfg.get('iod', 0.40), cfg.get('ey', 0.36), ee)
        r = rlib.norm(r, cfg['norm'])
        if mask is not None:
            r = np.where(mask > 0, r, 128).astype(np.uint8)
        rois.append(r); valid.append(True)
    return np.array(rois), np.array(valid)


def lbph_scores(H, trials):
    sc, y = [], []
    for pid, gal, gen, imp in trials:
        probes = np.concatenate([gen, imp])
        P = H[probes]
        dmin = np.full(len(probes), np.inf, np.float32)
        for g in gal:
            dmin = np.minimum(dmin, rlib.chisq_alt(P, H[g]))
        sc.append(-dmin); y.append(np.r_[np.ones(len(gen)), np.zeros(len(imp))])
    return np.concatenate(sc), np.concatenate(y)


def run(cfg):
    t = time.time()
    d = rlib.data()
    rois, valid = build_rois(cfg)
    H = rlib.lbph_hists(rois, cfg.get('radius', 1), 8, cfg.get('grid', 8))
    dev, _ = rlib.splits()
    res = []
    for seed in (0, 1):
        tr = rlib.make_trials(valid, d['target'], dev, k=cfg.get('k', 10), seed=seed)
        s, y = lbph_scores(H, tr)
        res.append(rlib.metrics(s, y))
    m = {k: float(np.mean([r[k] for r in res])) for k in ('auc', 'eer', 'tar1', 'tar5')}
    return cfg, m, time.time() - t


if __name__ == '__main__':
    t0 = time.time()
    e = eyes_cache()
    print('eyes found: %.3f  (%.0fs)' % (float(np.mean(~np.isnan(e[:, 0]))), time.time() - t0), flush=True)
    cfgs = []
    for crop, norm in itertools.product(['box', 'boxc', 'eye'], ['eqblur', 'eq', 'clahe4', 'tt']):
        cfgs.append(dict(crop=crop, norm=norm))
    for crop in ('boxc', 'eye'):
        for iod in (0.46, 0.52):
            for norm in ('eqblur', 'tt'):
                cfgs.append(dict(crop=crop, norm=norm, iod=iod))
    cfgs.append(dict(crop='box', norm='eqblur', pre_eq=True))     # exactly what the app does today (double equalisation)
    with Pool(4) as p:
        out = p.map(run, cfgs)
    out.sort(key=lambda r: r[1]['eer'])
    for cfg, m, dt in out:
        print('%-70s %s  (%.0fs)' % (json.dumps(cfg), rlib.fmt(m), dt))
    json.dump([(c, m) for c, m, _ in out], open(os.path.join(R, 'exp1.json'), 'w'), indent=1)
