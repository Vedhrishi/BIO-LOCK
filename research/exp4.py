"""Experiment 4 (DEV halves of both datasets): radius/grid, gallery size, mirror augmentation, crop choices."""
import os, json, itertools, time, sys
import numpy as np, cv2
from multiprocessing import Pool
import rlib, ds


def lbph_nn(H, trials, Hm=None):
    sc, y = [], []
    for pid, gal, gen, imp in trials:
        probes = np.concatenate([gen, imp])
        d = np.full(len(probes), np.inf, np.float32)
        for g in gal:
            d = np.minimum(d, rlib.chisq_alt(H[probes], H[g]))
            if Hm is not None:
                d = np.minimum(d, rlib.chisq_alt(H[probes], Hm[g]))
        sc.append(-d); y.append(np.r_[np.ones(len(gen)), np.zeros(len(imp))])
    return np.concatenate(sc), np.concatenate(y)


def run(job):
    dsname, cfg = job
    D = ds.lfw() if dsname == 'lfw' else ds.oliv()
    R, valid = ds.rois(D, cfg)
    H = rlib.lbph_hists(R, cfg.get('radius', 1), 8, cfg.get('grid', 8))
    Hm = rlib.lbph_hists(R[:, :, ::-1].copy(), cfg.get('radius', 1), 8, cfg.get('grid', 8)) if cfg.get('mirror') else None
    ms = []
    for seed in range(2 if dsname == 'lfw' else 4):
        tr = ds.trials(D, valid, 'dev', seed, k=cfg.get('k'))
        s, y = lbph_nn(H, tr, Hm)
        ms.append(rlib.metrics(s, y))
    return dsname, cfg, {k: float(np.mean([m[k] for m in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}


if __name__ == '__main__':
    base = dict(crop='eye', norm='tt', iod=0.46)
    cfgs = [dict(base, radius=r, grid=g) for r, g in ((3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (5, 7), (5, 6), (6, 6), (4, 6))]
    cfgs += [dict(base, radius=5, grid=8, mirror=True)]
    cfgs += [dict(base, radius=5, grid=8, iod=i) for i in (0.40, 0.52, 0.58)]
    cfgs += [dict(crop='boxc', norm='tt', iod=0.46, radius=5, grid=8), dict(crop='box', norm='tt', radius=5, grid=8),
             dict(crop='box', norm='eqblur', radius=1, grid=8), dict(crop='eye', norm='eqblur', iod=0.46, radius=5, grid=8),
             dict(crop='eye', norm='clahe4', iod=0.46, radius=5, grid=8)]
    jobs = [('lfw', c) for c in cfgs] + [('oliv', c) for c in cfgs]
    jobs += [('lfw', dict(base, radius=5, grid=8, k=20)), ('lfw', dict(base, radius=5, grid=8, k=5)), ('oliv', dict(base, radius=5, grid=8, k=3))]
    with Pool(3) as p:
        out = p.map(run, jobs)
    res = {}
    for dsname, cfg, m in out:
        res.setdefault(json.dumps(cfg), {})[dsname] = m
    rows = sorted(res.items(), key=lambda kv: kv[1].get('lfw', {'eer': 9})['eer'])
    for k, v in rows:
        print('%-78s' % k, ' | '.join('%s EER %5.1f%% TAR@1 %5.1f%% AUC %.3f' % (n, 100 * v[n]['eer'], 100 * v[n]['tar1'], v[n]['auc']) for n in ('lfw', 'oliv') if n in v))
    json.dump(rows, open(os.path.join(rlib.R, 'exp4.json'), 'w'), indent=1)
