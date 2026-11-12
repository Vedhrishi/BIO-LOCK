"""Experiment 5 (DEV halves): final candidates, more seeds, single-frame AND multi-frame (median of m probes)."""
import os, json, time, sys
import numpy as np
from multiprocessing import Pool
import rlib, ds


def model_scores(H, D, valid, split, seed, k):
    """For every identity in the split: enrol k random images, score ALL other valid images of the split.
    Returns list of (pid, probe_idx, probe_pid, score)."""
    rng = np.random.RandomState(seed)
    ids = [p for p in D[split]]
    pool = np.where(valid & np.isin(D['target'], ids))[0]
    out = []
    for p in ids:
        own = rng.permutation(np.where(valid & (D['target'] == p))[0])
        if len(own) < k + 3:
            continue
        gal = own[:k]
        probes = np.setdiff1d(pool, gal)
        d = np.full(len(probes), np.inf, np.float32)
        for g in gal:
            d = np.minimum(d, rlib.chisq_alt(H[probes], H[g]))
        out.append((p, probes, D['target'][probes], -d))
    return out


def single(ms, rng, max_gen, n_imp):
    s, y = [], []
    for p, probes, pp, sc in ms:
        g = np.where(pp == p)[0]
        g = rng.permutation(g)[:max_gen]
        others = np.unique(pp[pp != p])
        im = np.array([rng.choice(np.where(pp == q)[0]) for q in rng.choice(others, size=n_imp, replace=True)])
        s += [sc[g], sc[im]]; y += [np.ones(len(g)), np.zeros(len(im))]
    return np.concatenate(s), np.concatenate(y)


def multi(ms, rng, m, reps=10, max_imp_ids=30):
    s, y = [], []
    for p, probes, pp, sc in ms:
        g = np.where(pp == p)[0]
        if len(g) >= m:
            for _ in range(reps):
                s.append(np.median(sc[rng.choice(g, m, replace=False)])); y.append(1)
        others = np.unique(pp[pp != p])
        for q in rng.permutation(others)[:max_imp_ids]:
            qi = np.where(pp == q)[0]
            if len(qi) >= m:
                s.append(np.median(sc[rng.choice(qi, m, replace=False)])); y.append(0)
    return np.array(s), np.array(y)


def run(job):
    dsname, cfg, split = job
    D = ds.lfw() if dsname == 'lfw' else ds.oliv()
    R, valid = ds.rois(D, cfg)
    H = rlib.lbph_hists(R, cfg.get('radius', 1), 8, cfg.get('grid', 8))
    res = {}
    seeds = range(3) if dsname == 'lfw' else range(6)
    acc = {}
    for seed in seeds:
        ms = model_scores(H, D, valid, split, seed, cfg.get('k', D['k']))
        rng = np.random.RandomState(100 + seed)
        s, y = single(ms, rng, D['max_gen'], D['n_imp'])
        acc.setdefault('m1', []).append(rlib.metrics(s, y))
        for m in (3, 5):
            s, y = multi(ms, rng, m)
            if y.sum() > 0:
                acc.setdefault('m%d' % m, []).append(rlib.metrics(s, y))
    for k, v in acc.items():
        res[k] = {kk: float(np.mean([x[kk] for x in v])) for kk in ('auc', 'eer', 'tar1', 'tar5', 'n_gen', 'n_imp')}
    return dsname, cfg, res


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    cands = {
        'baseline(app today)': dict(crop='box', norm='eqblur', radius=1, grid=8),
        'eye  r5 g6': dict(crop='eye', norm='tt', iod=0.46, radius=5, grid=6),
        'boxc r5 g6': dict(crop='boxc', norm='tt', iod=0.46, radius=5, grid=6),
        'eye  r5 g8': dict(crop='eye', norm='tt', iod=0.46, radius=5, grid=8),
        'boxc r5 g8': dict(crop='boxc', norm='tt', iod=0.46, radius=5, grid=8),
        'eye  r4 g6': dict(crop='eye', norm='tt', iod=0.46, radius=4, grid=6),
        'boxc r4 g6': dict(crop='boxc', norm='tt', iod=0.46, radius=4, grid=6),
    }
    jobs = [(n, c, split) for c in cands.values() for n in ('lfw', 'oliv')]
    with Pool(3) as p:
        out = p.map(run, jobs)
    inv = {json.dumps(v): k for k, v in cands.items()}
    tab = {}
    for dsname, cfg, res in out:
        tab.setdefault(inv[json.dumps(cfg)], {})[dsname] = res
    for name in cands:
        for dsname in ('lfw', 'oliv'):
            r = tab[name][dsname]
            print('%-20s %-5s' % (name, dsname), ' | '.join('%s: EER %5.1f%% TAR@1%%FAR %5.1f%% AUC %.3f' % (k, 100 * r[k]['eer'], 100 * r[k]['tar1'], r[k]['auc']) for k in ('m1', 'm3', 'm5') if k in r))
    json.dump(tab, open(os.path.join(rlib.R, 'exp5_%s.json' % split), 'w'), indent=1)
