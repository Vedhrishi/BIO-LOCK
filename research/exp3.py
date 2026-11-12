"""Experiment 3: other components (template, HOG, spatial histograms) + radius 4/5 + fusion. DEV split only."""
import os, json, itertools, time, sys
import numpy as np
import cv2
import rlib, exp1


def unit(X):
    X = X.astype(np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def comp_scores(trials, feats):
    """feats: dict name -> (kind, array). Returns dict name -> scores (higher = more similar), and y."""
    out = {k: [] for k in feats}
    y = []
    for pid, gal, gen, imp in trials:
        probes = np.concatenate([gen, imp])
        y.append(np.r_[np.ones(len(gen)), np.zeros(len(imp))])
        for name, (kind, F) in feats.items():
            if kind == 'chi_nn':
                d = np.full(len(probes), np.inf, np.float32)
                for g in gal:
                    d = np.minimum(d, rlib.chisq_alt(F[probes], F[g]))
                out[name].append(-d)
            elif kind == 'chi_k3':
                D = np.stack([rlib.chisq_alt(F[probes], F[g]) for g in gal], axis=1)
                out[name].append(-np.sort(D, axis=1)[:, :3].mean(axis=1))
            elif kind == 'corr_mean':
                m = F[gal].mean(axis=0, keepdims=True)
                out[name].append((unit(F[probes]) @ unit(m).T)[:, 0])
            elif kind == 'corr_max':
                out[name].append((unit(F[probes]) @ unit(F[gal]).T).max(axis=1))
            elif kind == 'corr_k3':
                S = unit(F[probes]) @ unit(F[gal]).T
                out[name].append(np.sort(S, axis=1)[:, -3:].mean(axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(y)


if __name__ == '__main__':
    t0 = time.time()
    d = rlib.data()
    dev, _ = rlib.splits()
    base = dict(crop='eye', iod=0.46)
    rois_tt, valid = exp1.build_rois(dict(base, norm='tt'))
    rois_raw, _ = exp1.build_rois(dict(base, norm='none'))
    rois_cl, _ = exp1.build_rois(dict(base, norm='clahe4'))
    rois_ttb, _ = exp1.build_rois(dict(base, norm='ttb'))
    print('rois built %.0fs' % (time.time() - t0), flush=True)
    feats = {}
    for r in (3, 4, 5):
        feats['lbph_r%d_g8' % r] = ('chi_nn', rlib.lbph_hists(rois_tt, r, 8, 8))
    feats['lbph_r3_g8_k3'] = ('chi_k3', feats['lbph_r3_g8'][1])
    flat = lambda R: R.reshape(len(R), -1).astype(np.float32)
    for nm, R in (('tt', rois_tt), ('ttb', rois_ttb), ('raw', rois_raw), ('clahe', rois_cl)):
        feats['tmpl_mean_' + nm] = ('corr_mean', flat(R))
        feats['tmpl_max_' + nm] = ('corr_max', flat(R))
    for nm, R in (('tt', rois_tt), ('raw', rois_raw), ('clahe', rois_cl)):
        for cell in (10,):
            H = rlib.hog_desc(R, cell=cell)
            feats['hog%d_mean_%s' % (cell, nm)] = ('corr_mean', H)
            feats['hog%d_max_%s' % (cell, nm)] = ('corr_max', H)
    for nm, R in (('clahe', rois_cl), ('raw', rois_raw)):
        G = rlib.spatial_gray_hist(R, 4, 16)
        feats['ghist_mean_' + nm] = ('corr_mean', G)
        feats['ghist_max_' + nm] = ('corr_max', G)
    print('features built %.0fs' % (time.time() - t0), flush=True)
    allsc = []
    for seed in (0, 1):
        tr = rlib.make_trials(valid, d['target'], dev, k=10, seed=seed)
        sc, y = comp_scores(tr, feats)
        allsc.append((sc, y))
        print('seed', seed, 'scored %.0fs' % (time.time() - t0), flush=True)
    res = {}
    for name in feats:
        ms = [rlib.metrics(sc[name], y) for sc, y in allsc]
        res[name] = {k: float(np.mean([m[k] for m in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}
    for name, m in sorted(res.items(), key=lambda kv: kv[1]['eer']):
        print('%-22s %s' % (name, rlib.fmt(m)))
    np.savez(os.path.join(rlib.R, 'exp3_scores.npz'), y0=allsc[0][1], y1=allsc[1][1],
             **{'s0_' + k: v for k, v in allsc[0][0].items()}, **{'s1_' + k: v for k, v in allsc[1][0].items()})
