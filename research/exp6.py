"""Experiment 6 (DEV halves of LFW and Olivetti): does fusing LBPH with a histogram and a template score help?
Component definitions are the ones the app will ship: LBPH NN chi-square; HOG-descriptor correlation with the
mean enrolled descriptor; NCC with the mean enrolled face. NN variants of the last two are shown for reference."""
import os, json, itertools, sys
import numpy as np
import rlib, ds
from exp3 import unit

CFG = dict(crop=sys.argv[1] if len(sys.argv) > 1 else 'eye', norm='tt', iod=0.46, radius=5, grid=int(sys.argv[2]) if len(sys.argv) > 2 else 6)


def scores(D, split, seeds):
    R, valid = ds.rois(D, CFG)
    H = rlib.lbph_hists(R, CFG['radius'], 8, CFG['grid'])
    HOG = rlib.hog_desc(R, 10)
    T = R.reshape(len(R), -1).astype(np.float32)
    out = []
    for seed in seeds:
        tr = ds.trials(D, valid, split, seed)
        comp = {k: [] for k in ('lbph_d', 'hog_mean', 'hog_nn', 'tmpl_mean', 'tmpl_nn')}
        y = []
        for pid, gal, gen, imp in tr:
            pr = np.concatenate([gen, imp])
            y.append(np.r_[np.ones(len(gen)), np.zeros(len(imp))])
            d = np.full(len(pr), np.inf, np.float32)
            for g in gal:
                d = np.minimum(d, rlib.chisq_alt(H[pr], H[g]))
            comp['lbph_d'].append(d)
            comp['hog_mean'].append((unit(HOG[pr]) @ unit(HOG[gal].mean(0, keepdims=True)).T)[:, 0])
            comp['hog_nn'].append((unit(HOG[pr]) @ unit(HOG[gal]).T).max(1))
            comp['tmpl_mean'].append((unit(T[pr]) @ unit(T[gal].mean(0, keepdims=True)).T)[:, 0])
            comp['tmpl_nn'].append((unit(T[pr]) @ unit(T[gal]).T).max(1))
        out.append(({k: np.concatenate(v) for k, v in comp.items()}, np.concatenate(y)))
    return out


if __name__ == '__main__':
    L = scores(ds.lfw(), 'dev', range(3))
    O = scores(ds.oliv(), 'dev', range(6))
    np.savez(os.path.join(rlib.R, 'exp6_%s_g%d.npz' % (CFG['crop'], CFG['grid'])),
             **{'L%d_%s' % (i, k): v for i, (c, y) in enumerate(L) for k, v in c.items()}, **{'L%d_y' % i: y for i, (c, y) in enumerate(L)},
             **{'O%d_%s' % (i, k): v for i, (c, y) in enumerate(O) for k, v in c.items()}, **{'O%d_y' % i: y for i, (c, y) in enumerate(O)})
    print('CFG', CFG)
    for nm, S in (('lfw ', L), ('oliv', O)):
        for k in ('lbph_d', 'hog_mean', 'hog_nn', 'tmpl_mean', 'tmpl_nn'):
            ms = [rlib.metrics(-c[k] if k == 'lbph_d' else c[k], y) for c, y in S]
            print(nm, '%-10s' % k, rlib.fmt({kk: float(np.mean([m[kk] for m in ms])) for kk in ('auc', 'eer', 'tar1', 'tar5')}))
        allg = np.concatenate([c['lbph_d'][y == 1] for c, y in S]); alli = np.concatenate([c['lbph_d'][y == 0] for c, y in S])
        print(nm, 'LBPH distance  genuine pct[1,5,50,95,99] =', np.round(np.percentile(allg, [1, 5, 50, 95, 99]), 1),
              ' impostor pct[1,5,50,95,99] =', np.round(np.percentile(alli, [1, 5, 50, 95, 99]), 1))
        for k in ('hog_mean', 'tmpl_mean'):
            g = np.concatenate([c[k][y == 1] for c, y in S]); im = np.concatenate([c[k][y == 0] for c, y in S])
            print(nm, k, 'genuine pct[1,50,99] =', np.round(np.percentile(g, [1, 50, 99]), 3), ' impostor pct[1,50,99] =', np.round(np.percentile(im, [1, 50, 99]), 3))
