"""Experiment 7 (DEV halves): emulate a live enrolment (many near-duplicate frames) by jittering each enrolment photo.
Question: how do gallery size / jitter change accuracy and, importantly, the impostor distance tail used for the threshold?"""
import os, sys, json, math
import numpy as np, cv2
import rlib, ds

CFG = dict(crop='eye', norm='tt', iod=0.46, radius=5, grid=6)


def jitter_eyes(eyes, rng, rot=4.0, sc=0.04, sh=0.06):
    (lx, ly), (rx, ry) = eyes
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    iod = math.hypot(rx - lx, ry - ly)
    a = math.radians(rng.uniform(-rot, rot)); s = 1 + rng.uniform(-sc, sc)
    tx, ty = rng.uniform(-sh, sh) * iod, rng.uniform(-sh, sh) * iod
    def tr(x, y):
        dx, dy = (x - cx) * s, (y - cy) * s
        return (cx + dx * math.cos(a) - dy * math.sin(a) + tx, cy + dx * math.sin(a) + dy * math.cos(a) + ty)
    return tr(lx, ly), tr(rx, ry)


def crop_of(D, i, rng=None):
    g, b, e = D['images'][i], D['boxes'][i], D['eyes'][i]
    eyes = rlib.box_eyes(b) if np.isnan(e[0]) else ((e[0], e[1]), (e[2], e[3]))
    rotate = not np.isnan(e[0])
    if rng is not None:
        eyes = jitter_eyes(eyes, rng); rotate = True
    return rlib.norm(rlib.warp_eyes(np.asarray(g), eyes, 100, CFG['iod'], 0.36, rotate=rotate), 'tt')


def run(D, variants, seeds):
    R, valid = ds.rois(D, CFG)
    H = rlib.lbph_hists(R, CFG['radius'], 8, CFG['grid'])
    out = {}
    for v in variants:
        ms, G, I = [], [], []
        for seed in seeds:
            rng = np.random.RandomState(1000 + seed)
            sc, y = [], []
            for pid, gal, gen, imp in ds.trials(D, valid, 'dev', seed):
                pr = np.concatenate([gen, imp])
                d = np.full(len(pr), np.inf, np.float32)
                for g in gal:
                    d = np.minimum(d, rlib.chisq_alt(H[pr], H[g]))
                if v > 1:
                    aug = [crop_of(D, g, rng) for g in gal for _ in range(v - 1)]
                    Ha = rlib.lbph_hists(np.array(aug), CFG['radius'], 8, CFG['grid'])
                    for h in Ha:
                        d = np.minimum(d, rlib.chisq_alt(H[pr], h))
                sc.append(-d); y.append(np.r_[np.ones(len(gen)), np.zeros(len(imp))])
            s, yy = np.concatenate(sc), np.concatenate(y)
            ms.append(rlib.metrics(s, yy)); G.append(-s[yy == 1]); I.append(-s[yy == 0])
        m = {k: float(np.mean([x[k] for x in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}
        G, I = np.concatenate(G), np.concatenate(I)
        out[v] = (m, np.percentile(G, [1, 5, 50, 95, 99]), np.percentile(I, [0.1, 0.5, 1, 5, 50]))
        print('%s gallery x%-2d  %s | genuine d pct[1,5,50,95,99]=%s | impostor d pct[0.1,0.5,1,5,50]=%s' % (
            D['name'], v, rlib.fmt(m), np.round(out[v][1], 1), np.round(out[v][2], 1)), flush=True)
    return out


if __name__ == '__main__':
    run(ds.oliv(), (1, 9, 18), range(4))
    run(ds.lfw(), (1, 9), range(2))
