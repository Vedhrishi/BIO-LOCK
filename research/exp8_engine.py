"""Experiment 8 (DEV halves): does the shipped engine (biolock/engine.py) reproduce the research pipeline?

Same face boxes, eye positions, trials and seeds as experiment 7 (gallery x1), but every crop, feature and
distance comes from biolock.engine.FaceEngine. Expected (HANDOFF.md, experiment 7, gallery x1):
    Olivetti  EER 2.9 %  TAR@1%FAR 96.2 %     LFW  EER 17.2 %  TAR@1%FAR 61.1 %
Also checks that the numpy chi-square nearest neighbour equals cv2 LBPHFaceRecognizer.predict."""
import os, sys, json, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib, ds
from biolock.config import load_config
from biolock.engine import FaceEngine


def engine_rois(E, D):
    out, valid = [], []
    for g, b, e in zip(D['images'], D['boxes'], D['eyes']):
        if b[2] <= 0:
            out.append(np.zeros((E.size, E.size), np.uint8)); valid.append(False); continue
        eyes = None if np.isnan(e[0]) else ((float(e[0]), float(e[1])), (float(e[2]), float(e[3])))
        out.append(E.roi(np.asarray(g), tuple(int(v) for v in b), eyes)); valid.append(True)
    return np.array(out), np.array(valid)


def run(D, seeds, cfg):
    E = FaceEngine(cfg)
    t = time.time()
    R, valid = engine_rois(E, D)
    # the research crops, for a pixel-level comparison
    R0, _ = ds.rois(D, dict(crop='eye', norm='tt', iod=0.46, radius=5, grid=6))
    same = float(np.mean([np.array_equal(a, b) for a, b, v in zip(R, R0, valid) if v]))
    ms, G, I, worst = [], [], [], 0.0
    comp = {k: [] for k in ('lbph_distance', 'hist_corr', 'tmpl_corr')}
    Y = []
    for seed in seeds:
        sc, y = [], []
        for pid, gal, gen, imp in ds.trials(D, valid, 'dev', seed):
            E.train([R[i] for i in gal], save=False)
            ref = cv2.face.LBPHFaceRecognizer_create(radius=5, neighbors=8, grid_x=6, grid_y=6)
            ref.train([R[i] for i in gal], np.zeros(len(gal), np.int32))
            pr = np.concatenate([gen, imp])
            d = []
            for j, i in enumerate(pr):
                c = E.components(R[i])
                d.append(c['lbph_distance'])
                for k in comp:
                    comp[k].append(c[k])
                if j % 25 == 0:                                   # spot-check against OpenCV's own predict
                    worst = max(worst, abs(ref.predict(R[i])[1] - c['lbph_distance']))
            lab = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
            sc.append(-np.array(d)); y.append(lab); Y.append(lab)
        s, yy = np.concatenate(sc), np.concatenate(y)
        ms.append(rlib.metrics(s, yy)); G.append(-s[yy == 1]); I.append(-s[yy == 0])
    m = {k: float(np.mean([x[k] for x in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}
    G, I, Y = np.concatenate(G), np.concatenate(I), np.concatenate(Y)
    print('%-5s %s' % (D['name'], rlib.fmt(m)))
    print('      genuine d pct[1,5,50,95,99]=%s  impostor d pct[0.1,0.5,1,5,50]=%s' % (
        np.round(np.percentile(G, [1, 5, 50, 95, 99]), 1), np.round(np.percentile(I, [0.1, 0.5, 1, 5, 50]), 1)))
    print('      crops identical to research crops: %.4f   max |numpy - cv2.predict| = %.2e   (%.0fs)' % (
        same, worst, time.time() - t), flush=True)
    return m, {k: np.array(v) for k, v in comp.items()}, Y


if __name__ == '__main__':
    cfg = load_config(path='', use_env=False)
    print('engine config: radius %d grid %d roi %d' % (cfg['lbph_radius'], cfg['lbph_grid'], cfg['roi_size']))
    res = {}
    for D, seeds in ((ds.oliv(), range(4)), (ds.lfw(), range(2))):
        m, comp, y = run(D, seeds, cfg)
        res[D['name']] = m
    json.dump(res, open(os.path.join(rlib.R, 'exp8_engine.json'), 'w'), indent=1)
