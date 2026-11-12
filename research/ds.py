"""Dataset wrappers so the same code runs on LFW (unconstrained) and Olivetti/ORL (controlled)."""
import os, numpy as np, cv2
import rlib


def lfw():
    d = rlib.data()
    eyes = np.load(os.path.join(rlib.R, 'lfw_eyes.npz'))['eyes']
    ids = np.arange(len(d['names']))
    return dict(name='lfw', images=d['images'], target=d['target'], boxes=d['boxes'], eyes=eyes,
                dev=ids[0::2], test=ids[1::2], k=10, n_imp=150, max_gen=30)


def oliv():
    d = np.load(os.path.join(rlib.R, 'oliv_prep.npz'))
    ids = np.arange(40)
    return dict(name='oliv', images=d['images'], target=d['target'], boxes=d['boxes'], eyes=d['eyes'],
                dev=ids[0::2], test=ids[1::2], k=5, n_imp=120, max_gen=5)


def rois(D, cfg):
    out, valid = [], []
    n = cfg.get('out', 100)
    for g, b, e in zip(D['images'], D['boxes'], D['eyes']):
        if b[2] <= 0:
            out.append(np.zeros((n, n), np.uint8)); valid.append(False); continue
        ee = None if np.isnan(e[0]) else ((e[0], e[1]), (e[2], e[3]))
        r = rlib.crop(g, b, cfg['crop'], n, cfg.get('iod', 0.40), cfg.get('ey', 0.36), ee)
        out.append(rlib.norm(r, cfg['norm'])); valid.append(True)
    return np.array(out), np.array(valid)


def trials(D, valid, split, seed, k=None):
    return rlib.make_trials(valid, D['target'], D[split], k=k or D['k'], n_imp=D['n_imp'], max_gen=D['max_gen'], seed=seed)
