"""Experiment 2: LBPH parameters with eye alignment + Tan-Triggs (DEV split)."""
import os, json, itertools, time
import numpy as np
from multiprocessing import Pool
import rlib, exp1

if __name__ == '__main__':
    cfgs = []
    for radius, grid in itertools.product((1, 2, 3), (6, 8, 10)):
        cfgs.append(dict(crop='eye', norm='tt', iod=0.46, radius=radius, grid=grid))
    for radius, grid in ((2, 8), (2, 10), (3, 10)):
        cfgs.append(dict(crop='eye', norm='tt', iod=0.46, radius=radius, grid=grid, out=128))
    for norm in ('ttb', 'none'):
        cfgs.append(dict(crop='eye', norm=norm, iod=0.46))
    cfgs.append(dict(crop='eye', norm='tt', iod=0.46, mask=True))
    cfgs.append(dict(crop='eye', norm='tt', iod=0.46, ey=0.40))
    cfgs.append(dict(crop='eye', norm='tt', iod=0.46, ey=0.32))
    with Pool(3) as p:
        out = p.map(exp1.run, cfgs)
    out.sort(key=lambda r: r[1]['eer'])
    for cfg, m, dt in out:
        print('%-86s %s  (%.0fs)' % (json.dumps(cfg), rlib.fmt(m), dt))
    json.dump([(c, m) for c, m, _ in out], open(os.path.join(rlib.R, 'exp2.json'), 'w'), indent=1)
