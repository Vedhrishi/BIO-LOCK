import numpy as np, itertools, rlib
z = np.load('exp3_scores.npz')
def get(seed, name): return z['s%d_%s' % (seed, name)]
def zn(s, y):  # z-normalise with impostor statistics
    m, sd = s[y == 0].mean(), s[y == 0].std()
    return (s - m) / sd
names = ['lbph_r5_g8', 'hog10_max_tt', 'tmpl_max_ttb', 'ghist_max_clahe', 'hog10_mean_tt', 'tmpl_mean_ttb']
for combo in (['lbph_r5_g8', 'hog10_max_tt'], ['lbph_r5_g8', 'tmpl_max_ttb'], ['lbph_r5_g8', 'hog10_max_tt', 'tmpl_max_ttb'],
              ['lbph_r5_g8', 'hog10_max_tt', 'ghist_max_clahe'], ['lbph_r5_g8', 'hog10_mean_tt', 'tmpl_mean_ttb'],
              ['lbph_r3_g8', 'hog10_max_tt', 'tmpl_max_ttb']):
    best = None
    steps = np.arange(0, 1.0001, 0.05)
    for w in itertools.product(steps, repeat=len(combo) - 1):
        if sum(w) > 1 + 1e-9: continue
        ws = [1 - sum(w)] + list(w)
        f = sum(wi * zn(get(0, n), z['y0']) for wi, n in zip(ws, combo))
        m = rlib.metrics(f, z['y0'])
        key = (m['eer'] - 0.3 * m['tar1'])
        if best is None or key < best[0]: best = (key, ws, m)
    ws = best[1]
    f1 = sum(wi * zn(get(1, n), z['y1']) for wi, n in zip(ws, combo))
    print(combo, 'w=', np.round(ws, 2))
    print('   seed0(fit): ', rlib.fmt(best[2]))
    print('   seed1(chk): ', rlib.fmt(rlib.metrics(f1, z['y1'])), '  | LBPH alone seed1:', rlib.fmt(rlib.metrics(get(1, combo[0]), z['y1'])))
# correlation between components (impostor / genuine) to see complementarity
for a, b in (('lbph_r5_g8', 'hog10_max_tt'), ('lbph_r5_g8', 'tmpl_max_ttb'), ('hog10_max_tt', 'tmpl_max_ttb')):
    y = z['y0']
    print(a, b, 'corr genuine %.2f impostor %.2f' % (np.corrcoef(get(0, a)[y == 1], get(0, b)[y == 1])[0, 1], np.corrcoef(get(0, a)[y == 0], get(0, b)[y == 0])[0, 1]))
