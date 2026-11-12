"""Weight search on BOTH dev halves with fixed score calibration (the app's 0-100 mapping)."""
import numpy as np, itertools, json, rlib
z = np.load('exp6_eye_g6.npz')
def runs(prefix, n): return [({k: z['%s%d_%s' % (prefix, i, k)] for k in ('lbph_d', 'hog_mean', 'hog_nn', 'tmpl_mean', 'tmpl_nn')}, z['%s%d_y' % (prefix, i)]) for i in range(n)]
L, O = runs('L', 3), runs('O', 6)
# calibration constants: pooled dev percentiles (genuine 1st pct -> 100, impostor 99th pct -> 0 for distance; reverse for correlations)
def pool(k, lab): return np.concatenate([c[k][y == lab] for S in (L, O) for c, y in S])
cal = {'lbph_d_lo': float(np.percentile(pool('lbph_d', 1), 1)), 'lbph_d_hi': float(np.percentile(pool('lbph_d', 0), 99)),
       'hist_c_lo': float(np.percentile(pool('hog_mean', 0), 1)), 'hist_c_hi': float(np.percentile(pool('hog_mean', 1), 99)),
       'tmpl_c_lo': float(np.percentile(pool('tmpl_mean', 0), 1)), 'tmpl_c_hi': float(np.percentile(pool('tmpl_mean', 1), 99))}
print('calibration', {k: round(v, 3) for k, v in cal.items()})
s01 = lambda x, lo, hi: 100 * np.clip((x - lo) / (hi - lo), 0, 1)
def comps(c, hk='hog_mean', tk='tmpl_mean'):
    return (s01(cal['lbph_d_hi'] - c['lbph_d'], 0, cal['lbph_d_hi'] - cal['lbph_d_lo']), s01(c[hk], cal['hist_c_lo'], cal['hist_c_hi']), s01(c[tk], cal['tmpl_c_lo'], cal['tmpl_c_hi']))
def evalw(w, S, **kw):
    ms = []
    for c, y in S:
        a, b, t = comps(c, **kw)
        ms.append(rlib.metrics(w[0] * a + w[1] * b + w[2] * t, y))
    return {k: float(np.mean([m[k] for m in ms])) for k in ('auc', 'eer', 'tar1', 'tar5')}
grid = [(round(1 - i / 20 - j / 20, 2), i / 20, j / 20) for i in range(21) for j in range(21) if i + j <= 20]
rows = []
for w in grid:
    l, o = evalw(w, L), evalw(w, O)
    rows.append((w, l, o, (l['eer'] + o['eer']) / 2, (l['tar1'] + o['tar1']) / 2))
print('\nTop by mean EER over the two dev sets:')
for w, l, o, me, mt in sorted(rows, key=lambda r: r[3])[:8]:
    print('  w(lbph,hist,tmpl)=%s  LFW EER %.1f%% TAR@1 %.1f%% | Oliv EER %.1f%% TAR@1 %.1f%% | mean EER %.2f%%' % (w, 100 * l['eer'], 100 * l['tar1'], 100 * o['eer'], 100 * o['tar1'], 100 * me))
print('Top by mean TAR@1%FAR:')
for w, l, o, me, mt in sorted(rows, key=lambda r: -r[4])[:8]:
    print('  w=%s  LFW EER %.1f%% TAR@1 %.1f%% | Oliv EER %.1f%% TAR@1 %.1f%% | mean TAR@1 %.2f%%' % (w, 100 * l['eer'], 100 * l['tar1'], 100 * o['eer'], 100 * o['tar1'], 100 * mt))
for w in ((1.0, 0.0, 0.0), (0.5, 0.3, 0.2), (0.45, 0.2, 0.35), (0.8, 0.1, 0.1), (0.7, 0.2, 0.1)):
    l, o = evalw(w, L), evalw(w, O)
    print('  ref w=%s  LFW EER %.1f%% TAR@1 %.1f%% | Oliv EER %.1f%% TAR@1 %.1f%%' % (w, 100 * l['eer'], 100 * l['tar1'], 100 * o['eer'], 100 * o['tar1']))
# NN variants of hist/template for reference
rows2 = []
for w in grid:
    l, o = evalw(w, L, hk='hog_nn', tk='tmpl_nn'), evalw(w, O, hk='hog_nn', tk='tmpl_nn')
    rows2.append((w, l, o, (l['eer'] + o['eer']) / 2))
w, l, o, me = sorted(rows2, key=lambda r: r[3])[0]
print('NN variants best: w=%s LFW EER %.1f%% TAR@1 %.1f%% | Oliv EER %.1f%% TAR@1 %.1f%%' % (w, 100 * l['eer'], 100 * l['tar1'], 100 * o['eer'], 100 * o['tar1']))
json.dump(cal, open('calibration.json', 'w'), indent=1)
