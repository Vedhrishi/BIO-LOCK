"""Research library: alignment / normalisation / features / protocol / metrics for the matcher search.
Selection is done on the DEV identities only (even index in sorted-name order)."""
import os, time, math
import numpy as np
import cv2

R = os.path.dirname(os.path.abspath(__file__))
_D = None


def data():
    global _D
    if _D is None:
        d = np.load(os.path.join(R, 'lfw_meta.npz'))
        _D = {k: d[k] for k in d.files}
        _D['images'] = np.load(os.path.join(R, 'lfw_images.npy'), mmap_mode='r')
    return _D


EYE_CC = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# mean relative eye positions inside a Haar frontal-face box (measured on LFW, see eyes.py)
BOX_EYE_L = (0.312, 0.400)
BOX_EYE_R = (0.668, 0.392)


def find_eyes(gray, box, cc=EYE_CC, sf=1.05, mn=4):
    x, y, w, h = [int(v) for v in box]
    y0, y1 = y + int(0.15 * h), y + int(0.62 * h)
    band = gray[max(y0, 0):y1, max(x, 0):x + w]
    if band.size == 0:
        return None
    mins = max(8, int(0.12 * w))
    maxs = max(mins + 1, int(0.42 * w))
    ev = cc.detectMultiScale(band, sf, mn, minSize=(mins, mins), maxSize=(maxs, maxs))
    if len(ev) < 2:
        return None
    cands = [(ex + ew / 2.0, ey + eh / 2.0 + (y0 - y), ew) for ex, ey, ew, eh in ev]
    L = [c for c in cands if c[0] < 0.5 * w]
    Rr = [c for c in cands if c[0] >= 0.5 * w]
    if not L or not Rr:
        return None
    exp_l, exp_r = (BOX_EYE_L[0] * w, BOX_EYE_L[1] * h), (BOX_EYE_R[0] * w, BOX_EYE_R[1] * h)
    l = min(L, key=lambda c: (c[0] - exp_l[0]) ** 2 + (c[1] - exp_l[1]) ** 2)
    r = min(Rr, key=lambda c: (c[0] - exp_r[0]) ** 2 + (c[1] - exp_r[1]) ** 2)
    iod = math.hypot(r[0] - l[0], r[1] - l[1])
    ang = math.degrees(math.atan2(r[1] - l[1], r[0] - l[0]))
    if not (0.25 * w <= iod <= 0.60 * w) or abs(ang) > 22:
        return None
    return (x + l[0], y + l[1]), (x + r[0], y + r[1])


def box_eyes(box):
    x, y, w, h = box
    return (x + BOX_EYE_L[0] * w, y + BOX_EYE_L[1] * h), (x + BOX_EYE_R[0] * w, y + BOX_EYE_R[1] * h)


def warp_eyes(gray, eyes, out=100, iod=0.40, ey=0.36, rotate=True):
    """Similarity-warp so the eyes land at ((0.5-iod/2)*out, ey*out) and ((0.5+iod/2)*out, ey*out)."""
    (lx, ly), (rx, ry) = eyes
    dx, dy = rx - lx, ry - ly
    dist = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx)) if rotate else 0.0
    s = (iod * out) / max(dist, 1e-6)
    cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[0, 2] += 0.5 * out - cx
    M[1, 2] += ey * out - cy
    return cv2.warpAffine(gray, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def crop(gray, box, mode, out=100, iod=0.40, ey=0.36, eyes=None):
    if mode == 'box':
        x, y, w, h = box
        return cv2.resize(gray[y:y + h, x:x + w], (out, out), interpolation=cv2.INTER_AREA)
    if mode == 'boxc':
        return warp_eyes(gray, box_eyes(box), out, iod, ey, rotate=False)
    if mode == 'eye':
        return warp_eyes(gray, eyes if eyes is not None else box_eyes(box), out, iod, ey, rotate=eyes is not None)
    raise ValueError(mode)


# ---------------- normalisation ----------------
_CLAHE = {}


def norm(img, mode):
    if mode == 'eqblur':
        return cv2.GaussianBlur(cv2.equalizeHist(img), (3, 3), 0)
    if mode == 'eq':
        return cv2.equalizeHist(img)
    if mode.startswith('clahe'):
        t = int(mode[5:] or 4)
        c = _CLAHE.setdefault(t, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(t, t)))
        return c.apply(img)
    if mode == 'tt':
        return tan_triggs(img)
    if mode == 'ttb':
        return cv2.GaussianBlur(tan_triggs(img), (3, 3), 0)
    if mode == 'none':
        return img
    raise ValueError(mode)


def tan_triggs(img, gamma=0.2, s0=1.0, s1=2.0, alpha=0.1, tau=10.0):
    x = np.power(img.astype(np.float32) / 255.0 + 1e-6, gamma)
    x = cv2.GaussianBlur(x, (0, 0), s0) - cv2.GaussianBlur(x, (0, 0), s1)
    x = x / (np.mean(np.abs(x) ** alpha) ** (1.0 / alpha) + 1e-8)
    x = x / (np.mean(np.minimum(np.abs(x), tau) ** alpha) ** (1.0 / alpha) + 1e-8)
    x = tau * np.tanh(x / tau)
    lo, hi = -tau, tau
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def ellipse_mask(out=100, ax=0.48, ay=0.58, cy=0.5):
    m = np.zeros((out, out), np.uint8)
    cv2.ellipse(m, (out // 2, int(cy * out)), (int(ax * out), int(ay * out)), 0, 0, 360, 255, -1)
    return m


# ---------------- features ----------------
def lbph_hists(rois, radius=1, neighbors=8, grid=8):
    rec = cv2.face.LBPHFaceRecognizer_create(radius=radius, neighbors=neighbors, grid_x=grid, grid_y=grid)
    rec.train(list(rois), np.arange(len(rois), dtype=np.int32))
    return np.vstack(rec.getHistograms()).astype(np.float32)


def chisq_alt(P, g):
    """OpenCV HISTCMP_CHISQR_ALT between every row of P and vector g."""
    out = np.empty(len(P), np.float32)
    for a in range(0, len(P), 96):                      # chunked to keep temporaries small
        p = P[a:a + 96]
        d = p - g
        s = p + g
        np.multiply(d, d, out=d)
        np.divide(d, s, out=d, where=s > 0)
        d[s <= 0] = 0
        out[a:a + 96] = 2.0 * d.sum(axis=1)
    return out


def hog_desc(rois, cell=10, bins=9, out=100):
    block = 2 * cell
    hog = cv2.HOGDescriptor((out, out), (block, block), (cell, cell), (cell, cell), bins)
    return np.vstack([hog.compute(r).reshape(1, -1) for r in rois]).astype(np.float32)


def spatial_gray_hist(rois, grid=4, bins=16):
    N, H, W = rois.shape
    feats = np.zeros((N, grid * grid * bins), np.float32)
    q = (rois.astype(np.int32) * bins) // 256
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    k = 0
    for i in range(grid):
        for j in range(grid):
            cellq = q[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].reshape(N, -1)
            for b in range(bins):
                feats[:, k * bins + b] = (cellq == b).mean(axis=1)
            k += 1
    return feats


def rowcorr(P, g):
    """Pearson correlation between every row of P and vector g."""
    Pc = P - P.mean(axis=1, keepdims=True)
    gc = g - g.mean()
    return (Pc @ gc) / (np.linalg.norm(Pc, axis=1) * np.linalg.norm(gc) + 1e-9)


# ---------------- protocol ----------------
def splits():
    d = data()
    ids = np.arange(len(d['names']))
    return ids[0::2], ids[1::2]          # dev, test (sorted-name order, alternating)


def make_trials(valid, target, ids, k=10, n_imp=150, max_gen=30, seed=0):
    """Returns list of (pid, gallery_idx[k], genuine_idx[], impostor_idx[]) for identities in `ids`."""
    rng = np.random.RandomState(seed)
    ids = list(ids)
    by = {p: np.where((target == p) & valid)[0] for p in ids}
    out = []
    for p in ids:
        idx = rng.permutation(by[p])
        if len(idx) < k + 5:
            continue
        gal, gen = idx[:k], idx[k:k + max_gen]
        others = [q for q in ids if q != p and len(by[q]) > 0]
        imp = np.array([rng.choice(by[q]) for q in rng.choice(others, size=n_imp, replace=True)])
        out.append((p, gal, gen, imp))
    return out


def metrics(scores, y):
    """scores: higher = more genuine. Returns dict(auc, eer, tar@far1, tar@far5, thr_eer)."""
    order = np.argsort(-scores, kind='stable')
    s, g = scores[order], y[order].astype(np.float64)
    P, N = g.sum(), (1 - g).sum()
    tpr = np.concatenate([[0.0], np.cumsum(g) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - g) / N])
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    trap = getattr(np, 'trapezoid', None) or np.trapz

    def tar_at(far):
        j = np.searchsorted(fpr, far, side='right') - 1
        return float(tpr[max(j, 0)])
    return dict(auc=float(trap(tpr, fpr)), eer=float((fpr[i] + 1 - tpr[i]) / 2), thr_eer=float(s[max(i - 1, 0)]),
                tar1=tar_at(0.01), tar5=tar_at(0.05), n_gen=int(P), n_imp=int(N))


def fmt(m):
    return 'AUC %.3f  EER %5.1f%%  TAR@1%%FAR %5.1f%%  TAR@5%%FAR %5.1f%%' % (m['auc'], 100 * m['eer'], 100 * m['tar1'], 100 * m['tar5'])
