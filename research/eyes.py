"""Eye detection study on LFW: detection rate + relative eye geometry inside the Haar face box."""
import os, time
import numpy as np
import cv2

R = os.path.dirname(os.path.abspath(__file__))
import rlib
d = rlib.data()
imgs, boxes = d['images'], d['boxes']

CC = {n: cv2.CascadeClassifier(cv2.data.haarcascades + f) for n, f in {
    'eye': 'haarcascade_eye.xml', 'glasses': 'haarcascade_eye_tree_eyeglasses.xml',
    'left2': 'haarcascade_lefteye_2splits.xml', 'right2': 'haarcascade_righteye_2splits.xml'}.items()}


def find_eyes(gray, box, cc, sf=1.05, mn=4):
    x, y, w, h = [int(v) for v in box]
    y0, y1 = y + int(0.15 * h), y + int(0.62 * h)
    band = gray[y0:y1, x:x + w]
    if band.size == 0:
        return None
    mins = max(8, int(0.12 * w))
    maxs = max(mins + 1, int(0.42 * w))
    ev = cc.detectMultiScale(band, sf, mn, minSize=(mins, mins), maxSize=(maxs, maxs))
    if len(ev) < 2:
        return None
    cands = [(ex + ew / 2.0, ey + eh / 2.0 + (y0 - y), ew) for ex, ey, ew, eh in ev]   # face-box coords
    L = [c for c in cands if c[0] < 0.5 * w]
    Rr = [c for c in cands if c[0] >= 0.5 * w]
    if not L or not Rr:
        return None
    exp_l, exp_r = (0.30 * w, 0.38 * h), (0.70 * w, 0.38 * h)
    l = min(L, key=lambda c: (c[0] - exp_l[0]) ** 2 + (c[1] - exp_l[1]) ** 2)
    r = min(Rr, key=lambda c: (c[0] - exp_r[0]) ** 2 + (c[1] - exp_r[1]) ** 2)
    iod = np.hypot(r[0] - l[0], r[1] - l[1])
    ang = np.degrees(np.arctan2(r[1] - l[1], r[0] - l[0]))
    if not (0.25 * w <= iod <= 0.60 * w) or abs(ang) > 22:
        return None
    return (x + l[0], y + l[1]), (x + r[0], y + r[1])


for name in ('eye', 'glasses'):
    t = time.time()
    rel, n = [], 0
    for g, b in zip(imgs[::3], boxes[::3]):
        if b[2] <= 0:
            continue
        n += 1
        e = find_eyes(cv2.equalizeHist(g), b, CC[name])
        if e is None:
            continue
        (lx, ly), (rx, ry) = e
        x, y, w, h = b
        rel.append(((lx - x) / w, (ly - y) / h, (rx - x) / w, (ry - y) / h))
    rel = np.array(rel)
    print(name, 'rate %.3f' % (len(rel) / n), 'ms/img %.1f' % ((time.time() - t) * 1000 / n))
    print('  mean rel L(%.3f,%.3f) R(%.3f,%.3f)  std %s' % (*rel.mean(0), np.round(rel.std(0), 3)))
    print('  IOD/w mean %.3f std %.3f' % ((rel[:, 2] - rel[:, 0]).mean(), (rel[:, 2] - rel[:, 0]).std()))
