"""Research prep: load LFW (>=20 images/person) as uint8 gray, detect face + eyes once, cache."""
import os, sys, time
import numpy as np
import cv2

R = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(R, 'data')                       # dataset cache (git-ignored)
LFW = os.path.join(DATA, 'lfw_home', 'lfw_funneled')


def ensure_lfw():
    """Download LFW through scikit-learn (about 200 MB) if it is not cached yet."""
    if not os.path.isdir(LFW):
        from sklearn.datasets import fetch_lfw_people
        fetch_lfw_people(data_home=DATA, min_faces_per_person=20, resize=0.3)


def imread_gray(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    t0 = time.time()
    ensure_lfw()
    people = sorted(d for d in os.listdir(LFW) if os.path.isdir(os.path.join(LFW, d)))
    sel = []
    for p in people:
        files = sorted(f for f in os.listdir(os.path.join(LFW, p)) if f.lower().endswith('.jpg'))
        if len(files) >= 20:
            sel.append((p, files))
    print('identities', len(sel), 'images', sum(len(f) for _, f in sel))
    face_cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imgs, target, boxes, ndet = [], [], [], []
    for pid, (p, files) in enumerate(sel):
        for f in files:
            g = imread_gray(os.path.join(LFW, p, f))
            imgs.append(g)
            target.append(pid)
            eq = cv2.equalizeHist(g)
            det = face_cc.detectMultiScale(eq, 1.1, 5, minSize=(40, 40))
            ndet.append(len(det))
            if len(det) == 0:
                boxes.append((-1, -1, -1, -1))
                continue
            # LFW images are centred on the subject: prefer the detection nearest the centre, weighted by size
            c = np.array([125.0, 125.0])
            best = max(det, key=lambda b: b[2] * b[3] / (1.0 + np.linalg.norm(np.array([b[0] + b[2] / 2, b[1] + b[3] / 2]) - c) / 25.0))
            boxes.append(tuple(int(v) for v in best))
    imgs = np.array(imgs, dtype=np.uint8)
    np.save(os.path.join(R, 'lfw_images.npy'), imgs)
    np.savez(os.path.join(R, 'lfw_meta.npz'), target=np.array(target), boxes=np.array(boxes), ndet=np.array(ndet),
             names=np.array([p for p, _ in sel]))
    nd = np.array(ndet)
    print('detections: 0=%d 1=%d 2+=%d' % ((nd == 0).sum(), (nd == 1).sum(), (nd > 1).sum()), 'time %.0fs' % (time.time() - t0))
    b = np.array(boxes)[nd > 0]
    print('box width mean %.1f std %.1f  cx mean %.1f cy mean %.1f' % (b[:, 2].mean(), b[:, 2].std(), (b[:, 0] + b[:, 2] / 2).mean(), (b[:, 1] + b[:, 3] / 2).mean()))


if __name__ == '__main__':
    main()
