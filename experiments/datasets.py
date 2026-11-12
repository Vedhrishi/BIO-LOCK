"""Public face datasets for the benchmarks, shown to the engine the way the app sees a camera frame.

Two datasets (downloaded through scikit-learn on first use, cached in research/data/):

    LFW        Labeled Faces in the Wild, the 62 people with at least 20 photographs (3,023 images).
               Unconstrained press photographs: different days, light, pose, expression.
    Olivetti   AT&T / ORL, 40 people x 10 photographs taken in one laboratory. Controlled.

Each photograph is scaled so the face is about 190 px wide and placed in a 640x480 frame
(biolock.sim.render); from there on the benchmark runs the application's own code: frame
preparation, Haar face detection, eye detection, alignment, normalisation, matching.

Identities are split by position in sorted order: even index -> development half (used to choose
parameters), odd index -> test half (used only for the reported numbers).
"""

import os
import sys
import math

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from biolock import sim                      # noqa: E402
from biolock import session as S             # noqa: E402

DATA_HOME = os.path.join(ROOT, 'research', 'data')


def load_lfw(data_home=DATA_HOME, min_faces=20):
    lfw_dir = os.path.join(data_home, 'lfw_home', 'lfw_funneled')
    if not os.path.isdir(lfw_dir):                              # about 200 MB, once
        from sklearn.datasets import fetch_lfw_people
        fetch_lfw_people(data_home=data_home, min_faces_per_person=min_faces, resize=0.3)
    people = sorted(d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d)))
    photos, target, names = [], [], []
    for p in people:
        files = sorted(f for f in os.listdir(os.path.join(lfw_dir, p)) if f.lower().endswith('.jpg'))
        if len(files) < min_faces:
            continue
        for f in files:
            photos.append(cv2.imdecode(np.fromfile(os.path.join(lfw_dir, p, f), dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE))
            target.append(len(names))
        names.append(p)
    return dict(name='lfw', title='LFW (unconstrained)', photos=np.array(photos, dtype=np.uint8),
                target=np.array(target), names=names, scale=1.6, k=10, n_imp=150, max_gen=30, live_copies=9,
                seeds=(0, 1, 2), multi=(3, 5))


def load_olivetti(data_home=DATA_HOME):
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(data_home=data_home, shuffle=False)
    small = np.clip(np.rint(d.images * 255), 0, 255).astype(np.uint8)          # 400 x 64 x 64
    photos = []
    for g in small:
        up = cv2.resize(g, (230, 280), interpolation=cv2.INTER_CUBIC)          # restore the 92:112 aspect
        canvas = cv2.copyMakeBorder(up, 60, 60, 85, 85, cv2.BORDER_REPLICATE)
        photos.append(cv2.GaussianBlur(canvas, (0, 0), 0.8))                   # remove upscaling blockiness
    return dict(name='olivetti', title='Olivetti / ORL (controlled)', photos=np.array(photos, dtype=np.uint8),
                target=np.asarray(d.target), names=[f'subject_{i:02d}' for i in range(40)], scale=0.65, k=5,
                n_imp=120, max_gen=5, live_copies=18, seeds=(0, 1, 2, 3, 4, 5), multi=(3, 5))


LOADERS = {'lfw': load_lfw, 'olivetti': load_olivetti}


def split_ids(D, split):
    ids = np.arange(len(D['names']))
    return ids[0::2] if split == 'dev' else ids[1::2]


def frame_of(photo, scale, **pose):
    """The camera frame that shows `photo` (optionally with a small pose change)."""
    return sim.to_camera(sim.render(sim.scale_photo(photo, scale), **pose))


def pick_face(boxes, shape):
    """The dataset photographs are centred on one person: take the largest detection near the centre."""
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    return max(boxes, key=lambda b: b[2] * b[3] / (1.0 + math.hypot(b[0] + b[2] / 2.0 - cx,
                                                                    b[1] + b[3] / 2.0 - cy) / 40.0))


def face_crop(engine, frame):
    """Run the app's own front end on one frame: (normalised crop, box, eyes found?, faces detected) or None."""
    _disp, gray, eq = S.prepare(frame)
    boxes = engine.detect_faces(eq)
    if not boxes:
        return None
    box = boxes[0] if len(boxes) == 1 else pick_face(boxes, gray.shape)
    roi, _raw, eyes = engine.extract(gray, eq, box)
    return roi, box, eyes is not None, len(boxes), gray, eq


def extract(engine, D, progress=None, extra=None):
    """Normalised crop of every photograph, through the app's detection and alignment.

    Returns dict(rois[N,s,s], valid[N], eyes[N], ndet[N], boxw[N]). `extra(gray, eq, box)` may return a
    second crop of the same detection (stored as rois_extra), e.g. the preprocessing of an older version."""
    n = len(D['photos'])
    s = engine.size
    rois = np.zeros((n, s, s), np.uint8)
    valid = np.zeros(n, bool)
    eyes = np.zeros(n, bool)
    ndet = np.zeros(n, np.int32)
    boxw = np.zeros(n, np.int32)
    more = np.zeros((n, s, s), np.uint8) if extra else None
    for i, photo in enumerate(D['photos']):
        out = face_crop(engine, frame_of(photo, D['scale']))
        if out is not None:
            rois[i], box, eyes[i], ndet[i] = out[0], out[1], out[2], out[3]
            valid[i], boxw[i] = True, box[2]
            if extra:
                more[i] = extra(out[4], out[5], box)
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, n)
    res = dict(rois=rois, valid=valid, eyes=eyes, ndet=ndet, boxw=boxw)
    if extra:
        res['rois_extra'] = more
    return res


def make_trials(valid, target, ids, k, n_imp, max_gen, seed, min_gen=2):
    """Per identity: k random enrolment photographs, the rest (at most max_gen) as genuine probes, and n_imp
    impostor probes drawn uniformly over the other identities of the same half.

    Returns [(identity, enrol_idx[k], genuine_idx[], impostor_idx[n_imp])]. Identities with fewer than
    k + min_gen usable photographs are left out."""
    rng = np.random.RandomState(seed)
    ids = [int(p) for p in ids]
    by = {p: np.where((target == p) & valid)[0] for p in ids}
    out = []
    for p in ids:
        idx = rng.permutation(by[p])
        if len(idx) < k + min_gen:
            continue
        gal, gen = idx[:k], idx[k:k + max_gen]
        others = [q for q in ids if q != p and len(by[q]) > 0]
        imp = np.array([rng.choice(by[q]) for q in rng.choice(others, size=n_imp, replace=True)])
        out.append((p, gal, gen, imp))
    return out


def live_gallery(engine, D, F, gal, rng, noise=2.0):
    """Crops that emulate a live enrolment from the K enrolment photographs: each photograph itself plus
    (live_copies - 1) frames of it with a small random pose change (rotation +-4 deg, scale +-4 %, shift
    +-8 px) and sensor noise, every frame passed through detection and alignment again."""
    rois = []
    for i in gal:
        rois.append(F['rois'][i])
        base = sim.scale_photo(D['photos'][i], D['scale'])
        for _ in range(D['live_copies'] - 1):
            frame = sim.add_noise(sim.render(base, **sim.jitter_params(rng)), noise, rng)
            out = face_crop(engine, sim.to_camera(frame))
            if out is not None:
                rois.append(out[0])
    return rois


# ---- handing the arrays to worker processes ------------------------------------------------------------
def v9_roi(eq, box):
    """Version 9 preprocessing of a detected face: box crop of the equalised frame, equalised again,
    blurred, resized. Only used as the reference row of the benchmark."""
    x, y, w, h = box
    return cv2.resize(cv2.GaussianBlur(cv2.equalizeHist(eq[y:y + h, x:x + w]), (3, 3), 0), (100, 100))


def stage_photos(D, folder):
    """Write the photographs where worker processes can memory-map them; returns the picklable rest of D."""
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"{D['name']}_photos.npy"), D['photos'])
    np.save(os.path.join(folder, f"{D['name']}_target.npy"), D['target'])
    return {k: v for k, v in D.items() if k not in ('photos', 'target')}


def stage_features(info, F, folder):
    np.savez(os.path.join(folder, f"{info['name']}_feat.npz"), **F)


def unstage(info, folder, features=True):
    """(dataset dict, features dict) inside a worker process."""
    D = dict(info, photos=np.load(os.path.join(folder, f"{info['name']}_photos.npy"), mmap_mode='r'),
             target=np.load(os.path.join(folder, f"{info['name']}_target.npy")))
    if not features:
        return D, None
    z = np.load(os.path.join(folder, f"{info['name']}_feat.npz"))
    return D, {k: z[k] for k in z.files}


_ENGINE = []


def _extract_chunk(job):
    info, folder, idx, with_v9 = job
    if not _ENGINE:
        from biolock.config import load_config
        from biolock.engine import FaceEngine
        cv2.setNumThreads(1)
        _ENGINE.append(FaceEngine(load_config(path='', use_env=False)))
    engine = _ENGINE[0]
    D, _ = unstage(info, folder, features=False)
    out = []
    for i in idx:
        r = face_crop(engine, frame_of(D['photos'][int(i)], info['scale']))
        if r is None:
            out.append((int(i), None))
        else:
            roi, box, eyes, ndet, _gray, eq = r
            out.append((int(i), roi, int(box[2]), bool(eyes), int(ndet), v9_roi(eq, box) if with_v9 else None))
    return out


def extract_parallel(info, folder, n_photos, idx, workers, with_v9=False, size=100):
    """Like extract(), for the photograph indices `idx` only, spread over worker processes.
    The photographs must have been staged with stage_photos()."""
    from concurrent.futures import ProcessPoolExecutor
    F = dict(rois=np.zeros((n_photos, size, size), np.uint8), valid=np.zeros(n_photos, bool),
             eyes=np.zeros(n_photos, bool), ndet=np.zeros(n_photos, np.int32), boxw=np.zeros(n_photos, np.int32))
    if with_v9:
        F['rois_extra'] = np.zeros((n_photos, size, size), np.uint8)
    chunks = [c for c in np.array_split(np.asarray(idx), max(1, workers * 4)) if len(c)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_extract_chunk, [(info, folder, c, with_v9) for c in chunks]):
            for rec in part:
                if rec[1] is None:
                    continue
                i = rec[0]
                F['rois'][i], F['boxw'][i], F['eyes'][i], F['ndet'][i] = rec[1], rec[2], rec[3], rec[4]
                F['valid'][i] = True
                if with_v9:
                    F['rois_extra'][i] = rec[5]
    return F
