"""Face detection, alignment, feature extraction and matching (stages F and R of the pipeline).

Identity verification combines three scores, each mapped to 0-100:

    C = w_lbph * C_LBPH + w_hist * C_Hist + w_template * C_Temp

    C_LBPH   OpenCV LBPH: chi-square distance of the probe's LBP histogram to the nearest
             enrolled sample (identical to LBPHFaceRecognizer.predict, verified in the tests)
    C_Hist   correlation between the probe's block-wise gradient-orientation histograms
             and the mean enrolled descriptor
    C_Temp   normalised cross-correlation between the probe and the mean enrolled face

A decision uses the MEDIAN score of several frames, never a single frame.

Only features are stored for an enrolment (LBP histograms, one mean descriptor, one mean
normalised face); the file is a plain .npz that is loaded without pickle.
"""

import io
import os
import json
import math
import hashlib
import zipfile
import datetime

import cv2
import numpy as np

from . import winutil

MODEL_VERSION = 2

# Mean position of the eyes inside an OpenCV Haar frontal-face box, as a fraction of the
# box (measured on 2,200 LFW faces; used when the eyes themselves are not found).
BOX_EYE_L = (0.312, 0.400)
BOX_EYE_R = (0.668, 0.392)

# Geometry of the normalised crop: the eyes are placed EYE_IOD * size apart, at height EYE_Y * size.
EYE_IOD = 0.46
EYE_Y = 0.36

# Face width (pixels) at which the per-frame "is an eye visible?" check is run.
EYE_CHECK_WIDTH = 112

# Limits applied when a template is read from the (removable, untrusted) drive.
MAX_MODEL_BYTES = 32 * 1024 * 1024
MAX_SAMPLES = 2000


class EngineError(RuntimeError):
    pass


def _cascade(name):
    """Load a Haar cascade from memory, so that it also works when the cascade lives under a
    path with non-ASCII characters (OpenCV's own file loader cannot open those on Windows)."""
    path = os.path.join(cv2.data.haarcascades, name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        cc = cv2.CascadeClassifier()
        if cc.read(fs.getFirstTopLevelNode()) and not cc.empty():
            return cc
    except Exception:
        pass
    cc = cv2.CascadeClassifier(path)
    if cc.empty():
        raise EngineError(f'Could not load {name}.\nReinstall with:  pip install opencv-contrib-python')
    return cc


def tan_triggs(img, gamma=0.2, sigma0=1.0, sigma1=2.0, alpha=0.1, tau=10.0):
    """Tan & Triggs illumination normalisation: gamma -> difference of Gaussians -> contrast equalisation."""
    x = np.power(img.astype(np.float32) / 255.0 + 1e-6, gamma)
    x = cv2.GaussianBlur(x, (0, 0), sigma0) - cv2.GaussianBlur(x, (0, 0), sigma1)
    x = x / (np.mean(np.abs(x) ** alpha) ** (1.0 / alpha) + 1e-8)
    x = x / (np.mean(np.minimum(np.abs(x), tau) ** alpha) ** (1.0 / alpha) + 1e-8)
    x = tau * np.tanh(x / tau)
    return np.clip((x + tau) / (2.0 * tau) * 255.0, 0, 255).astype(np.uint8)


def lighting_quality(face_gray):
    """(score 0-100, hint) from the un-equalised grey face region."""
    mean = float(np.mean(face_gray))
    std = float(np.std(face_gray))
    score, msg = 100, 'Lighting OK'
    if mean < 50:
        score -= 50
        msg = 'Too dark - face a light source'
    elif mean > 215:
        score -= 50
        msg = 'Too bright - reduce direct light'
    if std < 18:
        score -= 30
        msg = 'Flat lighting - try light from the front'
    return max(0, score), msg


def sharpness(img):
    """Variance of the Laplacian: drops when the image is blurred (movement, defocus)."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _unit(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    v = v - v.mean()
    return v / (float(np.linalg.norm(v)) + 1e-9)


def calibrate(cfg, lbph_distance, hist_corr, tmpl_corr):
    """Map the three raw measurements to 0-100 component scores (works on numbers and on arrays)."""
    lb = 100.0 * np.clip((cfg['lbph_d_hi'] - lbph_distance) / (cfg['lbph_d_hi'] - cfg['lbph_d_lo']), 0.0, 1.0)
    hs = 100.0 * np.clip((hist_corr - cfg['hist_c_lo']) / (cfg['hist_c_hi'] - cfg['hist_c_lo']), 0.0, 1.0)
    tm = 100.0 * np.clip((tmpl_corr - cfg['tmpl_c_lo']) / (cfg['tmpl_c_hi'] - cfg['tmpl_c_lo']), 0.0, 1.0)
    return lb, hs, tm


def fuse(cfg, lbph, hist, template, weights=None):
    """C = w_lbph * C_LBPH + w_hist * C_Hist + w_template * C_Temp (numbers or arrays)."""
    w = weights if weights is not None else (cfg['w_lbph'], cfg['w_hist'], cfg['w_template'])
    return w[0] * lbph + w[1] * hist + w[2] * template


def template_digest(meta, G, hog_ref, tmpl_ref):
    """SHA-256 over the *content* of a template (parameters and every stored number).

    It does not depend on how the .npz container was written, so the same template always
    gives the same digest, and changing any stored value changes it."""
    h = hashlib.sha256()
    h.update(b'BioLock-template|')
    h.update(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    for name, arr in (('lbph', G), ('hog', hog_ref), ('tmpl', tmpl_ref)):
        a = np.ascontiguousarray(arr, dtype='<f4')
        h.update(f'|{name}|{a.shape}|'.encode('ascii'))
        h.update(a.tobytes())
    return h.digest()


class FaceEngine:

    def __init__(self, cfg, model_path=None):
        if not hasattr(cv2, 'face'):
            raise EngineError('This OpenCV build has no LBPH recogniser.\n'
                              'Install the contrib build:  pip install opencv-contrib-python')
        self.cfg = cfg
        self.model_path = model_path
        self.face_cc = _cascade('haarcascade_frontalface_default.xml')
        self.eye_cc = _cascade('haarcascade_eye.xml')
        self._set_params(cfg['lbph_radius'], cfg['lbph_neighbors'], cfg['lbph_grid'], cfg['roi_size'])
        self.trained = False
        self.load_error = None
        self.meta = {}
        self._G = None              # enrolled LBP histograms, one row per sample
        self._G_sum = None
        self._hog_ref = None        # unit-normalised mean HOG descriptor
        self._tmpl_ref = None       # unit-normalised mean face
        self._digest = None
        if model_path and os.path.isfile(model_path):
            self.load()

    # ---- parameters -------------------------------------------------------
    def _set_params(self, radius, neighbors, grid, roi):
        radius, neighbors, grid, roi = int(radius), int(neighbors), int(grid), int(roi)
        if roi % 10 or not 40 <= roi <= 400:
            raise EngineError('roi_size must be a multiple of 10 between 40 and 400')
        if not (1 <= radius <= roi // 4 and 4 <= neighbors <= 12 and 1 <= grid <= roi // 8):
            raise EngineError('LBPH parameters are out of range')
        self.radius, self.neighbors, self.grid, self.size = radius, neighbors, grid, roi
        self._lbph = cv2.face.LBPHFaceRecognizer_create(radius=radius, neighbors=neighbors,
                                                        grid_x=grid, grid_y=grid)
        self._hog = cv2.HOGDescriptor((roi, roi), (20, 20), (10, 10), (10, 10), 9)

    # ---- detection ----------------------------------------------------------
    def detect_faces(self, gray_eq, scale=0.5, min_size=30):
        """Haar frontal-face detection on a (down)scaled copy; boxes are returned in full-resolution pixels."""
        small = gray_eq if scale == 1.0 else cv2.resize(gray_eq, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_AREA)
        found = self.face_cc.detectMultiScale(small, 1.1, 5, minSize=(min_size, min_size),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        inv = 1.0 / scale
        return [tuple(int(round(v * inv)) for v in b) for b in found] if len(found) else []

    def _eye_band(self, gray_eq, box):
        x, y, w, h = [int(v) for v in box]
        y0, y1 = max(y + int(0.15 * h), 0), y + int(0.62 * h)
        x0 = max(x, 0)
        return gray_eq[y0:y1, x0:x + w], x0, y0

    def find_eyes(self, gray_eq, box):
        """Both eye centres ((lx, ly), (rx, ry)) in image pixels, or None when they are not found reliably."""
        band, x0, y0 = self._eye_band(gray_eq, box)
        w, h = int(box[2]), int(box[3])
        if band.size == 0 or w < 40:
            return None
        mins = max(8, int(0.12 * w))
        maxs = max(mins + 1, int(0.42 * w))
        ev = self.eye_cc.detectMultiScale(band, 1.05, 4, minSize=(mins, mins), maxSize=(maxs, maxs))
        if len(ev) < 2:
            return None
        bx, by = x0 - int(box[0]), y0 - int(box[1])                       # band offset inside the box
        cands = [(ex + ew / 2.0 + bx, ey + eh / 2.0 + by) for ex, ey, ew, eh in ev]
        left = [c for c in cands if c[0] < 0.5 * w]
        right = [c for c in cands if c[0] >= 0.5 * w]
        if not left or not right:
            return None
        el = (BOX_EYE_L[0] * w, BOX_EYE_L[1] * h)
        er = (BOX_EYE_R[0] * w, BOX_EYE_R[1] * h)
        l = min(left, key=lambda c: (c[0] - el[0]) ** 2 + (c[1] - el[1]) ** 2)
        r = min(right, key=lambda c: (c[0] - er[0]) ** 2 + (c[1] - er[1]) ** 2)
        iod = math.hypot(r[0] - l[0], r[1] - l[1])
        ang = math.degrees(math.atan2(r[1] - l[1], r[0] - l[0]))
        if not (0.25 * w <= iod <= 0.60 * w) or abs(ang) > 22:
            return None
        return (box[0] + l[0], box[1] + l[1]), (box[0] + r[0], box[1] + r[1])

    def eyes_visible(self, gray_eq, box):
        """True when at least one open eye is detected in the eye band (the blink signal).

        This runs on every frame of the blink step, so the band is first reduced to a face width
        of EYE_CHECK_WIDTH pixels: less than half the cost of the full-resolution search and the
        same answer on 98 % of frames (measured on the Olivetti development photographs)."""
        band, _x0, _y0 = self._eye_band(gray_eq, box)
        w = int(box[2])
        if band.size == 0 or w < 40:
            return False
        if w > EYE_CHECK_WIDTH:
            s = EYE_CHECK_WIDTH / float(w)
            band = cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return len(self.eye_cc.detectMultiScale(band, 1.08, 3, minSize=(20, 20))) >= 1

    # ---- normalised face crop -------------------------------------------------
    def crop(self, gray, box, eyes=None):
        """Aligned size x size crop of the raw grey image (no illumination normalisation).

        With `eyes` the face is rotated/scaled so the eyes sit at fixed positions; without, the
        same geometry is derived from the face box alone (no rotation)."""
        if eyes is None:
            x, y, w, h = box
            (lx, ly), (rx, ry) = ((x + BOX_EYE_L[0] * w, y + BOX_EYE_L[1] * h),
                                  (x + BOX_EYE_R[0] * w, y + BOX_EYE_R[1] * h))
            ang = 0.0
        else:
            (lx, ly), (rx, ry) = eyes
            ang = math.degrees(math.atan2(ry - ly, rx - lx))
        s = (EYE_IOD * self.size) / max(math.hypot(rx - lx, ry - ly), 1e-6)
        cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), ang, s)
        M[0, 2] += 0.5 * self.size - cx
        M[1, 2] += EYE_Y * self.size - cy
        return cv2.warpAffine(gray, M, (self.size, self.size),
                              flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def roi(self, gray, box, eyes=None):
        """Aligned, illumination-normalised crop: the input of every feature below."""
        return tan_triggs(self.crop(gray, box, eyes))

    def extract(self, gray, gray_eq, box):
        """(normalised crop, raw aligned crop, eyes or None) for one detected face.

        Eye alignment is used when both eyes are found, the box geometry otherwise."""
        eyes = self.find_eyes(gray_eq, box)
        raw = self.crop(gray, box, eyes)
        return tan_triggs(raw), raw, eyes

    # ---- features ---------------------------------------------------------------
    def lbph_hist(self, roi):
        self._lbph.train([roi], np.zeros(1, np.int32))
        return self._lbph.getHistograms()[0].astype(np.float32).ravel()

    def hog(self, roi):
        return self._hog.compute(roi).astype(np.float32).ravel()

    # ---- enrolment ----------------------------------------------------------------
    def train(self, rois, save=True):
        """Build the template from normalised crops. Returns the number of samples kept."""
        rois = [np.asarray(r, dtype=np.uint8) for r in rois]
        if not rois:
            raise EngineError('No samples to enrol')
        self._set_params(self.cfg['lbph_radius'], self.cfg['lbph_neighbors'], self.cfg['lbph_grid'],
                         self.cfg['roi_size'])
        if any(r.shape != (self.size, self.size) for r in rois):
            raise EngineError(f'Enrolment samples must be {self.size}x{self.size} crops')
        if len(rois) >= 20:
            # drop the 10 % of samples furthest from the mean face (bad crops / false detections)
            stack = np.array(rois, dtype=np.float32)
            dist = np.mean(np.abs(stack - stack.mean(axis=0)), axis=(1, 2))
            keep = dist <= np.percentile(dist, 90)
            rois = [r for r, k in zip(rois, keep) if k]
        self._lbph.train(rois, np.zeros(len(rois), np.int32))
        G = np.vstack(self._lbph.getHistograms()).astype(np.float32)
        hog_ref = _unit(np.mean([self.hog(r) for r in rois], axis=0))
        tmpl_ref = _unit(np.mean(np.array(rois, dtype=np.float32), axis=0))
        meta = {'version': MODEL_VERSION, 'radius': self.radius, 'neighbors': self.neighbors,
                'grid': self.grid, 'roi': self.size, 'eye_iod': EYE_IOD, 'eye_y': EYE_Y,
                'samples': len(rois), 'created': datetime.datetime.now().isoformat(timespec='seconds')}
        self._install(G, hog_ref, tmpl_ref, meta)
        if save and self.model_path:
            self.save()
        return len(rois)

    def _install(self, G, hog_ref, tmpl_ref, meta):
        self._G = np.ascontiguousarray(G, dtype=np.float32)
        self._G_sum = self._G.sum(axis=1)
        self._hog_ref = np.ascontiguousarray(hog_ref, dtype=np.float32).ravel()
        self._tmpl_ref = np.ascontiguousarray(tmpl_ref, dtype=np.float32).ravel()
        self.meta = dict(meta)
        self._digest = template_digest(self.meta, self._G, self._hog_ref, self._tmpl_ref)
        self.trained = True
        self.load_error = None

    def _serialise(self):
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.array(json.dumps(self.meta, sort_keys=True)),
                            lbph=self._G, hog=self._hog_ref, tmpl=self._tmpl_ref)
        return buf.getvalue()

    def save(self):
        if not self.trained:
            raise EngineError('Nothing to save: no face is enrolled')
        winutil.atomic_write(self.model_path, self._serialise())

    def load(self):
        """Load an enrolled template. Any problem leaves the engine untrained and sets load_error.

        The file comes from a removable drive, so it is treated as untrusted input: it is read
        without pickle, its unpacked size is limited, and every array is checked against the
        parameters it claims."""
        self.forget()
        try:
            if os.path.getsize(self.model_path) > MAX_MODEL_BYTES:
                raise EngineError('template file is too large')
            with open(self.model_path, 'rb') as f:
                data = f.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if sum(i.file_size for i in zf.infolist()) > MAX_MODEL_BYTES:
                    raise EngineError('template file unpacks to too much data')
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                G, hog_ref, tmpl_ref = z['lbph'], z['hog'], z['tmpl']
            if not isinstance(meta, dict) or meta.get('version') != MODEL_VERSION:
                raise EngineError('template version is not supported')
            self._set_params(meta['radius'], meta['neighbors'], meta['grid'], meta['roi'])
            dim = self.grid * self.grid * (2 ** self.neighbors)
            if G.ndim != 2 or G.shape[1] != dim or not 1 <= len(G) <= MAX_SAMPLES \
                    or tmpl_ref.size != self.size ** 2 or hog_ref.size != self._hog.getDescriptorSize():
                raise EngineError('template arrays do not match their parameters')
            if not (np.isfinite(G).all() and np.isfinite(hog_ref).all() and np.isfinite(tmpl_ref).all()) \
                    or float(G.min()) < 0:
                raise EngineError('template contains invalid numbers')
            self._install(G, hog_ref, tmpl_ref, meta)
            return True
        except Exception as e:                                    # corrupt, truncated or foreign file
            self.forget()
            self.load_error = f'{type(e).__name__}: {e}'
            self._set_params(self.cfg['lbph_radius'], self.cfg['lbph_neighbors'], self.cfg['lbph_grid'],
                             self.cfg['roi_size'])
            return False

    def digest(self):
        """SHA-256 of the enrolled template's content (feeds the vault-key derivation), or None."""
        return self._digest

    def forget(self):
        self.trained = False
        self._G = self._G_sum = self._hog_ref = self._tmpl_ref = None
        self._digest = None
        self.meta = {}

    # ---- matching -------------------------------------------------------------------
    def features(self, roi):
        """(LBP histogram, unit HOG descriptor, unit pixel vector) of one normalised crop."""
        return self.lbph_hist(roi), _unit(self.hog(roi)), _unit(roi)

    def lbph_distance(self, roi):
        """Chi-square (HISTCMP_CHISQR_ALT) distance to the nearest enrolled sample."""
        return self._nearest(self.lbph_hist(roi))

    def _nearest(self, h):
        nz = np.flatnonzero(h)
        g = self._G[:, nz]
        hv = h[nz]
        d = g - hv
        part = np.sum(d * d / (g + hv), axis=1)              # bins where the probe is non-zero
        rest = self._G_sum - g.sum(axis=1)                    # bins where it is zero contribute g
        return float(2.0 * np.min(part + rest))

    def compare(self, feat):
        """Raw measurements and the three 0-100 component scores for the features() of one crop."""
        if not self.trained:
            return {'lbph_distance': 999.0, 'hist_corr': 0.0, 'tmpl_corr': 0.0,
                    'lbph': 0.0, 'hist': 0.0, 'template': 0.0}
        h, hog_u, px_u = feat
        dist = self._nearest(h)
        hist_corr = float(hog_u @ self._hog_ref)
        tmpl_corr = float(px_u @ self._tmpl_ref)
        lb, hs, tm = calibrate(self.cfg, dist, hist_corr, tmpl_corr)
        return {'lbph_distance': dist, 'hist_corr': hist_corr, 'tmpl_corr': tmpl_corr,
                'lbph': float(lb), 'hist': float(hs), 'template': float(tm)}

    def components(self, roi):
        """compare(features(roi)): everything the matcher measures for one normalised crop."""
        return self.compare(self.features(roi)) if self.trained else self.compare(None)

    def fuse(self, comp):
        return float(fuse(self.cfg, comp['lbph'], comp['hist'], comp['template']))

    def score(self, roi):
        """(fused 0-100 match score, components) for one normalised crop."""
        comp = self.components(roi)
        return self.fuse(comp), comp

    def decide(self, scores):
        """(accepted, median score): the decision rule R = [median(C) >= T]."""
        if not len(scores):
            return False, 0.0
        med = float(np.median(scores))
        return med >= self.cfg['min_conf'], med
