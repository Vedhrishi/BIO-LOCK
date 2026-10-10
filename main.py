"""
BioLock SSD  v9.0  Final  (experiment-instrumented build)

Face-recognition based USB drive lock with liveness detection
and an encrypted vault.

Author: Pasupuleti Vedh Rishi

------------------------------------------------------------------
Runs only from external drives. See README.md and docs/ for the
architecture, the evaluation protocol and the project history.
------------------------------------------------------------------
"""

import os
import sys
import time
import json
import pickle
import hashlib
import secrets
import base64
import datetime
import threading
import subprocess
import shutil
import ctypes
import string
import math

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import load_config, save_config
from metrics import MetricsLogger

try:
    import winreg
    _WIN = True
except ImportError:
    _WIN = False

# --------------------------------------------------------------------------
# Configuration (externalised — see config.py / config.json)
# --------------------------------------------------------------------------
CFG = load_config()

APP_VERSION = '9.0'
ENROL_FRAMES = CFG['enrol_frames']
MIN_CONF = CFG['min_conf']
MAX_ATTEMPTS = CFG['max_attempts']
BLINKS_NEEDED = CFG['blinks_needed']
HEAD_TURN_PCT = CFG['head_turn_pct']
MOTION_THRESH = CFG['motion_thresh']
MOTION_FRAMES = CFG['motion_frames']
BLINK_MIN_CL = CFG['blink_min_cl']
PROCESS_EVERY = CFG['process_every']
AUTO_LOCK_M = CFG['auto_lock_mins']
W_LBPH = CFG['w_lbph']
W_HIST = CFG['w_hist']
W_TEMPLATE = CFG['w_template']
WATCHER_KEY = 'BioLockUSB'

# Theme
BG = '#060612'
SURFACE = '#0c0c20'
CARD = '#10102a'
BORDER = '#1e1e48'
ACCENT = '#6d28d9'
ACCENT_L = '#8b5cf6'
TEAL = '#0891b2'
TEAL_L = '#22d3ee'
GREEN = '#059669'
GREEN_L = '#34d399'
RED = '#dc2626'
RED_L = '#f87171'
AMBER = '#d97706'
AMBER_L = '#fbbf24'
FG = '#f5f5ff'
MUTED = '#9494b8'
FAINT = '#1f1f48'

CAM_NAMES = {0: 'Built-in webcam', 1: 'External / phone', 2: 'Virtual camera'}

NW = 134217728  # CREATE_NO_WINDOW, so 'attrib' calls don't flash a console

METRICS = MetricsLogger()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _base():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _p(*x):
    return os.path.join(_base(), *x)


LOCK_DIR = _p('.biolock')
MODEL_FILE = _p('.biolock', 'face.xml')
STATS_FILE = _p('.biolock', 'face_stats.pkl')
VAULT_KEY = _p('.biolock', 'vault.key')
STATE_FILE = _p('.biolock', 'state.json')
LOG_FILE = _p('.biolock', 'audit.log')
CFG_FILE = _p('.biolock', 'config.json')
SNAP_DIR = _p('.biolock', 'snapshots')
VAULT_DIR = _p('SecureVault')
DRIVE_ID = _p('.biolock', 'drive.id')


# --------------------------------------------------------------------------
# Small OS helpers
# --------------------------------------------------------------------------
def _hide(p):
    if p and os.path.exists(p):
        try:
            subprocess.run(['attrib', '+h', '+s', p], capture_output=True, creationflags=NW)
        except Exception:
            pass


def _unhide(p):
    if p and os.path.exists(p):
        try:
            subprocess.run(['attrib', '-h', '-s', p], capture_output=True, creationflags=NW)
        except Exception:
            pass


def _log(ev, detail='', level='INFO'):
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        _unhide(LOG_FILE)
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] [{level}] {ev}'
        if detail:
            line += f'  |  {detail}'
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        _hide(LOG_FILE)
    except Exception:
        pass


def _beep(f, d=880):
    try:
        if _WIN:
            ctypes.windll.kernel32.Beep(f, d)
    except Exception:
        pass


def _sound_ok():
    threading.Thread(target=lambda: [
        _beep(880, 80), time.sleep(0.05),
        _beep(1108, 80), time.sleep(0.05),
        _beep(1318, 150)], daemon=True).start()


def _sound_fail():
    threading.Thread(target=lambda: [
        _beep(400, 200), time.sleep(0.05),
        _beep(300, 300)], daemon=True).start()


def _sound_blink():
    threading.Thread(target=lambda: _beep(1200, 40), daemon=True).start()


def _drive_letter():
    try:
        drive = os.path.splitdrive(_base())[0]
        letter = drive.rstrip(':\\').upper()
        return letter if letter else '?'
    except Exception:
        return '?'


# Bus types reported by IOCTL_STORAGE_QUERY_PROPERTY that mean "plugged in from outside"
_EXTERNAL_BUSES = {7: 'USB', 4: '1394', 0xC: 'SD', 0xD: 'MMC'}


def _bus_type(letter):
    """Storage bus type of a drive letter (7 = USB, 17 = NVMe, 11 = SATA ...), or None if unknown."""
    try:
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.CreateFileW.restype = wintypes.HANDLE
        h = k.CreateFileW(f'\\\\.\\{letter}:', 0, 3, None, 3, 0, None)   # 0 access: no admin needed
        if h in (None, wintypes.HANDLE(-1).value, 0xFFFFFFFF, -1):
            return None
        try:
            query = (ctypes.c_ubyte * 12)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)   # StorageDeviceProperty, standard query
            out = (ctypes.c_ubyte * 1024)()
            got = wintypes.DWORD(0)
            ok = k.DeviceIoControl(h, 0x2D1400, query, 12, out, 1024, ctypes.byref(got), None)
            if not ok:
                return None
            return out[28]   # STORAGE_DEVICE_DESCRIPTOR.BusType
        finally:
            k.CloseHandle(h)
    except Exception:
        return None


def _is_external_drive(letter):
    """True only when the drive is positively identified as external/removable.
    Fails CLOSED: anything unknown or internal (SATA/NVMe/system disk) counts as internal."""
    if not _WIN:
        return False
    try:
        dt = ctypes.windll.kernel32.GetDriveTypeW(f'{letter}:\\')
    except Exception:
        return False
    if dt == 2:                                  # DRIVE_REMOVABLE (USB sticks, SD cards)
        return True
    if dt != 3:                                  # network / CD / RAM disk / unknown
        return False
    return _bus_type(letter) in _EXTERNAL_BUSES  # USB-attached SSD/HDD report as "fixed"


def _is_removable(letter):
    return _is_external_drive(letter)


def enforce_external_drive(root=None):
    """Called at start-up. The app refuses to run from an internal drive.
    Enforced for the packaged exe; from source set BIOLOCK_ENFORCE_EXTERNAL=1.
    BIOLOCK_ALLOW_INTERNAL=1 is a developer escape hatch."""
    if os.environ.get('BIOLOCK_ALLOW_INTERNAL') == '1':
        return True
    if not (getattr(sys, 'frozen', False) or os.environ.get('BIOLOCK_ENFORCE_EXTERNAL') == '1'):
        return True
    letter = _drive_letter()
    if _is_external_drive(letter):
        return True
    try:
        messagebox.showerror(
            'BioLock cannot run here',
            f'BioLock is running from drive {letter}:, which is an internal drive.\n\n'
            'This cannot be used to lock an internal drive, as it would endanger the '
            'storage components of the system.\n\n'
            'Copy BioLock to an external drive (USB flash drive, external SSD/HDD or SD card) '
            'and run it from there.')
    except Exception:
        pass
    return False


def _drive_free_gb(letter):
    try:
        st = shutil.disk_usage(f'{letter}:\\')
        return (st.free / 1073741824, st.total / 1073741824)
    except Exception:
        return (0, 0)


def _vault_size_mb():
    try:
        total = 0
        for dp, _dn, fn in os.walk(VAULT_DIR):
            for f in fn:
                total += os.path.getsize(os.path.join(dp, f))
        return total / 1048576
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Drive identity
# --------------------------------------------------------------------------
def _create_drive_id():
    os.makedirs(LOCK_DIR, exist_ok=True)
    uid = secrets.token_hex(16)
    _unhide(DRIVE_ID)
    with open(DRIVE_ID, 'w') as f:
        json.dump({
            'id': uid,
            'created': datetime.datetime.now().isoformat(),
            'version': APP_VERSION,
        }, f)
    _hide(DRIVE_ID)
    return uid


def _get_drive_id():
    if not os.path.exists(DRIVE_ID):
        return None
    try:
        _unhide(DRIVE_ID)
        with open(DRIVE_ID, 'r') as f:
            data = json.load(f)
        _hide(DRIVE_ID)
        return data.get('id')
    except Exception:
        return None


def _write_autorun():
    try:
        inf = _p('autorun.inf')
        _unhide(inf)
        with open(inf, 'w') as f:
            f.write('[autorun]\nopen=BioLock.exe\nicon=BioLock.exe,0\n'
                     'label=BioLock Secure Drive\naction=Unlock with BioLock\n')
        _hide(inf)
    except Exception:
        pass


def _register_autorun():
    if not _WIN:
        return
    try:
        exe = sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__)
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            'Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                            0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(k, WATCHER_KEY, 0, winreg.REG_SZ, f'"{exe}"')
        winreg.CloseKey(k)
    except Exception:
        pass


# --------------------------------------------------------------------------
# State / config persistence
# --------------------------------------------------------------------------
DEF_STATE = {
    'enrolled': False,
    'attempts': 0,
    'total_unlocks': 0,
    'last_unlock': None,
    'enrol_quality': 0,
}
DEF_CFG = {
    'auto_lock_mins': AUTO_LOCK_M,
    'cam_index': None,
}


def _load_state():
    try:
        _unhide(STATE_FILE)
        with open(STATE_FILE, 'r') as f:
            s = json.load(f)
        _hide(STATE_FILE)
        d = dict(DEF_STATE)
        d.update(s)
        return d
    except Exception:
        return dict(DEF_STATE)


def _save_state(s):
    os.makedirs(LOCK_DIR, exist_ok=True)
    _unhide(STATE_FILE)
    with open(STATE_FILE, 'w') as f:
        json.dump(s, f)
    _hide(STATE_FILE)
    _hide(LOCK_DIR)


def _load_cfg():
    try:
        _unhide(CFG_FILE)
        with open(CFG_FILE, 'r') as f:
            c = json.load(f)
        _hide(CFG_FILE)
        d = dict(DEF_CFG)
        d.update(c)
        return d
    except Exception:
        return dict(DEF_CFG)


def _save_cfg(c):
    os.makedirs(LOCK_DIR, exist_ok=True)
    _unhide(CFG_FILE)
    with open(CFG_FILE, 'w') as f:
        json.dump(c, f)
    _hide(CFG_FILE)


# --------------------------------------------------------------------------
# Vault encryption (face-derived key)
# --------------------------------------------------------------------------
def _fernet(fh, salt):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(fh)))


def save_vault_key(fh):
    salt = secrets.token_bytes(16)
    token = _fernet(fh, salt).encrypt(b'BIOLOCK_VAULT_OK')
    os.makedirs(LOCK_DIR, exist_ok=True)
    _unhide(VAULT_KEY)
    with open(VAULT_KEY, 'wb') as f:
        f.write(salt + token)
    _hide(VAULT_KEY)


def verify_vault_key(fh):
    if not os.path.exists(VAULT_KEY):
        return False
    try:
        _unhide(VAULT_KEY)
        with open(VAULT_KEY, 'rb') as f:
            data = f.read()
        salt, token = data[:16], data[16:]
        out = _fernet(fh, salt).decrypt(token)
        _hide(VAULT_KEY)
        return out == b'BIOLOCK_VAULT_OK'
    except Exception:
        return False


# --------------------------------------------------------------------------
# Haar cascades
# --------------------------------------------------------------------------
def _find_cc(name):
    cands = []
    try:
        cands.append(cv2.data.haarcascades + name)
        d = os.path.dirname(cv2.__file__)
        for sub in ('data', os.path.join('..', 'data'), ''):
            cands.append(os.path.normpath(os.path.join(d, sub, name)))
        for p in cands:
            if os.path.isfile(p):
                return p
        return None
    except Exception:
        return None


FACE_XML = _find_cc('haarcascade_frontalface_default.xml')
EYE_XML = _find_cc('haarcascade_eye.xml')


def _pre(img):
    return cv2.resize(cv2.GaussianBlur(cv2.equalizeHist(img), (3, 3), 0), (100, 100))


def _lighting_quality(face_gray):
    mean = float(np.mean(face_gray))
    std = float(np.std(face_gray))
    score = 100
    msg = 'Lighting OK'
    if mean < 60:
        score -= 40
        msg = 'Too dark — face a light source'
    if mean > 210:
        score -= 40
        msg = 'Too bright — reduce direct light'
    if std < 20:
        score -= 30
        msg = 'Flat lighting — try side light'
    return (max(0, score), msg)


# --------------------------------------------------------------------------
# Liveness detectors
# --------------------------------------------------------------------------
class BlinkDetector:
    """Counts natural blinks using the eye cascade: a blink is a short run
    of frames with no eyes detected, followed by eyes detected again."""

    def __init__(self):
        self.eye_cc = cv2.CascadeClassifier(EYE_XML) if EYE_XML else None
        self.blinks = 0
        self.closed = 0
        self.prev_mean = None
        self.is_done = False
        self._cool = 0

    def reset(self):
        self.__init__()

    def update(self, face_gray):
        if self.is_done:
            return (self.blinks, True, f'✓  {BLINKS_NEEDED} blinks confirmed')

        self._cool = max(0, self._cool - 1)
        eyes_open = True
        if self.eye_cc is not None and not self.eye_cc.empty():
            eyes = self.eye_cc.detectMultiScale(face_gray, 1.08, 3, minSize=(15, 15))
            eyes_open = len(eyes) >= 1

        if not eyes_open:
            self.closed += 1
        else:
            if self.closed >= BLINK_MIN_CL and self._cool == 0:
                self.blinks += 1
                self._cool = 6
                _sound_blink()
            self.closed = 0

        if self.blinks >= BLINKS_NEEDED:
            self.is_done = True
            return (self.blinks, True, f'✓  {BLINKS_NEEDED} blinks confirmed')
        return (self.blinks, False, f'Blink naturally  ({self.blinks}/{BLINKS_NEEDED})')


class HeadTurn:
    """Tracks the face box centre relative to a baseline: needs a turn to
    the LEFT followed by a turn to the RIGHT to pass."""

    L = 'left'
    R = 'right'
    D = 'done'

    def __init__(self):
        self.phase = self.L
        self.baseline = None
        self._buf = []
        self.is_done = False

    def reset(self):
        self.__init__()

    def update(self, box):
        if self.is_done:
            return (self.D, True, 'PASS Head movement', 'Head turn done')

        x, y, w, h = box
        cx = x + w / 2.0
        if self.baseline is None:
            self.baseline = cx

        delta_pct = (cx - self.baseline) / max(w, 1) * 100
        self._buf.append(delta_pct)
        if len(self._buf) > 200:
            self._buf.pop(0)

        if self.phase == self.L:
            if delta_pct <= -HEAD_TURN_PCT:
                self.phase = self.R
                return (self.phase, False, 'Turned LEFT — now turn RIGHT', 'Head turn: LEFT detected')
            return (self.phase, False, 'Turn your head slowly LEFT', 'Head turn: awaiting LEFT')

        if delta_pct >= HEAD_TURN_PCT:
            self.is_done = True
            return (self.D, True, 'PASS Head movement', 'Head turn done')
        return (self.phase, False, 'Now turn your head RIGHT', 'Head turn: awaiting RIGHT')


class MotionLive:
    """Frame-differencing liveness check on the face ROI: a static photo
    held up to the camera won't produce enough natural micro-motion.
    This is also the primary signal for telling a printed-photo
    presentation attack apart from a live or replayed one — see
    docs/EXPERIMENTS.md for the motion_thresh sweep."""

    def __init__(self):
        self._prev = None
        self._m = []
        self.is_live = False
        self._score = 0

    def reset(self):
        self.__init__()

    def update(self, gray, box):
        if self.is_live:
            return (True, 1, 'PASS Liveness', self._score)

        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]

        if self._prev is not None and self._prev.shape == roi.shape:
            diff = cv2.absdiff(roi, self._prev)
            self._score = float(np.mean(diff)) / 255.0
            self._m.append(self._score)
            if len(self._m) > MOTION_FRAMES:
                self._m.pop(0)
            avg = sum(self._m) / len(self._m)
            if avg >= MOTION_THRESH and len(self._m) >= MOTION_FRAMES // 2:
                self.is_live = True
                self._prev = roi
                return (True, 1, 'PASS Liveness', self._score)

        self._prev = roi
        prog = min(1.0, len(self._m) / max(MOTION_FRAMES, 1))
        return (False, prog, 'Stay still — checking presence…', self._score)


# --------------------------------------------------------------------------
# Face recognition engine — weighted fusion of LBPH + Histogram + Template
# --------------------------------------------------------------------------
class FaceEngine:
    """
    Match score is a configurable weighted fusion of three independent
    signals (weights W_LBPH / W_HIST / W_TEMPLATE, must sum to 1 — see
    config.py):

      - LBPH:      cv2.face.LBPHFaceRecognizer distance -> confidence
      - Histogram: grayscale histogram correlation against the mean
                   histogram of the enrolled samples
      - Template:  normalized cross-correlation against the enrolled
                   mean face image

    This is the only piece of matching logic that changed from the
    original single-method (LBPH-only) design, specifically so the
    weight-combination sweep for the experiments could be run. The
    4-step liveness pipeline around it (presence/blink/head-turn) and
    everything else is architecturally unchanged.
    """

    def __init__(self):
        if not FACE_XML:
            raise RuntimeError('Cascade not found.\npip install opencv-contrib-python==4.10.0.84')
        self.cc = cv2.CascadeClassifier(FACE_XML)
        self.lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.mean_face = None
        self.ref_hists = []
        self._mff = None
        self._fhc = None
        self.trained = self._load()

    def _load(self):
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(STATS_FILE):
                _unhide(MODEL_FILE)
                _unhide(STATS_FILE)
                self.lbph.read(MODEL_FILE)
                with open(STATS_FILE, 'rb') as f:
                    stats = pickle.load(f)
                self.mean_face = stats.get('mean_face')
                self.ref_hists = stats.get('ref_hists', [])
                self._fhc = stats.get('fhc')
                if self.mean_face is not None:
                    self._mff = self.mean_face.astype(np.float32)
                _hide(MODEL_FILE)
                _hide(STATS_FILE)
                return True
            return False
        except Exception:
            return False

    def detect_all(self, gray):
        # detect on a half-res copy (4x fewer pixels), then map boxes back to full-res
        # so ROIs used for training/matching are still cut from the full-res image
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        f = self.cc.detectMultiScale(small, 1.1, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return [tuple(int(v) * 2 for v in b) for b in f] if len(f) else []

    def detect(self, gray):
        f = self.detect_all(gray)
        if len(f) != 1:
            return None
        return f[0]

    def roi(self, gray, box):
        x, y, w, h = box
        return _pre(gray[y:y + h, x:x + w])

    def raw_roi(self, gray, box):
        x, y, w, h = box
        return gray[y:y + h, x:x + w]

    def train(self, samples, enrol_frames=None):
        n = enrol_frames or ENROL_FRAMES
        samples = samples[:n]
        if len(samples) >= 20:
            # drop the 10% of samples furthest from the mean face (bad crops / false detections)
            m = np.mean(np.array(samples, dtype=np.float32), axis=0)
            d = np.array([float(np.mean(np.abs(x.astype(np.float32) - m))) for x in samples])
            keep = d <= np.percentile(d, 90)
            samples = [x for x, k in zip(samples, keep) if k]
        arr = np.array(samples, dtype=np.uint8)
        lbl = np.zeros(len(samples), dtype=np.int32)
        self.lbph.train(arr, lbl)

        self.mean_face = np.mean(arr, axis=0).astype(np.uint8)
        self._mff = self.mean_face.astype(np.float32)

        self.ref_hists = [
            cv2.calcHist([s], [0], None, [256], [0, 256]).flatten() for s in samples
        ]
        self._fhc = np.mean(self.ref_hists, axis=0).astype(np.float32)

        os.makedirs(LOCK_DIR, exist_ok=True)
        _unhide(MODEL_FILE)
        _unhide(STATS_FILE)
        self.lbph.write(MODEL_FILE)
        with open(STATS_FILE, 'wb') as f:
            pickle.dump({'mean_face': self.mean_face, 'ref_hists': self.ref_hists, 'fhc': self._fhc}, f)
        _hide(MODEL_FILE)
        _hide(STATS_FILE)
        self.trained = True

    def _lbph_confidence(self, roi):
        if not self.trained:
            return 0.0, 999.0
        _label, dist = self.lbph.predict(roi)
        conf = max(0.0, min(100.0, 100.0 - dist))
        return conf, dist

    def _hist_confidence(self, roi):
        if self._fhc is None:
            return 0.0
        h = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten().astype(np.float32)
        corr = cv2.compareHist(h, self._fhc, cv2.HISTCMP_CORREL)  # -1..1
        return max(0.0, min(100.0, (corr + 1) / 2 * 100))

    def _template_confidence(self, roi):
        if self._mff is None:
            return 0.0
        a = roi.astype(np.float32)
        b = self._mff
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)  # -1..1, scalar since same size
        score = float(res.max())
        return max(0.0, min(100.0, (score + 1) / 2 * 100))

    def predict(self, roi):
        """Returns (fused_confidence, breakdown) where breakdown is
        (lbph_conf, hist_conf, template_conf, lbph_distance) — the
        individual scores are what you want logged per-attempt so the
        weight-combination sweep in the paper can be computed without
        re-running the camera pipeline for every combination (see
        experiments/run_experiments.py, which instead re-weights the
        same three logged scores)."""
        if not self.trained:
            return 0.0, (0.0, 0.0, 0.0, 999.0)
        lbph_conf, dist = self._lbph_confidence(roi)
        hist_conf = self._hist_confidence(roi)
        template_conf = self._template_confidence(roi)
        fused = W_LBPH * lbph_conf + W_HIST * hist_conf + W_TEMPLATE * template_conf
        return fused, (lbph_conf, hist_conf, template_conf, dist)

    def wipe(self):
        self.trained = False
        self.mean_face = None
        self.ref_hists = []
        self._mff = None
        self._fhc = None
        for n in ('face.xml', 'face_stats.pkl', 'vault.key', 'state.json',
                  'audit.log', 'config.json', 'drive.id'):
            p = _p('.biolock', n)
            _unhide(p)
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(SNAP_DIR):
            try:
                shutil.rmtree(SNAP_DIR)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Overlay animation (particles, scan-line, HUD text on the camera feed)
# --------------------------------------------------------------------------
class ParticleSystem:

    def __init__(self):
        self.active = False
        self.W = 0
        self.H = 0
        self.px = None
        self.py = None

    def burst(self, cx, cy, W, H, n=140):
        self.W = W
        self.H = H
        self.active = True
        ang = np.random.uniform(0, math.pi * 2, n)
        spd = np.random.uniform(2, 9, n)
        self.px = np.full(n, float(cx))
        self.py = np.full(n, float(cy))
        self.vx = np.cos(ang) * spd
        self.vy = np.sin(ang) * spd
        self.life = np.ones(n)
        self.decay = np.random.uniform(0.018, 0.045, n)
        self.sizes = np.random.randint(3, 10, n)
        pal = np.array([
            (34, 197, 94), (139, 92, 246), (34, 211, 238),
            (251, 191, 36), (240, 240, 255)], dtype=np.uint8)
        self.cols = pal[np.random.randint(0, len(pal), n)]

    def update(self, frame):
        if not self.active or self.px is None:
            return frame
        self.px = self.px + self.vx
        self.py = self.py + self.vy
        self.vy = self.vy + 0.15
        self.life = self.life - self.decay
        alive = self.life > 0
        if not np.any(alive):
            self.active = False
            return frame
        for i in range(len(self.px)):
            if not alive[i]:
                continue
            x, y = int(self.px[i]), int(self.py[i])
            if 0 <= x < self.W and 0 <= y < self.H:
                r = max(1, int(self.sizes[i] * self.life[i]))
                col = tuple(int(c) for c in self.cols[i])
                cv2.circle(frame, (x, y), r, col, -1, cv2.LINE_AA)
        self.px, self.py = self.px[alive], self.py[alive]
        self.vx, self.vy = self.vx[alive], self.vy[alive]
        self.life, self.decay = self.life[alive], self.decay[alive]
        self.sizes, self.cols = self.sizes[alive], self.cols[alive]
        return frame


_ASCII = str.maketrans({'—': '-', '–': '-', '…': '...', '✓': 'OK', '⚠': '!', '→': '->', '×': 'x'})


def _t(txt):
    return str(txt).translate(_ASCII)


class Anim:

    def __init__(self):
        self._t = time.time()
        self._scan = 0
        self._pulse = 0
        self._pd = 1
        self._flash = 0
        self.particles = ParticleSystem()

    def tick(self):
        dt = time.time() - self._t
        self._t = time.time()
        self._scan = (self._scan + int(240 * dt)) % 720
        self._pulse += self._pd * dt * 2
        if self._pulse >= 1:
            self._pulse = 1
            self._pd = -1
        elif self._pulse <= 0:
            self._pulse = 0
            self._pd = 1
        self._flash = max(0, self._flash - dt * 2.5)

    def flash_green(self):
        self._flash = 1

    def _bgr(self, h):
        h = h.lstrip('#')
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    def render(self, frame, all_boxes, box, stage, conf, steps, instr, label,
               col_hex, multi=False, quality=(True, 100)):
        self.tick()
        h, w = frame.shape[:2]
        col = self._bgr(col_hex)

        if all_boxes:
            for (bx, by, bw, bh) in all_boxes:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1, cv2.LINE_AA)

        if box is not None:
            x, y, bw, bh = box
            pulse = int(4 + self._pulse * 3)
            cv2.rectangle(frame, (x - pulse, y - pulse), (x + bw + pulse, y + bh + pulse), col, 2, cv2.LINE_AA)
            if bh:
                scan_y = y + int(self._scan % bh)
                cv2.line(frame, (x, scan_y), (x + bw, scan_y), col, 1, cv2.LINE_AA)

        if self._flash > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (34, 197, 94), -1)
            a = self._flash * 0.35
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

        cv2.rectangle(frame, (0, 0), (w, 34), (18, 18, 42), -1)
        cv2.putText(frame, _t(label), (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        if conf:
            cv2.putText(frame, f'{conf:.0f}%', (w - 70, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, h - 30), (w, h), (18, 18, 42), -1)
        cv2.putText(frame, _t(instr), (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 255), 1, cv2.LINE_AA)

        sx = 12
        for name, ok, active in steps:
            chip_col = (52, 211, 153) if ok else ((139, 92, 246) if active else (60, 60, 90))
            cv2.circle(frame, (sx + 5, 46), 5, chip_col, -1, cv2.LINE_AA)
            cv2.putText(frame, _t(name), (sx + 15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 245), 1, cv2.LINE_AA)
            sx += 16 + 9 * len(name)

        if multi:
            cv2.putText(frame, '! MULTIPLE FACES', (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self._bgr(RED), 2, cv2.LINE_AA)

        ok_q, score_q = quality if isinstance(quality, tuple) else (True, 100)
        if not ok_q:
            cv2.putText(frame, f'Lighting {score_q:.0f}%', (w - 140, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, self._bgr(AMBER), 1, cv2.LINE_AA)

        frame = self.particles.update(frame)
        return frame


class CameraWorker(threading.Thread):
    """Grabs frames and runs face detection off the Tk thread.

    The UI thread only ever picks up the newest finished packet, so a slow
    camera read or a slow detection pass can no longer freeze the window.
    Packet = (seq, small_bgr, gray, all_boxes, box, fresh)."""

    def __init__(self, cap, engine, process_every):
        super().__init__(daemon=True)
        self.cap = cap
        self.engine = engine
        self.process_every = max(1, int(process_every))
        self._lock = threading.Lock()
        self._packet = None
        self._stop_evt = threading.Event()
        self.failed = False
        self._seq = 0
        self._boxes = ([], None)

    def stop(self):
        self._stop_evt.set()

    def latest(self, last_seq):
        with self._lock:
            p = self._packet
        return p if p is not None and p[0] != last_seq else None

    def run(self):
        errs = 0
        while not self._stop_evt.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                errs += 1
                if errs > 30:
                    self.failed = True
                    return
                time.sleep(0.03)
                continue
            errs = 0
            try:
                frame = cv2.flip(frame, 1)
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                self._seq += 1
                fresh = self._seq % self.process_every == 0 or self._seq == 1
                if fresh:
                    all_boxes = self.engine.detect_all(gray)
                    box = all_boxes[0] if len(all_boxes) == 1 else None
                    self._boxes = (all_boxes, box)
                all_boxes, box = self._boxes
                with self._lock:
                    self._packet = (self._seq, small, gray, all_boxes, box, fresh)
            except Exception:
                time.sleep(0.01)


# --------------------------------------------------------------------------
# Camera selection
# --------------------------------------------------------------------------
def _scan_cams(max_idx=5):
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if _WIN else 0)
        try:
            if cap.isOpened():
                ok, _frame = cap.read()
                if ok:
                    found.append(i)
        finally:
            cap.release()
    return found


class CamPicker(tk.Toplevel):
    """Modal dialog letting the user pick which camera to use.
    Sets self.result to the chosen index (or None if cancelled)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title('Select camera')
        self.configure(bg=BG)
        self.geometry('380x280')
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None

        tk.Label(self, text='Choose a camera', font=('Segoe UI', 13, 'bold'),
                 fg=FG, bg=BG).pack(pady=(18, 10))

        cams = _scan_cams()
        if not cams:
            tk.Label(self, text='No camera detected.', fg=RED_L, bg=BG,
                      font=('Segoe UI', 10)).pack(pady=10)
        else:
            for idx in cams:
                name = CAM_NAMES.get(idx, f'Camera {idx}')
                b = tk.Button(self, text=f'{name}   (index {idx})', bg=CARD, fg=FG,
                              relief='flat', bd=0, pady=8, activebackground=ACCENT,
                              command=lambda i=idx: self._choose(i))
                b.pack(fill='x', padx=20, pady=4)

        tk.Button(self, text='Cancel', bg=BG, fg=MUTED, relief='flat', bd=0,
                  command=self.destroy).pack(pady=(16, 10))

    def _choose(self, idx):
        self.result = idx
        self.destroy()


# --------------------------------------------------------------------------
# Splash screen
# --------------------------------------------------------------------------
class Splash:

    def __init__(self, root):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=BG)
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        W, H = 560, 360
        self.win.geometry(f'{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}')
        self.win.attributes('-topmost', True)

        fr = tk.Frame(self.win, bg=ACCENT, padx=2, pady=2)
        fr.pack(fill='both', expand=True)
        inn = tk.Frame(fr, bg=BG)
        inn.pack(fill='both', expand=True)

        tk.Label(inn, text='🔒', font=('Segoe UI', 58), fg=ACCENT_L, bg=BG).pack(pady=(34, 4))
        tk.Label(inn, text='BioLock SSD', font=('Segoe UI', 26, 'bold'), fg=FG, bg=BG).pack()
        tk.Label(inn, text=f'Version {APP_VERSION}  ·  Final Year Project',
                 font=('Segoe UI', 11), fg=MUTED, bg=BG).pack(pady=(2, 16))

        self.bar_v = tk.DoubleVar(value=0)
        sty = ttk.Style()
        sty.theme_use('default')
        sty.configure('SP.Horizontal.TProgressbar', troughcolor=FAINT,
                      background=ACCENT_L, thickness=5, borderwidth=0)
        ttk.Progressbar(inn, style='SP.Horizontal.TProgressbar', variable=self.bar_v,
                        mode='determinate', length=400).pack()

        self.msg = tk.Label(inn, text='Initialising…', font=('Segoe UI', 10), fg=MUTED, bg=BG)
        self.msg.pack(pady=8)

        tk.Label(inn, text='Pasupuleti Vedh Rishi',
                 font=('Segoe UI', 8), fg=FAINT, bg=BG).pack()
        self.win.update()

    def update(self, pct, msg):
        try:
            self.bar_v.set(pct)
            self.msg.config(text=msg)
            self.win.update()
        except Exception:
            pass

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass


def show_about(root):
    win = tk.Toplevel(root)
    win.title('About BioLock')
    win.configure(bg=BG)
    win.geometry('420x320')
    win.resizable(False, False)
    win.transient(root)

    tk.Label(win, text='🔒', font=('Segoe UI', 40), fg=ACCENT_L, bg=BG).pack(pady=(22, 4))
    tk.Label(win, text='BioLock SSD', font=('Segoe UI', 18, 'bold'), fg=FG, bg=BG).pack()
    tk.Label(win, text=f'Version {APP_VERSION}  ·  Final Year Project',
             font=('Segoe UI', 10), fg=MUTED, bg=BG).pack(pady=(2, 14))
    tk.Label(win, text='Face-recognition based USB drive lock\nwith liveness detection and an encrypted vault.',
             font=('Segoe UI', 9), fg=MUTED, bg=BG, justify='center').pack()
    tk.Label(win, text='Pasupuleti Vedh Rishi',
             font=('Segoe UI', 8), fg=FAINT, bg=BG, justify='center').pack(pady=(18, 10))
    tk.Button(win, text='Close', command=win.destroy, bg=CARD, fg=FG,
              relief='flat', bd=0, padx=20, pady=6).pack(pady=6)


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
class BioLock:
    ST_MOTION = 0
    ST_BLINK = 1
    ST_HEAD = 2
    ST_FACE = 3

    MODE_IDLE = 'idle'
    MODE_ENROL = 'enrol'
    MODE_VERIFY = 'verify'

    TRIAL_LABELS = ['genuine', 'impostor', 'attack_photo', 'attack_phone', 'attack_replay']
    LIGHTING_TAGS = ['', 'bright', 'normal', 'dim', 'backlit']
    MOVEMENT_TAGS = ['', 'still', 'head_turn', 'walking', 'shaky_hand']

    def __init__(self, root):
        self.root = root
        self.root.title(f'BioLock SSD  v{APP_VERSION}')
        self.root.geometry('1060x700')
        self.root.minsize(920, 620)
        self.root.configure(bg=BG)

        self.drive_ltr = _drive_letter()
        self.state = _load_state()
        self.cfg = _load_cfg()
        self.engine = FaceEngine()
        self.motion = MotionLive()
        self.blink = BlinkDetector()
        self.head = HeadTurn()
        self.anim = Anim()

        self.mode = self.MODE_IDLE
        self.stage = self.ST_MOTION
        self.cap = None
        self.cam_idx = self.cfg.get('cam_index')
        self.unlocked = False
        self.frame_n = 0
        self._fsv = False
        self._last_act = time.time()
        self._quality = (True, 100)
        self._enrol_samples = []
        self._sharp_hist = []
        self._last_roi = None
        self._verify_hits = 0
        self.att_lbl = None
        self._attempt_token = None

        # Testing-mode tags — used only for experiment logging, never for
        # the actual accept/reject decision.
        self.trial_label = tk.StringVar(value='genuine')
        self.lighting_condition = tk.StringVar(value='')
        self.movement_condition = tk.StringVar(value='')

        self.mv_att = tk.StringVar(value=f'Failed unlock attempts: 0/{MAX_ATTEMPTS}')
        self.mv_trials = tk.StringVar(value='Trials logged: 0')
        self.no_lockout = tk.BooleanVar(value=False)
        self.mv_motion = tk.StringVar(value='0.000')
        self.mv_liveness = tk.StringVar(value='…')
        self.mv_step = tk.StringVar(value='—')
        self.mv_conf = tk.StringVar(value='—')
        self.sv = tk.StringVar(value='')

        self._build_ui()
        self._update_trials()

        self.root.bind('<F11>', lambda e: self._fs())
        self.root.bind('<Escape>', lambda e: self._esc())
        self.root.protocol('WM_DELETE_WINDOW', self._quit)

        if not os.path.exists(DRIVE_ID):
            _create_drive_id()
        if _WIN:
            _register_autorun()
            _write_autorun()

        self._refresh()
        self._session_check()
        self._update_drive_bar()

    # ---- window chrome -----------------------------------------------
    def _fs(self):
        self._fsv = not self._fsv
        self.root.attributes('-fullscreen', self._fsv)

    def _esc(self):
        if self._fsv:
            self._fsv = False
            self.root.attributes('-fullscreen', False)
            return
        if self.mode != self.MODE_IDLE:
            self._stop(reset=True)

    # ---- background timers ---------------------------------------------
    def _session_check(self):
        try:
            if self.unlocked:
                idle = time.time() - self._last_act
                mins = self.cfg.get('auto_lock_mins', AUTO_LOCK_M)
                rem = int(mins * 60 - idle)
                if rem <= 0:
                    self._lock()
                    _log('AUTO-LOCK', f'idle {mins}min')
                elif rem <= 60:
                    self.timer_lbl.config(text=f'⏱  Auto-lock {rem}s', fg=AMBER_L)
                else:
                    self.timer_lbl.config(text=f'⏱  Auto-lock {rem // 60}m', fg=FAINT)
            else:
                self.timer_lbl.config(text='')
            self.root.after(5000, self._session_check)
        except Exception:
            self.root.after(5000, self._session_check)

    def _update_drive_bar(self):
        try:
            free, total = _drive_free_gb(self.drive_ltr)
            vmb = _vault_size_mb()
            used = max(0, min(100, int((1 - free / max(total, 0.001)) * 100)))
            self.drive_info.config(text=f'{self.drive_ltr}:  {free:.1f} GB free  ·  Vault {vmb:.1f} MB')
            self.drive_bar['value'] = used
            self.root.after(30000, self._update_drive_bar)
        except Exception:
            self.root.after(30000, self._update_drive_bar)

    # ---- UI layout -------------------------------------------------------
    def _build_ui(self):
        top = tk.Canvas(self.root, height=3, bg=ACCENT, highlightthickness=0)
        top.pack(fill='x', side='top')
        top.bind('<Configure>', lambda e: self._grad(top))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True)

        # ---- Left sidebar ----
        side = tk.Frame(body, bg=SURFACE, width=320)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)

        self.rule = tk.Frame(side, bg=RED, height=4)
        self.rule.pack(fill='x')

        self.hl = tk.Label(side, text='Drive\nLocked', font=('Segoe UI', 20, 'bold'),
                            fg=FG, bg=SURFACE, justify='left')
        self.hl.pack(anchor='w', padx=18, pady=(18, 4))

        self.sl = tk.Label(side, text='', font=('Segoe UI', 9), fg=MUTED, bg=SURFACE,
                            wraplength=280, justify='left')
        self.sl.pack(anchor='w', padx=18)

        self.timer_lbl = tk.Label(side, text='', font=('Segoe UI', 9), fg=FAINT, bg=SURFACE)
        self.timer_lbl.pack(anchor='w', padx=18, pady=(4, 0))

        btn_wrap = tk.Frame(side, bg=SURFACE)
        btn_wrap.pack(fill='x', padx=18, pady=(16, 4))

        self.b_main = tk.Button(btn_wrap, text='Unlock with face ID', bg=ACCENT, fg='#fff',
                                 relief='flat', bd=0, pady=10, activebackground=ACCENT_L,
                                 command=self._primary)
        self.b_lock = tk.Button(btn_wrap, text='Lock vault', bg=CARD, fg=FG,
                                 relief='flat', bd=0, pady=8, command=self._lock)
        self.b_cam = tk.Button(btn_wrap, text='Change camera', bg=CARD, fg=MUTED,
                                relief='flat', bd=0, pady=8, command=self._pick_cam)
        self.b_log = tk.Button(btn_wrap, text='View audit log', bg=CARD, fg=MUTED,
                                relief='flat', bd=0, pady=8, command=self._view_log)
        self.b_wipe = tk.Button(btn_wrap, text='Wipe & reset', bg=CARD, fg=RED_L,
                                 relief='flat', bd=0, pady=8, command=self._wipe_prompt)
        self.b_main.pack(fill='x', pady=4)

        pf = tk.Frame(side, bg=SURFACE)
        pf.pack(fill='x', padx=18, pady=(10, 0))
        self.pbar = ttk.Progressbar(pf, orient='horizontal', mode='determinate', maximum=100)
        self.pbar.pack(fill='x', side='left', expand=True)
        self.pbar_pct = tk.Label(pf, text='0%', font=('Segoe UI', 8), fg=MUTED, bg=SURFACE)
        self.pbar_pct.pack(side='left', padx=(6, 0))

        steps_frame = tk.Frame(side, bg=SURFACE)
        steps_frame.pack(fill='x', padx=18, pady=(10, 4))
        self.step_labels = []
        for name in ('Presence', f'Blink ×{BLINKS_NEEDED}', 'Head L→R', 'Face match'):
            row = tk.Frame(steps_frame, bg=SURFACE)
            row.pack(fill='x', pady=1)
            ic = tk.Label(row, text='o', width=3, fg=FAINT, bg=SURFACE, font=('Consolas', 9))
            ic.pack(side='left')
            lb = tk.Label(row, text=name, fg=MUTED, bg=SURFACE, font=('Segoe UI', 9))
            lb.pack(side='left')
            self.step_labels.append((ic, lb))

        read_frame = tk.LabelFrame(side, text='Live signals', bg=SURFACE, fg=MUTED, font=('Segoe UI', 8))
        read_frame.pack(fill='x', padx=18, pady=(10, 4))
        for lbl_text, var in (('Motion', self.mv_motion), ('Liveness', self.mv_liveness),
                               ('Step', self.mv_step), ('Match score', self.mv_conf)):
            r = tk.Frame(read_frame, bg=SURFACE)
            r.pack(fill='x')
            tk.Label(r, text=lbl_text, fg=FAINT, bg=SURFACE, font=('Segoe UI', 8),
                     width=11, anchor='w').pack(side='left')
            tk.Label(r, textvariable=var, fg=FG, bg=SURFACE, font=('Consolas', 8)).pack(side='left')

        self.att_lbl = tk.Label(side, textvariable=self.mv_att, fg=ACCENT_L, bg=SURFACE, font=('Segoe UI', 8))
        self.att_lbl.pack(anchor='w', padx=18, pady=(6, 0))
        tk.Label(side, text='Match score = weighted similarity (0-100): LBPH 0.5, histogram 0.3,\n'
                            'template 0.2. It is a score, not a probability.',
                 fg=FAINT, bg=SURFACE, font=('Segoe UI', 7), justify='left',
                 wraplength=300).pack(anchor='w', padx=18, pady=(2, 0))

        # ---- Testing Mode panel (experiment logging tags only) ----
        test_frame = tk.LabelFrame(side, text='Testing mode (for experiment logging)',
                                    bg=SURFACE, fg=TEAL_L, font=('Segoe UI', 8))
        test_frame.pack(fill='x', padx=18, pady=(10, 4))

        tk.Label(test_frame, text='Trial label', fg=FAINT, bg=SURFACE, font=('Segoe UI', 8),
                 anchor='w').pack(fill='x')
        ttk.Combobox(test_frame, textvariable=self.trial_label, values=self.TRIAL_LABELS,
                     state='readonly', font=('Segoe UI', 8)).pack(fill='x', pady=(0, 4))

        tk.Label(test_frame, text='Lighting condition', fg=FAINT, bg=SURFACE, font=('Segoe UI', 8),
                 anchor='w').pack(fill='x')
        ttk.Combobox(test_frame, textvariable=self.lighting_condition, values=self.LIGHTING_TAGS,
                     state='readonly', font=('Segoe UI', 8)).pack(fill='x', pady=(0, 4))

        tk.Label(test_frame, text='Movement condition', fg=FAINT, bg=SURFACE, font=('Segoe UI', 8),
                 anchor='w').pack(fill='x')
        ttk.Combobox(test_frame, textvariable=self.movement_condition, values=self.MOVEMENT_TAGS,
                     state='readonly', font=('Segoe UI', 8)).pack(fill='x', pady=(0, 2))

        tk.Label(test_frame, textvariable=self.mv_trials, fg=TEAL_L, bg=SURFACE,
                 font=('Segoe UI', 8), anchor='w', justify='left', wraplength=300).pack(fill='x')
        tk.Checkbutton(test_frame, text='Disable lockout/wipe while testing', variable=self.no_lockout,
                       bg=SURFACE, fg=MUTED, selectcolor=CARD, activebackground=SURFACE,
                       activeforeground=FG, font=('Segoe UI', 8), anchor='w').pack(fill='x')
        tk.Label(test_frame,
                 text='Set these before each attempt so logs/attempts.csv\n'
                      'records what you actually tested.',
                 fg=FAINT, bg=SURFACE, font=('Segoe UI', 7), justify='left',
                 wraplength=260).pack(fill='x', pady=(2, 4))

        self.drive_info = tk.Label(side, text='', fg=MUTED, bg=SURFACE, font=('Segoe UI', 8))
        self.drive_info.pack(anchor='w', padx=18, pady=(10, 2))
        self.drive_bar = ttk.Progressbar(side, orient='horizontal', mode='determinate', maximum=100)
        self.drive_bar.pack(fill='x', padx=18)

        tk.Button(side, text='About', bg=SURFACE, fg=FAINT, relief='flat', bd=0,
                  command=lambda: show_about(self.root)).pack(side='bottom', pady=10)

        # ---- Right preview pane ----
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        stage_frame = tk.Frame(right, bg='#000000')
        stage_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.ph = tk.Label(stage_frame, text='', font=('Segoe UI', 11), fg=MUTED,
                            bg='#000000', justify='center')
        self.ph.place(relx=0.5, rely=0.5, anchor='center')

        self.vid = tk.Label(stage_frame, bg='#000000')

    def _grad(self, c):
        try:
            w = c.winfo_width()
            if w < 10:
                return
            c.delete('all')
            for i in range(w):
                t = i / w
                r = int(109 + -101 * t)
                g = int(40 + 105 * t)
                b = int(217 + -39 * t)
                c.create_line(i, 0, i, 3, fill=f'#{r:02x}{g:02x}{b:02x}')
        except Exception:
            pass

    # ---- small view helpers ---------------------------------------------
    def _show_ph(self, t):
        self.vid.place_forget()
        self.ph.config(text=t)
        self.ph.place(relx=0.5, rely=0.5, anchor='center')

    def _show_vid(self):
        self.ph.place_forget()
        self.vid.place(relx=0.5, rely=0.5, anchor='center')

    def _pset(self, v):
        self.pbar['value'] = v
        try:
            self.pbar_pct.config(text=f'{int(v)}%')
        except Exception:
            pass

    def _upd_steps(self, stage):
        for i, (ic, lb) in enumerate(self.step_labels):
            s = i + 1
            if stage > s:
                ic.config(text='OK', fg=GREEN_L)
                lb.config(fg=GREEN_L)
            elif stage == s:
                ic.config(text='>>', fg=ACCENT_L)
                lb.config(fg=FG)
            else:
                ic.config(text='o', fg=FAINT)
                lb.config(fg=MUTED)

    def _steps(self, s1, s2, s3, s4, a):
        return [
            ('Presence', s1, a == 1),
            (f'Blink ×{BLINKS_NEEDED}', s2, a == 2),
            ('Head L→R', s3, a == 3),
            ('Face match', s4, a == 4),
        ]

    def _refresh(self):
        for b in (self.b_main, self.b_lock, self.b_cam, self.b_log, self.b_wipe):
            b.pack_forget()

        atts = self.state.get('attempts', 0)
        self.mv_att.set(f'Failed unlock attempts: {atts}/{MAX_ATTEMPTS}')
        if self.att_lbl:
            if atts >= 3:
                self.att_lbl.config(fg=RED_L)
            elif atts > 0:
                self.att_lbl.config(fg=AMBER_L)
            else:
                self.att_lbl.config(fg=ACCENT_L)

        total = self.state.get('total_unlocks', 0)

        if self.unlocked:
            self.hl.config(text='Vault\nOpen')
            self.rule.config(bg=GREEN)
            self.sl.config(text=f'Session active  ·  {total} total unlock(s)')
            self.sv.set('Vault is open and accessible.')
            self._upd_steps(5)
            self.b_main.config(text='Open  SecureVault', bg=GREEN, fg='#000')
            self.b_main.pack(fill='x', pady=4)
            self.b_lock.pack(fill='x', pady=4)
            self.b_log.pack(fill='x', pady=4)
            self._show_ph('Vault unlocked  ✓\n\nClick  Open SecureVault  to see your files.\n'
                           'Click  Lock vault  when finished.\n\nAuto-locks after idle.')
            return

        if not self.state.get('enrolled'):
            self.hl.config(text='Welcome\nto BioLock')
            self.rule.config(bg=AMBER)
            self.sl.config(text=f'Drive {self.drive_ltr}:  ready  ·  Enrol your face once to secure this drive.')
            self.sv.set('First time setup — click  Begin face enrolment.')
            self._upd_steps(0)
            self.b_main.config(text='Begin face enrolment', bg=ACCENT, fg='#fff')
            self.b_main.pack(fill='x', pady=4)
            self.b_cam.pack(fill='x', pady=4)
            self._show_ph(f'BioLock is installed on drive  {self.drive_ltr}:\n\n'
                          f'One-time face enrolment:\n\n'
                          f'  1.  Click  Begin face enrolment\n'
                          f'  2.  Select your camera when asked\n'
                          f'  3.  Stay still — presence detected\n'
                          f'  4.  Blink  {BLINKS_NEEDED}  times\n'
                          f'  5.  Turn head slowly LEFT then RIGHT\n'
                          f'  6.  Hold still — {ENROL_FRAMES} frames captured\n\n'
                          f'After enrolment:\n'
                          f'  •  Your face is the ONLY key to this drive\n'
                          f'  •  {MAX_ATTEMPTS} failed attempts = automatic wipe\n'
                          f'  •  Auto-locks after {AUTO_LOCK_M} min idle')
            return

        rem = MAX_ATTEMPTS - atts
        self.hl.config(text='Drive\nLocked')
        self.rule.config(bg=RED)
        self.sl.config(text=f'4-step verification required  ·  {rem} attempt(s) remaining')
        self.sv.set('Click  Unlock with face ID')
        self._upd_steps(0)
        self.b_main.config(text='Unlock with face ID', bg=ACCENT, fg='#fff')
        self.b_main.pack(fill='x', pady=4)
        self.b_cam.pack(fill='x', pady=4)
        self.b_log.pack(fill='x', pady=4)
        self.b_wipe.pack(fill='x', pady=(14, 4))
        self._show_ph(f'Drive is locked.\n\nClick  Unlock with face ID  — camera opens here.\n\n'
                      f'4-step verification:\n'
                      f'  1.  Presence detected  (stay still)\n'
                      f'  2.  Blink  {BLINKS_NEEDED}  times\n'
                      f'  3.  Turn head left → right\n'
                      f'  4.  Face matched at  {MIN_CONF:.0f}%+  confidence\n\n'
                      f'⚠  Only the enrolled face can pass.\n'
                      f'    {rem} attempt(s) before automatic wipe.')

    # ---- primary action button ------------------------------------------
    def _primary(self):
        self._last_act = time.time()
        if self.unlocked:
            os.makedirs(VAULT_DIR, exist_ok=True)
            _unhide(VAULT_DIR)
            subprocess.Popen(f'explorer "{VAULT_DIR}"')
            return
        if not self.state.get('enrolled'):
            self._start(self.MODE_ENROL)
            return
        self._start(self.MODE_VERIFY)

    # ---- camera lifecycle -------------------------------------------------
    def _pick_cam(self):
        d = CamPicker(self.root)
        self.root.wait_window(d)
        if d.result is not None:
            self.cam_idx = d.result
            self.cfg['cam_index'] = d.result
            _save_cfg(self.cfg)

    def _start(self, mode):
        self.mode = mode
        self.stage = self.ST_MOTION
        self.frame_n = 0
        self.motion.reset()
        self.blink.reset()
        self.head.reset()
        self._enrol_samples = []
        self._sharp_hist = []
        self._last_roi = None
        self._verify_hits = 0
        self._live = {}
        self._verify_t0 = time.time()
        self._attempt_token = METRICS.start_attempt()

        if self.cam_idx is None:
            d = CamPicker(self.root)
            self.root.wait_window(d)
            if d.result is None:
                self.mode = self.MODE_IDLE
                return
            self.cam_idx = d.result
            self.cfg['cam_index'] = d.result
            _save_cfg(self.cfg)

        self._open_cam()

    def _open_cam(self):
        try:
            self.cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW if _WIN else 0)
            if not self.cap or not self.cap.isOpened():
                self._cam_failed('Could not open the selected camera.')
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # avoid stale queued frames
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self._cam_ready()
        except Exception as e:
            self._cam_failed(str(e))

    def _cam_ready(self):
        if self.mode == self.MODE_IDLE:
            return
        self._show_vid()
        self._worker = CameraWorker(self.cap, self.engine, PROCESS_EVERY)
        self._last_seq = 0
        self._worker.start()
        self.sv.set('Camera live — stay still to begin…')
        self.root.after(35, self._loop)

    def _cam_failed(self, msg):
        self._stop(reset=True)
        messagebox.showerror('Camera failed', msg)

    def _stop(self, reset=False):
        self.mode = self.MODE_IDLE
        cap = self.cap
        self.cap = None
        w = getattr(self, '_worker', None)
        self._worker = None
        if w is not None:
            w.stop()
            w.join(timeout=1.0)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if reset:
            self._refresh()

    def _loop(self):
        # single scheduler: _tick must never re-arm itself, otherwise every
        # frame spawns an extra loop and the Tk event queue floods.
        if self.mode == self.MODE_IDLE:
            return
        self._tick()
        if self.mode != self.MODE_IDLE:
            self.root.after(30, self._loop)

    def _current_cfg_snapshot(self):
        return {
            'min_conf': MIN_CONF, 'motion_thresh': MOTION_THRESH, 'enrol_frames': ENROL_FRAMES,
            'w_lbph': W_LBPH, 'w_hist': W_HIST, 'w_template': W_TEMPLATE,
        }

    def _tick(self):
        w = getattr(self, '_worker', None)
        if w is None:
            return
        if w.failed:
            self._stop()
            messagebox.showerror('Camera lost', 'Camera stopped.\nReconnect and try again.')
            self._refresh()
            return
        pkt = w.latest(self._last_seq)
        if pkt is None:
            return                      # no new frame yet; keep UI responsive
        self._last_seq, small, gray, all_boxes, box, fresh = pkt
        self.frame_n += 1

        if fresh:
            # only log when detection actually ran (avoids a CSV write per frame)
            METRICS.log_detection(self.mode, self.stage, len(all_boxes),
                                   self.lighting_condition.get(), self.movement_condition.get())

        multi = len(all_boxes) > 1
        sbox = box

        self._quality = (True, 100)
        if box is not None:
            raw = self.engine.raw_roi(gray, box)
            if raw.size:
                q, _qmsg = _lighting_quality(raw)
                self._quality = (q >= 50, q)

        if box is None and not multi:
            self.sv.set('Face not detected — center your face in frame.')

        if self.mode == self.MODE_ENROL:
            rendered, done = self._enrol(small, gray, all_boxes, box, sbox, multi)
        elif self.mode == self.MODE_VERIFY:
            rendered, done = self._verify(small, gray, all_boxes, box, sbox, multi)
        else:
            rendered, done = small, False

        img = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.vid.imgtk = imgtk
        self.vid.config(image=imgtk)

    def _accept_sample(self, gray, box, roi):
        """Quality gate for enrolment samples: only sharp, well-lit, adequately
        sized and non-duplicate crops reach the training set."""
        if box[2] < 80:
            return False, 'Move closer to the camera'
        if not self._quality[0]:
            return False, 'Improve lighting - face the light'
        sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        hist = self._sharp_hist
        hist.append(sharp)
        if len(hist) > 30:
            hist.pop(0)
        if len(hist) >= 8 and sharp < 0.6 * float(np.median(hist)):
            return False, 'Hold steady - image is blurry'
        last = getattr(self, '_last_roi', None)
        if last is not None:
            if float(np.mean(cv2.absdiff(roi, last))) < 1.2:
                return False, 'Slowly tilt/turn your head a little'
        return True, ''

    # ---- enrolment ----------------------------------------------------
    def _enrol(self, frame, gray, all_boxes, box, sbox, multi):
        if multi:
            return (self.anim.render(frame, all_boxes, box, 0, 0, [],
                                      'Only ONE person in frame',
                                      'ENROLMENT  —  MULTI-FACE DETECTED', RED, True), False)
        if box is None:
            return (frame, False)

        if self.stage == self.ST_MOTION:
            self._upd_steps(1)
            ml, mp, mm, ms = self.motion.update(gray, box)
            self.mv_motion.set(f'{ms:.3f}')
            self.mv_liveness.set('PASS' if ml else '…')
            self._pset(mp * 25)
            self.mv_step.set('PRESENCE')
            self.mv_conf.set('—')
            self.sv.set(mm)
            frame = self.anim.render(frame, all_boxes, box, 1, 0,
                                      self._steps(ml, False, False, False, 1), mm,
                                      'ENROLMENT  —  Step 1: Presence', AMBER, quality=self._quality)
            if ml:
                self.stage = self.ST_BLINK
            return (frame, False)

        if self.stage == self.ST_BLINK:
            self._upd_steps(2)
            roi = self.engine.raw_roi(gray, box)
            bl, bd, bm = self.blink.update(roi)
            self._pset(25 + (bl / BLINKS_NEEDED) * 25)
            self.mv_step.set('BLINK')
            self.sv.set(bm)
            frame = self.anim.render(frame, all_boxes, box, 2, 0,
                                      self._steps(True, bd, False, False, 2), bm,
                                      'ENROLMENT  —  Step 2: Blink', AMBER, quality=self._quality)
            if bd:
                self.stage = self.ST_HEAD
            return (frame, False)

        if self.stage == self.ST_HEAD:
            self._upd_steps(3)
            _hp, hd, hm, _ = self.head.update(box)
            self._pset(50 + (25 if hd else 12))
            self.mv_step.set('HEAD TURN')
            self.sv.set(hm)
            frame = self.anim.render(frame, all_boxes, box, 3, 0,
                                      self._steps(True, True, hd, False, 3), hm,
                                      'ENROLMENT  —  Step 3: Head turn', AMBER, quality=self._quality)
            if hd:
                self.stage = self.ST_FACE
            return (frame, False)

        # ST_FACE — capture training samples
        self._upd_steps(4)
        roi = self.engine.roi(gray, box)
        ok_s, why = self._accept_sample(gray, box, roi)
        if ok_s:
            self._enrol_samples.append(roi)
            self._last_roi = roi
        n = len(self._enrol_samples)
        self._pset(75 + (n / ENROL_FRAMES) * 25)
        self.mv_step.set('CAPTURING')
        self.mv_conf.set(f'{n}/{ENROL_FRAMES}')
        self.sv.set(f'Capturing face  ({n}/{ENROL_FRAMES})' if ok_s else why)
        frame = self.anim.render(frame, all_boxes, box, 4, 0,
                                  self._steps(True, True, True, n >= ENROL_FRAMES, 4),
                                  f'Capturing {n}/{ENROL_FRAMES}',
                                  'ENROLMENT  —  Step 4: Face capture', ACCENT, quality=self._quality)

        if n >= ENROL_FRAMES:
            self.engine.train(self._enrol_samples, enrol_frames=ENROL_FRAMES)
            fh = self.engine.mean_face.tobytes()
            save_vault_key(fh)
            self.state['enrolled'] = True
            self.state['enrol_quality'] = self._quality[1]
            self.state['attempts'] = 0
            _save_state(self.state)
            self.anim.flash_green()
            _sound_ok()
            _log('ENROLLED', f'drive={self.drive_ltr}:  frames={ENROL_FRAMES}')
            METRICS.log_attempt(self._attempt_token, trial_label='enrolment', outcome='enrolled',
                                 confidence=100.0, cfg=self._current_cfg_snapshot(),
                                 lighting_condition=self.lighting_condition.get(),
                                 movement_condition=self.movement_condition.get())
            self._stop()
            messagebox.showinfo('Enrolment complete',
                                 'Your face has been enrolled.\nBioLock is now active on this drive.')
            self._refresh()
            return (frame, True)

        return (frame, False)

    # ---- verification ---------------------------------------------------
    LIVENESS_TIMEOUT = 45   # seconds allowed to pass presence + blink + head turn

    def _update_trials(self):
        c = METRICS.trial_counts()
        total = sum(c.values())
        detail = ', '.join(f'{k} {v}' for k, v in sorted(c.items()))
        self.mv_trials.set(f'Trials logged: {total}' + (f'  ({detail})' if detail else ''))

    def _liveness_timeout(self, frame):
        names = {self.ST_MOTION: 'presence', self.ST_BLINK: 'blink', self.ST_HEAD: 'head'}
        stage = names.get(self.stage)
        if stage is None or time.time() - self._verify_t0 <= self.LIVENESS_TIMEOUT:
            return None
        live = {'presence': False, 'blink': False, 'head': False}
        live.update(self._live)
        METRICS.log_attempt(self._attempt_token, trial_label=self.trial_label.get(),
                             outcome='reject', confidence=0.0,
                             cfg=self._current_cfg_snapshot(),
                             lighting_condition=self.lighting_condition.get(),
                             movement_condition=self.movement_condition.get(),
                             liveness=live, failed_stage=stage)
        _log('LIVENESS_TIMEOUT', f'stage={stage}', 'WARN')
        self._stop()
        self._update_trials()
        messagebox.showwarning('Liveness check not completed',
                               f'The {stage} step was not completed in {self.LIVENESS_TIMEOUT}s.\n'
                               'Access denied (this does not count as a failed unlock attempt).')
        self._refresh()
        return (frame, True)

    def _verify(self, frame, gray, all_boxes, box, sbox, multi):
        timed_out = self._liveness_timeout(frame)
        if timed_out is not None:
            return timed_out
        if multi:
            return (self.anim.render(frame, all_boxes, box, 0, 0, [],
                                      'Remove extra people from frame',
                                      'VERIFICATION  —  REJECTED: MULTIPLE FACES', RED, True), False)
        if box is None:
            return (frame, False)

        if self.stage == self.ST_MOTION:
            self._upd_steps(1)
            ml, mp, mm, ms = self.motion.update(gray, box)
            self.mv_motion.set(f'{ms:.3f}')
            self.mv_liveness.set('PASS' if ml else '…')
            self._pset(mp * 25)
            self.mv_step.set('PRESENCE')
            self.mv_conf.set('—')
            self.sv.set(mm)
            frame = self.anim.render(frame, all_boxes, box, 1, 0,
                                      self._steps(ml, False, False, False, 1), mm,
                                      'VERIFICATION  —  Step 1: Presence', AMBER)
            if ml:
                self._live['presence'] = True
                self.stage = self.ST_BLINK
            return (frame, False)

        if self.stage == self.ST_BLINK:
            self._upd_steps(2)
            roi = self.engine.raw_roi(gray, box)
            bl, bd, bm = self.blink.update(roi)
            self._pset(25 + (bl / BLINKS_NEEDED) * 25)
            self.mv_step.set('BLINK')
            self.sv.set(bm)
            frame = self.anim.render(frame, all_boxes, box, 2, 0,
                                      self._steps(True, bd, False, False, 2), bm,
                                      'VERIFICATION  —  Step 2: Blink', AMBER)
            if bd:
                self._live['blink'] = True
                self.stage = self.ST_HEAD
            return (frame, False)

        if self.stage == self.ST_HEAD:
            self._upd_steps(3)
            _hp, hd, hm, _ = self.head.update(box)
            self._pset(50 + (25 if hd else 12))
            self.mv_step.set('HEAD TURN')
            self.sv.set(hm)
            frame = self.anim.render(frame, all_boxes, box, 3, 0,
                                      self._steps(True, True, hd, False, 3), hm,
                                      'VERIFICATION  —  Step 3: Head turn', AMBER)
            if hd:
                self._live['head'] = True
                self.stage = self.ST_FACE
            return (frame, False)

        # ST_FACE — match against the enrolled model
        self._upd_steps(4)
        roi = self.engine.roi(gray, box)
        conf, (lbph_c, hist_c, tmpl_c, _dist) = self.engine.predict(roi)
        self.mv_step.set('MATCHING')
        self.mv_conf.set(f'{conf:.0f}/100')
        match = conf >= MIN_CONF

        frame = self.anim.render(frame, all_boxes, box, 4, conf,
                                  self._steps(True, True, True, match, 4),
                                  f'Match score {conf:.0f}/100',
                                  'VERIFICATION  —  Step 4: Face match',
                                  GREEN if match else RED)

        if match:
            fh = self.engine.mean_face.tobytes()
            if verify_vault_key(fh):
                self.unlocked = True
                self.state['attempts'] = 0
                self.state['total_unlocks'] = self.state.get('total_unlocks', 0) + 1
                self.state['last_unlock'] = datetime.datetime.now().isoformat()
                _save_state(self.state)
                self._last_act = time.time()
                self.anim.flash_green()
                _sound_ok()
                _log('UNLOCKED', f'drive={self.drive_ltr}:')
                METRICS.log_attempt(self._attempt_token, trial_label=self.trial_label.get(),
                                     outcome='accept', confidence=conf,
                                     lbph_conf=lbph_c, hist_conf=hist_c, template_conf=tmpl_c,
                                     cfg=self._current_cfg_snapshot(),
                                     lighting_condition=self.lighting_condition.get(),
                                     movement_condition=self.movement_condition.get(),
                                     liveness=dict(self._live), failed_stage='')
                self._stop()
                self._update_trials()
                self._refresh()
                return (frame, True)

        # short grace window so a single bad frame doesn't burn an attempt
        self._verify_hits = getattr(self, '_verify_hits', 0) + (0 if match else 1)
        if self._verify_hits > 40:
            testing = self.no_lockout.get()
            if not testing:
                self.state['attempts'] = self.state.get('attempts', 0) + 1
                _save_state(self.state)
            _sound_fail()
            _log('FAILED', f'total={self.state.get("attempts", 0)}' + (' (testing, no lockout)' if testing else ''), 'WARN')
            METRICS.log_attempt(self._attempt_token, trial_label=self.trial_label.get(),
                                 outcome='reject', confidence=conf,
                                 lbph_conf=lbph_c, hist_conf=hist_c, template_conf=tmpl_c,
                                 cfg=self._current_cfg_snapshot(),
                                 lighting_condition=self.lighting_condition.get(),
                                 movement_condition=self.movement_condition.get(),
                                 liveness=dict(self._live), failed_stage='match')
            self._stop()
            self._update_trials()
            if testing:
                messagebox.showinfo('Rejected (testing mode)',
                                    f'Face not recognised. Match score {conf:.0f}/100 is below {MIN_CONF}.\n'
                                    'Lockout is disabled, so no attempt was counted.')
                self._refresh()
            elif self.state['attempts'] >= MAX_ATTEMPTS:
                self._do_wipe()
            else:
                messagebox.showwarning('Verification failed',
                    f'Face not recognised.\n{MAX_ATTEMPTS - self.state["attempts"]} attempt(s) remaining.')
                self._refresh()
            return (frame, True)

        return (frame, False)

    # ---- vault / housekeeping -----------------------------------------
    def _open_vault(self):
        self._last_act = time.time()
        os.makedirs(VAULT_DIR, exist_ok=True)
        _unhide(VAULT_DIR)
        subprocess.Popen(f'explorer "{VAULT_DIR}"')
        self._refresh()

    def _lock(self):
        _hide(VAULT_DIR)
        self.unlocked = False
        _sound_fail()
        _log('LOCKED', f'drive={self.drive_ltr}:')
        self.sv.set('Drive locked securely.')
        self._refresh()

    def _view_log(self):
        win = tk.Toplevel(self.root)
        win.title('Audit Log')
        win.geometry('720x500')
        win.configure(bg=BG)
        win.transient(self.root)

        tk.Label(win, text='Authentication Audit Log', font=('Segoe UI', 15, 'bold'),
                 fg=FG, bg=BG).pack(pady=(16, 4))
        frm = tk.Frame(win, bg=CARD)
        frm.pack(fill='both', expand=True, padx=16, pady=10)
        sb = tk.Scrollbar(frm)
        sb.pack(side='right', fill='y')
        txt = tk.Text(frm, bg=CARD, fg=FG, font=('Consolas', 10), relief='flat', bd=0,
                       yscrollcommand=sb.set)
        txt.pack(fill='both', expand=True, padx=8, pady=8)
        sb.config(command=txt.yview)

        try:
            _unhide(LOG_FILE)
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    txt.insert('1.0', f.read())
            else:
                txt.insert('1.0', 'No log entries yet.')
            _hide(LOG_FILE)
        except Exception as e:
            txt.insert('1.0', f'Could not read log: {e}')
        txt.config(state='disabled')

    def _wipe_prompt(self):
        if not messagebox.askyesno('⚠  Wipe & Reset',
                'This permanently deletes:\n\n  •  Face enrolment\n  •  ALL files in SecureVault\n'
                '  •  All config and logs\n\nCannot be undone.', icon='warning'):
            return
        if messagebox.askyesno('Final confirmation', 'ALL DATA LOST FOREVER. Sure?', icon='warning'):
            self._do_wipe()

    def _do_wipe(self):
        _log('WIPE', f'drive={self.drive_ltr}:', 'WARN')
        self._stop()
        _unhide(VAULT_DIR)
        if os.path.exists(VAULT_DIR):
            try:
                shutil.rmtree(VAULT_DIR)
            except Exception:
                pass
        self.engine.wipe()
        self.state = dict(DEF_STATE)
        _save_state(self.state)
        self.unlocked = False
        self.cam_idx = None
        messagebox.showinfo('Wipe complete', 'Drive reset. Enrol a new face.')
        self._refresh()

    def _quit(self):
        if self.mode != self.MODE_VERIFY and self.unlocked:
            pass  # closing an unlocked session is allowed without penalty
        if self.unlocked and messagebox.askyesno('Lock?', 'Lock vault before closing?'):
            self._lock()
        self._stop()
        self.root.destroy()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    if not enforce_external_drive(root):
        root.destroy()
        sys.exit(0)
    splash = Splash(root)
    for pct, msg in (
        (10, 'Checking environment…'),
        (25, 'Loading face engine…'),
        (45, 'Reading drive state…'),
        (65, 'Initialising security…'),
        (85, 'Building interface…'),
        (100, 'Starting BioLock…'),
    ):
        splash.update(pct, msg)
    splash.close()
    root.deiconify()
    BioLock(root)
    root.mainloop()
