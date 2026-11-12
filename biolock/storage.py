"""Everything BioLock keeps on the drive: file locations, state, per-drive settings, audit log, drive identity.

All writes are atomic (see winutil.atomic_write) because the target is a removable
drive that can be pulled out at any moment.
"""

import os
import json
import shutil
import secrets
import datetime

from . import winutil
from . import __version__


DEF_STATE = {
    'enrolled': False,
    'attempts': 0,              # failed face matches since the last unlock
    'live_fails': 0,            # liveness time-outs in a row
    'cooldown_until': 0,        # wall-clock time before which no new attempt may start
    'total_unlocks': 0,
    'last_unlock': None,
    'enrol_quality': 0,
}

DEF_LOCAL = {
    'cam_index': None,
    'cam_backend': None,
}

# files written by releases before 10.0 (LBPH XML + pickled statistics)
LEGACY_MODEL_FILES = ('face.xml', 'face_stats.pkl')


class Store:
    """File layout below `base` (the folder the app runs from):

        .biolock/face_model.npz   enrolled template (features only, no pickle)
        .biolock/vault.key        salt + encrypted sentinel
        .biolock/state.json       enrolled flag, failed-attempt counter, unlock statistics
        .biolock/config.json      per-drive settings (selected camera)
        .biolock/drive.id         random identifier created on first run
        .biolock/audit.log        security events
        SecureVault/              the protected folder
        logs/                     experiment logs (attempts.csv, detections.csv)
    """

    def __init__(self, base):
        self.base = os.path.abspath(base)
        self.lock_dir = os.path.join(self.base, '.biolock')
        self.model_file = os.path.join(self.lock_dir, 'face_model.npz')
        self.vault_key = os.path.join(self.lock_dir, 'vault.key')
        self.state_file = os.path.join(self.lock_dir, 'state.json')
        self.cfg_file = os.path.join(self.lock_dir, 'config.json')
        self.drive_id_file = os.path.join(self.lock_dir, 'drive.id')
        self.log_file = os.path.join(self.lock_dir, 'audit.log')
        self.snap_dir = os.path.join(self.lock_dir, 'snapshots')
        self.vault_dir = os.path.join(self.base, 'SecureVault')
        self.logs_dir = os.path.join(self.base, 'logs')
        self.letter = winutil.drive_letter(self.base)

    # ---- plumbing -------------------------------------------------------
    def _ensure_dir(self):
        os.makedirs(self.lock_dir, exist_ok=True)
        winutil.set_hidden(self.lock_dir, True)

    def _read_json(self, path, default):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            out = dict(default)
            if isinstance(loaded, dict):
                out.update(loaded)
            return out
        except Exception:
            return dict(default)

    def _write_json(self, path, obj):
        self._ensure_dir()
        winutil.atomic_write(path, json.dumps(obj).encode('utf-8'))

    def writable(self):
        """False when the drive is read-only or full: the app cannot keep state there."""
        try:
            self._ensure_dir()
            probe = os.path.join(self.lock_dir, '.write_test')
            with open(probe, 'wb') as f:
                f.write(b'ok')
            os.remove(probe)
            return True
        except Exception:
            return False

    # ---- state / settings -------------------------------------------------
    def load_state(self):
        return self._read_json(self.state_file, DEF_STATE)

    def save_state(self, state):
        self._write_json(self.state_file, state)

    def load_local(self):
        return self._read_json(self.cfg_file, DEF_LOCAL)

    def save_local(self, cfg):
        self._write_json(self.cfg_file, cfg)

    # ---- drive identity ---------------------------------------------------
    def drive_id(self, create=True):
        """Random identifier stored on the drive; created on first use."""
        try:
            with open(self.drive_id_file, 'r', encoding='utf-8') as f:
                uid = json.load(f).get('id')
            if uid:
                return uid
        except Exception:
            pass
        if not create:
            return None
        uid = secrets.token_hex(16)
        self._write_json(self.drive_id_file, {'id': uid, 'created': datetime.datetime.now().isoformat(),
                                              'version': __version__})
        return uid

    def volume_serial(self):
        return winutil.volume_serial(self.letter)

    # ---- audit log ----------------------------------------------------------
    def log(self, event, detail='', level='INFO'):
        try:
            self._ensure_dir()
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = f'[{ts}] [{level}] {event}' + (f'  |  {detail}' if detail else '') + '\n'
            winutil.set_hidden(self.log_file, False)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line)
            winutil.set_hidden(self.log_file, True)
        except Exception:
            pass

    def read_log(self):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ''

    # ---- vault folder -------------------------------------------------------
    def vault_show(self):
        os.makedirs(self.vault_dir, exist_ok=True)
        winutil.set_hidden(self.vault_dir, False)

    def vault_hide(self):
        if os.path.isdir(self.vault_dir):
            winutil.set_hidden(self.vault_dir, True)

    def vault_size_mb(self):
        total = 0
        try:
            for dp, _dn, fn in os.walk(self.vault_dir):
                for f in fn:
                    try:
                        total += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
        except Exception:
            pass
        return total / 1048576

    # ---- enrolment artefacts ------------------------------------------------
    def has_model(self):
        return os.path.isfile(self.model_file)

    def has_legacy_model(self):
        """An enrolment made by a release before 10.0 (cannot be verified by this version)."""
        return (not self.has_model()) and any(
            os.path.isfile(os.path.join(self.lock_dir, n)) for n in LEGACY_MODEL_FILES)

    def wipe(self, include_vault=True):
        """Application-level removal of BioLock's authentication artefacts (and, optionally, the vault).
        This deletes files; it does not overwrite the underlying storage sectors."""
        if include_vault and os.path.isdir(self.vault_dir):
            winutil.set_hidden(self.vault_dir, False)
            shutil.rmtree(self.vault_dir, ignore_errors=True)
        names = ('face_model.npz', 'vault.key', 'state.json', 'audit.log', 'config.json', 'drive.id') \
            + LEGACY_MODEL_FILES
        for n in names:
            p = os.path.join(self.lock_dir, n)
            if os.path.exists(p):
                winutil.set_hidden(p, False)
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            leftovers = [n for n in os.listdir(self.lock_dir) if n.endswith('.tmp') or n == '.write_test']
        except OSError:
            leftovers = []
        for n in leftovers:
            p = os.path.join(self.lock_dir, n)
            winutil.set_hidden(p, False)
            try:
                os.remove(p)
            except OSError:
                pass
        if os.path.isdir(self.snap_dir):
            shutil.rmtree(self.snap_dir, ignore_errors=True)
