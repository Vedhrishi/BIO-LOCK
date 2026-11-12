"""Application logic without a GUI: what is on the drive, and what a finished attempt does to it.

The UI (biolock/ui.py) only asks this object questions and forwards button presses, so the
complete enrol -> lock -> verify -> lock-out -> wipe behaviour can be tested without a
window or a camera.
"""

import os
import time
import datetime
import threading

from . import __version__
from . import vault
from . import winutil
from . import session as S
from .config import load_config, app_dir
from .engine import FaceEngine
from .metrics import MetricsLogger
from .storage import Store, DEF_STATE

# what `condition()` can return
NEW = 'new'                 # nothing enrolled on this drive
LOCKED = 'locked'           # enrolled, vault hidden
UNLOCKED = 'unlocked'       # enrolled, vault visible for this session
LEGACY = 'legacy'           # enrolment made by a release before 10.0 (cannot be verified)
DAMAGED = 'damaged'         # template or key file missing / unreadable
ORPHAN = 'orphan'           # a SecureVault with files but no enrolment

MAX_COOLDOWN_S = 300


class Refused(Exception):
    """An attempt cannot start right now; the message says why."""


class Controller:

    def __init__(self, base=None, cfg=None, clock=time.time):
        self.cfg = cfg if cfg is not None else load_config()
        self.store = Store(base if base is not None else app_dir())
        self.engine = FaceEngine(self.cfg, self.store.model_file)
        self.state = self.store.load_state()
        self.local = self.store.load_local()
        self.metrics = MetricsLogger(os.path.join(self.store.logs_dir, 'attempts.csv'))
        self.unlocked = False
        # Testing-mode tags: copied into the experiment log, never used for a decision
        self.testing = False
        self.trial_label = 'genuine'
        self.lighting = ''
        self.movement = ''
        self.no_lockout = False
        self.camera_desc = ''
        self._clock = clock
        self._lock = threading.RLock()
        self._last_activity = clock()
        winutil.set_hidden(self.store.lock_dir, True)     # no-op unless the folder exists and is visible
        if self.condition() == LOCKED:
            self.store.vault_hide()             # e.g. the drive was pulled out while the vault was open

    # ---- what is on the drive -------------------------------------------------------
    def vault_has_files(self):
        try:
            with os.scandir(self.store.vault_dir) as it:
                return any(True for _ in it)
        except OSError:
            return False

    def condition(self):
        with self._lock:
            if self.unlocked:
                return UNLOCKED
            has_key = os.path.isfile(self.store.vault_key)
            if self.engine.trained and has_key:
                return LOCKED
            if self.store.has_legacy_model():
                return LEGACY
            if self.store.has_model() or has_key:
                return DAMAGED
            if self.vault_has_files():
                return ORPHAN
            return NEW

    @property
    def attempts(self):
        return int(self.state.get('attempts', 0) or 0)

    @property
    def attempts_left(self):
        return max(0, self.cfg['max_attempts'] - self.attempts)

    def cooldown_left(self):
        """Seconds until another attempt may start (after repeated liveness time-outs)."""
        try:
            return max(0.0, float(self.state.get('cooldown_until') or 0) - self._clock())
        except (TypeError, ValueError):
            return 0.0

    # ---- attempts -------------------------------------------------------------------
    def begin(self, mode, clock=time.monotonic):
        """A Session for a new attempt, or Refused."""
        with self._lock:
            cond = self.condition()
            if mode == S.MODE_ENROL:
                if cond != NEW:
                    raise Refused('A face is already enrolled on this drive. Wipe & reset first.')
            elif cond != LOCKED:
                raise Refused('There is no enrolment to verify against.')
            left = self.cooldown_left()
            if left > 0:
                raise Refused(f'Too many incomplete liveness checks. Try again in {int(left) + 1} s.')
            if not self.store.writable():
                raise Refused(f'Drive {self.store.letter}: is read-only or full. BioLock needs to write to it.')
            self.metrics.start_attempt()
            self.touch()
            return S.Session(self.engine, self.cfg, mode, clock=clock)

    def finalize(self, result):
        """Apply a finished session: train and save, or check the vault key; update the counters and
        write the logs. Called on the camera worker thread, so key derivation never blocks the window."""
        with self._lock:
            if result.mode == S.MODE_ENROL:
                self._finish_enrol(result)
            else:
                self._finish_verify(result)
            result.extra.setdefault('attempts_left', self.attempts_left)
            result.extra.setdefault('cooldown_s', self.cooldown_left())
            label = 'enrolment' if result.mode == S.MODE_ENROL else (self.trial_label if self.testing else '')
            self.metrics.log_result(result, self.cfg, trial_label=label,
                                    lighting=self.lighting if self.testing else '',
                                    movement=self.movement if self.testing else '',
                                    camera=self.camera_desc, counted=result.extra.get('counted'),
                                    note=result.extra.get('note', ''), version=__version__)
            self.touch()
        return result

    def _finish_enrol(self, r):
        st = self.store
        if r.outcome != S.OUT_ENROLLED:
            st.log('ENROL_ABANDONED', f'{r.outcome} at {r.failed_stage or "-"}', 'WARN')
            return
        try:
            kept = self.engine.train(r.samples)                 # writes the template
            material = vault.key_material(self.engine.digest(), st.drive_id(create=True), st.volume_serial())
            vault.save_key(st.vault_key, material)
            self.state = dict(DEF_STATE, enrolled=True, enrol_quality=r.quality, enrol_samples=kept,
                              enrolled_at=datetime.datetime.now().isoformat(timespec='seconds'),
                              total_unlocks=self.state.get('total_unlocks', 0))
            st.save_state(self.state)
            st.vault_hide()
            r.extra['samples_kept'] = kept
            r.message = f'Enrolment complete ({kept} samples kept).'
            st.log('ENROLLED', f'drive={st.letter}:  samples={kept}')
        except Exception as e:                                   # full / unplugged / read-only drive
            self.engine.forget()
            for p in (st.model_file, st.vault_key):
                try:
                    os.remove(p)
                except OSError:
                    pass
            r.outcome = S.OUT_ERROR
            r.message = f'The enrolment could not be saved: {e}'
            st.log('ENROL_FAILED', f'{type(e).__name__}: {e}', 'WARN')
        finally:
            r.samples = []                                       # face crops are never kept in memory

    def _finish_verify(self, r):
        st, c = self.store, self.cfg
        if r.outcome == S.OUT_ACCEPT:
            material = vault.key_material(self.engine.digest(), st.drive_id(create=False), st.volume_serial())
            if vault.verify_key(st.vault_key, material):
                self.unlocked = True
                self.state.update(attempts=0, live_fails=0, cooldown_until=0,
                                  total_unlocks=int(self.state.get('total_unlocks', 0) or 0) + 1,
                                  last_unlock=datetime.datetime.now().isoformat(timespec='seconds'))
                st.save_state(self.state)
                st.vault_show()
                st.log('UNLOCKED', f'drive={st.letter}:  score={r.score:.0f}')
            else:
                r.outcome, r.failed_stage = S.OUT_REJECT, 'key'
                r.message = ('The face matched, but the enrolment on this drive does not belong to its key '
                             '(files were changed, or copied from another drive).')
                r.extra['counted'] = False
                st.log('KEY_MISMATCH', f'drive={st.letter}:', 'WARN')
        elif r.outcome == S.OUT_REJECT and r.failed_stage == 'match':
            counted = not (self.testing and self.no_lockout)
            r.extra['counted'] = counted
            if counted:
                self.state['attempts'] = self.attempts + 1
                st.save_state(self.state)
            shown = '-' if r.score is None else f'{r.score:.0f}'
            st.log('FAILED', f'score={shown}  attempts={self.attempts}/{c["max_attempts"]}'
                   + ('' if counted else '  (testing, not counted)'), 'WARN')
            if counted and self.attempts >= c['max_attempts']:
                self.wipe(reason=f'{c["max_attempts"]} failed face matches')
                r.extra['wiped'] = True
        elif r.outcome == S.OUT_TIMEOUT:
            r.extra['counted'] = False
            fails = int(self.state.get('live_fails', 0) or 0) + 1
            self.state['live_fails'] = fails
            if fails >= c['liveness_fail_limit'] and not (self.testing and self.no_lockout):
                wait = min(MAX_COOLDOWN_S, c['liveness_cooldown_s'] * 2 ** (fails - c['liveness_fail_limit']))
                self.state['cooldown_until'] = self._clock() + wait
            st.save_state(self.state)
            st.log('LIVENESS_TIMEOUT', f'stage={r.failed_stage}  in_a_row={fails}', 'WARN')
        else:
            r.extra['counted'] = False

    # ---- vault ------------------------------------------------------------------------
    def touch(self):
        self._last_activity = self._clock()

    def idle_seconds(self):
        return self._clock() - self._last_activity

    def auto_lock_in(self):
        """Seconds until the open vault locks itself; None while it is not open."""
        if not self.unlocked:
            return None
        return self.cfg['auto_lock_mins'] * 60.0 - self.idle_seconds()

    def open_vault(self):
        """Make sure the vault folder exists and is visible; returns its path (only while unlocked)."""
        with self._lock:
            if not self.unlocked:
                raise Refused('The vault is locked.')
            self.store.vault_show()
            self.touch()
            return self.store.vault_dir

    def lock(self, reason=''):
        with self._lock:
            if self.unlocked:
                self.store.vault_hide()
                self.unlocked = False
                self.store.log('LOCKED', f'drive={self.store.letter}:' + (f'  ({reason})' if reason else ''))

    def wipe(self, reason='requested'):
        """Application-level reset: enrolment, key, state, audit log and the contents of SecureVault."""
        with self._lock:
            self.engine.forget()
            self.store.wipe(include_vault=True)
            self.unlocked = False
            self.state = dict(DEF_STATE)
            self.local = self.store.load_local()
            try:
                self.store.save_state(self.state)
                self.store.log('WIPE', f'drive={self.store.letter}:  reason={reason}', 'WARN')
            except Exception:
                pass

    # ---- settings -------------------------------------------------------------------------
    def set_camera(self, index, backend=None, desc=''):
        with self._lock:
            self.local['cam_index'] = index
            self.local['cam_backend'] = backend
            self.camera_desc = desc
            try:
                self.store.save_local(self.local)
            except Exception:
                pass

    def trial_counts(self):
        return self.metrics.trial_counts()
