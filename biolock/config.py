"""Tunable parameters, kept out of the application logic so experiments can sweep them.

Load order when run from source:

    DEFAULTS  ->  config.json next to the app (if present)  ->  environment variables
    BIOLOCK_<NAME>  ->  explicit overrides passed by a script.

A packaged build (BioLock.exe) only accepts overrides for the keys in USER_KEYS. The
parameters that decide an unlock are fixed at build time there, so the lock cannot be
weakened by dropping a config.json next to the exe on the drive or by setting an
environment variable.

The matcher defaults (LBPH radius/grid, fusion weights, score calibration and the
acceptance threshold) were selected on the development halves of two public datasets;
the liveness thresholds (motion_thresh, blink_local_ratio, blink_min_change) come from
simulated sessions on the same development halves and still have to be confirmed with
live trials. Method and measured results: docs/RESULTS.md. Change these values only
together with a re-run of `python evaluate.py`.
"""

import os
import sys
import json


def app_dir():
    """Folder that holds the app's data: next to the exe when packaged, the project root from source."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CONFIG_PATH = os.path.join(app_dir(), 'config.json')

DEFAULTS = {
    # --- identity verification -------------------------------------------------
    'min_conf': 35.0,           # acceptance threshold T on the fused 0-100 match score
    'w_lbph': 1.0,              # fusion weights, must sum to 1
    'w_hist': 0.0,
    'w_template': 0.0,
    'match_frames': 15,         # frames whose scores are combined (median) into one decision
    'enrol_frames': 100,        # accepted samples that make up an enrolment
    'bind_liveness_identity': True,   # the face seen during the blink step must match as well

    # --- LBPH recogniser and face crop ------------------------------------------
    'lbph_radius': 5,
    'lbph_neighbors': 8,
    'lbph_grid': 6,
    'roi_size': 100,            # normalised face crop is roi_size x roi_size pixels

    # --- score calibration: component -> 0..100 (pooled development-set percentiles,
    #     research/calibration.json, rounded) ---------------------------------------
    'lbph_d_lo': 31.2,          # chi-square distance mapped to 100
    'lbph_d_hi': 76.4,          # chi-square distance mapped to 0
    'hist_c_lo': 0.114,         # correlation mapped to 0
    'hist_c_hi': 0.885,         # correlation mapped to 100
    'tmpl_c_lo': -0.083,
    'tmpl_c_hi': 0.649,

    # --- liveness ----------------------------------------------------------------
    'motion_thresh': 0.009,     # mean absolute frame difference (0-1, block-averaged) over the window
    'motion_frames': 20,        # size of the motion window
    'blinks_needed': 2,
    'blink_min_cl': 2,          # shortest eyes-closed run (frames) that counts as a blink
    'blink_max_cl': 24,         # longest eyes-closed run (frames) that still counts as a blink
    'blink_max_shift': 0.08,    # the face box may move at most this fraction of its width during a blink
    'blink_local_ratio': 2.5,   # when the eyes disappear the eye band must change this many times more
                                # than the rest of the face ...
    'blink_min_change': 0.011,  # ... and by at least this much (0-1)
    'blink_face_gap': 6,        # frames the face detector may miss during a blink before it is discarded
    'head_turn_pct': 10,        # head movement, % of face width, left then right
    'head_calib_obs': 10,       # observations used for the head-position baseline

    # --- session control ---------------------------------------------------------
    'process_every': 2,         # run face detection every N frames during the liveness steps
    'liveness_timeout_s': 45,   # presence + blink + head turn must finish within this time
    'match_timeout_s': 15,      # time allowed to gather match_frames usable frames
    'enrol_timeout_s': 240,     # a sample capture that takes longer is abandoned
    'enrol_gap_s': 0.12,        # shortest time between two accepted enrolment samples
    'min_face_px': 80,          # smallest face width (in a 640-wide frame) that is used
    'auto_lock_mins': 5,
    'max_attempts': 5,          # failed face matches before the automatic wipe
    'liveness_fail_limit': 3,   # liveness time-outs in a row before a cool-down starts
    'liveness_cooldown_s': 30,  # first cool-down; doubles with every further time-out (max 5 min)

    # --- experiment labelling (set from the Testing mode panel; never used for decisions) ---
    'trial_label': None,        # genuine | impostor | attack_photo | attack_phone | attack_replay
    'lighting_condition': None,
    'movement_condition': None,
}

# The only keys a packaged build lets config.json / the environment change.
USER_KEYS = ('auto_lock_mins', 'process_every', 'trial_label', 'lighting_condition', 'movement_condition')

_INT_KEYS = ('match_frames', 'enrol_frames', 'lbph_radius', 'lbph_neighbors', 'lbph_grid', 'roi_size',
             'motion_frames', 'blinks_needed', 'blink_min_cl', 'blink_max_cl', 'blink_face_gap',
             'head_calib_obs', 'process_every', 'max_attempts', 'min_face_px', 'liveness_fail_limit')
_POS_KEYS = ('liveness_timeout_s', 'match_timeout_s', 'enrol_timeout_s', 'auto_lock_mins', 'head_turn_pct',
             'liveness_cooldown_s')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(cfg):
    errs = []
    for k in ('min_conf', 'motion_thresh', 'w_lbph', 'w_hist', 'w_template', 'lbph_d_lo', 'lbph_d_hi',
              'hist_c_lo', 'hist_c_hi', 'tmpl_c_lo', 'tmpl_c_hi', 'blink_max_shift', 'blink_local_ratio',
              'blink_min_change', 'enrol_gap_s') + _POS_KEYS:
        if not _num(cfg[k]):
            errs.append(f'{k} must be a number (got {cfg[k]!r})')
    for k in _INT_KEYS:
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] < 1:
            errs.append(f'{k} must be a positive whole number (got {cfg[k]!r})')
    if not isinstance(cfg['bind_liveness_identity'], bool):
        errs.append('bind_liveness_identity must be true or false')
    if errs:
        raise ValueError('Invalid config:\n  - ' + '\n  - '.join(errs))

    if not (0 <= cfg['min_conf'] <= 100):
        errs.append(f"min_conf {cfg['min_conf']} must be within 0-100")
    if not (0 < cfg['motion_thresh'] < 1):
        errs.append(f"motion_thresh {cfg['motion_thresh']} must be within 0-1")
    for k in _POS_KEYS:
        if cfg[k] <= 0:
            errs.append(f'{k} must be greater than 0')
    wsum = cfg['w_lbph'] + cfg['w_hist'] + cfg['w_template']
    if abs(wsum - 1.0) > 1e-6 or min(cfg['w_lbph'], cfg['w_hist'], cfg['w_template']) < 0:
        errs.append(f'weights must be non-negative and sum to 1.0 (got lbph={cfg["w_lbph"]}, '
                    f'hist={cfg["w_hist"]}, template={cfg["w_template"]})')
    if cfg['lbph_d_hi'] <= cfg['lbph_d_lo']:
        errs.append('lbph_d_hi must be greater than lbph_d_lo')
    if cfg['hist_c_hi'] <= cfg['hist_c_lo'] or cfg['tmpl_c_hi'] <= cfg['tmpl_c_lo']:
        errs.append('each *_c_hi must be greater than its *_c_lo')
    if cfg['blink_max_cl'] < cfg['blink_min_cl']:
        errs.append('blink_max_cl must not be smaller than blink_min_cl')
    if not (0 < cfg['blink_max_shift'] <= 1):
        errs.append('blink_max_shift must be within 0-1')
    if cfg['blink_local_ratio'] < 0 or cfg['blink_min_change'] < 0:
        errs.append('blink_local_ratio and blink_min_change must not be negative')
    if cfg['enrol_gap_s'] < 0:
        errs.append('enrol_gap_s must not be negative')
    if errs:
        raise ValueError('Invalid config:\n  - ' + '\n  - '.join(errs))
    return cfg


def load_config(path=None, overrides=None, use_env=True, frozen=None):
    """Build the configuration. `path=''` skips the config file; `frozen` is for the tests."""
    frozen = getattr(sys, 'frozen', False) if frozen is None else frozen
    allowed = USER_KEYS if frozen else tuple(DEFAULTS)
    cfg = dict(DEFAULTS)
    path = CONFIG_PATH if path is None else path
    if path and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except (OSError, ValueError) as e:
            if not frozen:                      # from source a broken file should be noticed, not skipped
                raise ValueError(f'Cannot read {path}: {e}') from None
            loaded = {}
        if isinstance(loaded, dict):
            cfg.update({k: v for k, v in loaded.items() if k in allowed})
    if use_env:
        for key in allowed:
            raw = os.environ.get(f'BIOLOCK_{key.upper()}')
            if raw is not None:
                try:
                    cfg[key] = json.loads(raw)
                except json.JSONDecodeError:
                    cfg[key] = raw
    if overrides:
        unknown = [k for k in overrides if k not in DEFAULTS]
        if unknown:
            raise ValueError(f'Unknown config keys: {", ".join(sorted(unknown))}')
        cfg.update(overrides)
    return validate(cfg)


def save_config(cfg, path=None):
    with open(CONFIG_PATH if path is None else path, 'w', encoding='utf-8') as f:
        json.dump({k: cfg[k] for k in DEFAULTS if k in cfg}, f, indent=2)
