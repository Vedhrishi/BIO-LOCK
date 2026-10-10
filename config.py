"""
Tunable parameters for BioLock, split out so experiments can sweep them
without touching application logic.

Load order: DEFAULTS  ->  config.json (if present)  ->  environment
variables (BIOLOCK_<NAME>)  ->  values passed explicitly by a script
(e.g. the experiment runner) via load_config(overrides=...).

This file does not change BioLock's architecture — it only externalises
values that used to be hard-coded constants in main.py, so the same
logic can be re-run under different parameter combinations.
"""

import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

DEFAULTS = {
    # --- face match threshold sweep: 20-60 ---
    'min_conf': 40,

    # --- motion/liveness threshold sweep: 0.01-0.10 ---
    'motion_thresh': 0.03,
    'motion_frames': 20,

    # --- enrolment frame-count sweep: 10,20,40,60,80,100 ---
    'enrol_frames': 100,

    # --- fused match-score weights: LBPH / Histogram / Template, must sum to 1 ---
    'w_lbph': 0.5,
    'w_hist': 0.3,
    'w_template': 0.2,

    # --- liveness step settings (unchanged defaults from the original app) ---
    'blinks_needed': 2,
    'head_turn_pct': 10,
    'blink_min_cl': 2,

    # --- misc, unrelated to the experiment sweeps ---
    'process_every': 2,
    'auto_lock_mins': 5,
    'max_attempts': 5,

    # --- experiment/session labelling (set by the experiment runner or
    #     manually via the Testing Mode panel in the UI; not used for
    #     the live security logic itself) ---
    'trial_label': None,       # 'genuine' | 'impostor' | 'attack_photo' | 'attack_phone' | 'attack_replay'
    'lighting_condition': None,  # free-text tag, e.g. 'bright', 'dim', 'backlit'
    'movement_condition': None,  # free-text tag, e.g. 'still', 'head_turn', 'walking'
}


def validate(cfg):
    errs = []
    if not (20 <= cfg['min_conf'] <= 100):
        errs.append(f"min_conf {cfg['min_conf']} out of sane range (20-100)")
    if not (0 < cfg['motion_thresh'] < 1):
        errs.append(f"motion_thresh {cfg['motion_thresh']} out of sane range (0-1)")
    if cfg['enrol_frames'] < 1:
        errs.append('enrol_frames must be >= 1')
    wsum = cfg['w_lbph'] + cfg['w_hist'] + cfg['w_template']
    if abs(wsum - 1.0) > 1e-6:
        errs.append(f'weights must sum to 1.0 (got {wsum:.4f}: '
                    f"lbph={cfg['w_lbph']}, hist={cfg['w_hist']}, template={cfg['w_template']})")
    if errs:
        raise ValueError('Invalid config:\n  - ' + '\n  - '.join(errs))
    return cfg


def load_config(path=CONFIG_PATH, overrides=None):
    cfg = dict(DEFAULTS)

    if path and os.path.exists(path):
        with open(path, 'r') as f:
            cfg.update(json.load(f))

    # environment variable overrides: BIOLOCK_MIN_CONF=35 etc.
    for key in list(cfg.keys()):
        env_key = f'BIOLOCK_{key.upper()}'
        if env_key in os.environ:
            raw = os.environ[env_key]
            try:
                cfg[key] = json.loads(raw)
            except json.JSONDecodeError:
                cfg[key] = raw

    if overrides:
        cfg.update(overrides)

    return validate(cfg)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
