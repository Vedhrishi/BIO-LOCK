#!/usr/bin/env python3
"""
Summarizes REAL logs produced by running the actual BioLock app (main.py)
with its Testing Mode panel — use this instead of / alongside
run_experiments.py when you want numbers from the live camera pipeline
itself (full 4-step liveness included), rather than from a static-image
batch.

How to produce the logs this reads:
  1. Run `python main.py` as normal.
  2. Before each attempt, set the Testing Mode panel's "Trial label"
     (genuine / impostor / attack_photo / attack_phone / attack_replay)
     and the lighting/movement condition to match what you're about to
     do, THEN perform the attempt (live person, printed photo held up,
     phone screen, or replay video played at the camera).
  3. Repeat for as many attempts as you want statistical power (your
     mentors will have a view on N per condition — 20-30 per cell is a
     reasonable starting point for a project report).
  4. Every attempt is appended to logs/attempts.csv and every processed
     frame to logs/detections.csv automatically — nothing further to do.

Then run:
    python summarize_live_logs.py

which prints the same FAR/FRR/accuracy/precision/recall/F1 + latency/
CPU/memory tables as run_experiments.py, but computed from your real,
live, physically-performed attempts.
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as metrics_mod


def _print_table(title, rows, cols):
    print(f'\n=== {title} ===')
    if not rows:
        print('  (no rows)')
        return
    widths = {c: max(len(c), max(len(f'{r.get(c, "")}') for r in rows)) for c in cols}
    print('  ' + '  '.join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print('  ' + '  '.join(f'{r.get(c, "")}'.ljust(widths[c]) for c in cols))


def main():
    attempts = metrics_mod.load_attempts()
    if not attempts:
        print(f'No attempts logged yet at {metrics_mod.ATTEMPTS_CSV} — run main.py with the '
              f'Testing Mode panel set, perform some attempts, then re-run this script.')
        return

    detections_path = metrics_mod.DETECTIONS_CSV
    if os.path.exists(detections_path):
        with open(detections_path, newline='') as f:
            det_rows = list(csv.DictReader(f))
        tally = {'0': 0, '1': 0, '2+': 0}
        for r in det_rows:
            n = int(r['n_faces'])
            tally['0' if n == 0 else ('1' if n == 1 else '2+')] += 1
        print('\n=== Face-detection counts across all logged frames ===')
        print(f"  0 faces:        {tally['0']}")
        print(f"  1 face:         {tally['1']}")
        print(f"  2+ faces:       {tally['2+']}")

    # overall, by trial_label breakdown (sanity check on what was tested)
    by_label = {}
    for r in attempts:
        by_label.setdefault(r['trial_label'], []).append(r)
    label_rows = [{'trial_label': k, 'n_attempts': len(v)} for k, v in by_label.items()]
    _print_table('Attempts logged per trial label', label_rows, ['trial_label', 'n_attempts'])

    overall = [metrics_mod.compute_metrics(attempts)]
    _print_table('Overall (all logged attempts combined)', overall,
                 ['n', 'FAR', 'FRR', 'accuracy', 'precision', 'recall', 'f1',
                  'avg_latency_ms', 'avg_cpu_percent', 'avg_mem_mb'])

    by_condition = metrics_mod.summarize_by(group_fields=('lighting_condition', 'movement_condition'))
    _print_table('By lighting / movement condition', by_condition,
                 ['lighting_condition', 'movement_condition', 'n', 'FAR', 'FRR', 'accuracy',
                  'precision', 'recall', 'f1', 'avg_latency_ms'])

    by_minconf = metrics_mod.summarize_by(group_fields=('min_conf',))
    _print_table('By min_conf (whatever values you actually ran the app with)', by_minconf,
                 ['min_conf', 'n', 'FAR', 'FRR', 'accuracy', 'precision', 'recall', 'f1'])

    _print_table('Liveness per trial label (independent of face match)',
                 metrics_mod.liveness_breakdown(attempts),
                 ['trial_label', 'n', 'liveness_pass', 'liveness_pass_rate', 'stopped_presence',
                  'stopped_blink', 'stopped_head', 'rejected_match', 'accepted', 'accept_rate'])

    by_attack = metrics_mod.summarize_by(group_fields=('trial_label',))
    _print_table('By trial label (includes each attack type)', by_attack,
                 ['trial_label', 'n', 'FAR', 'FRR', 'accuracy', 'precision', 'recall', 'f1',
                  'avg_latency_ms', 'avg_cpu_percent', 'avg_mem_mb'])

    print(f'\nRaw per-attempt data: {metrics_mod.ATTEMPTS_CSV}')
    print(f'Raw per-frame data:   {metrics_mod.DETECTIONS_CSV}')
    print('Open either in Excel/pandas for any other cross-tab your mentors want.')


if __name__ == '__main__':
    main()
