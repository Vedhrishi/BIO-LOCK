#!/usr/bin/env python3
"""Summarise the live trials recorded by the app in logs/attempts.csv.

How the log is produced: run the app, tick "Testing mode", set the trial label (genuine, impostor,
attack_photo, attack_phone, attack_replay) and the lighting / movement tags before each attempt, then
do the attempt. One row is written per attempt. See docs/TESTING_GUIDE.md for the protocol.

    python experiments/summarize_live_logs.py [--csv logs/attempts.csv] [--json out.json]
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from biolock import metrics as M                    # noqa: E402


def pct(x):
    return '   -  ' if x != x else f'{100 * x:5.1f}%'


def ci(pair):
    lo, hi = pair
    return '      -      ' if lo != lo else f'[{100 * lo:4.1f}-{100 * hi:4.1f}%]'


def table(title, header, rows):
    print(f'\n{title}')
    if not rows:
        print('  (no rows)')
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print('  ' + '  '.join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print('  ' + '  '.join(str(v).ljust(w) for v, w in zip(r, widths)))


def rate_rows(groups, key_names):
    out = []
    for g in groups:
        out.append([g[k] for k in key_names] + [g['n_genuine'], g['n_negative'], pct(g['FRR']), ci(g['FRR_ci']),
                                                pct(g['FAR']), ci(g['FAR_ci'])])
    return out


def main():
    ap = argparse.ArgumentParser(description='Summarise logs/attempts.csv')
    ap.add_argument('--csv', default=os.path.join(ROOT, 'logs', 'attempts.csv'))
    ap.add_argument('--json', default=None, help='also write the summary to this file')
    a = ap.parse_args()

    rows = M.load_attempts(a.csv)
    trials = [r for r in rows if r.get('trial_label') in M.TRIAL_LABELS and r.get('mode') != 'enrol']
    if not trials:
        print(f'No labelled trials in {a.csv}.\nRun the app with Testing mode switched on, do some attempts, '
              'then run this again.')
        return 1

    overall = M.compute_metrics(trials)
    print(f'{len(trials)} labelled trials in {a.csv}  ({len(rows) - len(trials)} other rows ignored)')
    table('Overall', ['genuine', 'non-genuine', 'FRR', '95% CI', 'FAR', '95% CI'], rate_rows([overall], []))

    live = M.liveness_breakdown(trials)
    table('Per trial label: where attempts stopped',
          ['label', 'n', 'passed liveness', 'stopped: presence', 'blink', 'head', 'rejected: match', 'key',
           'accepted', 'accept rate', '95% CI'],
          [[g['trial_label'], g['n'], g['liveness_pass'], g['stopped_presence'], g['stopped_blink'],
            g['stopped_head'], g['rejected_match'], g['rejected_key'], g['accepted'], pct(g['accept_rate']),
            ci(g['accept_ci'])] for g in live])

    cond = M.summarize_by(trials, ('lighting_condition', 'movement_condition'))
    table('By lighting / movement tag', ['lighting', 'movement', 'genuine', 'non-genuine', 'FRR', '95% CI', 'FAR',
                                         '95% CI'],
          rate_rows(cond, ['lighting_condition', 'movement_condition']))

    used = sorted({r['min_conf'] for r in trials if r.get('min_conf')})
    sweep = M.threshold_sweep(trials, [20, 25, 30, 35, 40, 45, 50])
    table(f'Other thresholds, recomputed from the logged median scores (app ran with {", ".join(used) or "?"})',
          ['threshold', 'FRR', 'FAR'], [[s['threshold'], pct(s['FRR']), pct(s['FAR'])] for s in sweep])

    cost = ['latency_ms', 'proc_ms', 'fps', 'cpu_percent', 'mem_mb']
    done = [r for r in trials if r.get('outcome') in ('accept', 'reject')]
    means = []
    for k in cost:
        xs = M._floats(done, k)
        means.append('-' if not xs else f'{sum(xs) / len(xs):.1f}')
    table(f'Cost of the {len(done)} attempts that reached a decision (means)',
          ['time to decision (ms)', 'processing per frame (ms)', 'frames/s', 'CPU % of one core', 'memory (MB)'],
          [means])

    faces = [sum(int(float(r.get(k) or 0)) for r in trials) for k in ('faces_0', 'faces_1', 'faces_multi')]
    table('Face detection passes over all trials', ['no face', 'one face', 'several faces'], [faces])

    print('\nIntervals are Wilson 95% intervals. With 30 trials, 0 false accepts still allows a true rate of '
          'about 11%.')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'csv': a.csv, 'overall': overall, 'per_label': live, 'by_condition': cond,
                       'threshold_sweep': sweep}, f, indent=1)
        print(f'Saved {a.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
