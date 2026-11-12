#!/usr/bin/env python3
"""SIMULATED sessions: the real session state machine driven by animated photographs.

    python experiments/simulate.py --calibrate         liveness measures on the DEV halves (parameter selection)
    python experiments/simulate.py                     scenario sessions on the TEST halves (report)

WHAT THIS IS
    Photographs from the public datasets are animated (biolock/sim.py) and fed, frame by frame, to the same
    `Session` code the app runs behind the camera, with a clock that advances 1/30 s per frame. Every
    session goes through presence -> blink -> head -> match with the shipped configuration.

WHAT THIS IS NOT
    No person and no physical photograph, phone or screen was in front of a camera. An animated photograph
    performs the liveness cues by construction, so these runs show how the state machine and the matcher
    treat inputs that contain (or lack) the cues. Real liveness and presentation-attack rates need the live
    trials in docs/TESTING_GUIDE.md. Every table produced from this script must be labelled "simulation".

Scenarios (one enrolled identity at a time; enrolment as in experiments/benchmark.py, "live" gallery):
    genuine          an unseen photograph of the enrolled person, animated with all cues
    impostor         a photograph of another person, animated with all cues
    photo_static     an unseen photograph of the enrolled person that does not move (sensor noise 2 / 5 grey levels)
    photo_shaken     the same photograph trembling in a hand, moved left/right when asked, never blinking
    swap             all liveness steps done by another person's animated photograph, then the enrolled
                     person's photograph is shown for the match (run with and without identity binding)
A replayed video of the enrolled person that contains the cues is, for this pipeline, the same input as
`genuine`; it is therefore not a separate scenario, and nothing here suggests it would be stopped.

A photograph is only used where it can play its part: one detected face, and for a blinking role open eyes
that the eye check sees steadily and stops seeing once they are painted over. The others are counted and
reported, not silently dropped.

SELECTION RULES for the liveness thresholds (--calibrate, dev halves only)
    motion_thresh      geometric mean of the 99th percentile of the presence measure for a motionless
                       photograph with sensor noise 5 and its 1st percentile for a 10 px side-to-side sway
    blink_local_ratio  99th percentile of the eye-band / rest-of-face change ratio at the eye drop-outs of a
                       trembling photograph, rounded up to 0.5
    blink_min_change   99th percentile of the eye-band change at the eye drop-outs of a motionless
                       photograph with sensor noise 5, rounded up to 0.001
    They come from animated photographs, so they are starting values that live trials have to confirm.
"""
import os
import sys
import json
import time
import argparse
import tempfile
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import datasets as DS                                   # noqa: E402
from biolock import sim, liveness                       # noqa: E402
from biolock import session as S                        # noqa: E402
from biolock import __version__                         # noqa: E402
from biolock.config import load_config                  # noqa: E402
from biolock.engine import FaceEngine                   # noqa: E402
from biolock.metrics import wilson                      # noqa: E402

SWAY = 12.0             # px, side-to-side movement of the simulated person during the presence step
_W = {}                 # per-process cache: engine, dataset


def _worker(info, folder, overrides=None):
    key = (info['name'], folder, json.dumps(overrides or {}, sort_keys=True))
    if key not in _W:
        cv2.setNumThreads(1)
        cfg = load_config(path='', use_env=False, overrides=overrides)
        D, F = DS.unstage(info, folder)
        _W.clear()
        _W[key] = (cfg, FaceEngine(cfg), D, F)
    return _W[key]


# --------------------------------------------------------------------------------------------------------
# calibration of the liveness measures (dev halves)
# --------------------------------------------------------------------------------------------------------
class _Stage:
    """Minimal stand-in for a session Status, to put an Actor into a given stage."""
    head_calibrating = True
    head_phase = 'left'

    def __init__(self, stage):
        self.stage = stage


def _trace(E, actor, stage, n):
    """Per-frame liveness measurements, computed with the functions the session uses."""
    st, prev, box, out = _Stage(stage), None, None, []
    for t in range(n):
        _f, _g, eq = S.prepare(sim.to_camera(actor.frame(st)))
        if t % 2 == 0 or box is None:
            b = E.detect_faces(eq)
            box = b[0] if len(b) == 1 else box
        if prev is not None and box is not None:
            ch = liveness.local_change(prev, eq, box) or (0.0, 0.0)
            out.append((liveness.box_motion(prev, eq, box) or 0.0, E.eyes_visible(eq, box), ch[0], ch[1]))
        prev = eq
    return out


def _dropouts(tr, open_stable=3):
    """For every moment the eye detector loses stably visible eyes: the (eye change, rest change) pairs of
    the frames the blink detector looks at (the frame before, the first and the second frame without eyes)."""
    out, run = [], 0
    for i, (_m, vis, e, r) in enumerate(tr):
        if vis:
            run += 1
            continue
        if run >= open_stable:
            cands = [(e, r)]
            if i > 0:
                cands.append((tr[i - 1][2], tr[i - 1][3]))
            if i + 1 < len(tr) and not tr[i + 1][1]:
                cands.append((tr[i + 1][2], tr[i + 1][3]))
            out.append(cands)
        run = 0
    return out


def _best(cands):
    return max(cands, key=lambda c: c[0] / max(c[1], 1e-4))


def calibrate_job(job):
    info, folder, idx = job
    cfg, E, D, _F = _worker(info, folder)
    rows = []
    for i in idx:
        photo = D['photos'][int(i)]
        rec = {'i': int(i)}
        probe = sim.Actor(E, photo, D['scale'], np.random.RandomState(int(i)), kind='static', noise=0.0)
        if not probe.usable:
            continue
        for name, kw in (('static_n2', dict(kind='static', noise=2.0)), ('static_n5', dict(kind='static', noise=5.0)),
                         ('static_n8', dict(kind='static', noise=8.0)),
                         ('still_person', dict(kind='live', noise=2.0, sway=1.5)),
                         ('sway_5', dict(kind='live', noise=2.0, sway=5.0)),
                         ('sway_10', dict(kind='live', noise=2.0, sway=10.0)),
                         ('sway_20', dict(kind='live', noise=2.0, sway=20.0)),
                         ('shaken', dict(kind='shaken', noise=2.0))):
            a = sim.Actor(E, photo, D['scale'], np.random.RandomState(7919 * int(i) + len(name)), blinks=False, **kw)
            if not a.usable:
                continue
            tr = _trace(E, a, S.ST_PRESENCE, 26)
            if len(tr) >= 20:
                rec[name] = float(np.mean([t[0] for t in tr[-20:]]))
        # blink evidence: a simulated blink, a shaken photograph and a noisy static one
        a = sim.Actor(E, photo, D['scale'], np.random.RandomState(104729 + int(i)), kind='live', noise=2.0)
        if a.usable:
            rec['blink'] = _dropouts(_trace(E, a, S.ST_BLINK, 48))
        a = sim.Actor(E, photo, D['scale'], np.random.RandomState(1299709 + int(i)), kind='shaken', noise=2.0)
        rec['shaken_dropouts'] = _dropouts(_trace(E, a, S.ST_BLINK, 150))
        a = sim.Actor(E, photo, D['scale'], np.random.RandomState(15485863 + int(i)), kind='static', noise=5.0)
        rec['noise_dropouts'] = _dropouts(_trace(E, a, S.ST_BLINK, 90))
        rows.append(rec)
    return info['name'], rows


def _pct(x, q):
    return [round(float(v), 5) for v in np.percentile(x, q)] if len(x) else []


def calibrate(args, staged, folder):
    jobs = []
    for info, n in staged:
        idx = np.arange(n)[::info['calib_step']]
        jobs += [(info, folder, chunk) for chunk in np.array_split(idx, max(1, args.workers * 2))]
    rows = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, r in ex.map(calibrate_job, jobs):
            rows[name] += r
    allrows = [r for v in rows.values() for r in v]
    out = {'photos': {k: len(v) for k, v in rows.items()}, 'presence': {}, 'blink': {}}
    q = [1, 5, 50, 95, 99]
    print('\nPresence measure (block-averaged frame difference in the face box, mean of 20 frames)  pct[1,5,50,95,99]')
    for name in ('static_n2', 'static_n5', 'static_n8', 'still_person', 'sway_5', 'sway_10', 'sway_20', 'shaken'):
        v = [r[name] for r in allrows if name in r]
        out['presence'][name] = {'n': len(v), 'pct': _pct(v, q)}
        print('  %-14s n=%4d  %s' % (name, len(v), out['presence'][name]['pct']))
    hi_static = float(np.percentile([r['static_n5'] for r in allrows if 'static_n5' in r], 99))
    lo_sway = float(np.percentile([r['sway_10'] for r in allrows if 'sway_10' in r], 1))
    rule = float(np.sqrt(hi_static * lo_sway))
    out['presence']['rule'] = {'static_n5_p99': hi_static, 'sway_10_p1': lo_sway, 'geometric_mean': rule,
                               'motion_thresh': round(rule, 3)}
    out['presence']['raw'] = {name: [round(r[name], 5) for r in allrows if name in r]
                              for name in ('static_n2', 'static_n5', 'static_n8', 'still_person', 'sway_5',
                                           'sway_10', 'sway_20', 'shaken')}
    print('  rule: sqrt(P99 static, noise 5  x  P1 sway 10 px) = sqrt(%.4f x %.4f) = %.4f -> motion_thresh %.3f'
          % (hi_static, lo_sway, rule, round(rule, 3)))

    print('\nBlink evidence when the eye detector loses the eyes: change in the eye band, change in the rest of '
          'the face, ratio   pct[1,5,50,95,99]')
    for name, key in (('simulated blink', 'blink'), ('shaken photograph', 'shaken_dropouts'),
                      ('static photograph, noise 5', 'noise_dropouts')):
        ev = [_best(c) for r in allrows for c in r.get(key, [])]
        eye = [c[0] for c in ev]
        rest = [c[1] for c in ev]
        ratio = [c[0] / max(c[1], 1e-4) for c in ev]
        out['blink'][key] = {'events': len(ev), 'photos_with_events': sum(1 for r in allrows if r.get(key)),
                             'raw': [[[round(float(e), 5), round(float(x), 5)] for e, x in c]
                                     for r in allrows for c in r.get(key, [])],
                             'eye_pct': _pct(eye, q), 'rest_pct': _pct(rest, q), 'ratio_pct': _pct(ratio, q)}
        print('  %-28s events=%5d  eye %s  rest %s  ratio %s' % (name, len(ev), _pct(eye, q), _pct(rest, q),
                                                               [round(x, 2) for x in _pct(ratio, q)]))
    shaken = [_best(c) for r in allrows for c in r.get('shaken_dropouts', [])]
    noise = [_best(c) for r in allrows for c in r.get('noise_dropouts', [])]
    p99_ratio = float(np.percentile([c[0] / max(c[1], 1e-4) for c in shaken], 99)) if shaken else float('nan')
    p99_noise = float(np.percentile([c[0] for c in noise], 99)) if noise else float('nan')
    out['blink']['rule'] = {
        'shaken_ratio_p99': p99_ratio, 'blink_local_ratio': float(np.ceil(p99_ratio * 2) / 2) if shaken else None,
        'noise_eye_change_p99': p99_noise,
        'blink_min_change': float(np.ceil(p99_noise * 1000) / 1000) if noise else None}
    print('  rule: blink_local_ratio = P99 ratio of shaken-photo drop-outs, rounded up to 0.5: %.2f -> %s'
          % (p99_ratio, out['blink']['rule']['blink_local_ratio']))
    print('  rule: blink_min_change  = P99 eye-band change of noise drop-outs, rounded up to 0.001: %.4f -> %s'
          % (p99_noise, out['blink']['rule']['blink_min_change']))
    cfg = load_config(path='', use_env=False)
    for key in ('blink', 'shaken_dropouts', 'noise_dropouts'):
        ev = [c for r in allrows for c in r.get(key, [])]
        ok = [any(e >= cfg['blink_min_change'] and e >= cfg['blink_local_ratio'] * max(x, 1e-4) for e, x in c)
              for c in ev]
        out['blink'][key]['pass_shipped_rule'] = float(np.mean(ok)) if ok else None
        out['blink'][key]['pass_shipped_rule_count'] = [int(np.sum(ok)), len(ok)]
        print('  %-18s passes the shipped rule (ratio >= %g and eye change >= %g): %s' % (
            key, cfg['blink_local_ratio'], cfg['blink_min_change'],
            'n/a' if not ok else '%.3f (%d of %d)' % (np.mean(ok), int(np.sum(ok)), len(ok))))
    rules = {'motion_thresh': out['presence']['rule']['motion_thresh'],
             'blink_local_ratio': out['blink']['rule']['blink_local_ratio'],
             'blink_min_change': out['blink']['rule']['blink_min_change']}
    shipped = {k: cfg[k] for k in rules}
    out['rules'], out['shipped'] = rules, shipped
    out['shipped_matches_rules'] = all(rules[k] is not None and abs(rules[k] - shipped[k]) < 1e-9 for k in rules)
    print('  rules give %s; shipped %s; match: %s' % (rules, shipped, out['shipped_matches_rules']))
    return out


# --------------------------------------------------------------------------------------------------------
# scenario sessions (test halves)
# --------------------------------------------------------------------------------------------------------
def _run(E, cfg, next_frame):
    clock = sim.SimClock()
    return sim.run(S.Session(E, cfg, S.MODE_VERIFY, clock=clock), clock, next_frame)


def _row(scenario, pid, r, **extra):
    return dict(scenario=scenario, identity=int(pid), outcome=r.outcome, failed_stage=r.failed_stage,
                score=r.score, live_score=r.live_score, frames=r.frames, motion_avg=round(r.motion_avg, 5),
                blinks=r.blinks, blink_rejects=r.blink_rejects, proc_ms=round(r.proc_ms, 2),
                presence=r.passed['presence'], blink=r.passed['blink'], head=r.passed['head'], **extra)


def session_job(job):
    info, folder, split, seed, pids, n_gen, n_imp = job
    cfg, E, D, F = _worker(info, folder)
    cfg_free = dict(cfg, bind_liveness_identity=False)
    trials = DS.make_trials(F['valid'], D['target'], DS.split_ids(D, split), info['k'], max(info['n_imp'], n_imp),
                            info['max_gen'], seed)
    rows, skipped, reasons = [], collections.Counter(), collections.Counter()
    blinking = {}                       # photograph -> could it play a blinking role, or why not
    for pid, gal, gen, imp in trials:
        if pid not in pids:
            continue
        rng = np.random.RandomState(5000 + 97 * seed + pid)
        E.train(DS.live_gallery(E, D, F, gal, rng), save=False)

        def actor(i, kind, **kw):
            a = sim.Actor(E, D['photos'][int(i)], D['scale'], np.random.RandomState(int(rng.randint(1 << 30))),
                          kind=kind, sway=SWAY, **kw)
            if not a.usable:
                reasons[a.why] += 1
            if kind == 'live' and kw.get('blinks', True):
                blinking.setdefault(int(i), a.why or 'usable')
            return a if a.usable else None

        for i in gen[:n_gen]:
            a = actor(i, 'live')
            if a is None:
                skipped['genuine'] += 1
                continue
            rows.append(_row('genuine', pid, _run(E, cfg, a.frame), photo=int(i)))
        for i in imp[:n_imp]:
            a = actor(i, 'live')
            if a is None:
                skipped['impostor'] += 1
                continue
            rows.append(_row('impostor', pid, _run(E, cfg, a.frame), photo=int(i)))
        own = [int(i) for i in gen[:2]]
        for i, noise in ((own[0], 2.0), (own[-1], 5.0)):
            a = actor(i, 'static', noise=noise)
            if a is None:
                skipped['photo_static'] += 1
                continue
            rows.append(_row(f'photo_static_noise{noise:g}', pid, _run(E, cfg, a.frame), photo=i))
        for i in own:
            a = actor(i, 'shaken')
            if a is None:
                skipped['photo_shaken'] += 1
                continue
            rows.append(_row('photo_shaken', pid, _run(E, cfg, a.frame), photo=i))
        for i in imp[:2]:
            for name, c in (('swap_bound', cfg), ('swap_unbound', cfg_free)):
                a, b = actor(i, 'live'), actor(own[0], 'live', blinks=False)
                if a is None or b is None:
                    skipped[name] += 1
                    continue
                rows.append(_row(name, pid, _run(E, c, lambda st, a=a, b=b: (
                    b if st is not None and st.stage == S.ST_FACE else a).frame(st)), photo=int(i)))
    for row in rows:
        row['min_conf'] = cfg['min_conf']
    return info['name'], rows, dict(skipped), dict(reasons), blinking


def summarise(rows):
    out = {}
    for sc in sorted({r['scenario'] for r in rows}):
        rs = [r for r in rows if r['scenario'] == sc]
        n = len(rs)
        acc = sum(r['outcome'] == 'accept' for r in rs)
        stops = collections.Counter((r['failed_stage'] or 'accepted') for r in rs)
        lo, hi = wilson(acc, n)
        T = rs[0].get('min_conf')
        only_binding = sum(1 for r in rs if r['failed_stage'] == 'match' and r['score'] is not None
                           and T is not None and r['score'] >= T)
        out[sc] = dict(n=n, identities=len({r['identity'] for r in rs}), accepted=acc, accept_rate=acc / n,
                       accept_ci=[lo, hi], passed_presence=sum(r['presence'] for r in rs),
                       passed_blink=sum(r['blink'] for r in rs), passed_head=sum(r['head'] for r in rs),
                       stopped_at={k: stops[k] for k in ('presence', 'blink', 'head', 'match') if stops[k]},
                       rejected_only_by_identity_binding=only_binding,
                       mean_frames=float(np.mean([r['frames'] for r in rs])),
                       mean_proc_ms=float(np.mean([r['proc_ms'] for r in rs])))
    return out


def sessions(args, staged, folder):
    jobs = []
    for info, _n in staged:
        ids = [int(p) for p in DS.split_ids({'names': info['names']}, args.split)]
        if args.quick:
            ids = ids[:4]
        for chunk in np.array_split(np.array(ids), max(1, min(len(ids), args.workers * 2))):
            jobs.append((info, folder, args.split, args.seed, set(int(c) for c in chunk), args.genuine, args.impostor))
    rows, skipped = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    reasons = collections.defaultdict(collections.Counter)
    blinking = collections.defaultdict(dict)
    t = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, r, sk, why, blink in ex.map(session_job, jobs):
            rows[name] += r
            skipped[name].update(sk)
            reasons[name].update(why)
            for photo, verdict in blink.items():
                blinking[name].setdefault(photo, verdict)
    out = {}
    for name, r in rows.items():
        out[name] = {'summary': summarise(r), 'sessions_not_run': dict(skipped[name]),
                     'photographs_not_usable': dict(reasons[name]),
                     'photographs_tried_for_a_blinking_role': dict(collections.Counter(blinking[name].values())),
                     'sessions': r}
        print(f'\n{name}: {len(r)} simulated sessions ({time.time() - t:.0f}s)')
        print(f'  not run because a photograph could not play its part: {dict(skipped[name])}')
        print(f'  reasons: {dict(reasons[name])}')
        print(f"  distinct photographs tried for a blinking role: {out[name]['photographs_tried_for_a_blinking_role']}")
        print('  %-24s %5s %9s %-17s %-28s' % ('scenario', 'n', 'accepted', '95% CI', 'stopped at'))
        for sc, m in out[name]['summary'].items():
            print('  %-24s %5d %4d %4.1f%% [%4.1f-%4.1f%%]  %s' % (
                sc, m['n'], m['accepted'], 100 * m['accept_rate'], 100 * m['accept_ci'][0], 100 * m['accept_ci'][1],
                m['stopped_at']))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--calibrate', action='store_true', help='liveness measures on the dev halves')
    ap.add_argument('--split', default=None, choices=('dev', 'test'))
    ap.add_argument('--datasets', default='olivetti,lfw')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--genuine', type=int, default=4, help='genuine sessions per identity')
    ap.add_argument('--impostor', type=int, default=6, help='impostor sessions per identity')
    ap.add_argument('--workers', type=int, default=max(1, min(12, (os.cpu_count() or 2) * 3 // 5)))
    ap.add_argument('--quick', action='store_true', help='4 identities per dataset (smoke test)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    args = ap.parse_args()
    args.split = args.split or ('dev' if args.calibrate else 'test')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cfg = load_config(path='', use_env=False)
    E = FaceEngine(cfg)
    with tempfile.TemporaryDirectory() as folder:
        staged = []
        for name in args.datasets.split(','):
            D = DS.LOADERS[name]()
            keep = np.isin(D['target'], DS.split_ids(D, args.split))
            if args.calibrate:                                  # only the dev photographs are looked at
                D = dict(D, photos=D['photos'][keep], target=D['target'][keep])
                info = DS.stage_photos(D, folder)
                DS.stage_features(info, {'valid': np.ones(len(D['photos']), bool)}, folder)
            else:
                info = DS.stage_photos(D, folder)
                DS.stage_features(info, DS.extract_parallel(info, folder, len(D['photos']), np.flatnonzero(keep),
                                                            args.workers, size=E.size), folder)
            info['calib_step'] = 2 if name == 'olivetti' else 12
            if args.quick:
                info['calib_step'] *= 8
            staged.append((info, len(D['photos'])))
            print(f"{name}: {len(D['photos'])} photographs staged ({time.time() - t0:.0f}s)", flush=True)
        meta = dict(kind='SIMULATION', split=args.split, seed=args.seed, version=__version__,
                    opencv=cv2.__version__, numpy=np.__version__, fps=sim.FPS, sway_px=SWAY,
                    config={k: cfg[k] for k in sorted(cfg) if not k.endswith('_condition') and k != 'trial_label'})
        if args.calibrate:
            res = dict(meta=meta, calibration=calibrate(args, staged, folder))
            path = os.path.join(args.out, f'liveness_calibration_{args.split}.json')
        else:
            res = dict(meta=dict(meta, genuine_per_identity=args.genuine, impostor_per_identity=args.impostor),
                       datasets=sessions(args, staged, folder))
            path = os.path.join(args.out, f'simulation_{args.split}{"_quick" if args.quick else ""}.json')
    res['meta']['seconds'] = round(time.time() - t0, 1)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(f'\nSaved {path}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
