#!/usr/bin/env python3
"""One command for every number in docs/RESULTS.md.

    python evaluate.py              run everything, then write docs/RESULTS.md   (1 to 2 hours on a laptop)
    python evaluate.py --report     only rebuild docs/RESULTS.md from the JSON files in results/
    python evaluate.py --steps tests,perf      run some of the steps
    python evaluate.py --quick      small subset of every step, to check that the machinery works
                                    (development halves only, goes to results/quick/, no report)

Steps, in this order:
    tests             unit tests
    benchmark_dev     matcher on the development halves + the selection rules (experiments/benchmark.py)
    calibration       liveness measures on the development halves, simulation (experiments/simulate.py)
    benchmark_test    matcher on the test halves with the shipped configuration
    simulation_test   simulated sessions on the test halves
    perf              timing and memory on this machine (experiments/perf.py)

The test halves are only touched if the shipped configuration equals what the selection rules give on
the development halves. If it does not, this script stops and says which value to change.
"""
import os
import re
import sys
import json
import time
import argparse
import datetime
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = ('tests', 'benchmark_dev', 'calibration', 'benchmark_test', 'simulation_test', 'perf')
FILES = {'tests': 'tests.json', 'benchmark_dev': 'benchmark_dev.json', 'calibration': 'liveness_calibration_dev.json',
         'benchmark_test': 'benchmark_test.json', 'simulation_test': 'simulation_test.json', 'perf': 'perf.json'}


def load(out, step, quick=False):
    name = FILES[step]
    if quick and step in ('benchmark_dev', 'benchmark_test', 'simulation_test'):
        name = name.replace('.json', '_quick.json')
    try:
        with open(os.path.join(out, name), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run(step, cmd, out):
    log = os.path.join(out, step + '.log')
    print(f'[{time.strftime("%H:%M:%S")}] {step}: {" ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd[1:])}',
          flush=True)
    t = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f'           finished in {time.time() - t:.0f} s, exit code {p.returncode}, log {os.path.relpath(log, ROOT)}',
          flush=True)
    return p.returncode, time.time() - t, log


def run_steps(steps, out, quick):
    q = ['--quick'] if quick else []
    exp = os.path.join(ROOT, 'experiments')
    for step in steps:
        if step == 'tests':
            code, secs, log = run(step, [PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], out)
            with open(log, encoding='utf-8') as f:
                text = f.read()
            m = re.search(r'Ran (\d+) tests? in ([\d.]+)s', text)
            skipped = re.search(r'skipped=(\d+)', text)
            with open(os.path.join(out, FILES[step]), 'w', encoding='utf-8') as f:
                json.dump({'ran': int(m.group(1)) if m else 0, 'ok': code == 0, 'seconds': round(secs, 1),
                           'skipped': int(skipped.group(1)) if skipped else 0, 'python': sys.version.split()[0]}, f)
            if code != 0:
                sys.exit('The unit tests fail. Fix them before measuring anything (see results/tests.log).')
        elif step == 'benchmark_dev':
            code, _s, _l = run(step, [PY, os.path.join(exp, 'benchmark.py'), '--split', 'dev', '--out', out] + q, out)
            sel = (load(out, step, quick) or {}).get('selection', {})
            if code != 0 or (not quick and not sel.get('shipped_matches_rules')):
                sys.exit('Stopped before the test halves: the shipped matcher configuration is not what the '
                         'selection rules give on the development halves.\n'
                         f"  weights   rule {sel.get('weights', {}).get('chosen_by_rule')}  shipped "
                         f"{sel.get('weights', {}).get('shipped')}\n"
                         f"  threshold rule {sel.get('threshold', {}).get('chosen_by_rule')}  shipped "
                         f"{sel.get('threshold', {}).get('shipped')}\n"
                         'Change biolock/config.py and run again.')
        elif step == 'calibration':
            code, _s, _l = run(step, [PY, os.path.join(exp, 'simulate.py'), '--calibrate', '--out', out] + q, out)
            cal = (load(out, step) or {}).get('calibration', {})
            if code != 0 or (not quick and not cal.get('shipped_matches_rules')):
                sys.exit('Stopped before the test halves: the shipped liveness thresholds are not what the '
                         f"selection rules give on the development halves.\n  rules   {cal.get('rules')}\n"
                         f"  shipped {cal.get('shipped')}\nChange biolock/config.py and run again.")
        elif step == 'benchmark_test':
            if quick:                   # a quick run never looks at the test halves
                continue
            code, _s, _l = run(step, [PY, os.path.join(exp, 'benchmark.py'), '--split', 'test', '--out', out], out)
        elif step == 'simulation_test':
            split = 'dev' if quick else 'test'
            code, _s, _l = run(step, [PY, os.path.join(exp, 'simulate.py'), '--split', split, '--out', out] + q, out)
        elif step == 'perf':
            exe = os.path.join(ROOT, 'dist', 'BioLock.exe')
            marker = os.path.join(ROOT, 'dist', 'BUILD.json')        # written by the build of this version
            extra = ['--exe', exe] if os.path.isfile(exe) and os.path.isfile(marker) else []
            code, _s, _l = run(step, [PY, os.path.join(exp, 'perf.py'), '--out', out] + extra, out)
        else:
            sys.exit(f'unknown step {step!r}')
        if code != 0:
            sys.exit(f'Step {step} failed, see {os.path.join(out, step + ".log")}')


# --------------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------------
def pc(x, d=1):
    return 'n/a' if x is None or x != x else f'{100 * x:.{d}f} %'


def ci(pair, d=1):
    return 'n/a' if not pair or pair[0] != pair[0] else f'{100 * pair[0]:.{d}f} to {100 * pair[1]:.{d}f} %'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


SHORT = {'lfw': 'LFW', 'olivetti': 'Olivetti'}
TITLES = {'lfw': 'LFW (unconstrained)', 'olivetti': 'Olivetti / ORL (controlled)'}

DECISIONS = (('single', 'one photograph'), ('median3', 'median of 3 photographs'),
             ('median5', 'median of 5 photographs'))
GALLERIES = (('photos', 'K photographs'), ('live', 'emulated live enrolment'))


def matcher_rows(ds):
    rows = []
    for g, gname in GALLERIES:
        for d, dname in DECISIONS:
            m = ds['galleries'][g].get(d)
            if not m:
                continue
            a = m['at_threshold']
            rows.append([gname, dname, m['n_gen'], m['n_imp'], pc(m['eer']), f"{m['auc']:.3f}", pc(m['tar1']),
                         f"{pc(a['far'], 2)} ({ci(a['far_ci'], 2)})", f"{pc(a['frr'])} ({ci(a['frr_ci'])})"])
    return rows


def section_scope():
    return '\n'.join([
        '## 1. What these results cover', '',
        table(['Part of the system', 'Status', 'Source'], [
            ['Identity verification (detection, alignment, LBPH match, threshold)', 'measured',
             'two public face datasets, test halves, `experiments/benchmark.py`'],
            ['Session logic (presence, blink, head movement, match, identity binding)', 'simulated',
             'animated photographs, `experiments/simulate.py`'],
            ['Resistance to photographs / replay', 'simulated for photographs only',
             'animated photographs; a replay that shows the cues is not stopped'],
            ['Speed and memory', 'measured on one machine', '`experiments/perf.py`'],
            ['Software behaviour (lock-out, wipe, key check, drive rule)', 'unit tests', '`tests/`'],
            ['Real people, a real webcam, printed photos, phones, replayed video', '**not measured**',
             'needs the live trials in `docs/TESTING_GUIDE.md`']]), '',
        'A simulation is not a measurement of people. Section 5 says what the simulated sessions can and '
        'cannot show.'])


def section_protocol(bt):
    rows = []
    for name, ds in bt['datasets'].items():
        det = ds['detection']
        rows.append([ds['title'], ds['identities_in_half'], ds['photographs'], ds['k_enrol'], len(ds['seeds']),
                     pc(det['face_found']), det['several_faces'], pc(det['eyes_found_of_faces']),
                     f"{det['face_width_px_mean']:.0f} px"])
    return '\n'.join([
        '## 2. Protocol', '',
        '- Datasets: LFW (the 62 people with at least 20 photographs; unconstrained press photographs) and '
        'Olivetti / ORL (40 people, 10 photographs each, taken in one laboratory).',
        '- Identities are split by position in sorted order: even index = development half (parameters are '
        'chosen there), odd index = test half (reported here). Nothing was tuned on the test halves.',
        '- Each photograph is scaled so that the face is about 190 px wide and placed in a 640x480 frame. From '
        'there the application code runs unchanged: frame preparation, Haar face detection, eye detection, '
        'alignment, Tan-Triggs normalisation, LBPH (radius 5, 8 neighbours, 6x6 grid), threshold.',
        '- Per identity and seed, K random photographs enrol; the other photographs of that person are genuine '
        'probes (at most 30 for LFW, 5 for Olivetti); 150 (LFW) or 120 (Olivetti) impostor probes are drawn '
        'uniformly over the other identities of the same half.',
        '- Two galleries: the K photographs alone, and an emulated live enrolment (each photograph plus '
        'jittered, noisy copies that go through detection again, about 90 samples, like the 100-frame '
        'enrolment of the app). The threshold was set for the second one.',
        '- "Median of m" means the median score of m different photographs of one person. It is not the same '
        'as the 15 consecutive camera frames the app uses, which are much more alike.',
        '- A photograph without a detected face is left out. If several faces are detected the one nearest the '
        'centre is used (the app itself refuses such frames).',
        '- Intervals in brackets are 95 % intervals from resampling the enrolled identities 2,000 times. Trials '
        'of one identity are correlated, so per-trial intervals would be too narrow.', '',
        'Test halves:', '',
        table(['Dataset', 'Identities', 'Photographs', 'K', 'Seeds', 'Face found', 'Several faces', 'Both eyes found',
               'Mean face width'], rows)])


def section_selection(bd, cal):
    s = bd['selection']
    c = s['calibration']
    w = s['weights']
    lines = ['## 3. Parameters chosen on the development halves', '',
             'Rules were fixed before the test halves were run (they are written at the top of '
             '`experiments/benchmark.py` and `experiments/simulate.py`).', '',
             '**Score calibration** (maps each raw measure to 0-100; pooled 1st / 99th percentiles):', '',
             table(['Constant', 'Shipped', 'Re-derived through the app code'],
                   [[k, c['shipped'][k], f"{c['derived_through_engine'][k]:.3f}"] for k in c['shipped']]), '',
             '**Fusion weights** (LBPH, histogram, template). Rule: lowest mean equal-error rate over the two '
             'development sets on a 0.05 grid.', '',
             table(['Weights', 'Mean EER'] + [f'EER {SHORT.get(n, n)}' for n in w['datasets']],
                   [[f"{tuple(r['weights'])}", pc(r['mean_eer'], 2)] + [pc(e) for e in r['eer']] for r in w['top'][:5]]
                   + [[f"{tuple(w[k]['weights'])} {label}", pc(w[k]['mean_eer'], 2)] + [pc(e) for e in w[k]['eer']]
                      for k, label in (('paper', '(manuscript)'), ('v9', '(version 9)'))]), '']
    best = tuple(w['grid_minimum']['weights'])
    best_eer = w['grid_minimum']['mean_eer']
    if best == (1.0, 0.0, 0.0):
        worse = ' and '.join(f"{100 * (w[k]['mean_eer'] - best_eer):.1f} percentage points higher for the weights of "
                             f"{label}" for k, label in (('paper', 'the manuscript'), ('v9', 'version 9')))
        lines.append(f'The grid minimum is LBPH alone: no mix with the histogram or template score has a lower mean '
                     f'equal-error rate. The mean rate is {worse}. Shipped: (1, 0, 0).')
        per = w.get('per_dataset_minimum') or {}
        if per:
            lines.append('Each development set on its own: ' + '; '.join(
                f"{SHORT.get(n, n)}: best grid point {tuple(v['weights'])} with EER {pc(v['eer'], 2)}, LBPH alone "
                f"{pc(v['lbph_alone_eer'], 2)}" for n, v in per.items()) + '.')
    else:
        d = w['lbph_minus_minimum']
        lines.append(f"The grid minimum is {best}; LBPH alone is {100 * d['mean_eer_difference']:+.2f} percentage "
                     f"points away (95 % interval {100 * d['ci95'][0]:+.2f} to {100 * d['ci95'][1]:+.2f}). "
                     f"Chosen by the rule: {tuple(w['chosen_by_rule'])}.")
    t = s['threshold']
    lines += ['', '**Threshold.** Rule: the smallest whole number T such that at most 1 % of single-photograph '
                  'impostor probes score T or more, emulated live enrolment, on each development set.', '',
              table(['Development set', 'Impostor probes', 'Score exceeded by 1 %', 'T', 'FAR at T', 'FRR at T'],
                    [[SHORT.get(n, n), v['impostor_probes'], f"{v['score_with_1pct_far']:.2f}", v['threshold'],
                      pc(v['far_at_threshold'], 2), pc(v['frr_at_threshold'])] for n, v in t['per_dataset'].items()]),
              '', f"Shipped: T = {t['shipped']:g} (the larger of the two), which is an LBPH chi-square distance of "
                  f"{t['lbph_distance_at_threshold']:.1f}."]
    if cal:
        k = cal['calibration']
        pr, bl = k['presence'], k['blink']
        order = (('static_n2', 'motionless photograph, sensor noise 2'), ('static_n5', 'motionless photograph, noise 5'),
                 ('static_n8', 'motionless photograph, noise 8'), ('still_person', 'sway of 1.5 px'),
                 ('sway_5', 'sway of 5 px'), ('sway_10', 'sway of 10 px'), ('sway_20', 'sway of 20 px'),
                 ('shaken', 'trembling photograph'))
        lines += ['', '**Liveness thresholds (simulation).** These three values were set from animated photographs '
                      f"of the development halves ({sum(k['photos'].values())} photographs). They are starting "
                      'values: only live trials can confirm them.', '',
                  'Presence measure (mean absolute frame difference inside the face box on 8x8 block-averaged '
                  'pixels, mean of 20 frames):', '',
                  table(['Input', 'n', '1st pct', 'median', '99th pct'],
                        [[label, pr[key]['n'], f"{pr[key]['pct'][0]:.4f}", f"{pr[key]['pct'][2]:.4f}",
                          f"{pr[key]['pct'][4]:.4f}"] for key, label in order if key in pr]), '',
                  f"Rule: geometric mean of the 99th percentile for a motionless photograph with noise 5 "
                  f"({pr['rule']['static_n5_p99']:.4f}) and the 1st percentile for a 10 px sway "
                  f"({pr['rule']['sway_10_p1']:.4f}) = {pr['rule']['geometric_mean']:.4f}. "
                  f"Shipped `motion_thresh` = {k['shipped']['motion_thresh']}.", '',
                  'Blink evidence at the moment the eye detector loses the eyes (change in the eye band, change in '
                  'the rest of the face, their ratio; medians and 99th percentiles):', '',
                  table(['Event', 'Events', 'Eye band: median', '99th pct', 'Ratio: median', '99th pct',
                         'Counted as a blink by the shipped rule'],
                        [[label, bl[key]['events'], f"{bl[key]['eye_pct'][2]:.4f}", f"{bl[key]['eye_pct'][4]:.4f}",
                          f"{bl[key]['ratio_pct'][2]:.2f}", f"{bl[key]['ratio_pct'][4]:.2f}",
                          '%d of %d' % tuple(bl[key]['pass_shipped_rule_count'])]
                         for key, label in (('blink', 'simulated blink (eyes painted over)'),
                                            ('shaken_dropouts', 'trembling photograph'),
                                            ('noise_dropouts', 'motionless photograph, noise 5')) if bl[key]['events']]),
                  '', f"Rules: `blink_local_ratio` = 99th percentile of the ratio for a trembling photograph "
                      f"({bl['rule']['shaken_ratio_p99']:.2f}) rounded up to 0.5 = {k['shipped']['blink_local_ratio']}; "
                      f"`blink_min_change` = 99th percentile of the eye-band change from noise alone "
                      f"({bl['rule']['noise_eye_change_p99']:.4f}) rounded up = {k['shipped']['blink_min_change']}."]
    return '\n'.join(lines)


def section_matcher(bt, bd):
    T = bt['meta']['config']['min_conf']
    lines = ['## 4. Identity verification on the test halves', '',
             f"Shipped configuration: LBPH radius {bt['meta']['config']['lbph_radius']}, "
             f"{bt['meta']['config']['lbph_neighbors']} neighbours, grid {bt['meta']['config']['lbph_grid']}x"
             f"{bt['meta']['config']['lbph_grid']}, weights ({bt['meta']['config']['w_lbph']:g}, "
             f"{bt['meta']['config']['w_hist']:g}, {bt['meta']['config']['w_template']:g}), threshold T = {T:g}. "
             'FAR = impostor probes accepted, FRR = genuine probes rejected, both at T. EER, AUC and TAR at 1 % '
             'FAR describe the score itself, independent of T (mean over the seeds).']
    for name, ds in bt['datasets'].items():
        lines += ['', f"**{ds['title']}** ({ds['identities']} enrolled identities, {ds['models']} enrolments over "
                      f"{len(ds['seeds'])} seeds)", '',
                  table(['Gallery', 'Decision on', 'Genuine', 'Impostor', 'EER', 'AUC', 'TAR at 1 % FAR',
                         f'FAR at T={T:g}', f'FRR at T={T:g}'], matcher_rows(ds))]
    rows = []
    for name, ds in bt['datasets'].items():
        dv = bd['datasets'].get(name) if bd else None
        for g, gname in GALLERIES:
            t, d = ds['galleries'][g]['single'], (dv['galleries'][g]['single'] if dv else None)
            rows.append([ds['title'], gname, pc(d['eer']) if d else 'n/a', pc(t['eer']),
                         pc(d['at_threshold']['far'], 2) if d else 'n/a', pc(t['at_threshold']['far'], 2),
                         pc(d['at_threshold']['frr']) if d else 'n/a', pc(t['at_threshold']['frr'])])
    lines += ['', 'Development versus test (one photograph per decision). The threshold was chosen on the '
                  'development halves, so the test columns are the honest ones:', '',
              table(['Dataset', 'Gallery', 'EER dev', 'EER test', 'FAR dev', 'FAR test', 'FRR dev', 'FRR test'], rows)]

    names = list(bt['datasets'])
    keys = list(bt['datasets'][names[0]]['weightings'])
    rows = []
    for k in keys:
        row = [k]
        for n in names:
            m = bt['datasets'][n]['weightings'][k]
            row += [pc(m['eer']), pc(m['at_threshold']['far'], 2), pc(m['at_threshold']['frr'])]
        rows.append(row)
    for k in bt['datasets'][names[0]]['components']:
        row = [f'{k} alone']
        for n in names:
            row += [pc(bt['datasets'][n]['components'][k]['eer']), '', '']
        rows.append(row)
    head = ['Score']
    for n in names:
        head += [f'EER {SHORT.get(n, n)}', 'FAR at T', 'FRR at T']
    lines += ['', '**Does fusing three scores help?** One photograph per decision, K-photograph gallery, test '
                  'halves. The threshold T belongs to the shipped weights; the other rows show what the same T '
                  'would do with other weights.', '', table(head, rows), '']
    lines.append(fusion_finding(bt))
    rows = []
    for n in names:
        ds = bt['datasets'][n]
        v9, new = ds['v9_baseline'], ds['galleries']['photos']
        rows.append([ds['title'], pc(v9['single']['eer']), pc(new['single']['eer']), pc(v9['single']['tar1']),
                     pc(new['single']['tar1']),
                     pc(v9['median5']['eer']) if 'median5' in v9 else 'n/a',
                     pc(new['median5']['eer']) if 'median5' in new else 'n/a'])
    lines += ['', '**Against the version 9 matcher** (box crop, double histogram equalisation and blur, LBPH radius '
                  '1, 8x8 grid) on the same detections, K-photograph gallery:', '',
              table(['Dataset', 'EER v9', 'EER v10', 'TAR at 1 % FAR v9', 'TAR at 1 % FAR v10', 'EER v9, median of 5',
                     'EER v10, median of 5'], rows)]
    return '\n'.join(lines)


SCENARIOS = (('genuine', 'enrolled person, all cues'), ('impostor', 'another person, all cues'),
             ('photo_static_noise2', 'motionless photograph of the enrolled person, noise 2'),
             ('photo_static_noise5', 'motionless photograph, noise 5'),
             ('photo_shaken', 'trembling photograph, moved when asked, no blink'),
             ('swap_bound', 'liveness by another face, then the enrolled face for the match'),
             ('swap_unbound', 'the same with identity binding switched off'))


def section_simulation(sim):
    m = sim['meta']
    lines = ['## 5. Simulated sessions on the test halves (SIMULATION, not live trials)', '',
             'Photographs of the test halves are animated and fed frame by frame to the session code the app runs '
             f"behind the camera, with a clock that advances 1/{m['fps']:g} s per frame. Enrolment is the emulated "
             f"live enrolment of section 2. The simulated person sways {m['sway_px']:g} px during the presence step, "
             'blinks (the eye regions are painted over for 4 frames) when asked, moves left then right, then '
             'holds still. Time-outs and thresholds are the shipped ones.', '',
             '**What this can show:** whether the session logic stops inputs that lack a cue, what the matcher '
             'does at the end of a complete session, and that the steps are tied to one face.', '',
             '**What it cannot show:** how real faces, real blinks, a printed photograph, a phone screen or a '
             'camera behave. An animated photograph of the enrolled person that performs the cues is accepted '
             'like the person: for this pipeline it is the same input as a replayed video. The blink, motion '
             'and head-movement cues do not stop such a replay.']
    for name, ds in sim['datasets'].items():
        rows = []
        for key, label in SCENARIOS:
            s = ds['summary'].get(key)
            if not s:
                continue
            st = s['stopped_at']
            rows.append([label, s['n'], s['accepted'], f"{pc(s['accept_rate'])} ({ci(s['accept_ci'])})",
                         st.get('presence', 0), st.get('blink', 0), st.get('head', 0), st.get('match', 0)])
        lines += ['', f'**{TITLES.get(name, name)}**', '',
                  table(['Simulated input', 'Sessions', 'Accepted', 'Accept rate (95 % interval)', 'Stopped at presence',
                         'at blink', 'at head', 'Rejected at match'], rows)]
        g = ds['summary'].get('genuine')
        if g:
            lines += ['', f"Of the genuine sessions rejected at the match, {g['rejected_only_by_identity_binding']} had "
                          'a passing match-stage score and failed only the identity binding (the blink-stage frames '
                          'scored under the threshold).']
        tried = ds.get('photographs_tried_for_a_blinking_role') or {}
        if tried:
            total = sum(tried.values())
            lines += ['', f"A photograph can only play a blinking part if exactly one face is detected in it, the eye "
                          f"check sees its open eyes in at least 5 of 6 noisy frames, and it stops seeing them once "
                          f"they are painted over. Of {total} distinct photographs tried, {tried.get('usable', 0)} "
                          f"qualified; the others: "
                          + '; '.join(f'{v} {k}' for k, v in sorted(tried.items()) if k != 'usable') + '.']
        skipped = ds.get('sessions_not_run') or {}
        if skipped:
            labels = dict(SCENARIOS, photo_static='motionless photograph')
            lines += ['', 'Sessions that were planned but not run because a photograph did not qualify: '
                      + ', '.join(f'{v} x {labels.get(k, k)}' for k, v in sorted(skipped.items())) + '.']
    lines += ['', 'Intervals here are Wilson 95 % intervals over sessions. The genuine accept rate is mostly the '
                  'matcher on one unseen photograph per session (compare the FRR for "one photograph" in section 4); '
                  'it says nothing about how often a live person completes the blink step.', '',
              '"Open eyes not detected steadily" is a property of the Haar eye detector on these photographs (small '
              'or turned faces, glasses). It matters for live use: someone whose open eyes it does not see steadily '
              'cannot complete the blink step. How often that happens in front of a webcam has to be measured live.']
    return '\n'.join(lines)


def section_perf(perf):
    mc = perf['machine']
    lines = ['## 6. Speed and memory', '',
             f"Machine: {mc.get('cpu', mc.get('processor', 'unknown CPU'))}, {mc['logical_cpus']} logical CPUs, "
             f"{mc.get('ram_gb', '?')} GB RAM, {mc['platform']}, Python {mc['python']}, OpenCV {mc['opencv']}. "
             'Frames are 640x480 with one face about 270 px wide. Other machines will differ.', '',
             table(['Step', 'Median', '95th percentile'],
                   [[k, f"{v['median_ms']:.2f} ms", f"{v['p95_ms']:.2f} ms"] for k, v in perf['steps'].items()])]
    for mode, title in (('enrol', 'enrolment'), ('verify', 'verification')):
        s = perf['sessions'].get(mode)
        if not s:
            continue
        lines += ['', f"Processing per frame during a complete simulated {title} ({s['frames']} frames, "
                      f"{s['simulated_seconds']:g} s at 30 frames/s):", '',
                  table(['Stage', 'Frames', 'Median per frame', '95th percentile', 'Frames per second possible'],
                        [[k, v['frames'], f"{v['median_ms']:.1f} ms", f"{v['p95_ms']:.1f} ms", f"{v['frames_per_second']:.0f}"]
                         for k, v in s['stages'].items()])]
    lines += ['', table(['Memory', 'MB'], [[k, v] for k, v in perf['memory_mb'].items()]), '',
              table(['Start-up and one-off costs', 'Value'], [[k, v] for k, v in perf['startup'].items()
                                                             if k != 'build template from samples (ms)']
                    + ([['build the template from 100 samples (ms)', perf['sessions']['build template from samples (ms)']]]
                       if 'build template from samples (ms)' in perf['sessions'] else []))]
    return '\n'.join(lines)


def build_report(out):
    bd, bt = load(out, 'benchmark_dev'), load(out, 'benchmark_test')
    cal, sim, perf, tests = (load(out, k) for k in ('calibration', 'simulation_test', 'perf', 'tests'))
    version = (bt or bd or {}).get('meta', {}).get('version', '?')
    parts = [f'# BioLock {version}: measured results', '',
             f'Written by `python evaluate.py` on {datetime.date.today().isoformat()}. Every number below is read '
             'from the JSON files that command puts in `results/`; none was typed in. Run it again to check them.', '',
             section_scope()]
    if bt:
        parts += ['', section_protocol(bt)]
    if bd:
        parts += ['', section_selection(bd, cal)]
    if bt:
        parts += ['', section_matcher(bt, bd)]
    else:
        parts += ['', '## 4. Identity verification on the test halves', '', 'Not run yet.']
    parts += ['', section_simulation(sim) if sim else '## 5. Simulated sessions\n\nNot run yet.']
    parts += ['', section_perf(perf) if perf else '## 6. Speed and memory\n\nNot run yet.']
    if tests:
        parts += ['', '## 7. Unit tests', '',
                  f"{tests['ran']} tests, {'all passed' if tests['ok'] else 'FAILURES'}"
                  + (f" ({tests['skipped']} skipped)" if tests.get('skipped') else '')
                  + f", {tests['seconds']:.0f} s, Python {tests['python']}. They use drawn faces and fake cameras: "
                    'they check that the software does what it is meant to (median decision, lock-out and wipe, key '
                    'check, drive rule, template file checks), not how well it recognises people.']
    parts += ['', '## 8. What is still missing', '',
              '- Live trials with people and a webcam: genuine users, other people, a printed photograph, a phone '
              'screen, a replayed video. Until those exist there is no measured liveness or presentation-attack '
              'rate for this system, and the three liveness thresholds of section 3 are unconfirmed.',
              '- The datasets are photographs taken on different occasions, which is harder than one person in '
              'front of one webcam, and the Olivetti set is small. Treat the matcher numbers as a benchmark of '
              'the method, not as the accuracy a user will see.',
              '- All timings are from one machine.', '',
              '## 9. Reproduce', '', '```bash', 'pip install -r requirements.txt -r requirements-dev.txt',
              'python evaluate.py', '```', '',
              'The datasets are downloaded through scikit-learn on first use (about 200 MB, cached in '
              '`research/data/`). All random choices are seeded.', '']
    return '\n'.join(parts)


# --------------------------------------------------------------------------------------------------------
# short versions for README.md and docs/PAPER_ALIGNMENT.md (kept between <!-- name:start/end --> markers)
# --------------------------------------------------------------------------------------------------------
ORDER = ('olivetti', 'lfw')


def fusion_finding(bt):
    """One sentence, decided by the data: did any other weighting beat the shipped one on both test halves?"""
    names = list(bt['datasets'])
    keys = list(bt['datasets'][names[0]]['weightings'])
    ship = next(k for k in keys if k.startswith('shipped'))
    eer = lambda k, n: bt['datasets'][n]['weightings'][k]['eer']                    # noqa: E731
    better = [k for k in keys if k != ship and all(eer(k, n) < eer(ship, n) for n in names)]
    if better:
        return ('On the test halves ' + ', '.join(better) + ' has a lower equal-error rate than the shipped weights '
                'on both datasets. The shipped weights were chosen on the development halves and were not changed '
                'afterwards.')
    return ('On the test halves none of the other weightings tried has a lower equal-error rate than the shipped '
            'weights on both datasets. The weighted fusion described in the manuscript is not supported by these '
            'measurements.')


def sim_counts(sim):
    """{scenario: (accepted, sessions)} pooled over the datasets; the two static-photograph variants together."""
    out = {}
    for ds in sim['datasets'].values():
        for key, s in ds['summary'].items():
            key = 'photo_static' if key.startswith('photo_static') else key
            a, n = out.get(key, (0, 0))
            out[key] = (a + s['accepted'], n + s['n'])
    return out


def sim_sentence(sim):
    c = sim_counts(sim)
    part = lambda key, text: f'{text} in {c[key][0]} of {c[key][1]}' if key in c else None     # noqa: E731
    bits = [part('photo_static', 'a motionless photograph of the enrolled person was accepted'),
            part('photo_shaken', 'a trembling one that never blinks'),
            part('swap_bound', 'liveness done by one face and the match by another'),
            part('swap_unbound', 'the same with the identity binding switched off'),
            part('impostor', 'another person performing all cues'),
            part('genuine', 'an unseen photograph of the enrolled person performing all cues')]
    return '; '.join(b for b in bits if b) + ' sessions'


def summary_block(out):
    """What README.md says about the results."""
    bt, sim, perf = load(out, 'benchmark_test'), load(out, 'simulation_test'), load(out, 'perf')
    if not bt:
        return 'Not run yet.'
    T = bt['meta']['config']['min_conf']
    ds = [bt['datasets'][n] for n in ORDER if n in bt['datasets']]
    live = lambda d: d['galleries']['live']                                               # noqa: E731
    at = lambda d: live(d)['single']['at_threshold']                                      # noqa: E731
    rows = [['Equal-error rate'] + [pc(live(d)['single']['eer']) for d in ds],
            [f'Impostor photographs accepted at the shipped threshold ({T:g})']
            + [f"{pc(at(d)['far'], 2)} ({ci(at(d)['far_ci'], 2)})" for d in ds],
            ['Genuine photographs rejected at the shipped threshold']
            + [f"{pc(at(d)['frr'])} ({ci(at(d)['frr_ci'])})" for d in ds],
            ['Equal-error rate when the median of 5 photographs decides']
            + [pc(live(d)['median5']['eer']) if 'median5' in live(d) else 'n/a' for d in ds],
            ['Equal-error rate, version 9 matcher -> version 10 (gallery of K photographs)']
            + [f"{pc(d['v9_baseline']['single']['eer'])} -> {pc(d['galleries']['photos']['single']['eer'])}" for d in ds]]
    lines = ['**Identity verification**, test halves of two public face datasets, through the app\'s own code. '
             'Parameters were chosen on the development halves only. One photograph per decision, enrolment '
             'emulating the app\'s 100 frames, 95 % intervals in brackets:', '',
             table([''] + [d['title'] for d in ds], rows), '',
             '- The matcher works on controlled photographs and rejects about half of the genuine ones on '
             'unconstrained press photographs at this threshold. A webcam session is closer to the controlled case, '
             'but that has not been measured.' if len(ds) == 2 and at(ds[1])['frr'] > 0.4 and at(ds[0])['frr'] < 0.15
             else '- How often a genuine photograph is rejected depends strongly on the conditions.',
             '- ' + fusion_finding(bt)]
    if sim:
        lines.append('- **Simulated sessions** (animated photographs of the test halves, not people): '
                     + sim_sentence(sim) + '.')
        tried = [ds.get('photographs_tried_for_a_blinking_role') or {} for ds in sim['datasets'].values()]
        total = sum(sum(t.values()) for t in tried)
        unseen = sum(t.get('open eyes not detected steadily', 0) for t in tried)
        if total:
            lines.append(f'- The eye check that the blink step depends on did not see the open eyes steadily in '
                         f'{unseen} of {total} photographs tried ({pc(unseen / total, 0)}). Whether that also '
                         f'happens with a webcam has to be found out in the live trials.')
    if perf and 'verify' in perf.get('sessions', {}):
        st = perf['sessions']['verify']['stages']
        mc = perf['machine']
        lines.append(f"- **Speed** on a {mc.get('cpu', 'test machine')}: {st['blink']['median_ms']:.0f} ms per frame in "
                     f"the blink step and {st['face']['median_ms']:.0f} ms per frame while matching; "
                     f"{perf['memory_mb']['after two full sessions']:.0f} MB of memory.")
    return '\n'.join(lines)


def wording_block(out):
    """Section 3 of docs/PAPER_ALIGNMENT.md: a paragraph the manuscript can use, with the measured numbers."""
    bt, sim = load(out, 'benchmark_test'), load(out, 'simulation_test')
    if not bt:
        return 'Not run yet.'
    o, lf = bt['datasets'].get('olivetti'), bt['datasets'].get('lfw')
    if not o or not lf:
        return 'Both datasets are needed for this paragraph.'
    T = bt['meta']['config']['min_conf']
    s = lambda d: d['galleries']['live']['single']                                        # noqa: E731
    a = lambda d: s(d)['at_threshold']                                                    # noqa: E731
    w = lambda d, key: next(v for k, v in d['weightings'].items() if k.startswith(key))  # noqa: E731
    man_worse = all(w(d, 'manuscript')['eer'] > w(d, 'shipped')['eer'] for d in (o, lf))
    text = [
        'To characterise the identity-verification stage independently of the camera, the matcher was run through '
        'the application code on two public datasets: LFW (the 62 people with at least 20 photographs; '
        'unconstrained) and Olivetti/ORL (40 people, 10 photographs each; controlled). Identities were divided into '
        'a development half and a test half. Score calibration, fusion weights and the acceptance threshold were '
        'chosen on the development halves; the test halves were used only to report.',
        f"With an enrolment gallery that emulates the 100-frame enrolment and one photograph per decision, the "
        f"equal-error rate on the test halves was {pc(s(o)['eer'])} (Olivetti) and {pc(s(lf)['eer'])} (LFW). At the "
        f"shipped threshold ({T:g}) the false-accept rate was {pc(a(o)['far'], 2)} (95 % interval {ci(a(o)['far_ci'], 2)}) "
        f"and {pc(a(lf)['far'], 2)} ({ci(a(lf)['far_ci'], 2)}), and the false-reject rate {pc(a(o)['frr'])} "
        f"({ci(a(o)['frr_ci'])}) and {pc(a(lf)['frr'])} ({ci(a(lf)['frr_ci'])}).",
        (f"A weighted fusion of the LBPH score with the histogram and template scores did not improve on LBPH alone: "
         f"with the weights (0.45, 0.20, 0.35) the equal-error rate was {pc(w(o, 'manuscript')['eer'])} and "
         f"{pc(w(lf, 'manuscript')['eer'])}, against {pc(w(o, 'shipped')['eer'])} and {pc(w(lf, 'shipped')['eer'])} "
         f"for LBPH alone (gallery of K photographs).") if man_worse else
        'The comparison of fusion weights on the test halves is in RESULTS.md section 4; state it as measured.',
        f"Compared with the earlier matcher (LBPH radius 1, 8x8 grid, histogram equalisation) the equal-error rate "
        f"went from {pc(o['v9_baseline']['single']['eer'])} to {pc(o['galleries']['photos']['single']['eer'])} "
        f"(Olivetti) and from {pc(lf['v9_baseline']['single']['eer'])} to "
        f"{pc(lf['galleries']['photos']['single']['eer'])} (LFW) on the same detections.",
        'Liveness and presentation-attack performance were not measured with live subjects.'
        + (f" Sessions built from animated photographs were used only to check the session logic: "
           f"{sim_sentence(sim)}. A replayed video that shows the blinks and the head movement is not "
           f"distinguished from a person by these cues." if sim else '')]
    return '\n'.join('> ' + t for t in text)


def fill(path, name, text, inline=False):
    """Replace what stands between <!-- name:start --> and <!-- name:end --> in a document."""
    with open(path, encoding='utf-8') as f:
        doc = f.read()
    pattern = re.compile(rf'(<!-- {name}:start -->).*?(<!-- {name}:end -->)', re.S)
    if not pattern.search(doc):
        print(f'  {os.path.relpath(path, ROOT)}: no "{name}" markers, left alone')
        return False
    sep = '' if inline else '\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(pattern.sub(lambda m: m.group(1) + sep + text + sep + m.group(2), doc))
    return True


def main():
    ap = argparse.ArgumentParser(description='Run the evaluation and write docs/RESULTS.md')
    ap.add_argument('--steps', default=','.join(STEPS), help='comma-separated subset of: ' + ', '.join(STEPS))
    ap.add_argument('--report', action='store_true', help='only rebuild the report from existing results')
    ap.add_argument('--quick', action='store_true', help='small subsets; results/quick/, no report in docs/')
    a = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'quick') if a.quick else os.path.join(ROOT, 'results')
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if not a.report:
        steps = [s.strip() for s in a.steps.split(',') if s.strip()]
        bad = [s for s in steps if s not in STEPS]
        if bad:
            sys.exit(f'unknown step(s): {", ".join(bad)}')
        run_steps([s for s in STEPS if s in steps], out, a.quick)
    if a.quick:
        print(f'Quick run finished in {time.time() - t0:.0f} s. Results are in {os.path.relpath(out, ROOT)}; '
              'they are too small to report.')
        return
    path = os.path.join(ROOT, 'docs', 'RESULTS.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(build_report(out))
    print(f'Wrote {os.path.relpath(path, ROOT)}  ({time.time() - t0:.0f} s)')
    # the same numbers, in short, where the README and the manuscript check-list quote them
    fill(os.path.join(ROOT, 'README.md'), 'results', summary_block(out))
    tests = load(out, 'tests')
    if tests:
        fill(os.path.join(ROOT, 'README.md'), 'tests', f"{tests['ran']} tests.", inline=True)
    fill(os.path.join(ROOT, 'docs', 'PAPER_ALIGNMENT.md'), 'wording', wording_block(out))


if __name__ == '__main__':
    main()
