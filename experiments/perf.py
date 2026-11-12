#!/usr/bin/env python3
"""Timing and memory of the pipeline on this machine.

    python experiments/perf.py [--out results] [--exe path/to/BioLock.exe]

Measures the cost of each processing step on 640x480 frames, the per-frame cost of a whole session in
each stage, the frame rate that allows, memory use, and start-up times. The frames show a drawn face
(tests/synthface.py), so no camera, photograph or dataset is needed. Numbers depend on the machine:
the report records which one.
"""
import os
import sys
import json
import time
import argparse
import platform
import tempfile
import subprocess
import collections

t_import = time.perf_counter()
import numpy as np                                      # noqa: E402
import cv2                                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from biolock import __version__, sim, liveness, vault   # noqa: E402
from biolock import session as S                        # noqa: E402
from biolock.config import load_config                  # noqa: E402
from biolock.engine import FaceEngine, tan_triggs       # noqa: E402
from tests.synthface import make_face                   # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None

OWNER, SCALE = 6, 1.6


def rss_mb():
    return psutil.Process().memory_info().rss / 1048576.0 if psutil else float('nan')


def timeit(fn, seconds=0.6, least=20):
    """Median and 95th percentile of fn() in milliseconds."""
    fn()
    samples, t_end = [], time.perf_counter() + seconds
    while len(samples) < least or time.perf_counter() < t_end:
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
        if len(samples) >= 2000:
            break
    return {'median_ms': round(float(np.median(samples)), 3), 'p95_ms': round(float(np.percentile(samples, 95)), 3),
            'runs': len(samples)}


def machine():
    info = {'platform': platform.platform(), 'processor': platform.processor(), 'python': sys.version.split()[0],
            'opencv': cv2.__version__, 'numpy': np.__version__, 'logical_cpus': os.cpu_count(),
            'opencv_threads': cv2.getNumThreads()}
    if psutil:
        info['ram_gb'] = round(psutil.virtual_memory().total / 1073741824.0, 1)
        try:
            info['cpu_mhz'] = round(psutil.cpu_freq().max or psutil.cpu_freq().current)
        except Exception:
            pass
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                info['cpu'] = winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return info


def steps(E, cfg):
    """Cost of the individual operations on one 640x480 frame with one face."""
    a = sim.Actor(E, make_face(OWNER), SCALE, np.random.RandomState(0), kind='live')
    cam = sim.to_camera(cv2.cvtColor(a.frame(None), cv2.COLOR_GRAY2BGR))
    cam2 = sim.to_camera(cv2.cvtColor(a.frame(None), cv2.COLOR_GRAY2BGR))
    _f, gray, eq = S.prepare(cam)
    _f, _g, eq2 = S.prepare(cam2)
    box = E.detect_faces(eq)[0]
    eyes = E.find_eyes(eq, box)
    raw = E.crop(gray, box, eyes)
    roi = tan_triggs(raw)
    rng = np.random.RandomState(1)
    gallery = [tan_triggs(np.clip(raw.astype(int) + rng.randint(-12, 13, raw.shape), 0, 255).astype(np.uint8))
               for _ in range(cfg['enrol_frames'])]
    E.train(gallery, save=False)
    feat = E.features(roi)
    out = collections.OrderedDict()
    out['prepare frame (mirror, grey, equalise)'] = timeit(lambda: S.prepare(cam))
    out['face detection (half resolution)'] = timeit(lambda: E.detect_faces(eq))
    out['eye check for blinks (reduced band)'] = timeit(lambda: E.eyes_visible(eq, box))
    out['eye positions for alignment'] = timeit(lambda: E.find_eyes(eq, box))
    out['aligned crop + Tan-Triggs'] = timeit(lambda: tan_triggs(E.crop(gray, box, eyes)))
    out['features of one crop (LBP histogram, HOG)'] = timeit(lambda: E.features(roi))
    out[f'comparison with {E.meta["samples"]} enrolled samples'] = timeit(lambda: E.compare(feat))
    out['whole match of one frame (eyes, crop, score)'] = timeit(lambda: E.score(E.extract(gray, eq, box)[0]))
    out['presence measure'] = timeit(lambda: liveness.box_motion(eq, eq2, box))
    out['blink evidence measure'] = timeit(lambda: liveness.local_change(eq, eq2, box))
    return out, len(gallery)


def session_cost(cfg):
    """Per-frame processing time of a full simulated enrolment and verification, by stage."""
    E = FaceEngine(cfg)
    res = {}
    for mode in (S.MODE_ENROL, S.MODE_VERIFY):
        clock = sim.SimClock()
        ses = S.Session(E, cfg, mode, clock=clock)
        a = sim.Actor(E, make_face(OWNER), SCALE, np.random.RandomState(3 if mode == S.MODE_ENROL else 4), kind='live',
                      sway=12.0)
        per = collections.defaultdict(list)
        st = None
        for _ in range(6000):
            frame = sim.to_camera(a.frame(st))
            clock.tick()
            stage = ses.stage
            t = time.perf_counter()
            st = ses.process(frame)
            per[stage].append((time.perf_counter() - t) * 1000.0)
            if st.done:
                break
        r = ses.result
        if mode == S.MODE_ENROL and r is not None and r.outcome == S.OUT_ENROLLED:
            t = time.perf_counter()
            E.train(r.samples, save=False)
            res['build template from samples (ms)'] = round((time.perf_counter() - t) * 1000.0, 1)
        every = [v for vals in per.values() for v in vals]
        res[mode] = {'outcome': None if r is None else r.outcome, 'frames': len(every),
                     'simulated_seconds': round(clock.t, 1),
                     'stages': {k: {'frames': len(v), 'median_ms': round(float(np.median(v)), 2),
                                    'p95_ms': round(float(np.percentile(v, 95)), 2),
                                    'frames_per_second': round(1000.0 / float(np.mean(v)), 1)}
                                for k, v in per.items() if k in S.STAGES},
                     'all_stages_median_ms': round(float(np.median(every)), 2)}
    return res


def startup(exe=None):
    """Start-up costs: imports, engine, template load, key derivation, self-test of the source and the exe."""
    out = {}
    code = ('import time,sys; t=time.perf_counter(); import biolock.ui, biolock.controller; '
            'print(round((time.perf_counter()-t)*1000))')
    runs = []
    for _ in range(3):
        p = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        if p.returncode == 0:
            runs.append(int(p.stdout.strip().splitlines()[-1]))
    if runs:
        out['import all modules, fresh process (ms)'] = int(np.median(runs))
    cfg = load_config(path='', use_env=False)
    t = time.perf_counter()
    E = FaceEngine(cfg)
    out['create engine, load cascades (ms)'] = round((time.perf_counter() - t) * 1000.0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, '.biolock', 'face_model.npz')
        rng = np.random.RandomState(0)
        E2 = FaceEngine(cfg, path)
        E2.train([tan_triggs((rng.rand(100, 100) * 255).astype(np.uint8)) for _ in range(cfg['enrol_frames'])])
        out['template file (KB)'] = round(os.path.getsize(path) / 1024.0, 1)
        t = time.perf_counter()
        FaceEngine(cfg, path)
        out['create engine and load template (ms)'] = round((time.perf_counter() - t) * 1000.0, 1)
        key = os.path.join(tmp, '.biolock', 'vault.key')
        material = vault.key_material(E2.digest(), 'id', 1)
        t = time.perf_counter()
        vault.save_key(key, material)
        out['derive and store key, PBKDF2 600k (ms)'] = round((time.perf_counter() - t) * 1000.0)
        t = time.perf_counter()
        vault.verify_key(key, material)
        out['check key at unlock (ms)'] = round((time.perf_counter() - t) * 1000.0)
        for label, cmd in (('self-test from source (s)', [sys.executable, os.path.join(ROOT, 'main.py')]),
                           ('self-test of the packaged exe (s)', [exe] if exe and os.path.isfile(exe) else None)):
            if cmd is None:
                continue
            times, report = [], os.path.join(tmp, 'selftest.json')
            for _ in range(2):
                if os.path.exists(report):
                    os.remove(report)
                t = time.perf_counter()
                try:
                    p = subprocess.run(cmd + ['--selftest', report], cwd=tmp, capture_output=True, text=True,
                                       timeout=120)
                except subprocess.TimeoutExpired:       # e.g. an exe that does not know --selftest
                    times = []
                    break
                times.append(time.perf_counter() - t)
                if p.returncode != 0 or not os.path.exists(report):
                    times = []
                    break
            out[label] = round(min(times), 2) if times else 'failed'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--exe', default=None, help='a version-10 BioLock.exe to time as well (its --selftest)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(path='', use_env=False)
    base_rss = rss_mb()
    E = FaceEngine(cfg)
    step, n_gallery = steps(E, cfg)
    rss_enrolled = rss_mb()
    report = {'version': __version__, 'machine': machine(), 'steps': step, 'gallery_samples': n_gallery,
              'memory_mb': {'after imports': round(base_rss, 1), 'engine with an enrolled template': round(rss_enrolled, 1)},
              'sessions': session_cost(cfg), 'startup': startup(a.exe)}
    report['memory_mb']['after two full sessions'] = round(rss_mb(), 1)

    print('Machine:', report['machine'].get('cpu', report['machine']['processor']), '-', report['machine']['logical_cpus'],
          'logical CPUs,', report['machine'].get('ram_gb', '?'), 'GB RAM')
    print('\nStep                                               median ms    p95 ms')
    for k, v in step.items():
        print('  %-48s %8.2f  %8.2f' % (k, v['median_ms'], v['p95_ms']))
    for mode in (S.MODE_ENROL, S.MODE_VERIFY):
        s = report['sessions'][mode]
        print(f"\nSimulated {mode} session: {s['outcome']}, {s['frames']} frames")
        for k, v in s['stages'].items():
            print('  %-10s %5d frames   median %6.2f ms   p95 %6.2f ms   -> %5.1f frames/s' % (
                k, v['frames'], v['median_ms'], v['p95_ms'], v['frames_per_second']))
    print('\nMemory (MB):', report['memory_mb'])
    print('Start-up:')
    for k, v in report['startup'].items():
        print('  %-44s %s' % (k, v))
    path = os.path.join(a.out, 'perf.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
