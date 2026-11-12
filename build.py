#!/usr/bin/env python3
"""Build the packaged app and the release archives.

    python build.py             dist/BioLock.exe (one file), dist/BioLockFast/ (one folder, starts faster),
                                BioLock.zip, BioLock_Complete.zip
    python build.py --no-zip    executables only
    python build.py --zip-only  archives from the executables already in dist/ (after the documents changed)

Needs:  pip install -r requirements.txt -r requirements-dev.txt

Each executable is checked after it is built: `--selftest` must pass, and when the build machine's drive is
internal the app must show its "cannot run here" message and exit (that message appears on screen for a
moment). The outcome is written to dist/BUILD.json.
"""
import os
import sys
import json
import time
import shutil
import ctypes
import hashlib
import zipfile
import argparse
import datetime
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, 'dist')
WORK = os.path.join(ROOT, 'build')
EXCLUDE = ('matplotlib', 'scipy', 'pandas', 'IPython', 'pytest', 'sklearn', 'psutil.tests')
REFUSAL_TITLE = 'BioLock cannot run here'

README_FIRST = """BioLock {version} - complete package
====================================
BioLock only runs from an EXTERNAL drive (USB flash drive, external SSD/HDD, SD card). From an internal
drive it shows a warning and exits. Copy BioLock_Fast\\ (or BioLock.exe) to the external drive and run it there.

BioLock_Fast\\BioLockFast.exe  recommended: starts quickly (keep the whole folder together)
BioLock.exe                   single file; unpacks itself on every start, so it opens more slowly
Windows SmartScreen:          More info -> Run anyway (the program is not code-signed)

First use: Begin face enrolment, pick the camera that shows your face in colour, follow the prompts.
SecureVault is a hidden folder, not an encrypted one. A drive enrolled with version 9 has to be wiped and
enrolled again (move your files out with version 9 first).

source\\                       source code, tests, experiments and documents:
                              docs\\RESULTS.md          measured results and how they were obtained
                              docs\\TESTING_GUIDE.md    protocol for the live trials
                              docs\\PAPER_ALIGNMENT.md  where code and manuscript have to agree
Check an installation:        BioLock.exe --selftest report.json
Tests (from source\\):         python -m unittest discover -s tests -t .
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pyinstaller(name, onefile, log):
    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed', '--name', name,
           '--onefile' if onefile else '--onedir', '--collect-data', 'cv2', '--hidden-import', 'cv2.face',
           '--distpath', DIST, '--workpath', WORK, '--specpath', WORK]
    for mod in EXCLUDE:
        cmd += ['--exclude-module', mod]
    cmd.append(os.path.join(ROOT, 'main.py'))
    print(f'building {name} ({"one file" if onefile else "one folder"}) ...', flush=True)
    t = time.time()
    p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        sys.exit(f'PyInstaller failed for {name}; see build.log')
    return round(time.time() - t, 1)


def selftest(exe):
    """Run `exe --selftest`; returns (ok, seconds, report)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, 'selftest.json')
        t = time.time()
        try:
            p = subprocess.run([exe, '--selftest', report], cwd=tmp, timeout=240)
        except subprocess.TimeoutExpired:
            return False, 240.0, {'error': 'timed out'}
        secs = round(time.time() - t, 2)
        try:
            with open(report, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False, secs, {'error': f'no report, exit code {p.returncode}'}
        return bool(data.get('ok')) and p.returncode == 0 and bool(data.get('frozen')), secs, data


def guard_check(exe, wait=40.0):
    """Started from an internal drive, the app must show its refusal message and exit without doing anything."""
    sys.path.insert(0, ROOT)
    from biolock import winutil
    if os.name != 'nt':
        return 'skipped: not Windows'
    if winutil.is_external_drive(winutil.drive_letter(exe)):
        return 'skipped: this build folder is on an external drive'
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    lock_dir = os.path.join(os.path.dirname(exe), '.biolock')
    had_data = os.path.exists(lock_dir)
    p = subprocess.Popen([exe], cwd=os.path.dirname(exe))
    hwnd, deadline = None, time.time() + wait
    while time.time() < deadline and p.poll() is None:
        hwnd = user32.FindWindowW(None, REFUSAL_TITLE)
        if hwnd:
            break
        time.sleep(0.25)
    if not hwnd:
        p.kill()
        return 'FAILED: the refusal message did not appear'
    time.sleep(0.4)
    user32.PostMessageW(hwnd, 0x0010, 0, 0)                 # WM_CLOSE = pressing OK
    try:
        code = p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
        return 'FAILED: the app did not exit after the message'
    if not had_data and os.path.exists(lock_dir):
        return 'FAILED: the app wrote to the internal drive'
    return f'refused to run from the internal drive, exit code {code}, nothing written'


def source_files():
    """Files for the source folder of the complete archive: what git tracks or would track."""
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True).stdout.splitlines()
    skip = ('HANDOFF.md',)                                  # working notes, not part of the project
    return sorted(f for f in out if os.path.isfile(os.path.join(ROOT, f)) and f not in skip
                  and not f.startswith(('dist/', 'build/')))


def make_zips(version):
    single = os.path.join(ROOT, 'BioLock.zip')
    with zipfile.ZipFile(single, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(os.path.join(DIST, 'BioLock.exe'), 'BioLock.exe')
    full = os.path.join(ROOT, 'BioLock_Complete.zip')
    files = source_files()
    with zipfile.ZipFile(full, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(os.path.join(DIST, 'BioLock.exe'), 'BioLock_Complete/BioLock.exe')
        z.writestr('BioLock_Complete/README_FIRST.txt', README_FIRST.format(version=version))
        fast = os.path.join(DIST, 'BioLockFast')
        for folder, _dirs, names in os.walk(fast):
            for n in names:
                path = os.path.join(folder, n)
                z.write(path, 'BioLock_Complete/BioLock_Fast/' + os.path.relpath(path, fast).replace(os.sep, '/'))
        for f in files:
            z.write(os.path.join(ROOT, f), 'BioLock_Complete/source/' + f)
    return {os.path.basename(p): {'bytes': os.path.getsize(p), 'sha256': sha256(p)} for p in (single, full)}, len(files)


def main():
    ap = argparse.ArgumentParser(description='Build BioLock.exe, BioLockFast and the archives')
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--zip-only', action='store_true', help='only (re)make the archives from dist/')
    ap.add_argument('--no-guard-check', action='store_true', help='do not start the app to see its refusal message')
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from biolock import __version__
    import PyInstaller
    import cv2
    import numpy

    record = os.path.join(DIST, 'BUILD.json')
    if a.zip_only:
        try:
            with open(record, encoding='utf-8') as f:
                info = json.load(f)
        except (OSError, ValueError):
            sys.exit('dist/BUILD.json is missing: run a full build first')
        if info.get('version') != __version__:
            sys.exit(f"dist/ holds version {info.get('version')}, the source is {__version__}: run a full build")
        info['archives'], n = make_zips(__version__)
        info['archived'] = datetime.datetime.now().isoformat(timespec='seconds')
        with open(record, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
        print(f'archives written ({n} source files in the complete one)')
        return
    for folder in (DIST, WORK):
        shutil.rmtree(folder, ignore_errors=True)
    info = {'version': __version__, 'built': datetime.datetime.now().isoformat(timespec='seconds'),
            'python': sys.version.split()[0], 'pyinstaller': PyInstaller.__version__, 'opencv': cv2.__version__,
            'numpy': numpy.__version__, 'executables': {}}
    with open(os.path.join(ROOT, 'build.log'), 'w', encoding='utf-8') as log:
        times = {'BioLock': pyinstaller('BioLock', True, log), 'BioLockFast': pyinstaller('BioLockFast', False, log)}
    failed = False
    for name, exe in (('BioLock', os.path.join(DIST, 'BioLock.exe')),
                      ('BioLockFast', os.path.join(DIST, 'BioLockFast', 'BioLockFast.exe'))):
        ok, secs, report = selftest(exe)
        entry = {'path': os.path.relpath(exe, ROOT).replace(os.sep, '/'), 'bytes': os.path.getsize(exe),
                 'sha256': sha256(exe), 'build_seconds': times[name], 'selftest_ok': ok, 'selftest_seconds': secs,
                 'selftest_checks': [f"{c['name']}: {'ok' if c['ok'] else 'FAILED ' + str(c['detail'])}"
                                     for c in report.get('checks', [])] or report}
        if name == 'BioLockFast':
            size = sum(os.path.getsize(os.path.join(d, n)) for d, _s, ns in os.walk(os.path.dirname(exe)) for n in ns)
            entry['folder_bytes'] = size
        if not a.no_guard_check:
            entry['internal_drive_check'] = guard_check(exe)
            failed |= entry['internal_drive_check'].startswith('FAILED')
        failed |= not ok
        info['executables'][name] = entry
        print(f"  {name}: self-test {'passed' if ok else 'FAILED'} in {secs} s; "
              f"{entry.get('internal_drive_check', 'drive check not run')}", flush=True)
    shutil.copy(os.path.join(ROOT, 'autorun.inf'), os.path.join(DIST, 'autorun.inf'))
    if failed:
        with open(os.path.join(DIST, 'BUILD_FAILED.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
        sys.exit('The build did not pass its checks; see dist/BUILD_FAILED.json. No archives were made.')
    if not a.no_zip:
        info['archives'], n = make_zips(__version__)
        print(f'  archives written ({n} source files in the complete one)')
    with open(record, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    print('done: dist/BUILD.json')


if __name__ == '__main__':
    main()
