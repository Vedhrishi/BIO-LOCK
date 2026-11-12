"""`BioLock.exe --selftest [report.json]`: does this installation / packaged build work?

Runs every part that does not need a camera or a person: the libraries load, the Haar
cascades are found, a template can be built, saved, re-loaded and matched, the vault-key
check works, a session processes frames, and a window can be created. All files go to a
temporary folder that is removed afterwards; the drive BioLock runs from is not touched,
which is why this is allowed on an internal drive.
"""

import os
import sys
import json
import time
import tempfile
import platform


def run_selftest(report_path=None):
    t0 = time.perf_counter()
    report = {'ok': False, 'frozen': bool(getattr(sys, 'frozen', False)), 'python': sys.version.split()[0],
              'platform': platform.platform(), 'checks': []}

    def check(name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            report['checks'].append({'name': name, 'ok': True, 'ms': round((time.perf_counter() - t) * 1000, 1),
                                     'detail': detail})
            return True
        except Exception as e:
            report['checks'].append({'name': name, 'ok': False, 'ms': round((time.perf_counter() - t) * 1000, 1),
                                     'detail': f'{type(e).__name__}: {e}'})
            return False

    state = {}

    def libraries():
        import numpy
        import cv2
        import PIL
        import cryptography
        from biolock import __version__
        state.update(np=numpy, cv2=cv2)
        report['version'] = __version__
        return {'numpy': numpy.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__,
                'cryptography': cryptography.__version__, 'lbph': hasattr(cv2, 'face')}

    def engine():
        from biolock.config import load_config
        from biolock.engine import FaceEngine
        cfg = load_config(path='', use_env=False)
        state['cfg'] = cfg
        state['engine'] = FaceEngine(cfg, os.path.join(state['tmp'], '.biolock', 'face_model.npz'))
        return {'cascades': 'loaded', 'lbph': [cfg['lbph_radius'], cfg['lbph_neighbors'], cfg['lbph_grid']],
                'threshold': cfg['min_conf']}

    def template():
        np, E = state['np'], state['engine']
        from biolock.engine import tan_triggs, FaceEngine
        rng = np.random.RandomState(1)
        base = (rng.rand(100, 100) * 255).astype(np.uint8)
        rois = [tan_triggs(np.clip(base.astype(int) + rng.randint(-6, 7, base.shape), 0, 255).astype(np.uint8))
                for _ in range(24)]
        kept = E.train(rois)
        again = FaceEngine(state['cfg'], E.model_path)
        if not again.trained or again.digest() != E.digest():
            raise RuntimeError('the saved template did not load back identically')
        own = again.score(rois[0])[0]
        other = again.score(tan_triggs((rng.rand(100, 100) * 255).astype(np.uint8)))[0]
        if not own > other:
            raise RuntimeError(f'an enrolled sample scored {own:.1f}, a random image {other:.1f}')
        state['rois'] = rois
        return {'samples_kept': kept, 'enrolled_sample_score': round(own, 1), 'random_image_score': round(other, 1)}

    def key():
        from biolock import vault
        path = os.path.join(state['tmp'], '.biolock', 'vault.key')
        material = vault.key_material(state['engine'].digest(), 'drive-id', 1234)
        t = time.perf_counter()
        vault.save_key(path, material)
        ms = (time.perf_counter() - t) * 1000
        if not vault.verify_key(path, material):
            raise RuntimeError('the key did not verify')
        if vault.verify_key(path, vault.key_material(state['engine'].digest(), 'another-drive', 1234)):
            raise RuntimeError('the key verified for another drive')
        return {'pbkdf2_ms': round(ms)}

    def session():
        np = state['np']
        from biolock import session as S
        ses = S.Session(state['engine'], state['cfg'], S.MODE_VERIFY)
        rng = np.random.RandomState(2)
        st = None
        for _ in range(6):
            st = ses.process((rng.rand(480, 640, 3) * 255).astype(np.uint8))
        return {'frame': list(st.frame.shape), 'stage': st.stage, 'faces': len(st.boxes)}

    def storage():
        from biolock.storage import Store
        from biolock import winutil
        store = Store(state['tmp'])
        store.save_state({'enrolled': False})
        store.log('SELFTEST')
        letter = winutil.drive_letter(sys.executable)
        return {'writable': store.writable(), 'state': store.load_state().get('enrolled'),
                'run_drive': letter, 'run_drive_external': winutil.is_external_drive(letter)}

    def window():
        # builds the real main window (hidden) on an empty temporary "drive"
        import tkinter as tk
        from PIL import Image, ImageTk
        from biolock import ui, controller
        root = tk.Tk()
        root.withdraw()
        app = None
        try:
            img = ImageTk.PhotoImage(Image.fromarray(state['np'].zeros((48, 64, 3), 'uint8')))
            ctl = controller.Controller(os.path.join(state['tmp'], 'drive'), state['cfg'])
            app = ui.App(root, ctl, scan=lambda: [])
            root.update()
            if ctl.condition() != controller.NEW or app.b_main.cget('text') != 'Begin face enrolment':
                raise RuntimeError('the main window did not come up in its first-run state')
            return {'tk': root.tk.call('info', 'patchlevel'), 'image': [img.width(), img.height()],
                    'dpi_scale': round(root.winfo_fpixels('1i') / 96.0, 2), 'main_window': 'built'}
        finally:
            if app is not None:
                app.shutdown()
            else:
                root.destroy()

    with tempfile.TemporaryDirectory(prefix='biolock_selftest_') as tmp:
        state['tmp'] = tmp
        ok = check('libraries', libraries)
        ok = ok and check('face engine and cascades', engine)
        ok = ok and check('template build / save / load / match', template)
        ok = check('vault key', key) and ok if ok else False
        ok = check('session processes frames', session) and ok if ok else False
        ok = check('storage', storage) and ok if ok else False
        ok = check('window toolkit', window) and ok
        engine_obj = state.get('engine')
        if engine_obj is not None:
            engine_obj.forget()
    report['ok'] = bool(ok)
    report['seconds'] = round(time.perf_counter() - t0, 2)
    text = json.dumps(report, indent=2)
    if report_path:
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except OSError:
            pass
    try:
        print(text)
    except Exception:                       # a windowed exe has no console
        pass
    return 0 if ok else 1
