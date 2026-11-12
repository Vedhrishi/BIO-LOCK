"""The window. It draws what the Controller and the camera worker report and forwards button
presses; no image processing, key derivation or drive access runs on the Tk thread.
"""

import os
import sys
import queue
import threading
import subprocess

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from . import __version__
from . import winutil
from . import session as S
from . import controller as C
from .camera import CameraWorker, scan_cameras

# ---- theme ------------------------------------------------------------------
BG = '#060612'
SURFACE = '#0c0c20'
CARD = '#10102a'
ACCENT = '#6d28d9'
ACCENT_L = '#8b5cf6'
TEAL_L = '#22d3ee'
GREEN = '#059669'
GREEN_L = '#34d399'
RED = '#dc2626'
RED_L = '#f87171'
AMBER = '#d97706'
AMBER_L = '#fbbf24'
FG = '#f5f5ff'
MUTED = '#9494b8'
FAINT = '#55557a'
FONT = 'Segoe UI'
MONO = 'Consolas'

TRIAL_LABELS = ['genuine', 'impostor', 'attack_photo', 'attack_phone', 'attack_replay']
LIGHTING_TAGS = ['', 'bright', 'normal', 'dim', 'backlit']
MOVEMENT_TAGS = ['', 'still', 'head_turn', 'walking', 'shaky_hand']

STEP_NAMES = {S.MODE_VERIFY: ('Presence', 'Blink', 'Head left, right', 'Face match'),
              S.MODE_ENROL: ('Presence', 'Blink', 'Head left, right', 'Face capture')}
STAGE_TITLES = {S.ST_PRESENCE: 'Step 1 of 4:  presence', S.ST_BLINK: 'Step 2 of 4:  blink',
                S.ST_HEAD: 'Step 3 of 4:  head movement', S.ST_FACE: 'Step 4 of 4:  face', S.ST_DONE: 'Done'}

SOUND_OK = ((880, 80), (1108, 80), (1318, 150))
SOUND_FAIL = ((400, 200), (300, 300))
SOUND_STEP = ((1200, 40),)


def _bgr(hex_colour):
    h = hex_colour.lstrip('#')
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def enforce_external_drive(base, show=True):
    """BioLock refuses to run from an internal drive.

    Always enforced for the packaged exe. From source the check is off, so the code can be
    developed on an internal disk; set BIOLOCK_ENFORCE_EXTERNAL=1 to switch it on there."""
    if not (getattr(sys, 'frozen', False) or os.environ.get('BIOLOCK_ENFORCE_EXTERNAL') == '1'):
        return True
    letter = winutil.drive_letter(base)
    if winutil.is_external_drive(letter):
        return True
    if show:
        try:
            messagebox.showerror(
                'BioLock cannot run here',
                f'BioLock is running from drive {letter}:, which is an internal drive.\n\n'
                'This cannot be used to lock an internal drive, as it would endanger the '
                'storage components of the system.\n\n'
                'Copy BioLock to an external drive (USB flash drive, external SSD/HDD or SD card) '
                'and run it from there.')
        except Exception:
            pass
    return False


# --------------------------------------------------------------------------
# Drawing on the camera frame (runs on the camera worker thread)
# --------------------------------------------------------------------------
class Painter:
    """Turns a session Status into the RGB image shown in the window."""

    def __init__(self):
        self.target = (640, 480)        # size of the video area, set by the window
        self._flash = 0

    def flash(self, frames=8):
        self._flash = frames

    @staticmethod
    def _corners(img, box, colour, thick):
        x, y, w, h = box
        n = max(10, w // 5)
        for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(img, (px, py), (px + dx * n, py), colour, thick, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * n), colour, thick, cv2.LINE_AA)

    def __call__(self, status):
        img = status.frame.copy()
        for b in status.boxes:
            if b != status.box:
                cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (110, 110, 110), 1, cv2.LINE_AA)
        if status.box is not None:
            if status.result is not None:
                colour = GREEN_L if status.result.outcome in (S.OUT_ACCEPT, S.OUT_ENROLLED) else RED_L
            elif status.multi:
                colour = RED_L
            else:
                colour = ACCENT_L if status.stage == S.ST_FACE else AMBER_L
            self._corners(img, status.box, _bgr(colour), 2)
        if status.result is not None and status.result.outcome in (S.OUT_ACCEPT, S.OUT_ENROLLED):
            self._flash = max(self._flash, 1)
        if self._flash > 0:
            self._flash -= 1
            tint = img.copy()
            tint[:] = (94, 197, 34)
            cv2.addWeighted(tint, 0.22, img, 0.78, 0, img)
        tw, th = self.target
        h, w = img.shape[:2]
        k = min(tw / float(w), th / float(h))
        if k > 0 and abs(k - 1.0) > 0.02:
            img = cv2.resize(img, (max(2, int(w * k)), max(2, int(h * k))),
                             interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Small widgets
# --------------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    """A frame whose content scrolls vertically when the window is too short for it
    (1366x768 laptops, large display scaling)."""

    def __init__(self, parent, width, bg):
        super().__init__(parent, bg=bg, width=width)
        self.grid_propagate(False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, width=10)
        self.bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.body = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.body.bind('<Configure>', self._resize)
        self.canvas.bind('<Configure>', self._resize)
        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._wheel))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        self._bar_shown = False

    def _on_scroll(self, lo, hi):
        self.bar.set(lo, hi)
        need = float(lo) > 0.0 or float(hi) < 1.0
        if need != self._bar_shown:
            self._bar_shown = need
            if need:
                self.bar.grid(row=0, column=1, sticky='ns')     # the content gets narrower, never hidden
            else:
                self.bar.grid_remove()

    def _resize(self, _event=None):
        self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _wheel(self, event):
        if self._bar_shown:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    def overflow(self):
        """Pixels of content that do not fit (0 when everything is visible)."""
        self.update_idletasks()
        return max(0, self.body.winfo_reqheight() - self.canvas.winfo_height())


class CamPicker(tk.Toplevel):
    """Modal camera chooser. The scan runs on a thread; `result` is a CameraInfo or None."""

    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.result = None
        px = app.px
        self.title('Select camera')
        self.configure(bg=BG)
        self.geometry(f'{px(440)}x{px(320)}')
        self.resizable(False, False)
        self.transient(app.root)
        tk.Label(self, text='Choose a camera', font=(FONT, 13, 'bold'), fg=FG, bg=BG).pack(pady=(px(18), px(8)))
        self.status = tk.Label(self, text='Looking for cameras...', fg=MUTED, bg=BG, font=(FONT, 10))
        self.status.pack(pady=px(6))
        self.list = tk.Frame(self, bg=BG)
        self.list.pack(fill='both', expand=True, padx=px(20))
        tk.Button(self, text='Cancel', bg=BG, fg=MUTED, relief='flat', bd=0, command=self.destroy,
                  activebackground=BG, activeforeground=FG).pack(pady=(px(8), px(12)))
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        app.background(app.scan, self._show)
        self.grab_set()

    def _show(self, cams):
        if not self.winfo_exists():
            return
        px = self.app.px
        if not cams:
            self.status.config(text='No working camera was found.\nConnect one, close other apps that use it, '
                                    'and try again.', fg=RED_L)
            return
        self.status.config(text='Pick the camera that shows your face in colour.')
        for info in cams:
            tk.Button(self.list, text=info.describe(), bg=CARD, fg=FG, relief='flat', bd=0, pady=px(7),
                      activebackground=ACCENT, activeforeground='#fff', anchor='w', padx=px(10),
                      command=lambda i=info: self._choose(i)).pack(fill='x', pady=px(3))

    def _choose(self, info):
        self.result = info
        self.destroy()


class Splash:

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=ACCENT)
        k = root.winfo_fpixels('1i') / 96.0
        w, h = int(460 * k), int(250 * k)
        self.win.geometry(f'{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 2}')
        inner = tk.Frame(self.win, bg=BG)
        inner.pack(fill='both', expand=True, padx=2, pady=2)
        tk.Label(inner, text='BioLock SSD', font=(FONT, 24, 'bold'), fg=FG, bg=BG).pack(pady=(int(46 * k), 2))
        tk.Label(inner, text=f'Version {__version__}', font=(FONT, 11), fg=MUTED, bg=BG).pack()
        self.msg = tk.Label(inner, text='Starting...', font=(FONT, 10), fg=MUTED, bg=BG)
        self.msg.pack(pady=int(22 * k))
        tk.Label(inner, text='Pasupuleti Vedh Rishi', font=(FONT, 8), fg=FAINT, bg=BG).pack(side='bottom', pady=8)
        self.win.update()

    def say(self, text):
        try:
            self.msg.config(text=text)
            self.win.update()
        except tk.TclError:
            pass

    def close(self):
        try:
            self.win.destroy()
        except tk.TclError:
            pass


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class App:
    POLL_MS = 30

    def __init__(self, root, controller, open_source=None, scan=scan_cameras):
        self.root = root
        self.ctl = controller
        self.open_source = open_source      # tests: callable returning a fake camera
        self.scan = scan
        self.k = root.winfo_fpixels('1i') / 96.0
        self.worker = None
        self.painter = Painter()
        self._seq = 0
        self._photo = None
        self._seen = {}
        self._tasks = queue.Queue()
        self._after = {}                    # pending timer ids, cancelled when the window closes
        self._closing = False
        self._full = False
        self.last_result = None

        cfg = controller.cfg
        self.v_trial = tk.StringVar(value=controller.trial_label)
        self.v_light = tk.StringVar(value='')
        self.v_move = tk.StringVar(value='')
        self.v_testing = tk.BooleanVar(value=False)
        self.v_nolock = tk.BooleanVar(value=False)
        self.v_auto = tk.BooleanVar(value=False)
        self.v_trials = tk.StringVar(value='Trials logged: 0')
        self.v_signals = {k: tk.StringVar(value='-') for k in ('Motion', 'Blinks', 'Head', 'Match score')}
        self.v_attempts = tk.StringVar(value='')

        root.title(f'BioLock SSD  v{__version__}')
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(self.px(1080), sw - 60), min(self.px(700), sh - 110)
        root.geometry(f'{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}')
        root.minsize(min(self.px(860), sw - 60), min(self.px(520), sh - 110))
        self._style()
        self._build(cfg)
        root.bind('<F11>', lambda e: self._fullscreen())
        root.bind('<Escape>', lambda e: self._escape())
        root.protocol('WM_DELETE_WINDOW', self.quit)
        for seq in ('<Any-KeyPress>', '<Any-Button>'):
            root.bind_all(seq, lambda e: self.ctl.touch(), add='+')
        self.refresh()
        self._pump()
        self._timers()
        self._drive_usage()

    def px(self, n):
        """Scale a 96-dpi pixel size to this display."""
        return max(1, int(round(n * self.k)))

    # ---- plumbing ---------------------------------------------------------------
    def _later(self, key, ms, fn):
        """Schedule fn once; one pending timer per key."""
        self._after[key] = self.root.after(ms, fn)

    def background(self, fn, done):
        """Run fn() on a thread and hand its result to done() on the Tk thread."""
        def work():
            try:
                res = fn()
            except Exception:
                res = None
            self._tasks.put((done, res))
        threading.Thread(target=work, daemon=True).start()

    def _pump(self):
        try:
            while True:
                done, res = self._tasks.get_nowait()
                try:
                    done(res)
                except tk.TclError:
                    pass
        except queue.Empty:
            pass
        if not self._closing:
            self._later('pump', 150, self._pump)

    def _timers(self):
        if self._closing:
            return
        try:
            left = self.ctl.auto_lock_in()
            if left is None:
                self.timer.config(text='')
            elif left <= 0:
                self.ctl.lock('idle')
                winutil.beep_sequence(SOUND_FAIL)
                self.refresh()
            elif left <= 60:
                self.timer.config(text=f'Auto-lock in {int(left)} s', fg=AMBER_L)
            else:
                self.timer.config(text=f'Auto-lock in {int(left // 60) + 1} min', fg=FAINT)
            cool = self.ctl.cooldown_left()
            if cool > 0 and self.worker is None and self.ctl.condition() == C.LOCKED:
                self.sub.config(text=f'Too many incomplete liveness checks. Wait {int(cool) + 1} s.')
            elif self._cooling and self.worker is None:
                self.refresh()
            self._cooling = cool > 0
        except tk.TclError:
            return
        self._later('timers', 1000, self._timers)

    _cooling = False

    def _drive_usage(self):
        if self._closing:
            return
        store = self.ctl.store

        def measure():
            free, total = winutil.disk_usage_gb(store.letter)
            return free, total, store.vault_size_mb()

        def show(res):
            if res:
                free, total, vault_mb = res
                self.drive_info.config(text=f'{store.letter}:  {free:.1f} GB free of {total:.1f} GB')
                self.drive_bar['value'] = max(0, min(100, int((1 - free / max(total, 0.001)) * 100)))
        self.background(measure, show)
        self._later('drive', 30000, self._drive_usage)

    # ---- layout -------------------------------------------------------------------
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use('default')
        except tk.TclError:
            pass
        st.configure('B.Horizontal.TProgressbar', troughcolor=CARD, background=ACCENT_L, borderwidth=0,
                     thickness=self.px(6))
        st.configure('D.Horizontal.TProgressbar', troughcolor=CARD, background=TEAL_L, borderwidth=0,
                     thickness=self.px(4))
        st.configure('Vertical.TScrollbar', background=FAINT, troughcolor=SURFACE, bordercolor=SURFACE,
                     arrowcolor=FG, relief='flat', width=self.px(12), arrowsize=self.px(12))
        st.map('Vertical.TScrollbar', background=[('active', MUTED)])
        st.configure('TCombobox', fieldbackground=CARD, background=CARD, foreground=FG, arrowcolor=FG,
                     arrowsize=self.px(12))
        st.map('TCombobox', fieldbackground=[('readonly', CARD)], foreground=[('readonly', FG)])
        self.root.option_add('*TCombobox*Listbox.background', CARD)
        self.root.option_add('*TCombobox*Listbox.foreground', FG)

    def _button(self, parent, text, command, bg=CARD, fg=FG, bold=False):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, relief='flat', bd=0,
                         pady=self.px(8), font=(FONT, 10, 'bold' if bold else 'normal'),
                         activebackground=ACCENT_L, activeforeground='#fff', cursor='hand2')

    def _build(self, cfg):
        px = self.px
        tk.Frame(self.root, bg=ACCENT, height=px(3)).pack(fill='x', side='top')
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True)

        self.side = ScrollFrame(body, px(320), SURFACE)
        self.side.pack(side='left', fill='y')
        side = self.side.body
        pad = px(18)

        self.rule = tk.Frame(side, bg=RED, height=px(4))
        self.rule.pack(fill='x')
        self.head = tk.Label(side, text='', font=(FONT, 19, 'bold'), fg=FG, bg=SURFACE, justify='left')
        self.head.pack(anchor='w', padx=pad, pady=(px(14), px(2)))
        self.sub = tk.Label(side, text='', font=(FONT, 9), fg=MUTED, bg=SURFACE, wraplength=px(264),
                            justify='left')
        self.sub.pack(anchor='w', padx=pad)
        self.timer = tk.Label(side, text='', font=(FONT, 9), fg=FAINT, bg=SURFACE)
        self.timer.pack(anchor='w', padx=pad, pady=(px(2), 0))

        self.buttons = tk.Frame(side, bg=SURFACE)
        self.buttons.pack(fill='x', padx=pad, pady=(px(10), px(2)))
        self.b_main = self._button(self.buttons, '', self.primary, ACCENT, '#fff', bold=True)
        self.b_stop = self._button(self.buttons, 'Cancel', self.cancel)
        self.b_lock = self._button(self.buttons, 'Lock vault', self.lock)
        self.b_cam = self._button(self.buttons, 'Change camera', self.pick_camera, fg=MUTED)
        self.b_log = self._button(self.buttons, 'View audit log', self.view_log, fg=MUTED)
        self.b_wipe = self._button(self.buttons, 'Wipe & reset', self.wipe_prompt, fg=RED_L)

        bar = tk.Frame(side, bg=SURFACE)
        bar.pack(fill='x', padx=pad, pady=(px(8), 0))
        self.progress = ttk.Progressbar(bar, style='B.Horizontal.TProgressbar', mode='determinate', maximum=100)
        self.progress.pack(fill='x', side='left', expand=True)
        self.progress_pct = tk.Label(bar, text='0%', font=(FONT, 8), fg=MUTED, bg=SURFACE, width=4)
        self.progress_pct.pack(side='left', padx=(px(6), 0))

        steps = tk.Frame(side, bg=SURFACE)
        steps.pack(fill='x', padx=pad, pady=(px(8), px(2)))
        self.steps = []
        for _ in range(4):
            row = tk.Frame(steps, bg=SURFACE)
            row.pack(fill='x', pady=1)
            icon = tk.Label(row, text='o', width=3, fg=FAINT, bg=SURFACE, font=(MONO, 9))
            icon.pack(side='left')
            name = tk.Label(row, text='', fg=MUTED, bg=SURFACE, font=(FONT, 9))
            name.pack(side='left')
            self.steps.append((icon, name))

        sig = tk.LabelFrame(side, text='Live signals', bg=SURFACE, fg=MUTED, font=(FONT, 8), bd=1, relief='groove')
        sig.pack(fill='x', padx=pad, pady=(px(8), px(2)))
        for label, var in self.v_signals.items():
            row = tk.Frame(sig, bg=SURFACE)
            row.pack(fill='x', padx=px(6))
            tk.Label(row, text=label, fg=MUTED, bg=SURFACE, font=(FONT, 8), width=11, anchor='w').pack(side='left')
            tk.Label(row, textvariable=var, fg=FG, bg=SURFACE, font=(MONO, 8)).pack(side='left')

        self.attempts = tk.Label(side, textvariable=self.v_attempts, fg=ACCENT_L, bg=SURFACE, font=(FONT, 8))
        self.attempts.pack(anchor='w', padx=pad, pady=(px(6), 0))
        tk.Label(side, text=self._score_note(cfg), fg=FAINT, bg=SURFACE, font=(FONT, 7), justify='left',
                 wraplength=px(264)).pack(anchor='w', padx=pad, pady=(px(2), 0))

        # ---- Testing mode (collapsed by default; labels only, never used for a decision)
        self.b_test = tk.Checkbutton(side, text='Testing mode (label attempts for experiments)',
                                     variable=self.v_testing, command=self._testing_toggle, bg=SURFACE, fg=TEAL_L,
                                     selectcolor=CARD, activebackground=SURFACE, activeforeground=TEAL_L,
                                     font=(FONT, 8), anchor='w')
        self.b_test.pack(fill='x', padx=px(14), pady=(px(8), 0))
        self.test = tk.Frame(side, bg=SURFACE)
        for text, var, values in (('Trial label', self.v_trial, TRIAL_LABELS),
                                  ('Lighting condition', self.v_light, LIGHTING_TAGS),
                                  ('Movement condition', self.v_move, MOVEMENT_TAGS)):
            tk.Label(self.test, text=text, fg=MUTED, bg=SURFACE, font=(FONT, 8), anchor='w').pack(fill='x')
            box = ttk.Combobox(self.test, textvariable=var, values=values, state='readonly', font=(FONT, 8))
            box.pack(fill='x', pady=(0, px(3)))
            box.bind('<<ComboboxSelected>>', lambda e: self._tags())
        tk.Label(self.test, textvariable=self.v_trials, fg=TEAL_L, bg=SURFACE, font=(FONT, 8), anchor='w',
                 justify='left', wraplength=px(264)).pack(fill='x')
        self.b_nolock = tk.Checkbutton(self.test, text='Do not count failed matches (no lock-out / wipe)',
                                       variable=self.v_nolock, command=self._tags, bg=SURFACE, fg=MUTED,
                                       selectcolor=CARD, activebackground=SURFACE, activeforeground=FG,
                                       font=(FONT, 8), anchor='w', disabledforeground=FAINT)
        self.b_nolock.pack(fill='x')
        tk.Label(self.test, text='Can only be switched on while the vault is unlocked. Set the labels before '
                                 'each attempt; they are written to logs/attempts.csv.',
                 fg=FAINT, bg=SURFACE, font=(FONT, 7), justify='left', wraplength=px(264)).pack(fill='x')

        self.foot = tk.Frame(side, bg=SURFACE)
        self.foot.pack(fill='x', padx=pad, pady=(px(10), px(10)))
        self.drive_info = tk.Label(self.foot, text='', fg=MUTED, bg=SURFACE, font=(FONT, 8))
        self.drive_info.pack(anchor='w')
        self.drive_bar = ttk.Progressbar(self.foot, style='D.Horizontal.TProgressbar', mode='determinate',
                                         maximum=100)
        self.drive_bar.pack(fill='x', pady=(px(2), px(6)))
        exe = winutil.app_executable()
        if exe:                                         # packaged build only; never a silent registry write
            self.v_auto.set(winutil.autostart_enabled_for(exe))
            tk.Checkbutton(self.foot, text='Start BioLock when I sign in to Windows', variable=self.v_auto,
                           command=self._autostart, bg=SURFACE, fg=MUTED, selectcolor=CARD,
                           activebackground=SURFACE, activeforeground=FG, font=(FONT, 8),
                           anchor='w').pack(fill='x')
        tk.Button(self.foot, text='About', bg=SURFACE, fg=FAINT, relief='flat', bd=0, font=(FONT, 8),
                  activebackground=SURFACE, activeforeground=FG, command=self.about).pack(anchor='w')

        # ---- right: camera view
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)
        self.title = tk.Label(right, text='', font=(FONT, 12, 'bold'), fg=FG, bg=BG, anchor='w')
        self.title.pack(fill='x', padx=px(20), pady=(px(12), px(4)))
        self.prompt = tk.Label(right, text='', font=(FONT, 15, 'bold'), fg=FG, bg=BG, wraplength=px(680),
                               justify='center')
        self.prompt.pack(side='bottom', fill='x', padx=px(20), pady=(px(6), px(14)))
        self.stage = tk.Frame(right, bg='#000000')
        self.stage.pack(fill='both', expand=True, padx=px(20))
        self.placeholder = tk.Label(self.stage, text='', font=(FONT, 11), fg=MUTED, bg='#000000', justify='left')
        self.placeholder.place(relx=0.5, rely=0.5, anchor='center')
        self.video = tk.Label(self.stage, bg='#000000', bd=0)
        self.stage.bind('<Configure>', self._stage_size)

    @staticmethod
    def _score_note(cfg):
        w = (cfg['w_lbph'], cfg['w_hist'], cfg['w_template'])
        mix = 'LBPH only' if w == (1.0, 0.0, 0.0) else 'LBPH %g, histogram %g, template %g' % w
        return (f'Match score = similarity on a 0-100 scale ({mix}); the median of {cfg["match_frames"]} frames '
                f'must reach {cfg["min_conf"]:g}. It is a score, not a probability.')

    def _stage_size(self, event):
        self.painter.target = (max(160, event.width), max(120, event.height))
        self.prompt.config(wraplength=max(200, event.width))

    # ---- views ------------------------------------------------------------------------
    def _show_text(self, text):
        self.video.place_forget()
        self.placeholder.config(text=text)
        self.placeholder.place(relx=0.5, rely=0.5, anchor='center')

    def _show_video(self):
        self.placeholder.place_forget()
        self.video.place(relx=0.5, rely=0.5, anchor='center')

    def _set_steps(self, mode, done, active):
        for i, ((icon, name), text) in enumerate(zip(self.steps, STEP_NAMES[mode])):
            if i == 1:
                text = f'{text} x{self.ctl.cfg["blinks_needed"]}'
            if i < done:
                icon.config(text='OK', fg=GREEN_L)
                name.config(text=text, fg=GREEN_L)
            elif i == active:
                icon.config(text='>>', fg=ACCENT_L)
                name.config(text=text, fg=FG)
            else:
                icon.config(text='o', fg=FAINT)
                name.config(text=text, fg=MUTED)

    def _set_progress(self, fraction):
        v = int(round(100 * max(0.0, min(1.0, fraction))))
        if self._seen.get('progress') != v:
            self._seen['progress'] = v
            self.progress['value'] = v
            self.progress_pct.config(text=f'{v}%')

    def _pack_buttons(self, *buttons):
        for b in (self.b_main, self.b_stop, self.b_lock, self.b_cam, self.b_log, self.b_wipe):
            b.pack_forget()
        for b in buttons:
            b.pack(fill='x', pady=self.px(3))

    def refresh(self):
        """Redraw everything that depends on the state of the drive (not called per frame)."""
        ctl, cfg = self.ctl, self.ctl.cfg
        cond = ctl.condition()
        letter = ctl.store.letter
        self._set_progress(0)
        for var in self.v_signals.values():
            var.set('-')
        n = ctl.attempts
        self.v_attempts.set(f'Failed face matches: {n} of {cfg["max_attempts"]}' if cond in (C.LOCKED, C.UNLOCKED)
                            else '')
        self.attempts.config(fg=RED_L if n >= 3 else AMBER_L if n else ACCENT_L)
        self.b_nolock.config(state='normal' if (ctl.unlocked or ctl.no_lockout) else 'disabled')
        self.title.config(text='')
        self.prompt.config(text='', fg=FG)
        blinks = cfg['blinks_needed']
        steps = (f'  1.  Presence: move your head gently from side to side\n'
                 f'  2.  Hold still and blink {blinks} times, slowly\n'
                 f'  3.  Move your head to the left, then to the right\n')

        if cond == C.UNLOCKED:
            self.head.config(text='Vault\nopen')
            self.rule.config(bg=GREEN)
            self.sub.config(text=f'Session active  -  {ctl.state.get("total_unlocks", 0)} unlock(s) so far')
            self._set_steps(S.MODE_VERIFY, 4, -1)
            self._set_progress(1)
            self.b_main.config(text='Open SecureVault', bg=GREEN, fg='#fff')
            self._pack_buttons(self.b_main, self.b_lock, self.b_log)
            self._show_text('The vault is unlocked.\n\n'
                            'Open SecureVault shows your files in Explorer.\n'
                            'Lock vault hides them again when you have finished.\n\n'
                            f'It locks itself after {cfg["auto_lock_mins"]:g} minutes without activity here.')
        elif cond == C.NEW:
            self.head.config(text='Welcome\nto BioLock')
            self.rule.config(bg=AMBER)
            self.sub.config(text=f'Drive {letter}:  is ready. Enrol your face once to protect it.')
            self._set_steps(S.MODE_ENROL, 0, -1)
            self.b_main.config(text='Begin face enrolment', bg=ACCENT, fg='#fff')
            self._pack_buttons(self.b_main, self.b_cam)
            self._show_text(f'One-time face enrolment on drive {letter}:\n\n' + steps +
                            f'  4.  Follow the prompts while {cfg["enrol_frames"]} samples are captured\n\n'
                            'Afterwards:\n'
                            '  -  only the enrolled face opens the SecureVault folder\n'
                            f'  -  {cfg["max_attempts"]} failed face matches erase the enrolment and the vault\n'
                            f'  -  the vault locks itself after {cfg["auto_lock_mins"]:g} minutes without activity')
        elif cond == C.LOCKED:
            self.head.config(text='Drive\nlocked')
            self.rule.config(bg=RED)
            self.sub.config(text=f'4-step verification  -  {ctl.attempts_left} attempt(s) remaining')
            self._set_steps(S.MODE_VERIFY, 0, -1)
            self.b_main.config(text='Unlock with face ID', bg=ACCENT, fg='#fff')
            self._pack_buttons(self.b_main, self.b_cam, self.b_log, self.b_wipe)
            self._show_text('The drive is locked.\n\n' + steps +
                            f'  4.  Hold still while your face is matched  (score {cfg["min_conf"]:g} or more)\n\n'
                            f'{ctl.attempts_left} failed face match(es) left before the enrolment and the vault\n'
                            'are erased.')
        else:
            text = {
                C.LEGACY: ('Older\nenrolment',
                           'This drive was enrolled with an earlier BioLock version. That enrolment cannot be '
                           'verified by this version.',
                           'To keep the files in SecureVault, unlock the drive with the old version first and\n'
                           'move them somewhere else. Then use  Wipe & reset  here and enrol again.'),
                C.DAMAGED: ('Enrolment\ndamaged',
                            'The enrolment files on this drive are incomplete or unreadable.',
                            'The drive cannot be unlocked with these files.\n'
                            'Wipe & reset  erases them (and SecureVault) so that you can enrol again.'),
                C.ORPHAN: ('Vault without\nenrolment',
                           'This drive has a SecureVault folder but no enrolment.',
                           'A new face cannot be enrolled over an existing vault.\n'
                           'Wipe & reset  erases the SecureVault folder so that you can start again.'),
            }[cond]
            self.head.config(text=text[0])
            self.rule.config(bg=AMBER)
            self.sub.config(text=text[1])
            self._set_steps(S.MODE_VERIFY, 0, -1)
            self._pack_buttons(self.b_wipe, self.b_log)
            self._show_text(text[2])
        self._update_trials()

    # ---- testing mode -------------------------------------------------------------------
    def _testing_toggle(self):
        if self.v_testing.get():
            self.test.pack(fill='x', padx=self.px(18), pady=(self.px(2), 0), before=self.foot)
        else:
            self.test.pack_forget()
        self._tags()

    def _tags(self):
        ctl = self.ctl
        ctl.testing = self.v_testing.get()
        ctl.trial_label = self.v_trial.get()
        ctl.lighting = self.v_light.get()
        ctl.movement = self.v_move.get()
        if self.v_nolock.get() and not (ctl.unlocked or ctl.no_lockout):
            self.v_nolock.set(False)                    # can only be switched on by whoever unlocked the vault
        ctl.no_lockout = self.v_nolock.get()
        self.b_nolock.config(state='normal' if (ctl.unlocked or ctl.no_lockout) else 'disabled')
        self._update_trials()

    def _update_trials(self):
        if not self.v_testing.get():
            return

        def show(counts):
            counts = counts or {}
            detail = ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))
            self.v_trials.set(f'Trials logged: {sum(counts.values())}' + (f'  ({detail})' if detail else ''))
        self.background(self.ctl.trial_counts, show)

    def _autostart(self):
        exe = winutil.app_executable()
        if exe and not winutil.set_autostart(exe, self.v_auto.get()):
            self.v_auto.set(winutil.autostart_enabled_for(exe))
            messagebox.showwarning('Start with Windows', 'The setting could not be changed.')

    # ---- actions --------------------------------------------------------------------------
    def primary(self):
        if self.worker is not None:
            return
        cond = self.ctl.condition()
        if cond == C.UNLOCKED:
            try:
                folder = self.ctl.open_vault()
            except C.Refused:
                return self.refresh()
            try:
                subprocess.Popen(['explorer', folder])
            except OSError:
                messagebox.showinfo('SecureVault', folder)
        elif cond == C.NEW:
            self.start(S.MODE_ENROL)
        elif cond == C.LOCKED:
            self.start(S.MODE_VERIFY)

    def pick_camera(self, then=None):
        dlg = CamPicker(self)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            info = dlg.result
            self.ctl.set_camera(info.index, info.backend, info.describe())
            if then:
                then()

    def start(self, mode):
        if self.worker is not None:
            return
        if self.open_source is None and self.ctl.local.get('cam_index') is None:
            return self.pick_camera(then=lambda: self.start(mode))
        try:
            session = self.ctl.begin(mode)
        except C.Refused as e:
            messagebox.showwarning('Not now', str(e))
            return self.refresh()
        self._tags()
        source = self.open_source() if self.open_source else int(self.ctl.local['cam_index'])
        self.mode = mode
        self._seq = 0
        self._seen = {}
        self.last_result = None
        self.worker = CameraWorker(source, session, finalize=self.ctl.finalize, render=self.painter,
                                   prefer=self.ctl.local.get('cam_backend'))
        self.worker.start()
        self._pack_buttons(self.b_stop)
        self._set_steps(mode, 0, 0)
        self.head.config(text='Enrolment' if mode == S.MODE_ENROL else 'Verification')
        self.rule.config(bg=AMBER)
        self.sub.config(text='Follow the instruction under the camera picture.')
        self.title.config(text='Opening the camera...')
        self._show_text('Opening the camera...')
        self._later('poll', self.POLL_MS, self._poll)

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def _poll(self):
        """Single scheduler for the camera view: re-arms itself exactly once per call."""
        w = self.worker
        if w is None or self._closing:
            return
        packet = w.latest(self._seq)
        if packet is not None:
            self._seq, status, image = packet
            self._draw(status, image)
        if w.finished:
            return self._finished(w)
        if w.state == 'finalizing':
            self.title.config(text='Checking...')
        self._later('poll', self.POLL_MS, self._poll)

    def _draw(self, status, image):
        if self._seen.get('video') is None:
            self._seen['video'] = True
            self._show_video()
            if self.worker.info is not None:
                self.ctl.camera_desc = self.worker.info.describe()
                if self.worker.info.dark:
                    self.sub.config(text='The camera picture is black. Is the lens covered?')
        if image is not None:
            self._photo = ImageTk.PhotoImage(Image.fromarray(image))
            self.video.config(image=self._photo)
        seen = self._seen
        stage = status.stage
        if seen.get('stage') != stage:
            if seen.get('stage') is not None and stage != S.ST_DONE:
                winutil.beep_sequence(SOUND_STEP)
            seen['stage'] = stage
            self.title.config(text=('ENROLMENT    ' if status.mode == S.MODE_ENROL else 'VERIFICATION    ')
                              + STAGE_TITLES.get(stage, ''))
            order = S.STAGES.index(stage) if stage in S.STAGES else 4
            self._set_steps(status.mode, order, order)
        if seen.get('prompt') != status.prompt:
            seen['prompt'] = status.prompt
            self.prompt.config(text=status.prompt, fg=RED_L if status.multi else FG)
        if seen.get('blinks') != status.blinks:
            if status.blinks > seen.get('blinks', 0):
                winutil.beep_sequence(SOUND_STEP)
            seen['blinks'] = status.blinks
        self._set_progress(status.progress)
        cfg = self.ctl.cfg
        sig = (f'{status.motion_avg:.3f} / {cfg["motion_thresh"]:.3f}', f'{status.blinks} / {cfg["blinks_needed"]}',
               '-' if status.head_calibrating else f'{status.head_delta:+.0f} %',
               f'{status.samples} / {cfg["enrol_frames"]}' if status.mode == S.MODE_ENROL and stage == S.ST_FACE
               else ('-' if status.score is None else f'{status.score:.0f} / 100'))
        if seen.get('sig') != sig:
            seen['sig'] = sig
            for var, text in zip(self.v_signals.values(), sig):
                var.set(text)

    def _finished(self, w):
        self.worker = None
        result = w.result
        self.last_result = result
        if w.state == 'failed' or result is None:
            self.refresh()
            if not self._closing:
                again = messagebox.askyesno('Camera', (w.error or 'The camera could not be used.')
                                            + '\n\nChoose another camera?')
                if again:
                    self.pick_camera()
            return
        if w.error and not self._closing:
            messagebox.showwarning('Camera', w.error)
        self._announce(result)

    def _announce(self, r):
        cfg = self.ctl.cfg
        ok = r.outcome in (S.OUT_ACCEPT, S.OUT_ENROLLED)
        winutil.beep_sequence(SOUND_OK if ok else SOUND_FAIL)
        self.refresh()
        if self._closing:
            return
        if r.outcome == S.OUT_ENROLLED:
            messagebox.showinfo('Enrolment complete',
                                f'Your face is enrolled ({r.extra.get("samples_kept", "?")} samples kept).\n'
                                'The drive is now locked: unlock it with your face to open SecureVault.')
        elif r.outcome == S.OUT_ACCEPT:
            pass                                        # the unlocked view says it all
        elif r.outcome == S.OUT_ERROR:
            messagebox.showerror('BioLock', r.message)
        elif r.outcome == S.OUT_CANCEL:
            pass
        elif r.outcome == S.OUT_TIMEOUT:
            wait = r.extra.get('cooldown_s', 0)
            messagebox.showwarning(
                'Not completed', r.message + '\n\nThis is not counted as a failed face match.'
                + (f'\nAfter several incomplete attempts you have to wait {int(wait) + 1} s.' if wait > 0 else ''))
        elif r.failed_stage == 'key':
            messagebox.showerror('Access denied', r.message + '\n\nUse  Wipe & reset  to start again on this drive.')
        elif r.extra.get('wiped'):
            messagebox.showwarning('Drive reset',
                                   f'{cfg["max_attempts"]} face matches failed.\n'
                                   'The enrolment and the contents of SecureVault were erased.')
        elif r.extra.get('counted'):
            messagebox.showwarning('Verification failed',
                                   f'Face not recognised. {r.message}\n\n'
                                   f'{r.extra.get("attempts_left", "?")} attempt(s) remaining.')
        else:
            messagebox.showinfo('Rejected (testing mode)', f'Face not recognised. {r.message}\n\n'
                                                           'Lock-out is switched off, so nothing was counted.')

    def lock(self):
        self.ctl.lock()
        winutil.beep_sequence(SOUND_FAIL)
        self.refresh()

    def view_log(self):
        px = self.px
        win = tk.Toplevel(self.root)
        win.title('Audit log')
        win.geometry(f'{px(720)}x{px(480)}')
        win.configure(bg=BG)
        win.transient(self.root)
        tk.Label(win, text='Audit log', font=(FONT, 14, 'bold'), fg=FG, bg=BG).pack(pady=(px(14), px(4)))
        frame = tk.Frame(win, bg=CARD)
        frame.pack(fill='both', expand=True, padx=px(16), pady=px(10))
        bar = ttk.Scrollbar(frame)
        bar.pack(side='right', fill='y')
        text = tk.Text(frame, bg=CARD, fg=FG, font=(MONO, 9), relief='flat', bd=0, yscrollcommand=bar.set)
        text.pack(fill='both', expand=True, padx=px(8), pady=px(8))
        bar.config(command=text.yview)
        text.insert('1.0', self.ctl.store.read_log() or 'No entries yet.')
        text.see('end')
        text.config(state='disabled')

    def wipe_prompt(self):
        if self.worker is not None:
            return
        if not messagebox.askyesno('Wipe & reset',
                                   'This permanently deletes:\n\n  -  the face enrolment\n'
                                   '  -  ALL files in SecureVault\n  -  the audit log and settings\n\n'
                                   'It cannot be undone. Continue?', icon='warning', default='no'):
            return
        if messagebox.askyesno('Final confirmation', 'Everything in SecureVault will be lost. Wipe now?',
                               icon='warning', default='no'):
            self.ctl.wipe('requested in the app')
            self.v_nolock.set(False)
            self._tags()
            self.refresh()
            messagebox.showinfo('Wipe complete', 'The drive was reset. You can enrol a new face.')

    def about(self):
        px = self.px
        win = tk.Toplevel(self.root)
        win.title('About BioLock')
        win.configure(bg=BG)
        win.geometry(f'{px(430)}x{px(300)}')
        win.resizable(False, False)
        win.transient(self.root)
        tk.Label(win, text='BioLock SSD', font=(FONT, 18, 'bold'), fg=FG, bg=BG).pack(pady=(px(26), 0))
        tk.Label(win, text=f'Version {__version__}', font=(FONT, 10), fg=MUTED, bg=BG).pack(pady=(px(2), px(12)))
        tk.Label(win, text='Face-recognition lock for external drives with\nmotion, blink and head-movement '
                           'liveness cues.\n\nSecureVault is a hidden folder, not an encrypted one.\n'
                           'Use BitLocker or VeraCrypt if you need encryption.',
                 font=(FONT, 9), fg=MUTED, bg=BG, justify='center').pack()
        tk.Label(win, text='Pasupuleti Vedh Rishi', font=(FONT, 8), fg=FAINT, bg=BG).pack(pady=(px(16), px(6)))
        tk.Button(win, text='Close', command=win.destroy, bg=CARD, fg=FG, relief='flat', bd=0, padx=px(20),
                  pady=px(5)).pack()

    # ---- window ---------------------------------------------------------------------------------
    def _fullscreen(self):
        self._full = not self._full
        self.root.attributes('-fullscreen', self._full)

    def _escape(self):
        if self._full:
            self._fullscreen()
        elif self.worker is not None:
            self.cancel()

    def quit(self):
        if self._closing:
            return
        if self.ctl.unlocked:
            answer = messagebox.askyesnocancel(
                'Close BioLock', 'Lock the vault before closing?\n\n'
                                 'No leaves SecureVault visible until BioLock is started again.', default='yes')
            if answer is None:
                return
            if answer:
                self.ctl.lock('closing')
        self.shutdown()

    def shutdown(self):
        """Stop the timers and the camera, then close the window."""
        self._closing = True
        for timer in self._after.values():
            try:
                self.root.after_cancel(timer)
            except tk.TclError:
                pass
        w, self.worker = self.worker, None
        if w is not None:
            w.cancel()                  # an attempt that has reached the face match counts as failed
            w.join(timeout=3.0)
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def run(base=None):
    """Start the application. Returns the process exit code."""
    winutil.enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    from .config import app_dir
    base = base or app_dir()
    if not enforce_external_drive(base):
        root.destroy()
        return 0
    splash = Splash(root)
    try:
        splash.say('Loading the face engine...')
        ctl = C.Controller(base)
    except Exception as e:
        splash.close()
        messagebox.showerror('BioLock could not start', f'{type(e).__name__}: {e}')
        root.destroy()
        return 1
    splash.say('Building the window...')
    App(root, ctl)
    splash.close()
    root.deiconify()
    root.mainloop()
    return 0
