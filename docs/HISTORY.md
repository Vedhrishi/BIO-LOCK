# Project history

Three stages: the original program (source lost), version 9 (rebuilt from a compiled build, one 2,000-line
`main.py`), and version 10 (rewritten as the `biolock` package, with measured parameters).

## Version 10: what changed and why

Version 9 worked, but reading it closely and measuring it turned up the problems below. Version 10 fixes them.

| # | Version 9 | Version 10 |
|---|---|---|
| 1 | Unlocked on the first frame at or above the threshold (up to 41 frames tried) | Decision on the median of 15 frames |
| 2 | Squashed every camera frame to 640x360 | 640 wide, aspect ratio kept |
| 3 | "Histogram" score compared grey histograms of equalised crops (at chance) | Gradient-orientation (HOG) correlation; measured, and given weight 0 because it does not help |
| 4 | Loaded a pickle file from the removable drive (code execution risk) | Template is a plain `.npz`, loaded without pickle, sizes and values checked |
| 5 | Vault key derived from the stored mean face only | Key bound to the whole template, a drive identifier and the volume serial number |
| 6 | Wrote a Windows sign-in entry on every start | Only if the user ticks a box, packaged exe only |
| 7 | Started `attrib.exe` for every file read and write | File attributes set through the Windows API |
| 8 | One log row per processed frame on the USB drive | One row per attempt |
| 9 | Sidebar cut off on 1366x768 screens, blurry on high-DPI screens | Scrollable sidebar, DPI aware |
| 10 | Liveness ran on the UI thread; motion was only measured when two face boxes happened to have the same size, between crops at different positions | All per-frame work on a worker thread; motion measured inside the same box, on block-averaged pixels so that sensor noise does not count as movement |
| 11 | Lighting check on an equalised frame (always "fine"); prompt said "stay still" while the check needed motion | Lighting check on the raw frame; prompts say what is needed |
| 12 | Cascade files could not be loaded from a path with non-ASCII characters | Loaded from memory |
| 13 | Camera: DirectShow only, MJPG forced, opened on the UI thread | DirectShow, then Media Foundation, then OpenCV's default; black-frame check; opened on a worker thread |
| 14 | Unlimited liveness retries | Cool-down after repeated time-outs |

Found while rewriting, and also fixed:

- An eye-detector drop-out caused by a moving or blurred photograph counted as a blink. A blink now also needs
  the image to change mainly in the eye band.
- The liveness steps could be done by one face and the match by another (or by a photograph shown at the end).
  The face seen during the blink step now has to match as well.
- Deleting the state file made the app offer a new enrolment over the existing vault. Whether a drive is enrolled
  now follows from the template and key files, and a vault without an enrolment has to be wiped first.
- The lock-out could be switched off from the Testing panel by anyone. It can now only be switched off while the
  vault is unlocked.
- A `config.json` next to the exe or an environment variable could lower the threshold. The packaged exe now
  ignores both for everything that decides an unlock.
- Backing out during the face match avoided a counted failure. A cancelled or timed-out match now counts.

The matcher itself changed too: eye alignment, Tan-Triggs normalisation, LBPH radius 5 with a 6x6 grid, LBPH
alone instead of a weighted fusion, threshold 35 on a recalibrated score. `research/README.md` has the search,
`docs/RESULTS.md` the measurements.

Version 9 enrolments cannot be read by version 10 (different template, different key). The app says so and
offers a wipe; it never opens the old pickle file.

## How the first version was rebuilt

The original `.py` was lost; the first version of `main.py` was rebuilt from a compiled build. The raw decompiler
output used for that is not kept in this repository.

`BioLock.exe` was a PyInstaller-frozen build (Python 3.12) and the
original `.py` source wasn't available, so `main.py` in this repo was
rebuilt from the compiled app itself:

1. `pyinstxtractor-ng` extracted the PyInstaller archive from the `.exe`,
   recovering the compiled `main.pyc` bytecode .
2. [`pycdc`](https://github.com/zrax/pycdc) (built from source, since it's
   one of the few decompilers with any Python 3.12 opcode support) was
   run against that bytecode. 
3. `pycdas` produced a full bytecode disassembly for cross-reference:
   a disassembly (not kept).

## Why `main.py` isn't a byte-for-byte recovery

Python 3.12's bytecode (the adaptive/specializing interpreter, opcodes
like `CALL_INTRINSIC_1`, `LOAD_FAST_CHECK`, `POP_JUMP_IF_NOT_NONE`, the
newer exception-table format, etc.) isn't fully handled by public
decompilers yet. `pycdc`'s raw output recovered roughly 55-60% of the
app cleanly — all imports, constants, the theme/colour palette, simple
I/O helpers, state/config persistence, and the wipe logic came back
correct. The rest — notably the live camera loop, the animated overlay
renderer, the Tkinter layout, and the camera-picker dialog — came back
empty (`WARNING: Decompyle incomplete`) or corrupted, and were
**rewritten by hand**, based on the recovered function signatures, how
each return value is consumed elsewhere in the file, and standard
implementations of the same technique (Haar-cascade detection, LBPH
recognition, eye-cascade blink detection, frame-diff liveness).

| Area | Status |
|---|---|
| Imports, constants, theme colours, paths | Recovered cleanly |
| `_hide/_unhide/_log/_beep/*sound*` | Recovered cleanly |
| Drive helpers, drive-id, autorun | Recovered, minor cleanup |
| State/config load & save | Recovered, minor cleanup |
| Vault key derivation (Fernet/PBKDF2) | Recovered cleanly |
| Cascade lookup, lighting quality | Recovered cleanly |
| `BlinkDetector` / `HeadTurn` / `MotionLive` | **Reconstructed** |
| `FaceEngine` (`_load`, part of `train`) | **Reconstructed**; `predict()` additionally redesigned as a weighted LBPH+Histogram+Template fusion for the experiments (see `docs/EXPERIMENTS.md`) |
| `ParticleSystem` / `Anim.render` | **Reconstructed** |
| `_scan_cams` / `CamPicker` | **Reconstructed** |
| `BioLock.__init__` / `_build_ui` | **Reconstructed** (now also includes the Testing Mode panel used for experiment logging) |
| Camera loop (`_tick`, `_loop`, `_start`, `_open_cam`, `_enrol`, `_verify`) | **Reconstructed** |
| `Splash`, `show_about` | Recovered cleanly |

## If you want to double-check a specific function

Open a disassembly (not kept) and search for the
function name — you'll get the exact Python 3.12 bytecode for it,
which you can read instruction-by-instruction if you want to confirm
or correct a specific piece of logic in `main.py`.
